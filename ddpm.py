import argparse  # Parse command-line arguments for training configuration
import os  # Create output directories for experiment artifacts

import torch  # Core PyTorch tensors and autograd
from torch import nn  # Neural network layers and utilities
from torch.nn import functional as F  # Functional ops (mse_loss, pad, etc.)
from torch.utils.data import DataLoader  # Batch iterator over the dataset
from tqdm.auto import tqdm  # Progress bars for training and sampling

import matplotlib.pyplot as plt  # Plotting for saved sample images
import numpy as np  # Numpy arrays for frames/losses saving

import datasets  # Local module providing 2D toy datasets
from positional_embeddings import PositionalEmbedding  # Embeddings for inputs and timesteps


class Block(nn.Module):  # Residual feed-forward block used in the MLP denoiser
    def __init__(self, size: int):  # size: width of the hidden representation
        super().__init__()  # Initialize the nn.Module base class

        self.ff = nn.Linear(size, size)  # Linear layer keeping the same width
        self.act = nn.GELU()  # GELU non-linearity

    def forward(self, x: torch.Tensor):  # x: (batch, size) hidden activations
        return x + self.act(self.ff(x))  # Residual connection around linear + GELU


class MLP(nn.Module):  # Denoiser network predicting the noise added to a 2D point
    def __init__(self, hidden_size: int = 128, hidden_layers: int = 3, emb_size: int = 128,
                 time_emb: str = "sinusoidal", input_emb: str = "sinusoidal"):  # Model hyperparameters
        super().__init__()  # Initialize the nn.Module base class

        self.time_mlp = PositionalEmbedding(emb_size, time_emb)  # Embedding for the timestep t
        self.input_mlp1 = PositionalEmbedding(emb_size, input_emb, scale=25.0)  # Embedding for x coordinate
        self.input_mlp2 = PositionalEmbedding(emb_size, input_emb, scale=25.0)  # Embedding for y coordinate

        concat_size = len(self.time_mlp.layer) + \
            len(self.input_mlp1.layer) + len(self.input_mlp2.layer)  # Total width after concatenating embeddings
        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]  # Project embeddings to hidden width
        for _ in range(hidden_layers):  # Stack the requested number of residual blocks
            layers.append(Block(hidden_size))  # Add one residual block
        layers.append(nn.Linear(hidden_size, 2))  # Output layer predicts 2D noise vector
        self.joint_mlp = nn.Sequential(*layers)  # Compose all layers into one module

    def forward(self, x, t):  # x: (batch, 2) noisy points, t: (batch,) timesteps
        x1_emb = self.input_mlp1(x[:, 0])  # Embed the x coordinate
        x2_emb = self.input_mlp2(x[:, 1])  # Embed the y coordinate
        t_emb = self.time_mlp(t)  # Embed the timestep
        x = torch.cat((x1_emb, x2_emb, t_emb), dim=-1)  # Concatenate all embeddings
        x = self.joint_mlp(x)  # Run through the MLP to predict noise
        return x  # Return predicted noise of shape (batch, 2)


class NoiseScheduler():  # Precomputes DDPM schedule constants and implements forward/reverse steps
    def __init__(self,
                 num_timesteps=1000,  # Number of diffusion steps T
                 beta_start=0.0001,  # Beta at the first timestep
                 beta_end=0.02,  # Beta at the last timestep
                 beta_schedule="linear"):  # How betas are spaced between start and end

        self.num_timesteps = num_timesteps  # Store T for __len__ and training
        if beta_schedule == "linear":  # Evenly spaced betas
            self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float32)  # Linear beta schedule
        elif beta_schedule == "quadratic":  # Quadratically spaced betas
            self.betas = torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=torch.float32) ** 2  # Quadratic beta schedule

        self.alphas = 1.0 - self.betas  # alpha_t = 1 - beta_t
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)  # alpha_bar_t = prod of alphas up to t
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.)  # alpha_bar_{t-1}, with alpha_bar_{-1} = 1

        # required for self.add_noise
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5  # sqrt(alpha_bar_t) scales the clean data
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5  # sqrt(1 - alpha_bar_t) scales the noise

        # required for reconstruct_x0
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)  # sqrt(1 / alpha_bar_t)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(
            1 / self.alphas_cumprod - 1)  # sqrt(1 / alpha_bar_t - 1)

        # required for q_posterior
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)  # Coefficient on x_0 in posterior mean
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)  # Coefficient on x_t in posterior mean

    def reconstruct_x0(self, x_t, t, noise):  # Estimate clean x_0 from noisy x_t and predicted noise
        s1 = self.sqrt_inv_alphas_cumprod[t]  # Look up sqrt(1 / alpha_bar_t)
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t]  # Look up sqrt(1 / alpha_bar_t - 1)
        s1 = s1.reshape(-1, 1)  # Reshape for broadcasting over (batch, 2)
        s2 = s2.reshape(-1, 1)  # Reshape for broadcasting over (batch, 2)
        return s1 * x_t - s2 * noise  # x_0 = s1 * x_t - s2 * predicted noise

    def q_posterior(self, x_0, x_t, t):  # Mean of q(x_{t-1} | x_t, x_0)
        s1 = self.posterior_mean_coef1[t]  # Coefficient on x_0 at timestep t
        s2 = self.posterior_mean_coef2[t]  # Coefficient on x_t at timestep t
        s1 = s1.reshape(-1, 1)  # Reshape for broadcasting over (batch, 2)
        s2 = s2.reshape(-1, 1)  # Reshape for broadcasting over (batch, 2)
        mu = s1 * x_0 + s2 * x_t  # Posterior mean as weighted sum of x_0 and x_t
        return mu  # Return the mean of the previous-step distribution

    def get_variance(self, t):  # Variance of q(x_{t-1} | x_t, x_0) at timestep t
        if t == 0:  # No noise is added at the final (t=0) step
            return 0  # Zero variance at t=0

        variance = self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])  # DDPM posterior variance formula
        variance = variance.clip(1e-20)  # Avoid exactly zero for numerical stability
        return variance  # Return the clipped variance

    def step(self, model_output, timestep, sample):  # One reverse diffusion step: x_t -> x_{t-1}
        t = timestep  # Alias for readability
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)  # Estimate x_0 from predicted noise
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)  # Posterior mean for x_{t-1}

        variance = 0  # Default: deterministic step (used when t == 0)
        if t > 0:  # Add stochastic noise for all but the last step
            noise = torch.randn_like(model_output)  # Fresh Gaussian noise
            variance = (self.get_variance(t) ** 0.5) * noise  # Scale noise by posterior std dev

        pred_prev_sample = pred_prev_sample + variance  # Sample x_{t-1} = mean + noise

        return pred_prev_sample  # Return the denoised sample for the previous timestep

    def add_noise(self, x_start, x_noise, timesteps):  # Forward process: noise x_0 to x_t in one shot
        s1 = self.sqrt_alphas_cumprod[timesteps]  # sqrt(alpha_bar_t) per batch element
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]  # sqrt(1 - alpha_bar_t) per batch element

        s1 = s1.reshape(-1, 1)  # Reshape for broadcasting over (batch, 2)
        s2 = s2.reshape(-1, 1)  # Reshape for broadcasting over (batch, 2)

        return s1 * x_start + s2 * x_noise  # x_t = s1 * x_0 + s2 * noise

    def __len__(self):  # Allow len(scheduler) to give the number of timesteps
        return self.num_timesteps  # Return T


if __name__ == "__main__":  # Training entrypoint when run as a script
    parser = argparse.ArgumentParser()  # Build the CLI argument parser
    parser.add_argument("--experiment_name", type=str, default="base")  # Name for the output folder under exps/
    parser.add_argument("--dataset", type=str, default="dino", choices=["circle", "dino", "line", "moons"])  # Which 2D toy dataset to train on
    parser.add_argument("--train_batch_size", type=int, default=32)  # Batch size for training
    parser.add_argument("--eval_batch_size", type=int, default=1000)  # Number of points generated at eval time
    parser.add_argument("--num_epochs", type=int, default=200)  # Total training epochs
    parser.add_argument("--learning_rate", type=float, default=1e-3)  # AdamW learning rate
    parser.add_argument("--num_timesteps", type=int, default=50)  # Diffusion steps T
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "quadratic"])  # Beta spacing scheme
    parser.add_argument("--embedding_size", type=int, default=128)  # Width of positional embeddings
    parser.add_argument("--hidden_size", type=int, default=128)  # Width of MLP hidden layers
    parser.add_argument("--hidden_layers", type=int, default=3)  # Number of residual blocks
    parser.add_argument("--time_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "zero"])  # Timestep embedding type
    parser.add_argument("--input_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "identity"])  # Input coordinate embedding type
    parser.add_argument("--save_images_step", type=int, default=1)  # Generate eval samples every N epochs
    config = parser.parse_args()  # Parse CLI args into a config namespace

    dataset = datasets.get_dataset(config.dataset)  # Load the chosen 2D dataset
    dataloader = DataLoader(
        dataset, batch_size=config.train_batch_size, shuffle=True, drop_last=True)  # Shuffled training batches

    model = MLP(
        hidden_size=config.hidden_size,  # Hidden layer width
        hidden_layers=config.hidden_layers,  # Number of residual blocks
        emb_size=config.embedding_size,  # Embedding width
        time_emb=config.time_embedding,  # Timestep embedding type
        input_emb=config.input_embedding)  # Input embedding type

    noise_scheduler = NoiseScheduler(
        num_timesteps=config.num_timesteps,  # Diffusion steps T
        beta_schedule=config.beta_schedule)  # Beta spacing scheme

    optimizer = torch.optim.AdamW(
        model.parameters(),  # Optimize all model parameters
        lr=config.learning_rate,  # Learning rate from CLI
    )

    global_step = 0  # Total number of optimizer steps taken
    frames = []  # Generated samples per epoch (for visualizing learning)
    losses = []  # Per-step training losses
    print("Training model...")  # Announce training start
    for epoch in range(config.num_epochs):  # Loop over epochs
        model.train()  # Enable training mode
        progress_bar = tqdm(total=len(dataloader))  # Progress bar over batches
        progress_bar.set_description(f"Epoch {epoch}")  # Show epoch number
        for step, batch in enumerate(dataloader):  # Loop over training batches
            batch = batch[0]  # Unpack the (points,) tuple from TensorDataset
            noise = torch.randn(batch.shape)  # Sample Gaussian noise for this batch
            timesteps = torch.randint(
                0, noise_scheduler.num_timesteps, (batch.shape[0],)
            ).long()  # Random timestep per example

            noisy = noise_scheduler.add_noise(batch, noise, timesteps)  # Forward-noise the batch to x_t
            noise_pred = model(noisy, timesteps)  # Predict the added noise
            loss = F.mse_loss(noise_pred, noise)  # MSE between predicted and true noise
            loss.backward()  # Backpropagate gradients

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Clip gradient norm to 1.0
            optimizer.step()  # Update model parameters
            optimizer.zero_grad()  # Reset gradients for the next step

            progress_bar.update(1)  # Advance the progress bar
            logs = {"loss": loss.detach().item(), "step": global_step}  # Metrics to display
            losses.append(loss.detach().item())  # Record loss for saving later
            progress_bar.set_postfix(**logs)  # Show metrics on the progress bar
            global_step += 1  # Increment global step counter
        progress_bar.close()  # Finish the epoch's progress bar

        if epoch % config.save_images_step == 0 or epoch == config.num_epochs - 1:  # Periodically (and at the end) sample from the model
            # generate data with the model to later visualize the learning process
            model.eval()  # Switch to eval mode for sampling
            sample = torch.randn(config.eval_batch_size, 2)  # Start from pure Gaussian noise
            timesteps = list(range(len(noise_scheduler)))[::-1]  # Reverse timesteps: T-1 down to 0
            for i, t in enumerate(tqdm(timesteps)):  # Denoise step by step
                t = torch.from_numpy(np.repeat(t, config.eval_batch_size)).long()  # Broadcast t to the batch
                with torch.no_grad():  # No gradients needed during sampling
                    residual = model(sample, t)  # Predict noise at this timestep
                sample = noise_scheduler.step(residual, t[0], sample)  # Take one reverse step
            frames.append(sample.numpy())  # Save generated points for this epoch

    print("Saving model...")  # Announce checkpoint saving
    outdir = f"exps/{config.experiment_name}"  # Experiment output directory
    os.makedirs(outdir, exist_ok=True)  # Create it if it doesn't exist
    torch.save(model.state_dict(), f"{outdir}/model.pth")  # Save trained weights

    print("Saving images...")  # Announce image saving
    imgdir = f"{outdir}/images"  # Directory for per-epoch scatter plots
    os.makedirs(imgdir, exist_ok=True)  # Create it if it doesn't exist
    frames = np.stack(frames)  # Stack per-epoch samples into one array
    xmin, xmax = -6, 6  # X-axis limits for saved plots
    ymin, ymax = -6, 6  # Y-axis limits for saved plots
    for i, frame in enumerate(frames):  # Save one image per stored frame
        plt.figure(figsize=(10, 10))  # New square figure
        plt.scatter(frame[:, 0], frame[:, 1])  # Plot generated 2D points
        plt.xlim(xmin, xmax)  # Fix x limits for comparability
        plt.ylim(ymin, ymax)  # Fix y limits for comparability
        plt.savefig(f"{imgdir}/{i:04}.png")  # Save as zero-padded PNG
        plt.close()  # Close the figure to free memory

    print("Saving loss as numpy array...")  # Announce loss saving
    np.save(f"{outdir}/loss.npy", np.array(losses))  # Save per-step losses

    print("Saving frames...")  # Announce frames saving
    np.save(f"{outdir}/frames.npy", frames)  # Save generated samples for visualization
