"""
Reverse Diffusion Process and Sampling Algorithms for DDPM

This module implements the reverse diffusion process to generate images from noise.
Key concepts:
- Reverse diffusion: Iteratively denoise from pure Gaussian noise
- Sampling schedules: Different strategies for choosing timesteps
- Posterior variance: Sampling distribution for reverse process
"""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from tqdm import tqdm


class DDPMSampler:
    """
    Sampling algorithm for DDPM (Denoising Diffusion Probabilistic Models).

    Implements the reverse diffusion process:
    - Start with pure Gaussian noise x_T ~ N(0, I)
    - Iteratively denoise: x_{t-1} = reverse_step(x_t, t)
    - End with generated image x_0

    Mathematical formula for reverse step:
        x_{t-1} = (1/√α_t) * (x_t - (1-α_t)/√(1-ᾱ_t) * ε_θ(x_t, t)) + σ_t * z

    Where:
    - ε_θ: Trained noise prediction network (U-Net)
    - σ_t: Posterior variance
    - z: Gaussian noise (for stochastic sampling)
    """

    def __init__(self, noise_scheduler, num_inference_steps=1000):
        """
        Initialize DDPM sampler.

        Args:
            noise_scheduler: NoiseScheduler instance with pre-computed coefficients
            num_inference_steps: Number of reverse diffusion steps (T)
        """
        self.noise_scheduler = noise_scheduler
        self.num_inference_steps = num_inference_steps

        # Get all coefficients
        timesteps = torch.arange(0, noise_scheduler.num_timesteps)
        self.sqrt_alphas_cumprod = noise_scheduler.sqrt_alphas_cumprod
        self.sqrt_one_minus_alphas_cumprod = (
            noise_scheduler.sqrt_one_minus_alphas_cumprod
        )
        self.betas = noise_scheduler.betas
        self.alphas = noise_scheduler.alphas

        # Compute posterior variance
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1), noise_scheduler.alphas_cumprod[:-1]]
        )
        self.posterior_variance = (
            self.betas
            * (1.0 - alphas_cumprod_prev)
            / (1.0 - noise_scheduler.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            self.posterior_variance.clamp(min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas
            * torch.sqrt(alphas_cumprod_prev)
            / (1.0 - noise_scheduler.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - noise_scheduler.alphas_cumprod)
        )

    def set_timesteps(self, num_inference_steps: int, schedule: str = "linear"):
        """
        Set timesteps for sampling.

        Different schedules determine how many denoising steps to use:

        Args:
            num_inference_steps: Number of steps to use (e.g., 50, 100, 1000)
            schedule: "linear" (uniform spacing) or "cosine" (more steps early on)

        Schedule comparison:
        - linear: Evenly spaced timesteps (fast but less accurate)
        - cosine: More steps at high noise, fewer at low noise (better quality)

        Mathematical formulas:
        Linear: t_i = T * i / N
        Cosine: t_i = T * cos²(π * i / (2*N))
        """
        self.num_inference_steps = num_inference_steps
        self.device = None  # Will be set during sampling

        if schedule == "linear":
            # Linear spacing: evenly distributed through timesteps
            timesteps = torch.linspace(
                0,
                self.noise_scheduler.num_timesteps - 1,
                num_inference_steps,
                dtype=torch.long,
            )
        elif schedule == "cosine":
            # Cosine schedule: more steps at high noise, fewer at low noise
            # This focuses computational budget on high-noise (hard) steps
            s = 0.008  # Offset parameter (prevents very small timesteps)
            steps = torch.arange(num_inference_steps + 1, dtype=torch.float32)
            alphas_cumprod = (
                torch.cos((steps / num_inference_steps + s) / (1 + s) * torch.pi * 0.5)
                ** 2
            )
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

            # Map to actual timesteps
            timesteps = []
            for i in range(num_inference_steps):
                target_alpha = alphas_cumprod[i + 1]
                # Find closest timestep with this alpha value
                t = torch.argmin(
                    torch.abs(self.noise_scheduler.alphas_cumprod - target_alpha)
                )
                timesteps.append(t)
            timesteps = torch.tensor(
                timesteps[::-1], dtype=torch.long
            )  # Reverse: high to low noise
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        self.timesteps = timesteps
        self.sigmas = torch.sqrt(self.posterior_variance[timesteps])

    def step(
        self, model_output: torch.Tensor, timestep: torch.Tensor, sample: torch.Tensor
    ) -> torch.Tensor:
        """
        Single reverse diffusion step.

        Args:
            model_output: Predicted noise from U-Net (batch, channels, H, W)
            timestep: Current timestep t (batch,)
            sample: Current noisy image x_t (batch, channels, H, W)

        Returns:
            Denoised sample x_{t-1}

        Mathematical implementation:
        1. Compute mean: m_t = (1/√α_t) * (x_t - (1-α_t)/√(1-ᾱ_t) * ε_θ)
        2. Add variance: x_{t-1} = m_t + σ_t * z, where z ~ N(0, I)
        """
        # Get coefficients for this timestep
        sqrt_alpha = self.sqrt_alphas_cumprod[timestep]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timestep]

        # Reshape for broadcasting
        if len(sqrt_alpha.shape) == 1:
            sqrt_alpha = sqrt_alpha[:, None, None, None]
            sqrt_one_minus_alpha = sqrt_one_minus_alpha[:, None, None, None]

        # Posterior mean
        pred_original_sample = (
            sample - sqrt_one_minus_alpha * model_output
        ) / sqrt_alpha

        # Posterior mean coefficient
        coef1 = self.posterior_mean_coef1[timestep]
        coef2 = self.posterior_mean_coef2[timestep]

        if len(coef1.shape) == 1:
            coef1 = coef1[:, None, None, None]
            coef2 = coef2[:, None, None, None]

        mean = coef1 * pred_original_sample + coef2 * sample

        # Add variance (stochastic part)
        variance = self.sigmas[timestep]
        if len(variance.shape) == 1:
            variance = variance[:, None, None, None]

        z = torch.randn_like(sample)
        sample = mean + variance * z

        return sample

    def __call__(
        self,
        model: nn.Module,
        batch_size: int = 4,
        num_inference_steps: int = 50,
        schedule: str = "linear",
        device: torch.device = torch.device("cpu"),
        guidance_scale: Optional[float] = None,
        return_trajectory: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Generate images by sampling from the model.

        Args:
            model: Trained U-Net model
            batch_size: Number of images to generate
            num_inference_steps: Number of denoising steps
            schedule: Sampling schedule ("linear" or "cosine")
            device: Device to use for computation
            guidance_scale: Unconditional guidance scale (for future extensions)
            return_trajectory: If True, return intermediate steps for visualization

        Returns:
            samples: Generated images (batch, 1, 28, 28)
            trajectory: List of intermediate steps (if return_trajectory=True)

        Implementation:
        1. Start with pure Gaussian noise
        2. For each timestep (high to low):
            - Predict noise with U-Net
            - Take reverse step toward denoised image
        3. Return final denoised image
        """
        self.device = device
        self.set_timesteps(num_inference_steps, schedule)

        # Start with pure Gaussian noise
        sample = torch.randn(batch_size, 1, 28, 28, device=device)

        trajectory = [] if return_trajectory else None

        model.eval()
        with torch.no_grad():
            for i, t in enumerate(tqdm(self.timesteps, desc=f"Sampling ({schedule})")):
                # Prepare timestep tensor
                t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

                # Predict noise
                noise_pred = model(sample, t_tensor)

                # Take reverse step
                sample = self.step(noise_pred, t_tensor, sample)

                # Track trajectory
                if return_trajectory and i % max(1, len(self.timesteps) // 5) == 0:
                    trajectory.append(sample.cpu().clone())

        return sample, trajectory


class FastSampler(DDPMSampler):
    """
    Fast sampling with reduced number of steps.

    Key insight: We don't need all 1000 timesteps for reasonable samples.
    Using 50 steps instead of 1000 gives ~20x speedup with minor quality loss.

    Trade-offs:
    - Fast: 50 steps runs in ~1 second per image
    - Slow: 1000 steps runs in ~20 seconds per image
    - Quality: Minimal difference for most applications
    """

    def __call__(
        self,
        model: nn.Module,
        batch_size: int = 4,
        device: torch.device = torch.device("cpu"),
        schedule: str = "linear",
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Fast sampling with only 50 steps."""
        return super().__call__(
            model,
            batch_size=batch_size,
            num_inference_steps=50,
            schedule=schedule,
            device=device,
            return_trajectory=True,
        )


class SamplingComparison:
    """
    Utilities for comparing different sampling strategies.

    Enables comparison of:
    1. Different number of steps (50 vs 1000)
    2. Different schedules (linear vs cosine)
    3. Speed vs quality trade-offs
    """

    def __init__(self, noise_scheduler):
        """Initialize comparison utilities."""
        self.noise_scheduler = noise_scheduler
        self.sampler = DDPMSampler(noise_scheduler)

    def compare_schedules(
        self,
        model: nn.Module,
        num_steps: int = 50,
        device: torch.device = torch.device("cpu"),
    ) -> dict:
        """
        Compare linear vs cosine sampling schedules.

        Returns:
            Dictionary with results from both schedules
        """
        results = {}

        for schedule in ["linear", "cosine"]:
            samples, trajectory = self.sampler(
                model,
                batch_size=4,
                num_inference_steps=num_steps,
                schedule=schedule,
                device=device,
                return_trajectory=True,
            )
            results[schedule] = {"samples": samples, "trajectory": trajectory}

        return results

    def compare_step_counts(
        self,
        model: nn.Module,
        steps_list: List[int] = None,
        device: torch.device = torch.device("cpu"),
    ) -> dict:
        """
        Compare different number of sampling steps.

        Example:
            steps_list = [10, 50, 100, 1000]
            Results show quality improvement vs computational cost
        """
        if steps_list is None:
            steps_list = [10, 50, 100, 1000]

        results = {}

        for num_steps in steps_list:
            samples, _ = self.sampler(
                model,
                batch_size=4,
                num_inference_steps=num_steps,
                schedule="linear",
                device=device,
                return_trajectory=False,
            )
            results[num_steps] = samples

        return results

    @staticmethod
    def visualize_trajectory(
        trajectory: List[torch.Tensor], title: str = "Denoising Process"
    ):
        """
        Visualize denoising trajectory (noise → image).

        Args:
            trajectory: List of tensors showing denoising steps
            title: Title for visualization
        """
        fig, axes = plt.subplots(1, len(trajectory), figsize=(15, 3))

        for i, sample in enumerate(trajectory):
            img = sample[0, 0].cpu().numpy()
            axes[i].imshow(img, cmap="gray")
            axes[i].set_title(f"Step {i}")
            axes[i].axis("off")

        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        return fig

    @staticmethod
    def create_sample_grid(samples: torch.Tensor, nrow: int = 4) -> torch.Tensor:
        """
        Create grid of generated samples for visualization.

        Args:
            samples: Generated images (batch, channels, H, W)
            nrow: Number of samples per row

        Returns:
            Grid image tensor
        """
        # Normalize to [0, 1] for visualization
        samples_vis = (samples + 1) / 2
        grid = make_grid(samples_vis, nrow=nrow, normalize=False)
        return grid

    @staticmethod
    def plot_comparison_grid(results: dict, title: str = "Schedule Comparison"):
        """
        Plot side-by-side comparison of different sampling strategies.

        Args:
            results: Dictionary with results from different schedules
            title: Title for the figure
        """
        fig, axes = plt.subplots(len(results), 4, figsize=(12, 3 * len(results)))

        if len(results) == 1:
            axes = axes[np.newaxis, :]

        for row, (name, data) in enumerate(results.items()):
            samples = data["samples"]
            for col in range(4):
                img = samples[col, 0].cpu().numpy()
                axes[row, col].imshow(img, cmap="gray")
                axes[row, col].set_title(f"{name} - {col}")
                axes[row, col].axis("off")

        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        return fig


# ============================================================================
# Utility Functions
# ============================================================================


def load_pretrained_model(model_path: str, device: torch.device) -> nn.Module:
    """
    Load a pre-trained U-Net model.

    Args:
        model_path: Path to saved model checkpoint
        device: Device to load model onto

    Returns:
        Loaded model in eval mode
    """
    # Import model architecture (would come from module 16)
    # For now, placeholder
    state_dict = torch.load(model_path, map_location=device)
    # model = SimpleUNet(...)
    # model.load_state_dict(state_dict)
    # model = model.to(device).eval()
    # return model
    pass


def calculate_fid_score(
    real_samples: torch.Tensor, fake_samples: torch.Tensor
) -> float:
    """
    Calculate Fréchet Inception Distance between real and generated samples.

    Lower FID = better quality

    Note: Simplified version. Full FID requires Inception network.
    """
    # Compute feature statistics
    real_mean = real_samples.mean()
    real_var = real_samples.var()

    fake_mean = fake_samples.mean()
    fake_var = fake_samples.var()

    # Simplified FID (would need Inception features in practice)
    fid = ((real_mean - fake_mean) ** 2 + (real_var - fake_var) ** 2) ** 0.5
    return fid.item()


def visualize_denoise_steps(
    model: nn.Module, noise_scheduler, device: torch.device, num_steps: int = 10
) -> plt.Figure:
    """
    Visualize the complete denoising process from noise to image.

    Shows intermediate steps to understand how model gradually removes noise.

    Args:
        model: Trained U-Net
        noise_scheduler: Scheduler with coefficients
        device: Computation device
        num_steps: Number of intermediate steps to visualize

    Returns:
        Matplotlib figure with denoising progression
    """
    sampler = DDPMSampler(noise_scheduler, num_inference_steps=1000)
    samples, trajectory = sampler(
        model,
        batch_size=1,
        num_inference_steps=1000,
        schedule="linear",
        device=device,
        return_trajectory=True,
    )

    return SamplingComparison.visualize_trajectory(
        trajectory, title="Complete Denoising Process"
    )


if __name__ == "__main__":
    print("Diffusion Sampling Module")
    print("=" * 60)
    print("Key Components:")
    print("1. DDPMSampler: Main sampling algorithm")
    print("2. FastSampler: Accelerated sampling (50 steps)")
    print("3. SamplingComparison: Compare different strategies")
    print("4. Utility functions: Visualization and evaluation")
    print("\nUsage:")
    print("  sampler = DDPMSampler(noise_scheduler)")
    print("  samples = sampler(model, batch_size=4, schedule='cosine')")
