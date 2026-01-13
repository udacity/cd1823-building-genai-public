"""
Simple Diffusion Model Training
Dataset: MNIST (28x28 grayscale)

Key Differences from GANs:
1. MSE Loss: Deterministic gradient flow (no adversarial oscillation)
2. Fixed Scheduler: No learning required for variance schedule
3. Smooth Convergence: Loss decreases monotonically (typically)
4. Single Network: Only train one U-Net (no discriminator)

Training Goal:
- Minimize MSE between predicted noise and actual noise
- Network learns reverse diffusion process
- After training: Can generate images by pure noise → denoising

Comparison with Module 13 (cGAN):
- cGAN: Volatile loss, adversarial dynamics, two networks
- Diffusion: Stable loss, regression task, one network
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Tuple, List, Dict, Optional

from diffusion_model import (
    SimpleUNet, NoiseScheduler, add_noise, count_parameters, initialize_weights
)


class DDPMTrainer:
    """
    Trainer for Denoising Diffusion Probabilistic Models.
    
    Key Characteristics:
    - MSE loss for noise prediction
    - Fixed forward diffusion schedule
    - Single network training (U-Net)
    - Smooth, stable convergence curve
    
    Contrast with cGAN (Module 13):
    - cGAN uses binary cross-entropy (adversarial)
    - Diffusion uses MSE (regression)
    - cGAN has volatile loss curves
    - Diffusion has smooth, monotonic loss decrease
    """
    
    def __init__(
        self,
        model: SimpleUNet,
        scheduler: NoiseScheduler,
        learning_rate: float = 0.001,
        device: str = 'cpu',
        checkpoint_dir: Optional[str] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: SimpleUNet noise prediction network
            scheduler: NoiseScheduler for forward diffusion
            learning_rate: Adam optimizer learning rate (default: 0.001)
            device: 'cpu', 'cuda', or 'mps'
            checkpoint_dir: Directory to save checkpoints
        
        Key Differences from cGAN:
        - Single optimizer (only for model, no discriminator)
        - MSE loss (not adversarial)
        - Higher learning rate (0.001 vs 0.0002 for cGAN)
        """
        self.model = model
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer: Adam with higher learning rate than cGAN
        # Higher LR works because MSE loss is more stable
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Tracking
        self.train_losses: List[float] = []
        self.epoch_losses: List[float] = []
        self.best_loss = float('inf')
    
    def train_step(self, x_0: torch.Tensor) -> float:
        """
        Single training step for DDPM.
        
        Process:
        1. Sample random timesteps t ~ U(0, T)
        2. Sample random noise ε ~ N(0, I)
        3. Forward diffusion: x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε
        4. Predict noise: ε_pred = model(x_t, t)
        5. MSE loss: L = ||ε_pred - ε||²
        6. Backprop and optimize
        
        Args:
            x_0: Original images (batch, 1, 28, 28)
        
        Returns:
            loss: MSE loss value for this batch
        
        Key Insight:
        - This is a pure regression task (not adversarial)
        - Loss is deterministic and smooth (no min-max game)
        """
        # Sample random timesteps for this batch
        batch_size = x_0.shape[0]
        timesteps = torch.randint(
            0, self.scheduler.num_timesteps, (batch_size,),
            device=self.device
        )
        
        # Sample random noise
        noise = torch.randn_like(x_0)
        
        # Forward diffusion: x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε
        x_t, _ = add_noise(x_0, timesteps, self.scheduler, noise)
        
        # Predict noise from (x_t, t)
        predicted_noise = self.model(x_t, timesteps)
        
        # MSE loss: Minimize ||predicted_noise - actual_noise||²
        # This is the core training objective for diffusion models
        loss = F.mse_loss(predicted_noise, noise)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (optional, helps stability)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader for training data
        
        Returns:
            Average loss for epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        for images, _ in progress_bar:  # Ignore labels for unsupervised training
            images = images.to(self.device)
            
            # Train step
            loss = self.train_step(images)
            total_loss += loss
            num_batches += 1
            
            self.train_losses.append(loss)
            progress_bar.set_postfix({'loss': loss:.6f})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(
        self,
        train_loader: DataLoader,
        num_epochs: int = 50,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Key Observation:
        - Loss should decrease smoothly without oscillation
        - Compare with cGAN's volatile loss curves
        - No mode collapse issues (non-adversarial)
        
        Args:
            train_loader: Training data
            num_epochs: Number of training epochs
            val_loader: Optional validation data
        
        Returns:
            Dictionary with training history
        """
        print(f"\nTraining Diffusion Model for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {count_parameters(self.model):,}")
        print(f"Timesteps: {self.scheduler.num_timesteps}")
        print("="*60)
        
        for epoch in range(num_epochs):
            epoch_loss = self.train_epoch(train_loader)
            self.epoch_losses.append(epoch_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.6f}", end="")
            
            # Save checkpoint if best loss
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                print(" ✓ (Best)", end="")
                if self.checkpoint_dir:
                    self.save_checkpoint(epoch, epoch_loss, is_best=True)
            print()
        
        print("="*60)
        print(f"Training complete! Best loss: {self.best_loss:.6f}")
        
        return {
            'train_losses': self.train_losses,
            'epoch_losses': self.epoch_losses,
        }
    
    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save model checkpoint."""
        if not self.checkpoint_dir:
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'loss': loss,
        }
        
        if is_best:
            path = self.checkpoint_dir / 'best_model.pt'
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        
        torch.save(checkpoint, path)
        print(f"  Saved checkpoint: {path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        print(f"Loaded checkpoint from {checkpoint_path}")


class DDPMSampler:
    """
    Generate images using reverse diffusion process.
    
    Process:
    1. Start with pure noise x_T
    2. For t = T down to 1:
       - Predict noise: ε_pred = model(x_t, t)
       - Denoise: x_{t-1} = (x_t - sqrt(1 - ᾱ_t) * ε_pred) / sqrt(ᾱ_t) + noise
    3. Return x_0 (generated image)
    """
    
    def __init__(self, model: SimpleUNet, scheduler: NoiseScheduler, device: str = 'cpu'):
        """
        Initialize sampler.
        
        Args:
            model: Trained SimpleUNet
            scheduler: NoiseScheduler
            device: 'cpu', 'cuda', or 'mps'
        """
        self.model = model
        self.scheduler = scheduler
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def sample(self, num_samples: int = 16) -> torch.Tensor:
        """
        Generate images from pure noise.
        
        Args:
            num_samples: Number of images to generate
        
        Returns:
            Generated images (num_samples, 1, 28, 28)
        """
        # Start with pure noise x_T
        x_t = torch.randn(num_samples, 1, 28, 28, device=self.device)
        
        # Reverse diffusion: T → 1
        for t_idx in range(self.scheduler.num_timesteps - 1, -1, -1):
            t = torch.full((num_samples,), t_idx, dtype=torch.long, device=self.device)
            
            # Predict noise
            predicted_noise = self.model(x_t, t)
            
            # Get coefficients for this timestep
            alpha_t = self.scheduler.alphas[t_idx]
            alpha_cumprod_t = self.scheduler.alphas_cumprod[t_idx]
            alpha_cumprod_prev_t = self.scheduler.alphas_cumprod_prev[t_idx]
            
            # Denoise formula (based on DDPM paper)
            # x_{t-1} = 1/sqrt(α_t) * (x_t - (1 - α_t)/sqrt(1 - ᾱ_t) * ε_pred) + σ_t * z
            
            # Posterior variance
            posterior_variance = (
                (1 - alpha_cumprod_prev_t) / (1 - alpha_cumprod_t) * (1 - alpha_t)
            )
            
            # Coefficient for x_t
            coef_x_t = 1 / torch.sqrt(alpha_t)
            
            # Coefficient for predicted noise
            coef_noise = (1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)
            
            # Denoise step
            x_t = coef_x_t * (x_t - coef_noise * predicted_noise)
            
            # Add noise except at last step
            if t_idx > 0:
                noise = torch.randn_like(x_t)
                x_t = x_t + torch.sqrt(posterior_variance) * noise
        
        # Clip to valid range
        x_t = torch.clamp(x_t, -1.0, 1.0)
        
        return x_t


def visualize_diffusion_process(
    model: SimpleUNet,
    scheduler: NoiseScheduler,
    x_0: torch.Tensor,
    device: str = 'cpu',
    num_steps: int = 5,
):
    """
    Visualize forward diffusion process (image getting noisier).
    
    Args:
        model: SimpleUNet (used to get device)
        scheduler: NoiseScheduler
        x_0: Original image (1, 1, 28, 28)
        device: Device to use
        num_steps: Number of steps to visualize
    """
    x_0 = x_0.to(device)
    
    # Select timesteps to visualize
    timesteps = torch.linspace(0, scheduler.num_timesteps - 1, num_steps, dtype=torch.long)
    
    fig, axes = plt.subplots(1, num_steps, figsize=(12, 2))
    
    for idx, t in enumerate(timesteps):
        t_idx = int(t.item())
        t_tensor = torch.tensor([t_idx], device=device)
        
        # Add noise
        x_t, _ = add_noise(x_0, t_tensor, scheduler)
        
        # Plot
        img = x_t[0, 0].cpu().numpy()
        axes[idx].imshow(img, cmap='gray')
        axes[idx].set_title(f't={t_idx}')
        axes[idx].axis('off')
    
    plt.suptitle('Forward Diffusion Process (Noising)')
    plt.tight_layout()
    return fig


def plot_training_curves(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """
    Plot training loss curves.
    
    Key Feature:
    - MSE loss should decrease smoothly (unlike adversarial loss)
    - Compare with GAN's oscillating loss curves
    
    Args:
        history: Dictionary with 'train_losses' and 'epoch_losses'
        save_path: Optional path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Batch losses
    ax1.plot(history['train_losses'], linewidth=0.5, alpha=0.7)
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('MSE Loss per Batch (Smooth Convergence)')
    ax1.grid(True, alpha=0.3)
    
    # Epoch losses
    epochs = range(1, len(history['epoch_losses']) + 1)
    ax2.plot(epochs, history['epoch_losses'], marker='o', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Average MSE Loss')
    ax2.set_title('Average MSE Loss per Epoch')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Diffusion Model Training: Smooth Convergence vs cGAN Volatility', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved training curves to {save_path}")
    
    return fig


def create_and_preprocess_mnist_data(
    batch_size: int = 64,
    num_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Load MNIST and create data loaders.
    
    Args:
        batch_size: Batch size for training
        num_samples: Optional limit on samples (for faster testing)
    
    Returns:
        train_loader, val_loader
    """
    from torchvision import datasets, transforms
    
    # Preprocessing: Normalize to [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # [-1, 1]
    ])
    
    # Load MNIST
    print("Loading MNIST dataset...")
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform,
    )
    
    val_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform,
    )
    
    # Optionally limit samples (for faster testing)
    if num_samples:
        train_dataset = torch.utils.data.Subset(
            train_dataset, range(num_samples)
        )
        val_dataset = torch.utils.data.Subset(
            val_dataset, range(min(num_samples // 5, len(val_dataset)))
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    return train_loader, val_loader

