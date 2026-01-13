"""
DCGAN Training Script

Implements the training loop for DCGAN on CIFAR-10 dataset
- Uses Adam optimizer for both generator and discriminator
- Tracks losses and generates sample images during training
- Supports device auto-detection (MPS/CUDA/CPU)
"""

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Import DCGAN models
from dcgan import DCGANDiscriminator, DCGANGenerator
from torch.utils.data import DataLoader


class DCGANTrainer:
    """
    Trainer class for DCGAN
    Handles training loop, loss tracking, and checkpoint management
    """

    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        device: torch.device,
        lr_g: float = 0.0002,
        lr_d: float = 0.0002,
        beta1: float = 0.5,
        beta2: float = 0.999,
    ):
        """
        Args:
            generator: Generator model
            discriminator: Discriminator model
            device: Device to train on
            lr_g: Learning rate for generator
            lr_d: Learning rate for discriminator
            beta1: Beta1 for Adam optimizer
            beta2: Beta2 for Adam optimizer
        """
        self.generator = generator
        self.discriminator = discriminator
        self.device = device

        # Optimizers
        self.optimizer_g = optim.Adam(
            generator.parameters(), lr=lr_g, betas=(beta1, beta2)
        )
        self.optimizer_d = optim.Adam(
            discriminator.parameters(), lr=lr_d, betas=(beta1, beta2)
        )

        # Loss function
        self.criterion = nn.BCELoss()

        # Tracking
        self.g_losses = []
        self.d_losses = []

    def train_step(
        self,
        real_images: torch.Tensor,
        batch_size: int,
        latent_dim: int = 100,
    ) -> Tuple[float, float]:
        """
        Single training step

        Args:
            real_images: Real image batch (batch_size, 3, 32, 32)
            batch_size: Batch size
            latent_dim: Dimension of latent noise vector

        Returns:
            Tuple of (D_loss, G_loss)
        """
        # Labels for real/fake
        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)

        # =============== Train Discriminator ===============
        self.discriminator.zero_grad()

        # Real images
        real_images = real_images.to(self.device)
        output_real = self.discriminator(real_images)
        d_loss_real = self.criterion(output_real, real_labels)

        # Fake images
        z = torch.randn(batch_size, latent_dim, device=self.device)
        fake_images = self.generator(z)
        output_fake = self.discriminator(fake_images.detach())
        d_loss_fake = self.criterion(output_fake, fake_labels)

        # Total discriminator loss
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.optimizer_d.step()

        # =============== Train Generator ===============
        self.generator.zero_grad()

        z = torch.randn(batch_size, latent_dim, device=self.device)
        fake_images = self.generator(z)
        output = self.discriminator(fake_images)
        g_loss = self.criterion(output, real_labels)

        g_loss.backward()
        self.optimizer_g.step()

        return d_loss.item(), g_loss.item()

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        total_epochs: int,
        latent_dim: int = 100,
        log_interval: int = 50,
    ):
        """
        Train for one epoch

        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number
            total_epochs: Total number of epochs
            latent_dim: Dimension of latent noise vector
            log_interval: Log every N batches
        """
        self.generator.train()
        self.discriminator.train()

        for batch_idx, (real_images, _) in enumerate(train_loader):
            batch_size = real_images.size(0)
            d_loss, g_loss = self.train_step(real_images, batch_size, latent_dim)

            self.d_losses.append(d_loss)
            self.g_losses.append(g_loss)

            if (batch_idx + 1) % log_interval == 0:
                print(
                    f"Epoch [{epoch+1}/{total_epochs}] "
                    f"Batch [{batch_idx+1}/{len(train_loader)}] "
                    f"D_loss: {d_loss:.4f} | G_loss: {g_loss:.4f}"
                )

    def train(
        self,
        train_loader: DataLoader,
        num_epochs: int = 50,
        latent_dim: int = 100,
        log_interval: int = 50,
    ) -> Dict[str, List[float]]:
        """
        Complete training loop

        Args:
            train_loader: DataLoader for training data
            num_epochs: Number of epochs to train
            latent_dim: Dimension of latent noise vector
            log_interval: Log every N batches

        Returns:
            Dictionary with loss histories
        """
        print(f"\n{'='*80}")
        print(f"Starting DCGAN Training")
        print(f"{'='*80}")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Dataset size: {len(train_loader) * train_loader.batch_size}")
        print(f"{'='*80}\n")

        for epoch in range(num_epochs):
            self.train_epoch(
                train_loader,
                epoch,
                num_epochs,
                latent_dim,
                log_interval,
            )

            # Print epoch summary
            if (epoch + 1) % 10 == 0 or epoch == 0:
                avg_d_loss = np.mean(self.d_losses[-len(train_loader) :])
                avg_g_loss = np.mean(self.g_losses[-len(train_loader) :])
                print(
                    f"\n✓ Epoch {epoch+1} Complete - "
                    f"Avg D_loss: {avg_d_loss:.4f} | "
                    f"Avg G_loss: {avg_g_loss:.4f}\n"
                )

        print(f"\n{'='*80}")
        print(f"Training Complete!")
        print(f"{'='*80}\n")

        return {
            "g_losses": self.g_losses,
            "d_losses": self.d_losses,
        }

    def generate_samples(self, num_samples: int = 16, latent_dim: int = 100):
        """
        Generate sample images

        Args:
            num_samples: Number of samples to generate
            latent_dim: Dimension of latent noise vector

        Returns:
            Generated images tensor
        """
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, latent_dim, device=self.device)
            samples = self.generator(z)
        return samples


def initialize_weights(model: nn.Module):
    """
    Initialize model weights following DCGAN guidelines
    - Conv/ConvTranspose: Normal distribution (mean=0, std=0.02)
    - BatchNorm: Normal distribution (mean=1, std=0.02)
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, mean=1.0, std=0.02)
            nn.init.constant_(m.bias, 0.0)


if __name__ == "__main__":
    # Example usage
    from dcgan import create_dcgan_models

    # Determine device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Create models
    generator, discriminator = create_dcgan_models(
        latent_dim=100, num_channels=3, device=device
    )

    # Initialize weights
    initialize_weights(generator)
    initialize_weights(discriminator)

    # Create trainer
    trainer = DCGANTrainer(
        generator=generator,
        discriminator=discriminator,
        device=device,
    )

    print("✓ DCGAN Trainer initialized successfully!")
