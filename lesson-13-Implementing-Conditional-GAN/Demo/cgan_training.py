"""
Conditional GAN Training Script

Implements the training loop for cGAN on CIFAR-10 dataset
- Key difference from DCGAN: labels passed to both G and D
- Generator learns to create images matching the target class
- Discriminator learns to verify class-image consistency
- Supports device auto-detection (MPS/CUDA/CPU)
"""

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Import cGAN models
from cgan import ConditionalDiscriminator, ConditionalGenerator
from torch.utils.data import DataLoader


class ConditionalGANTrainer:
    """
    Trainer class for Conditional GAN
    Handles training loop with class labels, loss tracking, and class-specific generation
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
            generator: Conditional Generator model
            discriminator: Conditional Discriminator model
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
        self.g_losses_per_class = {i: [] for i in range(10)}
        self.d_losses_per_class = {i: [] for i in range(10)}

    def train_step(
        self,
        real_images: torch.Tensor,
        real_labels: torch.Tensor,
        batch_size: int,
        latent_dim: int = 100,
        num_classes: int = 10,
    ) -> Tuple[float, float]:
        """
        Single training step with class conditioning

        Args:
            real_images: Real image batch (batch_size, 3, 32, 32)
            real_labels: Real labels (batch_size,)
            batch_size: Batch size
            latent_dim: Dimension of latent noise vector
            num_classes: Number of classes

        Returns:
            Tuple of (D_loss, G_loss)
        """
        # Labels for real/fake
        real_labels_prob = torch.ones(batch_size, 1, device=self.device)
        fake_labels_prob = torch.zeros(batch_size, 1, device=self.device)

        # =============== Train Discriminator ===============
        self.discriminator.zero_grad()

        # Real images with real labels
        real_images = real_images.to(self.device)
        real_labels = real_labels.to(self.device)
        output_real = self.discriminator(real_images, real_labels)
        d_loss_real = self.criterion(output_real, real_labels_prob)

        # Fake images with fake labels (random classes)
        z = torch.randn(batch_size, latent_dim, device=self.device)
        fake_classes = torch.randint(0, num_classes, (batch_size,), device=self.device)
        fake_images = self.generator(z, fake_classes)
        output_fake = self.discriminator(fake_images.detach(), fake_classes)
        d_loss_fake = self.criterion(output_fake, fake_labels_prob)

        # Total discriminator loss
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.optimizer_d.step()

        # =============== Train Generator ===============
        self.generator.zero_grad()

        # Generate fake images with target class labels
        z = torch.randn(batch_size, latent_dim, device=self.device)
        target_classes = torch.randint(
            0, num_classes, (batch_size,), device=self.device
        )
        fake_images = self.generator(z, target_classes)

        # Try to fool discriminator: D should say fake images are real
        output = self.discriminator(fake_images, target_classes)
        g_loss = self.criterion(output, real_labels_prob)

        g_loss.backward()
        self.optimizer_g.step()

        return d_loss.item(), g_loss.item()

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        total_epochs: int,
        latent_dim: int = 100,
        num_classes: int = 10,
        log_interval: int = 50,
    ):
        """
        Train for one epoch with conditional labels

        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number
            total_epochs: Total number of epochs
            latent_dim: Dimension of latent noise vector
            num_classes: Number of classes
            log_interval: Log every N batches
        """
        self.generator.train()
        self.discriminator.train()

        for batch_idx, (real_images, real_labels) in enumerate(train_loader):
            batch_size = real_images.size(0)
            d_loss, g_loss = self.train_step(
                real_images,
                real_labels,
                batch_size,
                latent_dim,
                num_classes,
            )

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
        num_classes: int = 10,
        log_interval: int = 50,
    ) -> Dict[str, List[float]]:
        """
        Complete training loop with conditional generation

        Args:
            train_loader: DataLoader for training data
            num_epochs: Number of epochs to train
            latent_dim: Dimension of latent noise vector
            num_classes: Number of classes
            log_interval: Log every N batches

        Returns:
            Dictionary with loss histories
        """
        print(f"\n{'='*80}")
        print(f"Starting Conditional GAN (cGAN) Training")
        print(f"{'='*80}")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Classes: {num_classes}")
        print(f"Dataset size: {len(train_loader) * train_loader.batch_size}")
        print(f"{'='*80}\n")

        for epoch in range(num_epochs):
            self.train_epoch(
                train_loader,
                epoch,
                num_epochs,
                latent_dim,
                num_classes,
                log_interval,
            )

            # Print epoch summary
            if (epoch + 1) % 10 == 0 or epoch == 0:
                avg_d_loss = np.mean(self.d_losses[-len(train_loader) :])
                avg_g_loss = np.mean(self.g_losses[-len(train_loader) :])
                print(
                    f"\n Epoch {epoch+1} Complete - "
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

    def generate_class_samples(
        self,
        target_class: int,
        num_samples: int = 8,
        latent_dim: int = 100,
    ) -> torch.Tensor:
        """
        Generate samples from a specific class

        Args:
            target_class: Class label to generate (0-9)
            num_samples: Number of samples to generate
            latent_dim: Dimension of latent noise vector

        Returns:
            Generated images tensor
        """
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, latent_dim, device=self.device)
            labels = torch.full(
                (num_samples,), target_class, dtype=torch.long, device=self.device
            )
            samples = self.generator(z, labels)
        return samples

    def generate_all_classes_grid(
        self,
        num_classes: int = 10,
        samples_per_class: int = 10,
        latent_dim: int = 100,
        shared_z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate a grid showing class-specific generation
        Each row: same noise vector, different classes (shows class disentanglement)

        Args:
            num_classes: Number of classes
            samples_per_class: Number of samples per class (creates grid width)
            latent_dim: Dimension of latent noise vector
            shared_z: Optional shared noise vector (for class comparison)

        Returns:
            Generated images grid (num_classes * samples_per_class, 3, 32, 32)
        """
        self.generator.eval()
        with torch.no_grad():
            if shared_z is None:
                # Use same noise for all classes (shows class control)
                shared_z = torch.randn(1, latent_dim, device=self.device)

            grid_images = []
            for class_id in range(num_classes):
                # Expand shared noise for this class
                z_class = shared_z.expand(samples_per_class, -1)
                labels = torch.full(
                    (samples_per_class,), class_id, dtype=torch.long, device=self.device
                )
                samples = self.generator(z_class, labels)
                grid_images.append(samples)

            # Stack all images: (num_classes * samples_per_class, 3, 32, 32)
            all_images = torch.cat(grid_images, dim=0)

        return all_images


def initialize_weights(model: nn.Module):
    """
    Initialize model weights following DCGAN guidelines
    - Conv/ConvTranspose: Normal distribution (mean=0, std=0.02)
    - BatchNorm: Normal distribution (mean=1, std=0.02)
    - Linear: Normal distribution (mean=0, std=0.02)
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, mean=1.0, std=0.02)
            nn.init.constant_(m.bias, 0.0)


if __name__ == "__main__":
    # Example usage
    from cgan import create_cgan_models

    # Determine device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Create models
    generator, discriminator = create_cgan_models(
        latent_dim=100,
        num_classes=10,
        label_dim=50,
        num_channels=3,
        device=device,
    )

    # Initialize weights
    initialize_weights(generator)
    initialize_weights(discriminator)

    # Create trainer
    trainer = ConditionalGANTrainer(
        generator=generator,
        discriminator=discriminator,
        device=device,
    )

    print("✓ Conditional GAN Trainer initialized successfully!")
