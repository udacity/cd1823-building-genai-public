"""
GAN Training Module: Implement adversarial training for GANs

This module provides the core training functions for GANs:
- train_discriminator_step: One optimization step for the discriminator
- train_generator_step: One optimization step for the generator
- train_gan: Full training loop with loss tracking

Key concepts:
- Alternating optimization: D and G take turns improving
- Separate losses: Discriminator loss + Generator loss tracked separately
- Target labels: D wants (real=1, fake=0), G wants (fake=1 to fool D)
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import BCELoss
from torch.optim import Adam


def train_discriminator_step(
    discriminator: nn.Module,
    generator: nn.Module,
    real_images: torch.Tensor,
    d_optimizer: Adam,
    device: str = "cpu",
) -> float:
    """
    One training step for the discriminator.

    The discriminator learns to distinguish real from fake images.

    Args:
        discriminator: Discriminator model
        generator: Generator model
        real_images: Batch of real images from dataset
        d_optimizer: Optimizer for discriminator
        device: 'cpu' or 'cuda'

    Returns:
        discriminator_loss: Loss value for this step

    Training logic:
    1. Get real images, label them as "real" (target=1)
    2. Generate fake images, label them as "fake" (target=0)
    3. Forward pass on both through discriminator
    4. Compute BCE loss for both
    5. Backpropagate combined loss
    6. Update discriminator weights
    """
    batch_size = real_images.size(0)

    # Create labels
    real_labels = torch.ones(batch_size, 1, device=device)
    fake_labels = torch.zeros(batch_size, 1, device=device)

    # === Train on Real Images ===
    d_optimizer.zero_grad()

    # Forward pass on real images
    real_predictions = discriminator(real_images)
    real_loss = BCELoss()(real_predictions, real_labels)

    # === Train on Fake Images ===
    # Generate fake images
    noise = torch.randn(batch_size, 100, device=device)
    with torch.no_grad():
        fake_images = generator(noise)

    # Reshape fake images from (batch, 784) to (batch, 1, 28, 28)
    fake_images = fake_images.view(-1, 1, 28, 28)

    # Forward pass on fake images
    fake_predictions = discriminator(fake_images)
    fake_loss = BCELoss()(fake_predictions, fake_labels)

    # === Combined Loss ===
    # Total discriminator loss = average of real and fake losses
    d_loss = (real_loss + fake_loss) / 2

    # Backpropagate
    d_loss.backward()
    d_optimizer.step()

    return d_loss.item()


def train_generator_step(
    discriminator: nn.Module,
    generator: nn.Module,
    g_optimizer: Adam,
    batch_size: int,
    device: str = "cpu",
) -> float:
    """
    One training step for the generator.

    The generator learns to create fake images that fool the discriminator.

    Key trick: We label fake images as "real" (target=1) so the generator
    learns to maximize discriminator's confusion.

    Args:
        discriminator: Discriminator model (frozen, not updated)
        generator: Generator model
        g_optimizer: Optimizer for generator
        batch_size: Number of images to generate
        device: 'cpu' or 'cuda'

    Returns:
        generator_loss: Loss value for this step

    Training logic:
    1. Generate batch of fake images
    2. Label them as "real" (target=1) - this tricks the discriminator
    3. Forward pass through discriminator
    4. Compute BCE loss (want predictions close to 1)
    5. Backpropagate to generator only
    6. Update generator weights
    """
    # Zero gradients
    g_optimizer.zero_grad()

    # Generate fake images
    noise = torch.randn(batch_size, 100, device=device)
    fake_images = generator(noise)

    # Reshape from (batch, 784) to (batch, 1, 28, 28)
    fake_images = fake_images.view(-1, 1, 28, 28)

    # Forward pass through discriminator
    fake_predictions = discriminator(fake_images)

    # Create "real" labels for fake images
    # This tricks the discriminator into giving high scores
    real_labels = torch.ones(batch_size, 1, device=device)

    # Compute loss
    # Generator wants discriminator to output close to 1 for fake images
    g_loss = BCELoss()(fake_predictions, real_labels)

    # Backpropagate
    g_loss.backward()
    g_optimizer.step()

    return g_loss.item()


def train_gan(
    generator: nn.Module,
    discriminator: nn.Module,
    train_loader,
    num_epochs: int = 50,
    device: str = "cpu",
    lr: float = 0.0002,
    beta1: float = 0.5,
    checkpoint_interval: int = 5,
) -> Tuple[List[float], List[float], List[torch.Tensor]]:
    """
    Train a GAN for multiple epochs.

    This implements the alternating optimization algorithm:
    For each batch:
        1. Train discriminator (one step)
        2. Train generator (one step)

    Args:
        generator: Generator model
        discriminator: Discriminator model
        train_loader: DataLoader with training images
        num_epochs: Number of training epochs
        device: 'cpu' or 'cuda'
        lr: Learning rate for optimizers
        beta1: Beta1 parameter for Adam optimizer
        checkpoint_interval: Save generated samples every N epochs

    Returns:
        d_losses: List of discriminator loss values (one per batch)
        g_losses: List of generator loss values (one per batch)
        generated_samples: List of generated image grids at checkpoints

    Loss interpretation:
    - Ideal: Both losses hover around 0.5-0.7 (discriminator confused)
    - Bad: D loss → 0 (discriminator too good, G can't learn)
    - Bad: D loss → 1 (generator too good early, no challenge for D)
    """
    # Set up optimizers
    d_optimizer = Adam(
        discriminator.parameters(),
        lr=lr,
        betas=(beta1, 0.999),
    )
    g_optimizer = Adam(
        generator.parameters(),
        lr=lr,
        betas=(beta1, 0.999),
    )

    # Put models in training mode
    generator.train()
    discriminator.train()

    # Track losses
    d_losses = []
    g_losses = []
    generated_samples = []

    print(f"Starting GAN training for {num_epochs} epochs...")
    print(f"Device: {device}\n")

    for epoch in range(num_epochs):
        epoch_d_loss = 0
        epoch_g_loss = 0
        num_batches = 0

        for real_images, _ in train_loader:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # Flatten images from (batch, 1, 28, 28) to (batch, 784)
            real_images = real_images.view(-1, 784)

            # Train discriminator
            d_loss = train_discriminator_step(
                discriminator,
                generator,
                real_images,
                d_optimizer,
                device,
            )

            # Train generator
            g_loss = train_generator_step(
                discriminator,
                generator,
                g_optimizer,
                batch_size,
                device,
            )

            d_losses.append(d_loss)
            g_losses.append(g_loss)

            epoch_d_loss += d_loss
            epoch_g_loss += g_loss
            num_batches += 1

        # Average loss for epoch
        avg_d_loss = epoch_d_loss / num_batches
        avg_g_loss = epoch_g_loss / num_batches

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f}"
        )

        # Generate samples at checkpoint intervals
        if (epoch + 1) % checkpoint_interval == 0:
            generator.eval()
            with torch.no_grad():
                noise = torch.randn(16, 100, device=device)
                samples = generator(noise)
                samples = samples.view(-1, 1, 28, 28)
                generated_samples.append(samples.cpu())
            generator.train()

    print("\n✓ Training complete!")
    return d_losses, g_losses, generated_samples


def visualize_losses(d_losses: List[float], g_losses: List[float]):
    """
    Plot discriminator and generator losses over training.

    Args:
        d_losses: List of discriminator losses
        g_losses: List of generator losses
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot losses
    ax.plot(d_losses, label="Discriminator Loss", alpha=0.7, linewidth=2)
    ax.plot(g_losses, label="Generator Loss", alpha=0.7, linewidth=2)

    # Add 0.693 baseline (random guessing)
    ax.axhline(
        y=0.693,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label="Random guessing baseline",
    )

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("GAN Training: Loss Curves", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def analyze_convergence(d_losses: List[float], g_losses: List[float]):
    """
    Analyze GAN convergence and detect common failure modes.

    Args:
        d_losses: List of discriminator losses
        g_losses: List of generator losses

    Outputs:
        Analysis of whether GAN reached Nash Equilibrium
    """
    # Last 100 batches for analysis
    recent_d = np.array(d_losses[-100:])
    recent_g = np.array(g_losses[-100:])

    avg_d = recent_d.mean()
    avg_g = recent_g.mean()

    print("\n" + "=" * 60)
    print("CONVERGENCE ANALYSIS")
    print("=" * 60)

    print(f"\nRecent average losses (last 100 batches):")
    print(f"  Discriminator: {avg_d:.4f}")
    print(f"  Generator: {avg_g:.4f}")

    print(f"\nLoss stability (std dev):")
    print(f"  Discriminator: {recent_d.std():.4f}")
    print(f"  Generator: {recent_g.std():.4f}")

    # Check for failure modes
    print(f"\nFailure mode detection:")

    if avg_d < 0.1:
        print(f"   DISCRIMINATOR TOO GOOD: D loss {avg_d:.4f} (< 0.1)")
        print(f"     Generator may have mode collapse or stopped learning")
    elif avg_d > 0.9:
        print(f"   GENERATOR TOO GOOD: D loss {avg_d:.4f} (> 0.9)")
        print(f"     Discriminator can't provide useful feedback")
    else:
        print(f"   BALANCED: D loss {avg_d:.4f}")
        print(f"     Discriminator and Generator are in competition")

    if avg_g > 1.5:
        print(f"   HIGH GENERATOR LOSS: {avg_g:.4f}")
        print(f"     Generator struggles to fool discriminator")
    else:
        print(f"   Generator loss reasonable: {avg_g:.4f}")

    print(f"\nNash Equilibrium characteristics:")
    print(f"  - D loss should stabilize around 0.5-0.7")
    print(f"  - G loss should decrease over time")
    print(f"  - Both losses should not oscillate wildly")
    print(f"  - Generated samples should improve with training")

    print("=" * 60)
