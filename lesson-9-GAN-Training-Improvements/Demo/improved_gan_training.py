"""
improved_gan_training.py
------------------------
GAN Training Stability Improvements: Label Smoothing & Feature Matching

This module implements two essential stabilization techniques:

1. Label Smoothing:
   - Replace hard labels (0/1) with soft labels (e.g., 0.1/0.9)
   - Reduces discriminator overconfidence
   - Prevents vanishing gradients for the generator
   - Empirically improves training stability

2. Feature Matching:
   - Match intermediate discriminator features between real and fake data
   - Prevents mode collapse by encouraging feature diversity
   - Forces generator to learn representative features, not memorize
   - Add MSE loss on intermediate layers to generator objective

Classes:
    - BaselineGAN: Standard BCE loss only
    - LabelSmoothingGAN: With label smoothing (0.9/0.1 instead of 1/0)
    - FeatureMatchingGAN: With feature matching loss on intermediate layer
    - ComparisonTrainer: Run and compare all three approaches
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import BCELoss, MSELoss
from torch.optim import Adam


class BaselineGAN:
    """
    Standard GAN training with unmodified BCE loss.

    Loss calculation:
    - Discriminator: BCE(D(real), 1) + BCE(D(fake), 0)
    - Generator: BCE(D(fake), 1)
    """

    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        lr: float = 0.0002,
        beta1: float = 0.5,
        device: str = "cpu",
    ):
        """Initialize baseline GAN trainer."""
        self.generator = generator
        self.discriminator = discriminator
        self.device = device

        self.d_optimizer = Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.g_optimizer = Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.criterion = BCELoss()

        self.d_losses = []
        self.g_losses = []
        self.name = "Baseline (No Stabilization)"

    def train_step(
        self, real_images: torch.Tensor, batch_size: int
    ) -> Tuple[float, float]:
        """
        One training step (D step + G step).

        Args:
            real_images: Batch of real images (batch, 1, 28, 28)
            batch_size: Size of batch

        Returns:
            (d_loss, g_loss): Loss values for this step
        """
        # ========== Discriminator Step ==========
        self.d_optimizer.zero_grad()

        # Real images with hard label 1
        real_pred = self.discriminator(real_images)
        real_labels = torch.ones(batch_size, 1, device=self.device)
        d_loss_real = self.criterion(real_pred, real_labels)

        # Fake images with hard label 0
        noise = torch.randn(batch_size, 100, device=self.device)
        with torch.no_grad():
            fake_images = self.generator(noise)
        fake_images = fake_images.view(-1, 1, 28, 28)
        fake_pred = self.discriminator(fake_images)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)
        d_loss_fake = self.criterion(fake_pred, fake_labels)

        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        self.d_optimizer.step()

        # ========== Generator Step ==========
        self.g_optimizer.zero_grad()

        noise = torch.randn(batch_size, 100, device=self.device)
        fake_images = self.generator(noise)
        fake_images = fake_images.view(-1, 1, 28, 28)
        fake_pred = self.discriminator(fake_images)

        # Fool discriminator with hard label 1
        real_labels = torch.ones(batch_size, 1, device=self.device)
        g_loss = self.criterion(fake_pred, real_labels)

        g_loss.backward()
        self.g_optimizer.step()

        self.d_losses.append(d_loss.item())
        self.g_losses.append(g_loss.item())

        return d_loss.item(), g_loss.item()


class LabelSmoothingGAN:
    """
    GAN training with label smoothing.

    Label smoothing replaces hard labels with soft labels:
    - Real label: 1.0 → 0.9  (discriminator slightly less confident)
    - Fake label: 0.0 → 0.1  (generator gets stronger gradients)

    Benefits:
    - Prevents discriminator from becoming overconfident
    - Encourages generator to explore more diverse samples
    - Reduces vanishing gradient problem early in training
    - More stable loss curves
    """

    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        smooth_real: float = 0.9,
        smooth_fake: float = 0.1,
        lr: float = 0.0002,
        beta1: float = 0.5,
        device: str = "cpu",
    ):
        """
        Initialize label smoothing GAN trainer.

        Args:
            generator: Generator model
            discriminator: Discriminator model
            smooth_real: Soft label for real images (default 0.9)
            smooth_fake: Soft label for fake images (default 0.1)
            lr: Learning rate
            beta1: Adam beta1 parameter
            device: 'cpu' or 'cuda'
        """
        self.generator = generator
        self.discriminator = discriminator
        self.device = device

        self.smooth_real = smooth_real
        self.smooth_fake = smooth_fake

        self.d_optimizer = Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.g_optimizer = Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.criterion = BCELoss()

        self.d_losses = []
        self.g_losses = []
        self.name = f"Label Smoothing ({smooth_real}/{smooth_fake})"

    def train_step(
        self, real_images: torch.Tensor, batch_size: int
    ) -> Tuple[float, float]:
        """
        One training step with smoothed labels.

        Args:
            real_images: Batch of real images (batch, 1, 28, 28)
            batch_size: Size of batch

        Returns:
            (d_loss, g_loss): Loss values for this step
        """
        # ========== Discriminator Step ==========
        self.d_optimizer.zero_grad()

        # Real images with smoothed label (0.9 instead of 1.0)
        real_pred = self.discriminator(real_images)
        real_labels = torch.full((batch_size, 1), self.smooth_real, device=self.device)
        d_loss_real = self.criterion(real_pred, real_labels)

        # Fake images with smoothed label (0.1 instead of 0.0)
        noise = torch.randn(batch_size, 100, device=self.device)
        with torch.no_grad():
            fake_images = self.generator(noise)
        fake_images = fake_images.view(-1, 1, 28, 28)
        fake_pred = self.discriminator(fake_images)
        fake_labels = torch.full((batch_size, 1), self.smooth_fake, device=self.device)
        d_loss_fake = self.criterion(fake_pred, fake_labels)

        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        self.d_optimizer.step()

        # ========== Generator Step ==========
        self.g_optimizer.zero_grad()

        noise = torch.randn(batch_size, 100, device=self.device)
        fake_images = self.generator(noise)
        fake_images = fake_images.view(-1, 1, 28, 28)
        fake_pred = self.discriminator(fake_images)

        # Fool discriminator with hard label 1 (generator still uses hard target)
        real_labels = torch.ones(batch_size, 1, device=self.device)
        g_loss = self.criterion(fake_pred, real_labels)

        g_loss.backward()
        self.g_optimizer.step()

        self.d_losses.append(d_loss.item())
        self.g_losses.append(g_loss.item())

        return d_loss.item(), g_loss.item()


class FeatureMatchingGAN:
    """
    GAN training with feature matching.

    Feature matching adds an auxiliary loss on intermediate discriminator features:
    - Extract features from intermediate layer (e.g., before final classification)
    - Compute MSE between real and fake feature distributions
    - Add this loss to generator objective with weight λ

    Benefits:
    - Forces generator to match feature statistics, not just fooling classifier
    - Reduces mode collapse by encouraging feature diversity
    - Prevents generator from learning spurious high-frequency patterns
    - More stable and visually better samples

    Loss: G_loss = BCE(D(fake), 1) + λ * MSE(features_real, features_fake)
    """

    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        feature_layer: str = "layer3",  # Name of intermediate layer to match
        feature_weight: float = 10.0,  # Weight of feature matching loss
        lr: float = 0.0002,
        beta1: float = 0.5,
        device: str = "cpu",
    ):
        """
        Initialize feature matching GAN trainer.

        Args:
            generator: Generator model
            discriminator: Discriminator model
            feature_layer: Name of layer to extract features from
            feature_weight: Weight of feature matching loss (lambda)
            lr: Learning rate
            beta1: Adam beta1 parameter
            device: 'cpu' or 'cuda'
        """
        self.generator = generator
        self.discriminator = discriminator
        self.device = device
        self.feature_weight = feature_weight
        self.feature_layer = feature_layer

        self.d_optimizer = Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.g_optimizer = Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))

        self.criterion = BCELoss()
        self.feature_criterion = MSELoss()

        # Register forward hook to extract intermediate features
        self.features = {}
        self._register_feature_hook()

        self.d_losses = []
        self.g_losses = []
        self.feature_losses = []
        self.name = "Feature Matching"

    def _register_feature_hook(self):
        """Register forward hook to capture intermediate features."""

        def get_features(name):
            def hook(model, input, output):
                self.features[name] = output.detach()

            return hook

        # Attach hook to last layer before output (adjust based on your discriminator)
        # Assuming discriminator has structure: features -> final_layer
        # Hook to the features before the final classification layer
        for name, module in self.discriminator.named_modules():
            if "layer" in name or "conv" in name:
                # Attach to penultimate layer
                if name == self.feature_layer:
                    module.register_forward_hook(get_features(name))

    def train_step(
        self, real_images: torch.Tensor, batch_size: int
    ) -> Tuple[float, float, float]:
        """
        One training step with feature matching.

        Args:
            real_images: Batch of real images (batch, 1, 28, 28)
            batch_size: Size of batch

        Returns:
            (d_loss, g_loss, fm_loss): Discriminator loss, generator loss, feature matching loss
        """
        # ========== Discriminator Step ==========
        self.d_optimizer.zero_grad()

        # Real images
        real_pred = self.discriminator(real_images)
        real_labels = torch.ones(batch_size, 1, device=self.device)
        d_loss_real = self.criterion(real_pred, real_labels)

        # Fake images
        noise = torch.randn(batch_size, 100, device=self.device)
        with torch.no_grad():
            fake_images = self.generator(noise)
        fake_images = fake_images.view(-1, 1, 28, 28)
        fake_pred = self.discriminator(fake_images)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)
        d_loss_fake = self.criterion(fake_pred, fake_labels)

        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        self.d_optimizer.step()

        # ========== Generator Step with Feature Matching ==========
        self.g_optimizer.zero_grad()

        # Generate fake images
        noise = torch.randn(batch_size, 100, device=self.device)
        fake_images = self.generator(noise)
        fake_images = fake_images.view(-1, 1, 28, 28)

        # Forward through discriminator to get features
        fake_pred = self.discriminator(fake_images)
        fake_features = self.features.get(self.feature_layer, None)

        # Get real features
        real_pred_for_features = self.discriminator(real_images)
        real_features = self.features.get(self.feature_layer, None)

        # Adversarial loss (fool discriminator)
        real_labels = torch.ones(batch_size, 1, device=self.device)
        g_loss_adv = self.criterion(fake_pred, real_labels)

        # Feature matching loss
        fm_loss = torch.tensor(0.0, device=self.device)
        if real_features is not None and fake_features is not None:
            # Average features across spatial dimensions if needed
            if len(real_features.shape) > 2:
                real_features_flat = real_features.view(real_features.size(0), -1)
                fake_features_flat = fake_features.view(fake_features.size(0), -1)
            else:
                real_features_flat = real_features
                fake_features_flat = fake_features

            fm_loss = self.feature_criterion(
                real_features_flat.mean(dim=0), fake_features_flat.mean(dim=0)
            )

        # Total generator loss
        g_loss = g_loss_adv + self.feature_weight * fm_loss

        g_loss.backward()
        self.g_optimizer.step()

        self.d_losses.append(d_loss.item())
        self.g_losses.append(g_loss_adv.item())
        self.feature_losses.append(fm_loss.item())

        return d_loss.item(), g_loss_adv.item(), fm_loss.item()


class ComparisonTrainer:
    """
    Train and compare three GAN variants:
    1. Baseline (standard BCE)
    2. Label Smoothing
    3. Feature Matching
    """

    def __init__(self, device: str = "cpu"):
        """Initialize comparison trainer."""
        self.device = device
        self.results = {}

    def train_all_variants(
        self,
        generator_class,
        discriminator_class,
        train_loader,
        num_epochs: int = 50,
        lr: float = 0.0002,
        beta1: float = 0.5,
    ) -> Dict:
        """
        Train all three GAN variants on the same dataset.

        Args:
            generator_class: Class to instantiate generators
            discriminator_class: Class to instantiate discriminators
            train_loader: DataLoader with training images
            num_epochs: Number of training epochs
            lr: Learning rate
            beta1: Adam beta1 parameter

        Returns:
            Dictionary with results for each variant
        """
        variants = []

        # 1. Baseline GAN
        print("=" * 60)
        print("Training Baseline GAN (No Stabilization)...")
        print("=" * 60)
        gen1 = generator_class(latent_dim=100)
        disc1 = discriminator_class()
        gen1.to(self.device)
        disc1.to(self.device)
        baseline_trainer = BaselineGAN(
            gen1, disc1, lr=lr, beta1=beta1, device=self.device
        )
        baseline_losses = self._train_variant(
            baseline_trainer, train_loader, num_epochs
        )
        variants.append(("Baseline", baseline_trainer))
        self.results["Baseline"] = baseline_losses

        # 2. Label Smoothing GAN
        print("\n" + "=" * 60)
        print("Training Label Smoothing GAN (0.9/0.1)...")
        print("=" * 60)
        gen2 = generator_class(latent_dim=100)
        disc2 = discriminator_class()
        gen2.to(self.device)
        disc2.to(self.device)
        ls_trainer = LabelSmoothingGAN(
            gen2, disc2, lr=lr, beta1=beta1, device=self.device
        )
        ls_losses = self._train_variant(ls_trainer, train_loader, num_epochs)
        variants.append(("Label Smoothing", ls_trainer))
        self.results["Label Smoothing"] = ls_losses

        # 3. Feature Matching GAN
        print("\n" + "=" * 60)
        print("Training Feature Matching GAN...")
        print("=" * 60)
        gen3 = generator_class(latent_dim=100)
        disc3 = discriminator_class()
        gen3.to(self.device)
        disc3.to(self.device)
        fm_trainer = FeatureMatchingGAN(
            gen3, disc3, lr=lr, beta1=beta1, device=self.device
        )
        fm_losses = self._train_variant(fm_trainer, train_loader, num_epochs)
        variants.append(("Feature Matching", fm_trainer))
        self.results["Feature Matching"] = fm_losses

        return self.results

    def _train_variant(self, trainer, train_loader, num_epochs: int) -> Dict:
        """Train a single GAN variant."""
        trainer.generator.train()
        trainer.discriminator.train()

        for epoch in range(num_epochs):
            epoch_d_loss = 0
            epoch_g_loss = 0
            num_batches = 0

            for real_images, _ in train_loader:
                real_images = real_images.to(self.device)
                batch_size = real_images.size(0)

                # Flatten images
                real_images = real_images.view(-1, 784)

                # Train step
                if isinstance(trainer, FeatureMatchingGAN):
                    d_loss, g_loss, fm_loss = trainer.train_step(
                        real_images, batch_size
                    )
                else:
                    d_loss, g_loss = trainer.train_step(real_images, batch_size)

                epoch_d_loss += d_loss
                epoch_g_loss += g_loss
                num_batches += 1

            avg_d_loss = epoch_d_loss / num_batches
            avg_g_loss = epoch_g_loss / num_batches

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f}"
                )

        print(f"✓ Training complete for {trainer.name}\n")

        return {
            "d_losses": trainer.d_losses,
            "g_losses": trainer.g_losses,
            "name": trainer.name,
        }

    def get_stability_metrics(self) -> Dict:
        """
        Compute stability metrics for each variant.

        Returns:
            Dictionary with stability analysis for each variant
        """
        metrics = {}

        for variant_name, losses in self.results.items():
            d_losses = np.array(losses["d_losses"])
            g_losses = np.array(losses["g_losses"])

            # Last 100 batches for stability analysis
            recent_d = d_losses[-100:] if len(d_losses) > 100 else d_losses
            recent_g = g_losses[-100:] if len(g_losses) > 100 else g_losses

            metrics[variant_name] = {
                "avg_d_loss": recent_d.mean(),
                "std_d_loss": recent_d.std(),
                "avg_g_loss": recent_g.mean(),
                "std_g_loss": recent_g.std(),
                "d_oscillation": np.std(np.diff(recent_d)),
                "g_oscillation": np.std(np.diff(recent_g)),
            }

        return metrics

    def print_comparison_report(self):
        """Print detailed comparison report."""
        metrics = self.get_stability_metrics()

        print("\n" + "=" * 80)
        print("GAN STABILITY COMPARISON REPORT")
        print("=" * 80)

        for variant_name, metric_dict in metrics.items():
            print(f"\n{variant_name}:")
            print(f"  Discriminator Loss:")
            print(f"    Average: {metric_dict['avg_d_loss']:.4f}")
            print(f"    Std Dev: {metric_dict['std_d_loss']:.4f}")
            print(f"    Oscillation: {metric_dict['d_oscillation']:.6f}")
            print(f"  Generator Loss:")
            print(f"    Average: {metric_dict['avg_g_loss']:.4f}")
            print(f"    Std Dev: {metric_dict['std_g_loss']:.4f}")
            print(f"    Oscillation: {metric_dict['g_oscillation']:.6f}")

        print("\n" + "=" * 80)
        print("ANALYSIS:")
        print("=" * 80)

        # Find best variant for each metric
        best_d_stability = min(metrics.items(), key=lambda x: x[1]["std_d_loss"])
        best_g_stability = min(metrics.items(), key=lambda x: x[1]["std_g_loss"])
        best_d_balance = min(
            metrics.items(), key=lambda x: abs(x[1]["avg_d_loss"] - 0.5)
        )

        print(f"\n✓ Best D Loss Stability: {best_d_stability[0]}")
        print(f"  Std Dev: {best_d_stability[1]['std_d_loss']:.4f}")

        print(f"\n✓ Best G Loss Stability: {best_g_stability[0]}")
        print(f"  Std Dev: {best_g_stability[1]['std_g_loss']:.4f}")

        print(f"\n✓ Best D Loss Balance (closest to 0.5): {best_d_balance[0]}")
        print(f"  Average D Loss: {best_d_balance[1]['avg_d_loss']:.4f}")

        print("\n" + "=" * 80)
