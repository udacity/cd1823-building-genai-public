# Training GANs Step by Step: Complete Guide

## Overview

This module teaches you how to train a complete GAN on Fashion MNIST for 50 epochs, monitor training dynamics through loss curves, and analyze whether Nash Equilibrium was reached. By the end, you'll understand adversarial training, convergence detection, and failure mode diagnosis.

---

## Key Concepts

### Adversarial Training

GANs are based on a game-theoretic framework where two networks compete:

**Discriminator's Goal:**
- Learn to classify images as real or fake
- Maximize classification accuracy on both real and generated images
- **Loss objective:** Minimize BCE on real images (target=1) and fake images (target=0)

**Generator's Goal:**
- Learn to create realistic images that fool the discriminator
- Generate images that discriminator classifies as "real"
- **Loss objective:** Minimize BCE where generator's images have target=1 (trick D into thinking they're real)

**The Game:**
- Each network tries to defeat the other
- Equilibrium occurs when both networks reach stable performance
- This is called **Nash Equilibrium**

### Loss Curve Interpretation

#### Random Baseline
Binary cross-entropy for random guessing: **0.693** (= log(2))
- Either network randomly assigning labels would get this loss
- Both D and G should start near this point

#### Discriminator Loss Analysis

**What D loss tells you:**

| D Loss Value | Interpretation | Status |
|---|---|---|
| 0.0 - 0.1 | D too good, G failing | ✗ Mode Collapse |
| 0.1 - 0.3 | D strong, G struggling | Imbalanced |
| 0.3 - 0.7 | Balanced competition |  Ideal |
| 0.7 - 0.9 | G too strong | Imbalanced |
| 0.9 - 1.0 | D can't learn |  D Failure |

**Expected trajectory:**
- Starts near 0.693 (random)
- Fluctuates during training (both networks adapting)
- Stabilizes around 0.5-0.7 (balanced)

#### Generator Loss Analysis

**What G loss tells you:**

| G Loss Value | Interpretation |
|---|---|
| Decreasing |  G learning to fool D |
| Increasing |  G getting worse |
| Constant |  No learning occurring |

**Expected trajectory:**
- Starts high (G generates random noise)
- Decreases over epochs (G improves)
- Stabilizes when G reaches equilibrium with D

### Failure Modes

#### Mode Collapse
**Symptom:** D loss drops below 0.1

**Cause:** Generator learns to produce only a few types of images that fool D, instead of diverse clothing types

**Visual indicator:** Generated samples look like repeated/similar items

**Recovery:** Adjust architecture, use batch normalization, or gradient penalty


#### Discriminator Overpowering
**Symptom:** D loss above 0.9

**Cause:** Discriminator network too strong; G can't generate convincing fakes

**Recovery:** Reduce D learning rate or simplify D architecture


#### Divergence
**Symptom:** Losses oscillate wildly without stabilizing

**Cause:** Learning rates too high or architectural mismatch

**Recovery:** Reduce learning rate, use Adam optimizer with proper betas


### Nash Equilibrium

**Definition:** Both networks reach a stable state where:
- Discriminator can't improve further (balanced at ~0.5-0.7 loss)
- Generator can't improve further (loss stable or slightly increasing)
- Neither network can make progress without degrading opponent's performance

**Indicators of Nash Equilibrium:**
-  D loss stabilizes around 0.5-0.7
-  G loss stabilizes (no significant improvement)
-  Loss curves smooth (not wild oscillations)
-  Generated samples improve each epoch then plateau
-  Training statistics consistent over final batches

---

## Dataset: Fashion MNIST

**What is it?**
- 60,000 training images of clothing items
- 10 classes: T-shirt, Trouser, Pullover, Coat, Dress, Shirt, Sneaker, Bag, Ankle Boot, Boot
- 28×28 grayscale images
- Similar structure to MNIST but more challenging for GANs

**Normalization Applied:**
- Raw images: [0, 255] or [0, 1] after ToTensor()
- Normalized for GAN: [-1, 1] using Normalize((0.5,), (0.5,))
- Why [-1, 1]? Better for GAN training with tanh activation in generator

---

## Training Algorithm

### Step 1: Discriminator Training
```python
# For each batch of real images:

# Forward pass on real images
real_output = D(real_images)  # Should output ~1.0 (real)
real_loss = BCE(real_output, target=1.0)

# Forward pass on generated images
fake_images = G(noise)
fake_output = D(fake_images)  # Should output ~0.0 (fake)
fake_loss = BCE(fake_output, target=0.0)

# Combined loss
d_loss = (real_loss + fake_loss) / 2

# Backprop and optimize
d_loss.backward()
optimizer_d.step()
```

### Step 2: Generator Training
```python
# Generate batch of fake images
noise = torch.randn(batch_size, latent_dim)
fake_images = G(noise)

# Trick discriminator by labeling fake as "real"
fake_output = D(fake_images)
g_loss = BCE(fake_output, target=1.0)  # NOTE: target=1, not 0!

# Backprop and optimize
g_loss.backward()
optimizer_g.step()
```

### Full Training Loop
```python
for epoch in range(num_epochs):
    for batch_idx, (real_images, _) in enumerate(train_loader):
        # Train discriminator
        d_loss = train_discriminator_step(D, G, real_images, optimizer_d, device)
        
        # Train generator
        g_loss = train_generator_step(D, G, optimizer_g, batch_size, device)
        
        # Track losses
        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())
        
        # Save samples every checkpoint_interval epochs
        if (epoch + 1) % checkpoint_interval == 0:
            samples = generate_sample_grid(G, device)
            generated_samples.append(samples)
```

---

## Expected Results

### After 50 Epochs on Fashion MNIST

**Loss Curve Expectations:**
- D loss: 0.693 → stabilizes at 0.4-0.7 (fluctuations normal)
- G loss: 1.5+ → decreases to 0.8-1.2

**Generated Sample Quality:**
- Epochs 1-10: Random noise, no structure
- Epochs 10-20: Blurry shapes, barely recognizable
- Epochs 20-40: Clear clothing outlines, distinguishable items
- Epochs 40-50: Realistic Fashion MNIST clothing items

**Convergence Indicators:**
- Final D loss: 0.3-0.8 (any value in this range is acceptable)
- Final G loss: 0.8-1.5 (should have decreased from initial)
- Loss curves: Smooth, not oscillating wildly
- Samples: Diverse clothing types (not all one class)

---

## Key Hyperparameters

| Parameter | Value | Why |
|---|---|---|
| Learning Rate | 0.0002 | Standard for GANs; higher causes divergence |
| Beta1 (Adam) | 0.5 | Recommended for GANs; default 0.9 is too high |
| Batch Size | 64 | Balance between stability and computation |
| Latent Dim | 100 | Standard dimension for random noise |
| Epochs | 50 | Sufficient for Fashion MNIST convergence |
| Checkpoint | Every 5 epochs | 10 checkpoints to track progress |

---

## Analyzing Your Results

### Checklist for Successful Training

After training, check these indicators:

- [ ] **D loss stabilized?** Look for convergence to 0.4-0.8 range
- [ ] **G loss decreased?** Compare initial vs final value
- [ ] **Curves smooth?** Few wild jumps or oscillations?
- [ ] **No mode collapse?** D loss didn't drop below 0.1?
- [ ] **Samples improved?** Visual quality better at epoch 50 vs epoch 5?
- [ ] **Diverse outputs?** Generated samples cover multiple clothing types?

### If Training Failed

**D loss → 0 (Mode Collapse)**
- Solution: Increase D learning rate, add gradient penalty, or use spectral normalization
- Try: learning_rate = 0.0004 for D, keep G at 0.0002

**D loss → 1 (Discriminator Too Good)**
- Solution: Decrease D learning rate or simplify D architecture
- Try: learning_rate = 0.00005 for D

**Wild oscillations**
- Solution: Reduce learning rate, use feature matching, or add batch normalization
- Try: learning_rate = 0.0001 for both networks

**All samples look the same (Mode Collapse)**
- Solution: Use Wasserstein loss (WGAN), gradient penalty, or spectral normalization
- This requires architectural changes beyond this module


## Resources

**Recommended Reading:**
- Goodfellow et al., "Generative Adversarial Networks" (2014) - Original GAN paper
- Radford et al., "Unsupervised Representation Learning with Deep Convolutional GANs" (DCGAN)
- Miyato et al., "Spectral Normalization for GAN Discriminators" (stabilization technique)

**PyTorch GAN Resources:**
- PyTorch Generative Models Tutorial
- GAN Zoo (collection of GAN implementations)
- TorchGAN library for standardized training

---

## Summary

This module teaches you:
1. **How GANs train** through adversarial competition
2. **How to track training** via loss curves and convergence analysis
3. **How to detect problems** through failure mode diagnosis
4. **What Nash Equilibrium is** and how to identify it
5. **How to interpret results** and iterate on hyperparameters


