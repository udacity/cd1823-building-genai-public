# Discriminator Exercise: Starter Guide

## Quick Start

### Run The Interactive Notebook
```bash
jupyter notebook exercise_notebook.ipynb
```

## What You'll Build

A function to create **mixed batches** of real and fake images for training a discriminator.

```python
mixed_images, mixed_targets = create_mixed_batch(
    generator, 
    batch_size=16, 
    latent_dim=100, 
    device='cpu'
)
```

## TODOs in Order

### TODO 1: Split the batch
Split `batch_size` in half to have equal real and fake images.

**Hint**: Use integer division `//`

### TODO 2: Generate fake images
Use the generator to create `half_batch` fake images.

**Hint**:
- Sample noise from `torch.randn(half_batch, latent_dim, device)`
- Pass through generator
- Reshape from (batch, 784) to (batch, 1, 28, 28)
- Use `torch.no_grad()` since you don't need gradients

### TODO 3: Create "real" images
Create placeholder real images (in practice, these come from MNIST).

**Hint**:
- Use `torch.rand(half_batch, 1, 28, 28, device)` for random images
- Real images should have shape `(half_batch, 1, 28, 28)`

### TODO 4: Mix real and fake
Concatenate real and fake images along the batch dimension.

**Hint**:
- Use `torch.cat([real_images, fake_images], dim=0)`
- Result should have shape `(batch_size, 1, 28, 28)`

### TODO 5: Create labels
Create ground truth labels: 1 for real, 0 for fake.

**Hint**:
- Real labels: `torch.ones(half_batch, 1, device)`
- Fake labels: `torch.zeros(half_batch, 1, device)`
- Concatenate them

### TODO 6: Return results
Return both the mixed images and targets.

## How to Test

After implementing `create_mixed_batch()`, the `main()` function will test it.

**Expected output**:
- Predictions shape: `(16, 1)`
- Loss value: around 0.68 (random guessing)
- Accuracy: around 50%

## Common Mistakes

###  Wrong shape
**Problem**: Output shape is `(16, 784)` instead of `(16, 1, 28, 28)`  
**Solution**: Reshape fake images: `fake_images.view(-1, 1, 28, 28)`

###  Gradient errors
**Solution**: Use `torch.no_grad()` when sampling from generator

###  Label mismatch
**Solution**: Ensure targets shape is `(batch_size, 1)`

###  Device mismatch
**Solution**: Ensure all tensors use same device
