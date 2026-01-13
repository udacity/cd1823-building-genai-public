# Module 9: GAN Training Improvements - Demo

## Overview

This demo introduces two essential stabilization techniques for training GANs:

1. **Label Smoothing**: Use soft labels (0.9/0.1) instead of hard labels (1/0)
2. **Feature Matching**: Add auxiliary loss on intermediate discriminator features

## Learning Objectives

By the end of this demo, you'll understand:
- Why GANs are unstable and what causes failures
- How label smoothing prevents discriminator overconfidence
- How feature matching prevents mode collapse
- How to measure training stability empirically
- When to use each technique (and when to use both)

## Key Concepts

### Label Smoothing

**Problem**: Hard labels (1/0) make the discriminator extremely confident, leading to:
- Vanishing gradients for the generator
- Early discriminator overconfidence
- Generator stuck learning trivial features

**Solution**: Use soft labels (0.9/0.1):
```python
# Instead of:
real_labels = torch.ones(batch_size, 1)  # Hard: 1.0

# Use:
real_labels = torch.full((batch_size, 1), 0.9)  # Soft: 0.9
fake_labels = torch.full((batch_size, 1), 0.1)  # Soft: 0.1
```

**Benefits**:
- Discriminator stays uncertain longer
- Generator gets stronger gradients early
- More stable training curves

### Feature Matching

**Problem**: Standard GAN loss only cares about fooling the final classifier
- Generator can produce artifacts that fool just the final layer
- Mode collapse: generator learns to produce only a few types of images
- Poor feature diversity

**Solution**: Add loss on intermediate discriminator features:
```python
# Standard loss: G wants D(fake) ≈ 1
g_loss_adv = BCE(D(fake), 1)

# Feature matching loss: G wants to match real/fake feature distributions
real_features = D_intermediate(real_images)
fake_features = D_intermediate(fake_images)
fm_loss = MSE(real_features.mean(dim=0), fake_features.mean(dim=0))

# Combined:
g_loss = g_loss_adv + λ * fm_loss
```

**Benefits**:
- Forces generator to learn representative features
- Prevents mode collapse
- Improves sample quality and diversity


## Key Metrics

For each variant, we measure (over last 100 batches):

1. **avg_d_loss**: Average discriminator loss
   - Ideal: ~0.5 (balanced competition)
   - Bad: <0.1 (discriminator too good, mode collapse)
   - Bad: >0.9 (generator too good, no feedback)

2. **std_d_loss**: Stability (lower is better)
   - Measures variance in loss
   - More stable = more consistent learning

3. **d_oscillation**: Smoothness (lower is better)
   - Measures how much loss bounces around
   - Smoother = better gradient flow

4. **avg_g_loss**: Generator loss trend
   - Should decrease or stabilize
   - Increasing = generator struggling

## Expected Results

Typical patterns observed:

| Metric | Baseline | Label Smoothing | Feature Matching |
|--------|----------|-----------------|------------------|
| D Loss Stability | Unstable | Better | Best |
| D Loss Balance | Often <0.1 | ~0.4 | ~0.3 |
| Sample Quality | Poor | Good | Excellent |
| Training Time | Fast | Same | +20% |

## Experiments to Try

1. **Different smoothing values**: Try 0.8/0.2 or 0.95/0.05
2. **Different feature weights**: Vary λ from 1 to 100
3. **Combination**: Use both label smoothing AND feature matching
4. **Measure quality**: Compare FID scores or visual inspection
5. **Different datasets**: Apply techniques to MNIST, CIFAR-10, etc.

## Troubleshooting

**Q: Training takes forever**
- A: Reduce batch size or epochs in demo (line change only)

**Q: Losses are NaN**
- A: Learning rate too high, reduce from 0.0002 to 0.0001

**Q: All losses are identical across variants**
- A: Check that models aren't reusing same random seed

**Q: Feature matching breaks with error**
- A: Ensure your discriminator has a "layer3" attribute; adjust layer name in code
