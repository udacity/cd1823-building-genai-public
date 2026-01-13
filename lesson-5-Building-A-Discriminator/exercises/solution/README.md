# Discriminator Exercise: Solution Walkthrough

## Overview

This directory contains the **complete, working solution** for the discriminator exercise.

## Files

### `exercise_solution.ipynb`
The full implementation with all TODOs completed.

#### Key Functions

**`create_mixed_batch()`**
Creates a balanced training batch with:
- `batch_size // 2` real images (labeled 1)
- `batch_size // 2` fake generated images (labeled 0)

**`analyze_predictions()`**
Evaluates discriminator performance:
- Real accuracy: % of real images predicted as real (>0.5)
- Fake accuracy: % of fake images predicted as fake (<0.5)

**`main()`**
Complete workflow:
1. Create models (generator + discriminator)
2. Create mixed batch
3. Get predictions
4. Compute BCE loss
5. Analyze performance

## How to Run

```bash
jupyter notebook exercise_solution.ipynb
```

### Expected Output

The script will show:
- Models created successfully
- Mixed batch shapes verified
- Discriminator predictions
- BCE loss value (around 0.68-0.70 for untrained model)
- Accuracy metrics (~50% since discriminator is guessing)
- Visualization plots

## Implementation Details

### Tensor Shapes Through Pipeline

```
Input: 
  - Noise: (batch//2, 100)
  
Generator output:
  - Flat images: (batch//2, 784)
  - Reshaped: (batch//2, 1, 28, 28)
  
Mixed batch:
  - Images: (batch, 1, 28, 28)
  - Targets: (batch, 1)
  
Discriminator:
  - Flattens: (batch, 784)
  - Hidden: (batch, 512) → (batch, 256)
  - Output: (batch, 1)  [probability 0-1]
```

### Key Patterns

**Generating Images**
```python
with torch.no_grad():
    fake_images = generator(noise)
    fake_images = fake_images.view(-1, 1, 28, 28)
```

**Creating Labels**
```python
real_targets = torch.ones(half_batch, 1, device=device)
fake_targets = torch.zeros(half_batch, 1, device=device)
```

**Computing Loss**
```python
criterion = BCELoss()
loss = criterion(predictions, mixed_targets)
```

## What You Should Understand

After studying this, you should understand:

1. **Binary Classification**: Discriminator outputs 0-1 probability
2. **Mixed Batches**: Combining real/fake with proper labels
3. **BCE Loss**: Why 0.693 indicates random guessing
4. **Tensor Shapes**: How images transform through the pipeline
5. **Model Evaluation**: Analyzing accuracy before training



