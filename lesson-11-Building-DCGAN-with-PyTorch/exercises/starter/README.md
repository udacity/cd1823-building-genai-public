# Exercise: DCGAN on CIFAR-10 (Starter)

Complete the 10 TODO sections to build and train a full DCGAN pipeline.

##  Exercise Objectives

1. Set up PyTorch device detection
2. Load and preprocess CIFAR-10 dataset
3. Initialize DCGAN models
4. Create training loop using DCGANTrainer
5. Train for 20 epochs
6. Visualize loss curves
7. Generate synthetic images
8. Compare real vs generated
9. Compute training statistics
10. Answer analysis questions


## TODO Items

### TODO 1: Set Device (2 min)
Detect MPS/CUDA/CPU and move models accordingly.
```python
# Check: torch.backends.mps.is_available(), torch.cuda.is_available()
```

### TODO 2: Load CIFAR-10 (5 min)
Create transforms and DataLoader.
```python
# Create: transforms.Normalize with mean/std (0.5, 0.5, 0.5)
# Load: datasets.CIFAR10 with transform
# DataLoader: batch_size=64, shuffle=True
```

### TODO 3: Create Models (5 min)
Instantiate generator and discriminator.
```python
# Call: create_dcgan_models() with latent_dim=100, num_channels=3
# Initialize: initialize_weights() on both models
```

### TODO 4: Create Trainer (3 min)
Initialize DCGANTrainer with models and hyperparameters.
```python
# Create: DCGANTrainer with default hyperparams
```

### TODO 5: Train DCGAN (25 min)
Run the training loop for 20 epochs.
```python
# Call: trainer.train() with num_epochs=20
```

### TODO 6: Visualize Losses (8 min)
Plot D and G loss curves.
```python
# Create: 2 subplot figure
# Plot: D loss with reference line at 0.5
# Plot: G loss curve
```

### TODO 7: Generate Images (5 min)
Use generator to create 32 synthetic images.
```python
# Generate: z vectors and fake images
# Denormalize: from [-1, 1] to [0, 1]
# Display: in 4×8 grid
```

### TODO 8: Real vs Generated (8 min)
Side-by-side comparison visualization.
```python
# Load: real batch from dataset
# Generate: fake batch
# Display: alternating rows of real and generated
```

### TODO 9: Statistics (5 min)
Compute and print training metrics.
```python
# Calculate: mean, std, min, max for D and G losses
# Print: formatted statistics table
```

### TODO 10: Analysis Questions (10 min)
Think critically about DCGAN properties.
```python
# Write answers to 5 analysis questions
# Include reasoning and observations
```


##  How to Run

###  Jupyter Notebook
```bash
jupyter notebook exercise_notebook.ipynb
```
