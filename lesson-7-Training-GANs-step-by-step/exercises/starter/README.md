# Starter Exercises: Training GANs

## Quick Start


###  Interactive Notebook 
**File:** `exercise_notebook.ipynb`

```bash
jupyter notebook exercise_notebook.ipynb
```

---


## What You'll Do

### Core Task: Train a GAN for 50 Epochs

**Given:**
- Fashion MNIST dataset (60,000 clothing images)
- Generator and Discriminator classes from previous modules
- Training framework (`gan_training.py`)

**Your Job:**
1. Load and prepare data (normalize to [-1, 1])
2. Create models and optimizers
3. Implement training loop calling `train_gan()`
4. Track discriminator and generator losses
5. Analyze convergence and detect failure modes
6. Visualize progression from epoch 5 to epoch 50

**Expected Output:**
- Loss curves showing convergence
- Generated Fashion MNIST items of increasing quality
- Nash Equilibrium analysis
- Failure mode diagnosis (if any)

---

## Key TODOs Explained

### TODO 1: Load Fashion MNIST
```python
# Create transforms
transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),  # [-1, 1] range
])

# Load dataset
datasets.FashionMNIST(root='./data', train=True, download=True, transform=...)

# Create DataLoader
DataLoader(dataset, batch_size=64, shuffle=True)
```

**Why [-1, 1]?** GAN generators use tanh activation which outputs [-1, 1].

---

### TODO 2: Create Models
```python
from generator_model import create_generator
from discriminator_model import create_discriminator

generator = create_generator(latent_dim=100)
discriminator = create_discriminator()

generator.to(device)
discriminator.to(device)
```

---

### TODO 3: Train GAN (Most Important!)
```python
from gan_training import train_gan

d_losses, g_losses, generated_samples = train_gan(
    generator=generator,
    discriminator=discriminator,
    train_loader=train_loader,
    num_epochs=50,
    device=device,
    learning_rate=0.0002,
    beta1=0.5,
    checkpoint_interval=5,
    verbose=True
)
```

**This will:**
- Train both networks for 50 epochs
- Alternate D and G optimization
- Save sample grids every 5 epochs
- Print progress
- Return losses and samples

**Training time:** 5-10 minutes on GPU, 15-30 on CPU

---

### TODO 4-9: Analysis
Once training completes:

1. **Visualize losses** - Plot D and G loss curves
2. **Print statistics** - Initial, final, min, max, average
3. **Analyze convergence** - Detect failure modes and Nash Equilibrium
4. **Visualize samples** - Show progression from epoch 5 to 50
5. **Answer questions** - Reflect on results

---

## Expected Results

### Good Training
```
Discriminator Loss: 0.693 → 0.52 (stabilized)
Generator Loss: 1.50 → 0.89 (decreasing)

✓ Nash Equilibrium reached
✓ Samples: Epoch 5 (random) → Epoch 50 (realistic clothing)
```

### Warning Signs
```
D Loss → 0.0   [Mode collapse - generator only generating few items]
D Loss → 1.0   [Discriminator too good - generator can't fool it]
Wild oscillations [Unstable training - adjust learning rate]
```

---

## Common Mistakes

**Normalization to [0, 1] instead of [-1, 1]**
```python
# WRONG
Normalize((0.5,), (0.5,))  # This gives [0, 1]

# RIGHT - The setup already handles this, just use it!
```

**Forgetting to move models to device**
```python
# WRONG
generator = create_generator()
discriminator = create_discriminator()
# Models still on CPU!

# RIGHT
generator.to(device)
discriminator.to(device)
```

**Not using shuffle=True in DataLoader**
```python
# WRONG
DataLoader(dataset, batch_size=64)  # shuffle=False by default

# RIGHT
DataLoader(dataset, batch_size=64, shuffle=True)
```

