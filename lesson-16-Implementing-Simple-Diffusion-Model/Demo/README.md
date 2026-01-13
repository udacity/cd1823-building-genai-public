Module 16 Demo: Detailed Code Explanation
==========================================

This document provides detailed walkthroughs of the demo implementation.

Reference: Demo/diffusion_model.py and Demo/diffusion_training.py

## File Structure

### diffusion_model.py 
Contains all model components:
- NoiseScheduler
- TimeEmbedding
- ResidualBlock
- AttentionBlock (optional)
- SimpleUNet
- Utility functions

### diffusion_training.py 
Contains training infrastructure:
- DDPMTrainer class
- DDPMSampler class
- Training utilities
- Visualization functions

### demo_notebook.ipynb (10 sections)
Interactive walkthrough with visualizations.

---

## Part 1: NoiseScheduler Deep Dive

### What It Does
Manages the fixed noise schedule for forward diffusion.

### Mathematical Background

**Beta Schedule (Noise Variance):**
```
β_t = β_start + t * (β_end - β_start) / T

Linear schedule from 0.0001 to 0.02 over 1000 steps.
```

**Alpha Schedule:**
```
α_t = 1 - β_t  (retention rate at step t)
```

**Cumulative Products:**
```
ᾱ_t = ∏_{s=1}^t α_s  (cumulative retention)

At t=0: ᾱ_0 ≈ 1    (mostly original image)
At t=T: ᾱ_T ≈ 0.003 (mostly noise)
```

**Forward Diffusion Coefficients:**
```
√ᾱ_t: Weight for original image (decreases)
√(1-ᾱ_t): Weight for noise (increases)

x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
```

### Implementation Details

```python
class NoiseScheduler:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        # Create linear schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        
        # Pre-compute for efficiency
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Pre-compute square roots (used during training/sampling)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
```

**Why Pre-compute?**
- Avoid repeated computation during training
- Ensures consistent values
- Faster forward/backward passes

### Usage

```python
scheduler = NoiseScheduler(num_timesteps=1000)

# Get coefficients for batch of timesteps
t = torch.tensor([0, 100, 500, 999], device='cuda')  # 4 timesteps
sqrt_alpha, sqrt_one_minus_alpha = scheduler.get_coefficients(t)

# Shape: (4, 1, 1, 1) for broadcasting with images
# Apply: x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
```

---

## Part 2: Forward Diffusion Process

### add_noise Function

```python
def add_noise(x_0, timestep, scheduler, noise=None):
    \"\"\"Apply forward diffusion.\"\"\"
    if noise is None:
        noise = torch.randn_like(x_0)
    
    sqrt_alpha, sqrt_one_minus_alpha = scheduler.get_coefficients(timestep)
    x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
    return x_t, noise
```

### Step-by-Step Example

**Input:** 
- x_0: Original digit image (batch=1, channels=1, h=28, w=28)
- t: timestep = 500
- scheduler: NoiseScheduler instance

**Process:**

1. Get coefficients:
```
√ᾱ_500 ≈ 0.1  (10% original image)
√(1-ᾱ_500) ≈ 0.99  (99% noise)
```

2. Sample random noise:
```
ε ~ N(0, I)  (shape: 1, 1, 28, 28)
```

3. Mix:
```
x_500 = 0.1 * x_0 + 0.99 * ε
       = 0.1 * (original digit) + 0.99 * (random noise)
       ≈ mostly noise with hint of original digit
```

4. Return (x_500, ε):
```
x_500: The noisy image input to model
ε: The target noise for training
```

**Training Objective:**
```
Model(x_500, t=500) ≈ ε  (predict the noise)
Loss = MSE(Model output, ε)
```

### Why This Works

**Key Insight:** At different timesteps, the model must denoise different levels:
- Early steps: Fine details (small noise)
- Late steps: Coarse structure (heavy noise)
- By training on all steps equally, network learns multi-scale denoising

---

## Part 3: Time Embedding

### TimeEmbedding Class

```python
class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim
    
    def forward(self, timestep):
        \"\"\"Convert timestep to 128-dim embedding.\"\"\"
        device = timestep.device
        half_dim = 64
        
        # Frequency schedule
        freqs = torch.exp(-math.log(10000) * torch.arange(half_dim) / half_dim)
        
        # Multiply by timestep
        args = timestep[:, None].float() * freqs[None, :]
        
        # Apply sine and cosine
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding  # (batch_size, 128)
```

### Why Sinusoidal?

**Inspiration:** Transformer Positional Encoding

Different frequencies capture position at different scales:
```
High frequency: Captures fine details (local position)
Low frequency: Captures coarse structure (global position)

Model learns which frequencies are important.
```

### Example

```
timestep = 500

Frequency 0 (lowest): sin(500 * 0.1) = sin(50) ≈ -0.26
Frequency 1:          sin(500 * 0.105) ≈ 0.84
...
Frequency 63 (highest): sin(500 * 10000) ≈ -0.49

Result: [sin(0), sin(1), ..., sin(63), cos(0), cos(1), ..., cos(63)]
        = 128-dimensional embedding
```

**Why This Works:**
- Smooth variation with timestep (nearby timesteps have similar embeddings)
- Captures both coarse (low freq) and fine (high freq) information
- Model can learn to extract relevant information

---

## Part 4: ResidualBlock with FiLM

### FiLM: Feature-wise Linear Modulation

Basic idea: Multiply feature maps by timestep-dependent scale.

```python
class ResidualBlock(nn.Module):
    def forward(self, x, time_embedding):
        # Path 1: Conv
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # FiLM: Time conditioning
        time_scale = self.time_mlp(time_embedding)  # (batch, channels)
        time_scale = time_scale[:, :, None, None]  # (batch, channels, 1, 1)
        h = h * time_scale  # Modulate!
        
        # Path 2: Conv
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        # Skip connection
        return h + self.skip(x)
```

### Why FiLM?

**Key Insight:** Different timesteps need different processing.

```
Early timestep (t small, little noise):
  - Network should preserve details
  - FiLM scales: Keep high magnitudes

Late timestep (t large, heavy noise):
  - Network should focus on coarse structure
  - FiLM scales: Adjust adaptively
```

### Implementation Breakdown

1. **Main Conv Path:**
```python
h = self.conv1(F.silu(self.norm1(x)))
```

2. **Generate Time Scale:**
```python
# 2-layer MLP maps time_embedding to channel-wise scale
time_scale = self.time_mlp(time_embedding)  # (batch, channels)

# Reshape for broadcasting
time_scale = time_scale[:, :, None, None]  # (batch, channels, 1, 1)
```

3. **Apply Scale:**
```python
h = h * time_scale  # Multiply each channel by its scale
```

4. **Reason for this approach:**
- Different channels focus on different features
- Time modulation allows adaptive behavior
- Mathematically: Element-wise multiplication = diagonal linear transformation

---

## Part 5: SimpleUNet Architecture

### Overall Structure

```
Input (batch, 1, 28, 28)
  ↓
Initial Conv (batch, 64, 28, 28)
  ↓ ENCODER
ResBlock (64 → 128) + Downsample (stride=2)
  → (batch, 128, 14, 14)
  ↓
ResBlock (128 → 128) + Downsample (stride=2)
  → (batch, 128, 7, 7)
  ↓ MIDDLE
ResBlock (128 → 128) at bottleneck
  → (batch, 128, 7, 7)
  ↓ DECODER
Upsample + ResBlock (128 → 64)
  → (batch, 64, 14, 14)
  ↓
Upsample + ResBlock (64 → 64)
  → (batch, 64, 28, 28)
  ↓
Final Conv (64 → 1)
  → (batch, 1, 28, 28)
Output (predicted noise, same shape as input)
```

### Why This Structure?

**Encoder (Downsampling):**
- Reduces spatial resolution (28 → 14 → 7)
- Increases channel capacity (64 → 128)
- Captures hierarchical features

**Bottleneck:**
- Most compressed representation
- All information must flow through this
- ResBlock for efficient processing

**Decoder (Upsampling):**
- Restores spatial resolution (7 → 14 → 28)
- Decreases channel capacity (128 → 64)
- Reconstructs detailed features

### Time Conditioning

Time embedding is conditioned at EVERY ResBlock:

```python
for block in encoder:
    if isinstance(block, ResidualBlock):
        h = block(h, time_emb)  # ← Pass time_emb
    else:
        h = block(h)  # Pooling, etc.
```

**Why at every block?**
- Allows different operations at different timesteps
- Network can adapt its denoising strategy
- Critical for good quality

---

## Part 6: Training Loop

### DDPMTrainer Class

```python
class DDPMTrainer:
    def __init__(self, model, scheduler, learning_rate=0.001, device='cpu'):
        self.model = model
        self.scheduler = scheduler
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.train_losses = []
```

### Single Training Step

```python
def train_step(self, x_0):
    batch_size = x_0.shape[0]
    
    # 1. Sample random timesteps
    t = torch.randint(0, self.scheduler.num_timesteps, (batch_size,))
    
    # 2. Sample random noise
    noise = torch.randn_like(x_0)
    
    # 3. Forward diffusion
    x_t, _ = add_noise(x_0, t, self.scheduler, noise)
    
    # 4. Predict noise
    predicted_noise = self.model(x_t, t)
    
    # 5. MSE Loss (MSE not adversarial!)
    loss = F.mse_loss(predicted_noise, noise)
    
    # 6. Backward
    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
    self.optimizer.step()
    
    return loss.item()
```

### Key Differences from cGAN

**cGAN (Module 13):**
```
Step 1: Update Discriminator (maximize real, minimize fake)
Step 2: Update Generator (maximize fake)
Problem: Conflicting objectives → Oscillations
```

**Diffusion (Module 16):**
```
Single Step: Update U-Net (minimize noise prediction MSE)
Advantage: Single objective → Smooth convergence
```

### Training Epoch

```python
def train_epoch(self, dataloader):
    total_loss = 0.0
    for images, _ in dataloader:
        loss = self.train_step(images)
        total_loss += loss
    return total_loss / len(dataloader)
```

### Full Training Loop

```python
for epoch in range(num_epochs):
    epoch_loss = self.train_epoch(train_loader)
    print(f"Epoch {epoch} | Loss: {epoch_loss:.6f}")
```

**Expected Behavior:**
- Loss should decrease smoothly
- No sudden jumps or oscillations
- Consistent improvement across epochs

---

## Part 7: Generation (Reverse Diffusion)

### DDPMSampler Class

```python
class DDPMSampler:
    @torch.no_grad()
    def sample(self, num_samples=16):
        # Start with pure noise
        x_t = torch.randn(num_samples, 1, 28, 28, device=self.device)
        
        # Reverse diffusion: T → 1
        for t_idx in range(self.scheduler.num_timesteps - 1, -1, -1):
            t = torch.full((num_samples,), t_idx, device=self.device)
            
            # Predict noise at this step
            predicted_noise = self.model(x_t, t)
            
            # Denoise one step
            # x_{t-1} = (x_t - (1-α_t)/√(1-ᾱ_t) * ε_pred) / √α_t + noise
            
            alpha_t = self.scheduler.alphas[t_idx]
            alpha_cumprod_t = self.scheduler.alphas_cumprod[t_idx]
            
            x_t = (x_t - (1 - alpha_t) / √(1 - alpha_cumprod_t) * predicted_noise) / √alpha_t
            
            # Add noise except at last step
            if t_idx > 0:
                z = torch.randn_like(x_t)
                x_t = x_t + √posterior_variance * z
        
        return x_t  # Generated image
```

### Sampling Algorithm

```
Input: Pure noise x_T
Output: Generated digit x_0

For each timestep from T down to 1:
    1. Feed x_t and timestep to trained U-Net
    2. Get predicted noise ε_pred
    3. Remove predicted noise from x_t:
       x_{t-1} ≈ x_t - (noise contribution of step t)
    4. Add small noise (except at last step)
       x_{t-1} += random noise
    5. Continue with x_{t-1}
```

### Why This Works

**Key Insight:** The U-Net learned to reverse the forward process.

```
Forward:  x_0 → x_1 → x_2 → ... → x_T (clean → noisy)
Reverse:  x_T → x_{T-1} → ... → x_0 (noise → clean)
```

The model learned at each step t:
- What noise was ADDED during forward process
- How to REMOVE it during reverse process

---

## Part 8: Visualization and Analysis

### Loss Curve Visualization

```python
def plot_training_curves(history):
    fig, axes = plt.subplots(1, 2)
    
    # Batch-level losses
    axes[0].plot(history['train_losses'], linewidth=0.5, alpha=0.7)
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Loss per Batch (Smooth!)')
    
    # Epoch-level losses
    axes[1].plot(history['epoch_losses'], marker='o')
    axes[1].set_ylabel('Average Loss')
    axes[1].set_title('Loss per Epoch (Monotonic!)')
```

**Expected Pattern:**
```
Epoch 1: Loss = 0.5000
Epoch 2: Loss = 0.4500
Epoch 3: Loss = 0.4200
Epoch 4: Loss = 0.4000  ← Smooth decrease
...
Epoch 10: Loss = 0.3500

Compare with cGAN:
Epoch 1: Loss = 0.5000
Epoch 2: Loss = 0.3200  ← Big jump
Epoch 3: Loss = 0.6100  ← Up!
Epoch 4: Loss = 0.4500  ← Down!  (Oscillating)
...
```

### Convergence Metrics

```python
# Compute smoothness
epoch_diffs = [abs(loss[i+1] - loss[i]) for i in range(len(loss)-1)]

print(f"Average change: {np.mean(epoch_diffs):.6f}")
print(f"Max increase: {max(epoch_diffs):.6f}")

# Check for monotonicity
if max(epoch_diffs) == 0:
    print("✓ Perfectly monotonic!")
else:
    print(f"✓ Mostly monotonic ({sum(1 for d in epoch_diffs if d > 0)} increases)")
```

---

## Hyperparameter Tuning Guide

### Learning Rate
- **Too high (0.1):** Loss diverges
- **Good (0.001):** Smooth convergence
- **Too low (0.00001):** Very slow convergence

```python
# Recommendation
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

### Batch Size
- **Too small (8):** Noisy gradients
- **Good (64):** Stable training
- **Too large (512):** May overfit

```python
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
```

### Number of Timesteps
- **100 steps:** Fast training, lower quality
- **1000 steps (default):** Good balance
- **5000 steps:** Very high quality, slow

```python
scheduler = NoiseScheduler(num_timesteps=1000)
```

### Model Size
- **base_channels=32:** Small model, fast
- **base_channels=64 (default):** Balanced
- **base_channels=128:** Large model, slower

```python
model = SimpleUNet(base_channels=64)
```

---


## Performance Tips

### 1. Gradient Accumulation
```python
# Train on larger effective batch size
accumulation_steps = 4
for step, (x, _) in enumerate(dataloader):
    loss = train_step(x)
    (loss / accumulation_steps).backward()
    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 2. Mixed Precision Training
```python
from torch.amp import autocast, GradScaler

scaler = GradScaler()
with autocast(device_type='cuda'):
    loss = F.mse_loss(predicted_noise, noise)
scaler.scale(loss).backward()
scaler.step(optimizer)
```

### 3. Data Caching
```python
# Pre-normalize data offline
# Load from cache during training
# Saves computation time
```

---

## Summary

**Key Components:**
1. NoiseScheduler: Fixed variance schedule
2. TimeEmbedding: Sinusoidal timestep encoding
3. ResidualBlock: FiLM-based time conditioning
4. SimpleUNet: Encoder-decoder for noise prediction
5. Training: Single MSE objective
6. Sampling: Iterative denoising

**Why It Works:**
- Pure regression (not adversarial) → Smooth convergence
- Fixed noise schedule → Efficient training
- U-Net architecture → Multi-scale features
- Time conditioning → Adaptive denoising

**Comparison with cGAN:**
- cGAN: 2 networks, adversarial loss, volatile curves
- Diffusion: 1 network, MSE loss, smooth curves
