"""
Exercise: Training a Basic Diffusion Model on MNIST

"""

## Overview

Your task is to implement a complete DDPM (Denoising Diffusion Probabilistic Model) 
training pipeline on MNIST. This exercise tests your understanding of:

1.  Fixed noise schedules
2.  Forward diffusion process
3.  Time embeddings
4.  U-Net architecture
5.  MSE loss training
6.  Smooth convergence analysis

## The 8 TODOs Explained

### TODO 1: NoiseScheduler.__init__

**What to do:**
Implement the constructor for the NoiseScheduler class.

**Requirements:**
- Create linear beta schedule from beta_start to beta_end
- Compute alphas (1 - beta)
- Compute cumulative products (alphas_cumprod)
- Pre-compute square roots for efficiency

**Expected code structure:**
```python
def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
    self.num_timesteps = num_timesteps
    
    # Step 1: Create betas using torch.linspace
    self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
    
    # Step 2: Compute alphas
    self.alphas = 1.0 - self.betas
    
    # Step 3: Cumulative product
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])
    
    # Step 4: Pre-compute square roots
    self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
    self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
```

**Key insight:**
- Linear schedule: β_t increases uniformly from 0.0001 to 0.02
- Alpha cumprod: ᾱ_t decreases (image becomes noisier)
- Pre-compute: Avoid repeated computation during training

**Hint:**
- Use torch.linspace() for uniform spacing
- torch.cumprod() computes products along dimension
- sqrt() operations are used in forward diffusion formula

---

### TODO 2: add_noise (Forward Diffusion)

**What to do:**
Implement the forward diffusion function that adds noise to images.

**Requirements:**
- Sample random noise (if not provided)
- Get coefficients from scheduler
- Apply forward diffusion formula: x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
- Return noisy image and noise

**Expected code structure:**
```python
def add_noise(x_0, timestep, scheduler, noise=None):
    # Step 1: Sample noise if needed
    if noise is None:
        noise = torch.randn_like(x_0)
    
    # Step 2: Get coefficients
    sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod = scheduler.get_coefficients(timestep)
    
    # Step 3: Forward diffusion formula
    x_t = sqrt_alphas_cumprod * x_0 + sqrt_one_minus_alphas_cumprod * noise
    
    # Step 4: Return
    return x_t, noise
```

**Key insight:**
- At t=0: x_t ≈ x_0 (mostly original)
- At t=T: x_t ≈ noise (mostly random)
- Formula mixes image and noise based on timestep

**Hint:**
- torch.randn_like(x) creates noise with same shape as x
- Element-wise operations work with broadcasting

---

### TODO 3: TimeEmbedding.forward

**What to do:**
Implement sinusoidal time embedding (like Transformer position encoding).

**Requirements:**
- Create frequency schedule
- Multiply timestep by frequencies
- Apply sine and cosine
- Concatenate results

**Expected code structure:**
```python
def forward(self, timestep):
    device = timestep.device
    half_dim = self.embedding_dim // 2
    
    # Step 1: Frequency schedule
    freqs = torch.exp(-math.log(10000) * torch.arange(half_dim, device=device) / half_dim)
    
    # Step 2: Multiply by timestep
    args = timestep[:, None].float() * freqs[None, :]
    
    # Step 3: Apply sine and cosine
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    
    return embedding
```

**Key insight:**
- Different frequencies capture different scales
- Sine/cosine encode position smoothly
- Embedding enables time conditioning

**Hint:**
- math.log(10000) creates exponential frequency spacing
- [:, None] and [None, :] reshape for broadcasting
- torch.cat() concatenates along dimension

---

### TODO 4: ResidualBlock.forward

**What to do:**
Implement the forward pass with FiLM time conditioning.

**Requirements:**
- Apply convolution path with GroupNorm and activation
- Apply FiLM conditioning (multiply by time scale)
- Apply second convolution
- Add skip connection

**Expected code structure:**
```python
def forward(self, x, time_embedding):
    # Step 1: First path
    h = self.norm1(x)
    h = F.silu(h)
    h = self.conv1(h)
    
    # Step 2: Time conditioning (FiLM)
    time_scale_shift = self.time_mlp(time_embedding)  # (batch, channels)
    time_scale_shift = time_scale_shift[:, :, None, None]  # (batch, channels, 1, 1)
    h = h * time_scale_shift
    
    # Step 3: Second path
    h = self.norm2(h)
    h = F.silu(h)
    h = self.conv2(h)
    
    # Step 4: Skip connection
    return h + self.skip(x)
```

**Key insight:**
- FiLM multiplies features by time-dependent scale
- Different channels get different scales
- Skip connection preserves original signal

**Hint:**
- time_mlp outputs (batch, channels)
- Reshape to (batch, channels, 1, 1) for broadcasting
- Skip handles dimension mismatch via 1x1 conv

---

### TODO 5: SimpleUNet.forward

**What to do:**
Implement the U-Net forward pass (encoder → bottleneck → decoder).

**Requirements:**
- Create time embedding
- Apply encoder (downsampling blocks)
- Apply middle (bottleneck block)
- Apply decoder (upsampling blocks)
- Apply final convolution

**Expected code structure:**
```python
def forward(self, x, timestep):
    # Step 1: Time embedding
    time_emb = self.time_embedding(timestep)
    
    # Step 2: Initial conv
    h = self.init_conv(x)
    
    # Step 3: Encoder
    h_down1 = self.down_res1(h, time_emb)
    h = self.down_conv1(h_down1)  # Downsample
    
    h_down2 = self.down_res2(h, time_emb)
    h = self.down_conv2(h_down2)  # Downsample
    
    # Step 4: Middle
    h = self.middle_res(h, time_emb)
    
    # Step 5: Decoder
    h = self.up_conv2(h)  # Upsample
    h = self.up_res2(h, time_emb)
    
    h = self.up_conv1(h)  # Upsample
    h = self.up_res1(h, time_emb)
    
    # Step 6: Final
    h = self.final_norm(h)
    h = F.silu(h)
    h = self.final_conv(h)
    
    return h
```

**Key insight:**
- Time embedding passed to each ResBlock
- Encoder reduces spatial size, increases channels
- Decoder increases spatial size, decreases channels
- Final output same shape as input (noise map)

**Hint:**
- Pass time_emb to every ResidualBlock
- Follow sequence: down1, down2, middle, up2, up1
- Make sure spatial shapes are 28→14→7→14→28

---

### TODO 6: train_step

**What to do:**
Implement single training iteration with MSE loss.

**Requirements:**
- Move data to device
- Sample random timesteps
- Sample random noise
- Apply forward diffusion
- Predict noise with model
- Compute MSE loss
- Backward pass

**Expected code structure:**
```python
def train_step(model, optimizer, x_0, scheduler, device):
    model.train()
    x_0 = x_0.to(device)
    batch_size = x_0.shape[0]
    
    # Step 1: Random timesteps
    timesteps = torch.randint(0, scheduler.num_timesteps, (batch_size,), device=device)
    
    # Step 2: Random noise
    noise = torch.randn_like(x_0)
    
    # Step 3: Forward diffusion
    x_t, _ = add_noise(x_0, timesteps, scheduler, noise)
    
    # Step 4: Predict noise
    predicted_noise = model(x_t, timesteps)
    
    # Step 5: MSE loss (main advantage over cGAN!)
    loss = F.mse_loss(predicted_noise, noise)
    
    # Step 6: Backward pass
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return loss.item()
```

**Key insight:**
- MSE loss is simple and stable (unlike adversarial)
- No competing objectives (single network)
- Gradient clipping helps stability

**Hint:**
- torch.randint() for random timesteps
- torch.randn_like() for noise
- F.mse_loss() computes mean squared error
- clip_grad_norm_ prevents gradient explosion

---

### TODO 7: train_epoch

**What to do:**
Implement training for one epoch (loop over all batches).

**Requirements:**
- Set model to training mode
- Loop through dataloader
- Call train_step for each batch
- Accumulate and return average loss

**Expected code structure:**
```python
def train_epoch(model, optimizer, train_loader, scheduler, device):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # Step 1: Loop through batches
    for images, _ in tqdm(train_loader, desc="Training"):
        # Step 2: Train step
        loss = train_step(model, optimizer, images, scheduler, device)
        
        # Step 3: Accumulate
        total_loss += loss
        num_batches += 1
    
    # Step 4: Return average
    return total_loss / num_batches
```

**Key insight:**
- Loop processes all training data once
- tqdm shows progress
- Average loss = total_loss / num_batches

**Hint:**
- Ignore labels: `for images, _ in train_loader`
- Use tqdm for progress display
- Return average, not total

---

### TODO 8: main() Training Loop

**What to do:**
Implement complete training script with device setup, data loading, and training loop.

**Requirements:**
- Determine device (CUDA/MPS/CPU)
- Load MNIST with proper preprocessing
- Create model and scheduler
- Setup optimizer
- Run training loop
- Plot results and analyze convergence

**Expected code structure:**
```python
def main():
    # Step 1: Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")
    
    # Step 2: Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    
    # Step 3: Model
    model = SimpleUNet(image_channels=1, base_channels=64).to(device)
    scheduler = NoiseScheduler(num_timesteps=1000)
    
    # Step 4: Optimizer (higher LR than cGAN because MSE is stable)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Step 5: Training
    num_epochs = 10
    epoch_losses = []
    for epoch in range(num_epochs):
        loss = train_epoch(model, optimizer, train_loader, scheduler, device)
        epoch_losses.append(loss)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss:.6f}")
    
    # Step 6: Visualize and analyze
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('DDPM Training: Smooth Convergence (Unlike cGAN)')
    plt.grid(True)
    plt.show()
    
    # Step 7: Convergence analysis
    print("\nConvergence Analysis:")
    print(f"  Initial loss: {epoch_losses[0]:.6f}")
    print(f"  Final loss: {epoch_losses[-1]:.6f}")
    print(f"  Improvement: {(epoch_losses[0]-epoch_losses[-1])/epoch_losses[0]*100:.1f}%")
```

**Key insight:**
- Complete pipeline from data to visualization
- MSE loss should decrease smoothly
- Analysis compares with cGAN's volatile curves

**Hint:**
- Use device detection for portability
- Normalize MNIST to [-1, 1]
- lr=0.001 works well (higher than cGAN's 0.0002)
- Plot loss curve to verify smooth convergence

---

## Common Mistakes and How to Avoid Them

### Mistake 1: Shape Mismatches
```python
# Wrong
time_scale = self.time_mlp(time_embedding)  # (batch, channels)
h = h * time_scale  # h is (batch, channels, 28, 28) ← Broadcast fails!

# Correct
time_scale = time_scale[:, :, None, None]  # (batch, channels, 1, 1)
h = h * time_scale  # Now broadcasts correctly
```

### Mistake 2: Forgetting Device Movement
```python
# Wrong
model = SimpleUNet()
model.to('cuda')
x_0 = x_0  # Still on CPU!
x_t = model(x_0, t)  # Error: tensors on different devices

# Correct
x_0 = x_0.to(device)
x_t = model(x_0, t)  # Same device
```

### Mistake 3: Not Passing Time to Every Block
```python
# Wrong
h = self.down_res1(h)  # No time conditioning!

# Correct
h = self.down_res1(h, time_emb)  # Pass time_emb
```

### Mistake 4: Forgetting Model.train() Mode
```python
# Wrong
model.eval()  # ← Wrong mode!
loss = train_step(model, optimizer, x, scheduler, device)

# Correct
model.train()  # Enables dropout, batch norm updates, etc.
loss = train_step(model, optimizer, x, scheduler, device)
```

---

## Expected Outputs

### After Completing All TODOs:

**Console Output:**
```
Device: cuda (or mps/cpu)

Loading MNIST dataset...
Train samples: 60000
Val samples: 10000

Creating model...
U-Net Parameters: 1,234,567

Starting training...

Epoch 1/10 | Loss: 0.245123
Epoch 2/10 | Loss: 0.180456
Epoch 3/10 | Loss: 0.145789
Epoch 4/10 | Loss: 0.125634
Epoch 5/10 | Loss: 0.110234
Epoch 6/10 | Loss: 0.100123
Epoch 7/10 | Loss: 0.092456
Epoch 8/10 | Loss: 0.086789
Epoch 9/10 | Loss: 0.082134
Epoch 10/10 | Loss: 0.078901

Training Complete!
Initial Loss: 0.245123
Final Loss: 0.078901
Improvement: 67.8%
```

**Loss Curve Characteristics:**
-  Smooth, continuous decrease
-  No sudden jumps or oscillations
-  Consistent improvement per epoch
-  ~68% reduction in 10 epochs

**Comparison with cGAN:**
```
DDPM (This exercise):
  Loss = [0.245, 0.180, 0.146, 0.126, ...]  ← Smooth!

cGAN (Module 13):
  Loss = [0.5, 0.32, 0.61, 0.45, 0.38, ...]  ← Volatile!
```

---

## Debugging Guide

### Issue: Model Not Learning (Loss Stays High)

**Check:**
1. Is learning rate too high or too low?
   ```python
   optimizer = optim.Adam(model.parameters(), lr=0.001)  # Should be around here
   ```

2. Is batch size appropriate?
   ```python
   train_loader = DataLoader(..., batch_size=64)  # 64 is good
   ```

3. Is model actually being updated?
   ```python
   print(list(model.parameters())[0][0, :3])  # Should change each epoch
   ```

### Issue: CUDA Out of Memory

**Solution:**
- Reduce batch size: `batch_size=32` instead of 64
- Reduce model size: `base_channels=32` instead of 64
- Use gradient accumulation

### Issue: Loss NaN or Inf

**Causes:**
- Gradients too large (not clipped)
- Learning rate too high
- Numerical instability

**Solution:**
```python
# Already in code, but verify:
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

# Reduce learning rate:
optimizer = optim.Adam(model.parameters(), lr=0.0005)
```

### Issue: Generation Produces Noise

**Expected:** After 10 epochs, samples will be low quality (still partly noisy)
- This is normal! Would improve with 50+ epochs
- For better results, train longer


---

## Considerations

**Q: Why is my loss higher than expected?**
A: Normal variation. With only 10 epochs, final loss depends on exact implementation.

**Q: How long should training take?**
A: ~2-5 minutes for 10 epochs on GPU, ~10-20 minutes on CPU.

**Q: Can I use more epochs?**
A: Yes! 50+ epochs will give much better results.

**Q: What if I get shape mismatches?**
A: Check that time_embedding is reshaped correctly and passed to all ResBlocks.

---

## Summary

By completing these 8 TODOs, you will:
+ Understand noise schedules and forward diffusion
+ Implement time embeddings for conditioning
+ Build a complete U-Net architecture
+ Train with MSE loss (demonstrating stable convergence)
+ Analyze results and compare with adversarial approaches
+ Generate images from pure noise


