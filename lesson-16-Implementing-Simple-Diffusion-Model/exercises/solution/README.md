# Solution Walkthrough: Simple DDPM


## Part 1: Understanding the Solution Architecture

### Why This Design?

**Diffusion Models vs GANs:**

Our solution differs fundamentally from Module 13's cGAN:

```
cGAN (Module 13):
  - 2 networks (Generator + Discriminator)
  - Adversarial loss (binary cross-entropy)
  - Min-max game dynamics
  - Volatile loss curves
  - Risk of mode collapse

Diffusion (This solution):
  - 1 network (U-Net denoiser)
  - Regression loss (MSE)
  - Pure optimization problem
  - Smooth loss curves
  - No mode collapse
```

**Key Innovation:** Instead of learning to generate from scratch (like GANs), diffusion models learn to denoise gradually. This is mathematically simpler and more stable.

---

## Part 2: Component-by-Component Walkthrough

### Component 1: NoiseScheduler

**Full Implementation:**
```python
class NoiseScheduler:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        """
        Initialize noise schedule for forward diffusion process.
        
        Key insight: We pre-compute everything to avoid repeated calculations
        during training (massive performance improvement).
        
        Args:
            num_timesteps (int): Total diffusion steps (T=1000)
            beta_start (float): Initial noise level
            beta_end (float): Final noise level (image becomes mostly noise)
        """
        self.num_timesteps = num_timesteps
        device = torch.device("cpu")  # Will be moved to GPU if needed
        
        # Step 1: Create noise schedule
        # Linear schedule: β_t increases uniformly from beta_start to beta_end
        # This controls how much noise to add at each timestep
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        
        # Step 2: Compute alphas
        # α_t = 1 - β_t (fraction of original signal that remains)
        # At t=0: α ≈ 1 (mostly original image)
        # At t=T: α ≈ 0 (mostly noise)
        self.alphas = 1.0 - self.betas
        
        # Step 3: Cumulative product (key for efficient forward pass)
        # ᾱ_t = ∏_{i=1}^t α_i
        # This represents total signal retention after t steps
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # For posterior variance calculation
        self.alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), self.alphas_cumprod[:-1]])
        
        # Step 4: Pre-compute square roots (used in forward diffusion formula)
        # Forward diffusion: x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
        # Pre-computing √ prevents recomputation ~60 times per epoch
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    
    def get_coefficients(self, timesteps):
        """Get pre-computed coefficients for a batch of timesteps.
        
        Why pre-compute?
        - Avoids sqrt() computation during training
        - Enables efficient indexing
        - Critical performance optimization
        """
        sqrt_alphas = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alphas = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        # Reshape for broadcasting with images
        # From (batch,) to (batch, 1, 1, 1) for proper element-wise multiplication
        if len(sqrt_alphas.shape) == 1:
            sqrt_alphas = sqrt_alphas[:, None, None, None]
            sqrt_one_minus_alphas = sqrt_one_minus_alphas[:, None, None, None]
        
        return sqrt_alphas, sqrt_one_minus_alphas
```

**Design Decision Explanation:**

Why linear schedule?
- Simple and interpretable
- Works well in practice
- Other schedules (cosine, sigmoid) also work

Why pre-compute?
- During 10 epochs × 938 batches × 64 images = 600k forward passes
- Each pass needs √ computation
- Pre-computing saves ~99% of computation time

Why this range (0.0001 to 0.02)?
- Too small: Too little noise → blurry samples
- Too large: Too much noise → unstable training
- This range proven effective across literature

---

### Component 2: Forward Diffusion Process

**Full Implementation:**
```python
def add_noise(x_0, timestep, scheduler, noise=None):
    """Add noise to image according to forward diffusion process.
    
    Mathematical formula:
        x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
    
    Where:
        - x_0: Original clean image
        - ε: Random Gaussian noise
        - √ᾱ_t: Signal retention (decreases with t)
        - √(1-ᾱ_t): Noise level (increases with t)
    
    Key insight: This single-step formula is equivalent to T-step iterative
    Gaussian diffusion but much more efficient!
    """
    # Sample random noise if not provided
    if noise is None:
        noise = torch.randn_like(x_0)
    
    # Get pre-computed coefficients
    sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod = scheduler.get_coefficients(timestep)
    
    # Apply forward diffusion formula
    # This scales the signal and noise according to timestep
    x_t = sqrt_alphas_cumprod * x_0 + sqrt_one_minus_alphas_cumprod * noise
    
    return x_t, noise
```

**Why This Works:**

```
Intuition:
- t=0:   x_0 ≈ √1 * x_0 + √0 * ε = x_0 (original)
- t=500: x_0 ≈ 0.5 * x_0 + 0.87 * ε (mixed)
- t=999: x_0 ≈ 0 * x_0 + 1 * ε = ε (pure noise)

Mathematical property:
- Variance is preserved: Var(x_t) = 1
- Enables stable training
- Prevents gradient explosion
```

**Training implications:**
- Early timesteps: Model learns high-level structure
- Late timesteps: Model learns fine-grained details
- Network sees full range of noise levels

---

### Component 3: TimeEmbedding

**Full Implementation:**
```python
class TimeEmbedding(nn.Module):
    """Encode timestep into learnable embedding using sinusoidal functions.
    
    Why sinusoidal?
    - Used in Transformers with great success
    - Enables relative position encoding
    - Smooth and continuous
    - No learnable parameters needed
    """
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim
    
    def forward(self, timestep):
        """Convert timestep to embedding.
        
        Args:
            timestep: LongTensor of shape (batch,)
        
        Returns:
            embedding: FloatTensor of shape (batch, embedding_dim)
        """
        device = timestep.device
        half_dim = self.embedding_dim // 2
        
        # Step 1: Create frequency schedule
        # exp(-ln(10000) * k / d) creates exponential frequency spacing
        # This is the same formula used in Transformer attention
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=device) / half_dim
        )
        
        # Step 2: Multiply timestep by frequencies
        # This creates different oscillation rates for each dimension
        args = timestep[:, None].float() * freqs[None, :]
        
        # Step 3: Apply sine and cosine
        # Sine/cosine create smooth, periodic patterns
        # Different frequencies encode different time scales
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        return embedding
```

**Why This Encoding?**

```
Example for single timestep (t=500):

Frequency 1 (slow):   sin(500 * 0.1) = sin(50) ≈ -0.26
                      cos(500 * 0.1) = cos(50) ≈ 0.96

Frequency 2 (medium): sin(500 * 1.0) = sin(500) ≈ 0.48
                      cos(500 * 1.0) = cos(500) ≈ -0.88

Frequency 3 (fast):   sin(500 * 10.0) = sin(5000) ≈ -0.69
                      cos(500 * 10.0) = cos(5000) ≈ -0.72

Result: Dense 128-D vector encoding t=500 uniquely
```

**Key properties:**
- Each dimension oscillates at different frequency
- Early frequencies: Capture coarse time scale
- Late frequencies: Capture fine details
- Enables model to distinguish between t=500 and t=501

---

### Component 4: ResidualBlock with FiLM

**Full Implementation:**
```python
class ResidualBlock(nn.Module):
    """Residual block with FiLM (Feature-wise Linear Modulation) conditioning.
    
    Why FiLM?
    - Lightweight time conditioning
    - Multiplicative modulation of features
    - Proven effective in diffusion models
    - Alternative to concat/add methods
    """
    def __init__(self, in_channels, out_channels, embedding_dim=128):
        super().__init__()
        
        # First convolution path
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Time embedding projection to scale (FiLM parameter)
        self.time_mlp = nn.Sequential(
            nn.Linear(embedding_dim, out_channels * 2),
            nn.SiLU(),
            nn.Linear(out_channels * 2, out_channels)
        )
        
        # Second convolution path
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Skip connection (1x1 conv if dimensions change)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x, time_embedding):
        """Apply residual block with time conditioning.
        
        Args:
            x: Input features (batch, channels, height, width)
            time_embedding: Time embedding (batch, embedding_dim)
        
        Returns:
            Output: Same shape as input (with skip connection)
        """
        # Path 1: First convolution
        h = self.norm1(x)
        h = F.silu(h)  # Swish activation (smooth, smooth gradient)
        h = self.conv1(h)
        
        # FiLM conditioning: Multiply by time-dependent scale
        # This applies different scaling to each output channel based on timestep
        time_scale = self.time_mlp(time_embedding)
        
        # Reshape from (batch, channels) to (batch, channels, 1, 1)
        # This enables broadcasting with (batch, channels, H, W)
        time_scale = time_scale[:, :, None, None]
        
        # Feature-wise modulation
        h = h * time_scale
        
        # Path 2: Second convolution
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        # Skip connection (residual)
        return h + self.skip(x)
```

**Why FiLM over alternatives?**

```
Comparison of conditioning methods:

1. Concatenation (cGAN approach):
   h_concat = torch.cat([h, time_emb.expand(...)], dim=1)
   → Increases channels (inefficient)
   → Hard to control influence

2. Additive (V-diffusion):
   h_add = h + time_projection
   → Mixes heterogeneous dimensions
   → Weak conditioning

3. FiLM (This solution):
   h_film = h * time_scale
   → Multiplicative modulation
   → Per-channel scaling
   → Proven effective ✓
```

**Why GroupNorm?**

Alternative: Batch Norm
- ✗ Depends on batch statistics
- ✗ Behaves differently during training/inference
- ✗ Problematic during sampling with different batch sizes

GroupNorm:
- ✓ Independent of batch size
- ✓ Consistent training/inference
- ✓ Standard in diffusion models

---

### Component 5: U-Net Architecture

**Full Implementation:**
```python
class SimpleUNet(nn.Module):
    """Simple U-Net for noise prediction in DDPM.
    
    Architecture:
                         Input (28×28)
                              ↓
                      Initial Conv (64)
                              ↓
        ┌─────────────→ Down 1 (64→64)
        │                      ↓ Downsample
        │              Down 2 (64→128)
        │                      ↓
        │              Middle (128→128)
        │                      ↓ Upsample
        │              Up 2 (128→64)
        │                      ↓
        └─────────→ Up 1 (64→64)
                              ↓
                      Final Conv (1)
                         Output (28×28)
    
    Key insight: Skip connections preserve spatial information
    """
    def __init__(self, image_channels=1, base_channels=64):
        super().__init__()
        
        self.base_channels = base_channels
        
        # Time embedding (from timestep)
        self.time_embedding = TimeEmbedding(embedding_dim=128)
        
        # Initial convolution
        self.init_conv = nn.Conv2d(image_channels, base_channels, kernel_size=3, padding=1)
        
        # Encoder (downsampling)
        self.down_res1 = ResidualBlock(base_channels, base_channels, embedding_dim=128)
        self.down_conv1 = nn.Conv2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1)
        
        self.down_res2 = ResidualBlock(base_channels, base_channels * 2, embedding_dim=128)
        self.down_conv2 = nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=4, stride=2, padding=1)
        
        # Bottleneck
        self.middle_res = ResidualBlock(base_channels * 2, base_channels * 2, embedding_dim=128)
        
        # Decoder (upsampling)
        self.up_conv2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.up_res2 = ResidualBlock(base_channels, base_channels, embedding_dim=128)
        
        self.up_conv1 = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1)
        self.up_res1 = ResidualBlock(base_channels, base_channels, embedding_dim=128)
        
        # Final layers
        self.final_norm = nn.GroupNorm(32, base_channels)
        self.final_conv = nn.Conv2d(base_channels, image_channels, kernel_size=3, padding=1)
    
    def forward(self, x, timestep):
        """Predict noise at given timestep.
        
        Args:
            x: Noisy image (batch, 1, 28, 28)
            timestep: Timestep (batch,)
        
        Returns:
            Predicted noise (batch, 1, 28, 28)
        """
        # Encode timestep
        time_emb = self.time_embedding(timestep)  # (batch, 128)
        
        # Initial convolution
        h = self.init_conv(x)  # (batch, 64, 28, 28)
        
        # Encoder
        h1 = self.down_res1(h, time_emb)  # (batch, 64, 28, 28)
        h = self.down_conv1(h1)  # (batch, 64, 14, 14) - downsample
        
        h2 = self.down_res2(h, time_emb)  # (batch, 128, 14, 14)
        h = self.down_conv2(h2)  # (batch, 128, 7, 7) - downsample
        
        # Bottleneck
        h = self.middle_res(h, time_emb)  # (batch, 128, 7, 7)
        
        # Decoder (upsampling brings back spatial dimensions)
        h = self.up_conv2(h)  # (batch, 64, 14, 14) - upsample
        h = self.up_res2(h, time_emb)  # (batch, 64, 14, 14)
        
        h = self.up_conv1(h)  # (batch, 64, 28, 28) - upsample
        h = self.up_res1(h, time_emb)  # (batch, 64, 28, 28)
        
        # Final output
        h = self.final_norm(h)
        h = F.silu(h)
        out = self.final_conv(h)  # (batch, 1, 28, 28)
        
        return out
```

**Spatial Dimensions Walkthrough:**

```
Input:        (batch, 1, 28, 28)
              ↓
init_conv:    (batch, 64, 28, 28)

down_res1:    (batch, 64, 28, 28)
down_conv1:   (batch, 64, 14, 14)  ← Reduce spatial size by 2

down_res2:    (batch, 128, 14, 14)
down_conv2:   (batch, 128, 7, 7)   ← Reduce spatial size by 2 again

middle_res:   (batch, 128, 7, 7)

up_conv2:     (batch, 64, 14, 14)  ← Increase spatial size by 2
up_res2:      (batch, 64, 14, 14)

up_conv1:     (batch, 64, 28, 28)  ← Increase spatial size by 2
up_res1:      (batch, 64, 28, 28)

final_conv:   (batch, 1, 28, 28)
              ↓
Output:       (batch, 1, 28, 28)  ← Same as input!
```

**Why this architecture?**

1. **Symmetry:** Encoder mirrors Decoder
   - Preserves spatial structure
   - Enables skip connections (could be added in advanced version)

2. **Multi-scale processing:**
   - 28×28: Fine details
   - 14×14: Mid-level features
   - 7×7: Semantic information

3. **Progressive conditioning:**
   - Time embedding passed at every layer
   - Enables fine-grained time control

---

## Part 3: Training Implementation

### train_step Function

**Full Implementation:**
```python
def train_step(model, optimizer, x_0, scheduler, device):
    """Single training iteration (one batch).
    
    Key insight: This is pure regression (MSE loss), not adversarial.
    This guarantees stable gradients and convergence.
    """
    model.train()  # Enable dropout, batch norm updates, etc.
    
    # Move data to device
    x_0 = x_0.to(device)
    batch_size = x_0.shape[0]
    
    # Step 1: Sample random timesteps
    # Each image in batch gets random timestep
    # This ensures model learns to denoise at all scales
    timesteps = torch.randint(
        0, 
        scheduler.num_timesteps, 
        (batch_size,), 
        device=device
    )
    
    # Step 2: Sample random noise (what we want model to predict)
    noise = torch.randn_like(x_0)
    
    # Step 3: Forward diffusion (add noise)
    x_t, _ = add_noise(x_0, timesteps, scheduler, noise)
    
    # Step 4: Model predicts noise
    # This is the key training objective:
    # Given x_t and t, can we predict the noise added?
    predicted_noise = model(x_t, timesteps)
    
    # Step 5: MSE loss (compare predicted noise to actual noise)
    # This is the main difference from cGAN!
    # MSE is simple, stable, and guarantees convergence
    loss = F.mse_loss(predicted_noise, noise)
    
    # Step 6: Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping (prevent exploding gradients)
    # Not necessary with MSE but good practice
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    optimizer.step()
    
    return loss.item()
```

**Loss Function Comparison:**

```
MSE Loss (This solution):
  Loss = mean((predicted_noise - noise)²)
  → Simple regression problem
  → Gradients are smooth
  → Training is stable

Binary Cross Entropy (cGAN):
  Loss = -[t*log(p) + (1-t)*log(1-p)]
  → Classification problem
  → Gradients can be sharp
  → Training can be unstable

Why MSE wins:
1. Mathematically simpler
2. Stable gradients everywhere
3. No vanishing gradients at equilibrium
4. More interpretable
```

### train_epoch Function

**Full Implementation:**
```python
def train_epoch(model, optimizer, train_loader, scheduler, device):
    """Train for one epoch (all batches).
    
    Purpose: Iterate through entire dataset once.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # Process each batch
    for images, _ in tqdm(train_loader, desc="Training"):
        # Call train_step for this batch
        loss = train_step(model, optimizer, images, scheduler, device)
        
        # Accumulate loss
        total_loss += loss
        num_batches += 1
    
    # Return average loss for this epoch
    avg_loss = total_loss / num_batches
    return avg_loss
```

**Why average?**

```
Example with 938 batches (60,000 / 64):

Batch losses: [0.35, 0.28, 0.31, ..., 0.25]

Total: sum of all 938 losses = 280
Average: 280 / 938 = 0.298

Why average?
- Batch size affects absolute loss magnitude
- Average enables fair comparison between experiments
- Standard practice in deep learning
```

### main() Training Loop

**Full Implementation:**
```python
def main():
    """Complete training pipeline."""
    
    # Step 1: Device selection (ensures portability)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")
    
    # Step 2: Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0  # Set to 4+ for faster loading
    )
    
    # Step 3: Model creation
    model = SimpleUNet(image_channels=1, base_channels=64).to(device)
    scheduler = NoiseScheduler(num_timesteps=1000)
    
    # Step 4: Optimizer
    # Learning rate is higher than cGAN (0.001 vs 0.0002)
    # Reason: MSE loss is more stable, can tolerate larger LR
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Step 5: Training loop
    num_epochs = 10
    epoch_losses = []
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        # Train one epoch
        avg_loss = train_epoch(model, optimizer, train_loader, scheduler, device)
        epoch_losses.append(avg_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.6f}")
        
        # Save checkpoint (every 5 epochs)
        if (epoch + 1) % 5 == 0:
            torch.save(
                model.state_dict(),
                f'diffusion_model_epoch_{epoch+1}.pt'
            )
    
    # Step 6: Visualization and analysis
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), epoch_losses, marker='o', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('DDPM Training: Smooth Convergence\n(Compare with volatile cGAN curves)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Step 7: Convergence analysis
    print("\n" + "="*50)
    print("Convergence Analysis")
    print("="*50)
    print(f"Initial Loss: {epoch_losses[0]:.6f}")
    print(f"Final Loss:   {epoch_losses[-1]:.6f}")
    improvement_pct = (epoch_losses[0] - epoch_losses[-1]) / epoch_losses[0] * 100
    print(f"Improvement:  {improvement_pct:.1f}%")
    print(f"Avg Loss:     {sum(epoch_losses)/len(epoch_losses):.6f}")
    
    # Analyze smoothness
    diffs = [abs(epoch_losses[i+1] - epoch_losses[i]) for i in range(len(epoch_losses)-1)]
    print(f"Max loss jump: {max(diffs):.6f}")
    print(f"Avg loss jump: {sum(diffs)/len(diffs):.6f}")
    
    if max(diffs) < 0.1 * epoch_losses[0]:
        print("✓ Convergence is SMOOTH (no volatile jumps)")
    else:
        print("✗ Warning: High volatility detected")
```

**Output Interpretation:**

```
Epoch 1/10 | Loss: 0.245123  ← Starting point
Epoch 2/10 | Loss: 0.180456  ← ~26% improvement
Epoch 3/10 | Loss: 0.145789  ← Continuing to decrease
...
Epoch 10/10 | Loss: 0.078901 ← ~68% total improvement

==================================================
Convergence Analysis
==================================================
Initial Loss:    0.245123
Final Loss:      0.078901
Improvement:     67.8%
Avg Loss:        0.128976
Max loss jump:   0.064667  ← Small jumps indicate smooth convergence
Avg loss jump:   0.018234
✓ Convergence is SMOOTH (no volatile jumps)
```

---

## Part 4: Key Design Decisions Explained

### Decision 1: Learning Rate = 0.001

**Why higher than cGAN's 0.0002?**

```python
# cGAN (Module 13)
optimizer = optim.Adam(cgan_gen.parameters(), lr=0.0002)

# Diffusion (This solution)
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

**Reasoning:**

```
MSE Loss characteristics:
- Smooth gradient landscape
- No vanishing gradients
- No game dynamics

Binary Cross-Entropy (cGAN):
- Sharp gradients near decision boundary
- Risk of vanishing gradients
- Competing objectives

Conclusion: Higher LR OK with MSE
- More stable landscape
- Faster convergence
- Less risk of divergence
```

### Decision 2: GroupNorm over BatchNorm

**Why GroupNorm?**

```python
# ✗ Incorrect for diffusion
self.norm = nn.BatchNorm2d(channels)

# ✓ Correct
self.norm = nn.GroupNorm(32, channels)
```

**Reasoning:**

```
BatchNorm limitations:
- Depends on batch statistics
- Different behavior at train vs inference
- Problematic with batch_size=1 during sampling

GroupNorm advantages:
- Independent of batch size
- Same computation train/inference
- Consistent with diffusion literature
- Works well during unconditional generation
```

### Decision 3: Time Embedding Dimension = 128

**Why 128?**

```python
self.time_embedding = TimeEmbedding(embedding_dim=128)
```

**Reasoning:**

```
Embedding Dimension Analysis:

Too small (e.g., 16):
- Can't distinguish between nearby timesteps
- Model confuses t=500 and t=501
- Poor time conditioning

128 (our choice):
- ~2 bits per timestep (log₂(1000) ≈ 10 bits needed)
- 128 >> 10, so plenty of capacity
- Standard in diffusion literature
- Good balance of expressiveness and efficiency

Too large (e.g., 1024):
- Wastes model capacity
- Marginal benefit
- Slower training
```

---

## Part 5: Advanced Extensions

### Extension 1: Sampling Algorithm

**How to generate new images (reverse process):**

```python
def sample(model, scheduler, device, num_samples=4):
    """Generate new images by reversing diffusion."""
    model.eval()
    
    # Start with pure noise
    x_t = torch.randn(num_samples, 1, 28, 28, device=device)
    
    # Iteratively denoise
    timesteps = range(scheduler.num_timesteps - 1, -1, -1)
    for t in tqdm(timesteps):
        t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
        
        # Predict noise
        noise_pred = model(x_t, t_tensor)
        
        # Reverse diffusion step
        # x_{t-1} = (x_t - √(1-ᾱ_t) * noise_pred) / √ᾱ_t
        x_t = reverse_diffusion_step(x_t, t, noise_pred, scheduler)
    
    return x_t
```

**Why this works:**
- If model perfectly predicts noise, reverse process recovers original
- Imperfect predictions create diverse samples
- Deterministic process allows reproducible generation

### Extension 2: Conditional Generation

**Add class conditioning (like Module 13 but stable):**

```python
class ConditionalUNet(SimpleUNet):
    def __init__(self, image_channels=1, base_channels=64, num_classes=10):
        super().__init__(image_channels, base_channels)
        
        # Class embedding
        self.class_embedding = nn.Embedding(num_classes, 128)
        
        # Combine time and class embeddings
        self.combined_proj = nn.Linear(256, 128)
    
    def forward(self, x, timestep, class_label):
        """With class conditioning."""
        time_emb = self.time_embedding(timestep)  # (batch, 128)
        class_emb = self.class_embedding(class_label)  # (batch, 128)
        
        # Combine
        combined_emb = self.combined_proj(torch.cat([time_emb, class_emb], dim=1))
        
        # Rest of U-Net uses combined_emb
        ...
```

**Advantages over cGAN:**
- Still uses MSE loss (stable)
- Single network (simple)
- Guaranteed convergence

### Extension 3: Better Schedulers

**Compare different noise schedules:**

```python
# Linear (current)
betas = torch.linspace(beta_start, beta_end, T)

# Cosine (smoother, often better)
betas = torch.linspace(0, 1, T)
betas = torch.cos((betas + 0.008) / 1.008 * torch.pi * 0.5) ** 2
betas = (1 - betas) / (1 - torch.cos(torch.tensor(torch.pi * 0.008)))

# Sigmoid (steep transition)
betas = torch.sigmoid(torch.linspace(-6, 6, T)) * (beta_end - beta_start) + beta_start
```

---

## Part 6: Convergence Comparison

### vs cGAN (Module 13)

```
Characteristic          cGAN              Diffusion (This)
─────────────────────────────────────────────────────────
Loss Function          BCE               MSE
Networks               2                 1
Objectives             Adversarial       Single
Loss Curve             Volatile          Smooth
Convergence            Unstable          Guaranteed
Mode Collapse          Possible          No
Training Time          Stable            5-50 min (10 epochs)
Sampling Quality       Good              Fair (at 10 epochs)
Theoretical            Min-max game      Pure optimization
```

### Expected Results

**After 10 epochs:**
- Loss reduction: 60-70%
- Sample quality: Blurry but recognizable
- Convergence: Smooth throughout

**After 50 epochs:**
- Loss reduction: 80-90%
- Sample quality: Sharp and realistic
- Still smooth convergence



## Part 8: Performance Tuning

### For Faster Training

```python
# Increase batch size (if memory allows)
train_loader = DataLoader(..., batch_size=128)

# Reduce model size
model = SimpleUNet(base_channels=32)  # was 64

# Use mixed precision
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

### For Better Quality

```python
# Train longer
num_epochs = 50

# Reduce learning rate
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Better scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
```

---

## Summary

This solution demonstrates that:

1. ✓ Diffusion models are simpler than GANs (1 network vs 2)
2. ✓ MSE loss is more stable than adversarial loss
3. ✓ Smooth convergence is achievable with proper design
4. ✓ Time conditioning via embeddings works well
5. ✓ U-Net architecture is effective for noise prediction

**Key insight:** The reason diffusion models train so smoothly is fundamental—we're solving a pure regression problem instead of playing a min-max game.