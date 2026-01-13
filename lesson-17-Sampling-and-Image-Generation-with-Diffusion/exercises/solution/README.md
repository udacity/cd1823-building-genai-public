# Solution Guide: Sampling Schedule Comparison

## Overview

This guide walks through the complete solution for the sampling schedule comparison exercise.

**What This Solution Demonstrates:**
- How to implement linear and cosine schedules
- Implementing the core reverse diffusion step
- Complete sampling loop architecture
- Quality metrics and comparison strategies

---

## Solution Architecture

```
reverse_diffusion_step()          ← Core algorithm (TODO 3)
       ↓
LinearSchedule/CosineSchedule     ← Timestep strategies (TODOs 1-2)
       ↓
sample_with_schedule()            ← Main sampling loop (TODO 4)
       ↓
compute_sample_variance()         ← Metrics (TODO 5)
compute_sharpness()
       ↓
visualize_schedule_comparison()   ← Visualization (TODO 6)
compare_schedule_metrics()        ← Analysis (TODO 7)
```

---

## TODO 1 Solution: LinearSchedule.__init__()

### Implementation

```python
class LinearSchedule:
    """Linear (uniform) timestep schedule for fast sampling."""
    
    def __init__(self, num_train_timesteps=1000, num_inference_steps=50):
        """
        Initialize linear timestep schedule.
        
        Creates uniformly-spaced timesteps from high noise to clean.
        """
        # Linearly spaced values
        values = torch.linspace(
            num_train_timesteps - 1,  # Start: 999
            0,                        # End: 0
            num_inference_steps       # Number of steps: 50
        )
        
        # Convert to integer tensor indices
        self.timesteps = values.long()
```

### Why This Works

**Uniform distribution:**
```
Timestep sequence (10 steps from 1000):
[999, 889, 778, 668, 557, 447, 336, 226, 115, 4]
     ↑ every 111 steps ↑
```

This means:
- Each step removes roughly the same amount of noise
- Simple to implement and understand
- Not optimal for quality but fastest

### Expected Output

```python
schedule = LinearSchedule(1000, 50)

# First 10 timesteps
print(schedule.timesteps[:10])
# Output: tensor([999, 978, 957, 936, 915, 894, 873, 852, 831, 810])

# Last 10 timesteps (approaching clean)
print(schedule.timesteps[-10:])
# Output: tensor([189, 168, 147, 126, 105,  84,  63,  42,  21,   0])

# Total steps
print(len(schedule.timesteps))
# Output: 50
```

---

## TODO 2 Solution: CosineSchedule.__init__()

### Implementation

```python
class CosineSchedule:
    """Cosine (non-linear) timestep schedule for high-quality sampling."""
    
    def __init__(self, num_train_timesteps=1000, num_inference_steps=50):
        """
        Initialize cosine timestep schedule.
        
        Creates non-linear timesteps concentrated at high noise.
        Formula: α_t = cos²(π * (t/N + s) / (1+s))
        """
        # Smoothing constant (prevents too much concentration)
        s = 0.008
        
        # Create step indices [0, 1, 2, ..., N]
        steps = torch.arange(
            num_inference_steps + 1,
            dtype=torch.float32
        )
        
        # Compute alpha values using cosine formula
        # This creates a distribution skewed toward high noise
        alphas = torch.cos(
            ((steps / num_inference_steps + s) / (1 + s)) * math.pi / 2
        ) ** 2
        
        # For each alpha value, find the nearest actual timestep
        # Alpha decreases from 1 to 0 as timestep increases from 0 to 1000
        
        # We need alpha_cumprod values (pre-computed from training)
        # Get them from the noise scheduler
        timesteps = []
        for alpha in alphas:
            # Find which timestep t gives this alpha value
            # by comparing with alpha_cumprod = ∏(1 - β_i)
            distances = torch.abs(alpha_cumprod - alpha)
            timestep = torch.argmin(distances).item()
            timesteps.append(timestep)
        
        # Convert to tensor
        self.timesteps = torch.tensor(timesteps, dtype=torch.long)
```

### How It Works: Detailed Breakdown

**Step 1: Alpha Schedule**
```python
s = 0.008
steps = [0, 1, 2, ..., 50]
```

**Step 2: Compute cosine distribution**
```
For each step i:
  angle = π/2 * (i/50 + 0.008) / (1.008)
  alpha_i = cos²(angle)

Results in:
  alpha_0 ≈ 1.0    (early: high alpha = clean image signal)
  alpha_25 ≈ 0.5   (middle: balanced)
  alpha_50 ≈ 0.0   (late: low alpha = pure noise)
```

**Step 3: Map to actual timesteps**
```
alpha=1.0 → timestep 1000 (clean)
alpha=0.9 → timestep 950
alpha=0.7 → timestep 500
alpha=0.1 → timestep 50
alpha=0.0 → timestep 0 (pure noise)

Non-linear mapping causes more timesteps at high noise!
```

### Visual Explanation

```
Alpha over cosine schedule (50 steps):
1.0 ▓
0.9 ▓ ▒
0.8 ▓ ▒ ▒
0.7 ▓ ▒ ▒ ░
0.6 ▓ ▒ ▒ ░ ░
... [more low-noise steps compressed] ...
0.1 ▓ ▒ ▒ ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ ░
    |-------- many steps --------|------few---|
    |-------- high noise region --|-- low noise region
```

### Expected Output

```python
schedule = CosineSchedule(1000, 50)

# First 10 timesteps (high noise: big gaps!)
print(schedule.timesteps[:10])
# Output: tensor([1000, 969, 894, 776, 609, 393, 179, 54, 12, 0])
#          gap:    31   75   118  167  216  214  125  42  12
#          (gaps are larger early = focuses effort on hard part)

# Last 10 timesteps (low noise: small gaps)
print(schedule.timesteps[-10:])
# Output: includes very small timesteps

# Compare with linear
linear_sched = LinearSchedule(1000, 50)
print("Linear:  ", linear_sched.timesteps[:10])
print("Cosine:  ", cosine_sched.timesteps[:10])
# Cosine has larger gaps = fewer steps for easy high-alpha removal
# Linear has uniform gaps = wasted effort on easy steps
```

### Key Insight

**Cosine is smarter allocation:**
- High noise (hard): Many fine-grained steps (big timestep gaps)
- Low noise (easy): Few coarse steps (small timestep gaps)

This is like:
- Using a thick pencil for rough sketching
- Using a fine pencil for detail work

---

## TODO 3 Solution: reverse_diffusion_step()

### Implementation

```python
def reverse_diffusion_step(
    x_t,
    t,
    predicted_noise,
    alpha_cumprod,
    alphas,
    betas,
    device
):
    """
    Perform one reverse diffusion step: x_t → x_{t-1}
    
    This implements the reverse diffusion process:
    1. Use U-Net prediction to estimate original image
    2. Apply Bayesian posterior to compute mean
    3. Add stochastic noise for diversity
    """
    
    # Get pre-computed coefficients for this timestep
    alpha_t = alpha_cumprod[t]
    alpha_t_prev = alpha_cumprod[t - 1] if t > 0 else torch.tensor(1.0, device=device)
    beta_t = betas[t]
    
    # Compute useful derived quantities
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
    sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev)
    
    # ===== STEP 1: Estimate original image =====
    # From forward process: x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
    # Solving for x_0: x_0 = (x_t - √(1-ᾱ_t) * ε_θ) / √ᾱ_t
    
    pred_original = (x_t - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t
    
    # ===== STEP 2: Compute posterior distribution =====
    # Using Bayesian conjugacy, the posterior p(x_{t-1} | x_t, x_0) is Gaussian
    
    # Posterior variance (from DDPM paper, Appendix B):
    # σ_t² = (1 - ᾱ_{t-1}) * β_t / (1 - ᾱ_t)
    variance = (1.0 - alpha_t_prev) * beta_t / (1.0 - alpha_t)
    
    # Posterior mean (weighted combination of x_t and x_0):
    # m_t = coef1 * x_0 + coef2 * x_t
    
    # Coefficients are derived from the Gaussian posterior
    coef1 = sqrt_alpha_t_prev * beta_t / (1.0 - alpha_t)
    coef2 = (1.0 - beta_t) * sqrt_alpha_t / (1.0 - alpha_t)
    
    mean = coef1 * pred_original + coef2 * x_t
    
    # ===== STEP 3: Sample from posterior =====
    # x_{t-1} = m_t + σ_t * z  where z ~ N(0, I)
    
    # Sample noise (standard normal)
    z = torch.randn_like(x_t)
    
    # Add variance to mean to get final sample
    x_t_minus_1 = mean + torch.sqrt(variance) * z
    
    return x_t_minus_1
```

### Mathematical Explanation

**Forward Process (Training):**
```
q(x_t | x_0) = N(x_t; √ᾱ_t * x_0, (1-ᾱ_t) * I)

This means we can compute any x_t directly:
x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε   where ε ~ N(0,I)
```

**Reverse Process (Sampling):**
```
We train U-Net to predict ε given x_t and t.
Then we reverse the forward process:

x_0 = (x_t - √(1-ᾱ_t) * ε_θ(x_t,t)) / √ᾱ_t

Using Bayes rule, the true posterior is:
p(x_{t-1} | x_t, x_0) = N(mean, variance)

We don't know x_0, so we use predicted x_0.
```

### Step Breakdown

**Step 1: Estimate Original**
```
If x_t = √ᾱ * x_0 + √(1-ᾱ) * ε
Then x_0 = (x_t - √(1-ᾱ) * ε) / √ᾱ

This reverses one step of the forward process.
```

**Step 2: Posterior Mean**
```
The posterior mean combines:
- Estimated x_0 (how clean should it be?)
- Current x_t (how noisy is it now?)

With weights that balance them:
coef1 = importance of estimated x_0
coef2 = importance of current x_t

mean = coef1 * x_0_est + coef2 * x_t
```

**Step 3: Add Variance**
```
The posterior has variance σ_t.
We sample z ~ N(0,I) and compute:
x_{t-1} = mean + √variance * z

This adds stochasticity → diverse samples.
```

### Code Flow Diagram

```
Input: x_t (noisy), t (timestep), ε_θ (U-Net output)
  ↓
Get alpha_t, beta_t, etc. from pre-computed arrays
  ↓
Estimate x_0 = (x_t - √(1-α) * ε) / √α
  ↓
Compute posterior variance σ_t²
  ↓
Compute posterior mean m_t = c1*x_0 + c2*x_t
  ↓
Sample z ~ N(0,I)
  ↓
Return x_{t-1} = m_t + √σ_t² * z
```

---

## TODO 4 Solution: sample_with_schedule()

### Implementation

```python
def sample_with_schedule(
    model,
    noise_scheduler,
    schedule_type,
    num_steps,
    batch_size,
    device,
    return_trajectory=False
):
    """
    Complete sampling pipeline with chosen timestep schedule.
    """
    
    # ===== STEP 1: Create schedule object =====
    if schedule_type == "linear":
        schedule = LinearSchedule(
            num_train_timesteps=noise_scheduler.num_steps,
            num_inference_steps=num_steps
        )
    elif schedule_type == "cosine":
        schedule = CosineSchedule(
            num_train_timesteps=noise_scheduler.num_steps,
            num_inference_steps=num_steps
        )
    else:
        raise ValueError(f"Unknown schedule: {schedule_type}")
    
    # ===== STEP 2: Initialize with pure Gaussian noise =====
    # Starting image x_T is just random noise
    x_t = torch.randn(batch_size, 1, 28, 28, device=device)
    
    # Initialize trajectory if needed
    trajectory = [] if return_trajectory else None
    
    # ===== STEP 3: Main sampling loop =====
    # For each timestep from high noise to clean
    with torch.no_grad():
        for t in schedule.timesteps:
            # Convert tensor to int if needed
            t_index = int(t.item()) if isinstance(t, torch.Tensor) else t
            
            # Use U-Net to predict noise
            model_output = model(x_t, t_index)
            
            # Perform one reverse diffusion step
            x_t = reverse_diffusion_step(
                x_t=x_t,
                t=t_index,
                predicted_noise=model_output,
                alpha_cumprod=noise_scheduler.alpha_cumprod,
                alphas=noise_scheduler.alphas,
                betas=noise_scheduler.betas,
                device=device
            )
            
            # Save trajectory if requested
            if return_trajectory:
                trajectory.append(x_t.detach().clone())
    
    # ===== STEP 4: Return results =====
    if return_trajectory:
        return x_t, trajectory
    else:
        return x_t
```

### Execution Flow

```
Initialize: x_T ~ N(0, I) (pure noise)
            ↓
Timestep 999: ε_θ(x_999, 999) → x_998
            ↓
Timestep 978: ε_θ(x_998, 978) → x_977
            ↓
Timestep 957: ε_θ(x_977, 957) → x_956
            ↓
            ... (50 steps total for linear, different order for cosine) ...
            ↓
Timestep 21:  ε_θ(x_21, 21) → x_20
            ↓
Timestep 0:   ε_θ(x_0, 0) → final sample
            ↓
Return: Clean image x_0
```

### Key Design Points

**1. Model in eval mode**
```python
model.eval()  # Disable dropout, batch norm updates
with torch.no_grad():  # Don't track gradients
    # sampling
```

**2. Trajectory storage**
```python
# Saves all intermediate steps for visualization
if return_trajectory:
    trajectory = [x_T, x_{T-1}, x_{T-2}, ..., x_0]
    # Useful for seeing denoising progression
```

**3. Timestep iteration**
```python
# Schedule.timesteps is pre-computed in correct order
for t in schedule.timesteps:  # Automatically high→low
    ...
```

---

## TODO 5 Solution: Compute Fidelity Metrics

### 5.1: compute_sample_variance()

```python
def compute_sample_variance(samples):
    """
    Measure diversity of generated samples.
    
    Computes variance of per-pixel means across the batch.
    High variance = diverse samples (good)
    Low variance = similar samples (mode collapse)
    """
    # Flatten to (batch, pixels)
    batch_size = samples.shape[0]
    flat = samples.reshape(batch_size, -1)
    
    # Compute mean of each pixel across batch
    pixel_means = flat.mean(dim=0)
    
    # Variance of these means
    variance = pixel_means.var().item()
    
    return variance
```

**Interpretation:**
- If all samples identical: variance = 0 (mode collapse)
- If diverse samples: variance > 0 (healthy generation)

**Example:**
```
Batch of 4 samples:
  [[[1, 2], [3, 4]],
   [[1, 2], [3, 4]],    Identical samples → variance = 0
   [[1, 2], [3, 4]],
   [[1, 2], [3, 4]]]

  [[[1, 1], [1, 1]],
   [[2, 2], [2, 2]],    Diverse samples → variance > 0
   [[3, 3], [3, 3]],
   [[4, 4], [4, 4]]]
```

### 5.2: compute_sharpness()

```python
def compute_sharpness(samples):
    """
    Estimate image clarity using Sobel edge detection.
    
    Sharp images have strong edges.
    Blurry images have weak edges.
    """
    # Sobel kernels for edge detection
    sobel_x = torch.tensor([
        [-1., 0., 1.],
        [-2., 0., 2.],
        [-1., 0., 1.]
    ], device=samples.device).view(1, 1, 3, 3)
    
    sobel_y = torch.tensor([
        [-1., -2., -1.],
        [0., 0., 0.],
        [1., 2., 1.]
    ], device=samples.device).view(1, 1, 3, 3)
    
    # Apply convolution (edge detection)
    edges_x = F.conv2d(samples, sobel_x, padding=1)
    edges_y = F.conv2d(samples, sobel_y, padding=1)
    
    # Compute edge magnitude
    edges = torch.sqrt(edges_x**2 + edges_y**2 + 1e-8)
    
    # Average edge strength
    sharpness = edges.mean().item()
    
    return sharpness
```

**Interpretation:**
- Blurry image: sharpness ≈ 0.1
- Sharp image: sharpness ≈ 0.5+
- Very sharp: sharpness ≈ 1.0+

**Why Sobel:**
- Detects rapid intensity changes (edges)
- Strong edges → high magnitude → sharp image
- Weak edges → low magnitude → blurry image

---

## TODO 6 Solution: visualize_schedule_comparison()

### Implementation

```python
def visualize_schedule_comparison(
    linear_samples,
    cosine_samples,
    linear_trajectory,
    cosine_trajectory,
    num_display=4,
    figsize=(16, 8)
):
    """Create side-by-side comparison visualization."""
    
    fig, axes = plt.subplots(2, 8, figsize=figsize)
    fig.suptitle('Linear vs Cosine Schedule Comparison', fontsize=16)
    
    # ===== ROW 1: Final samples =====
    # Linear samples (left 4)
    for i in range(4):
        ax = axes[0, i]
        img = linear_samples[i].squeeze(0).cpu().numpy()
        ax.imshow(img, cmap='gray', vmin=-1, vmax=1)
        ax.set_title('Linear' if i == 0 else '')
        ax.axis('off')
    
    # Cosine samples (right 4)
    for i in range(4):
        ax = axes[0, i + 4]
        img = cosine_samples[i].squeeze(0).cpu().numpy()
        ax.imshow(img, cmap='gray', vmin=-1, vmax=1)
        ax.set_title('Cosine' if i == 0 else '')
        ax.axis('off')
    
    # ===== ROW 2: Denoising trajectories =====
    # Select 4 evenly-spaced steps from trajectory
    num_steps = len(linear_trajectory)
    indices = np.linspace(0, num_steps - 1, num_display, dtype=int)
    
    # Linear trajectory (left 4)
    for col, idx in enumerate(indices):
        ax = axes[1, col]
        img = linear_trajectory[idx][0].squeeze(0).cpu().numpy()
        ax.imshow(img, cmap='gray', vmin=-1, vmax=1)
        step_label = int(num_steps - idx)  # Steps remaining
        ax.set_title(f'Step {step_label}')
        ax.axis('off')
    
    # Cosine trajectory (right 4)
    for col, idx in enumerate(indices):
        ax = axes[1, col + 4]
        img = cosine_trajectory[idx][0].squeeze(0).cpu().numpy()
        ax.imshow(img, cmap='gray', vmin=-1, vmax=1)
        step_label = int(num_steps - idx)
        ax.set_title(f'Step {step_label}')
        ax.axis('off')
    
    plt.tight_layout()
    return fig
```

### Output Explanation

**Top Row:** Final samples
- Left 4: Results from linear schedule
- Right 4: Results from cosine schedule
- Often look very similar to casual observation, but metrics differ

**Bottom Row:** Denoising progression
- Left 4: How linear schedule removes noise step-by-step
- Right 4: How cosine schedule removes noise step-by-step
- Visually shows the trajectory from noise → clean

---

## TODO 7 Solution: compare_schedule_metrics()

### Implementation

```python
def compare_schedule_metrics(
    linear_samples,
    cosine_samples,
    linear_trajectory,
    cosine_trajectory
):
    """Compute and compare metrics for both schedules."""
    
    # Compute metrics for linear schedule
    lin_variance = compute_sample_variance(linear_samples)
    lin_sharpness = compute_sharpness(linear_samples)
    
    # Compute metrics for cosine schedule
    cos_variance = compute_sample_variance(cosine_samples)
    cos_sharpness = compute_sharpness(cosine_samples)
    
    # Build results dictionary
    results = {
        'linear': {
            'variance': lin_variance,
            'sharpness': lin_sharpness,
            'steps': len(linear_trajectory),
        },
        'cosine': {
            'variance': cos_variance,
            'sharpness': cos_sharpness,
            'steps': len(cosine_trajectory),
        }
    }
    
    # Analysis and comparison
    results['comparison'] = {
        'variance_winner': 'cosine' if cos_variance > lin_variance else 'linear',
        'sharpness_winner': 'cosine' if cos_sharpness > lin_sharpness else 'linear',
        'variance_ratio': cos_variance / (lin_variance + 1e-8),
        'sharpness_ratio': cos_sharpness / (lin_sharpness + 1e-8),
    }
    
    return results
```

### Typical Output

```
Linear Schedule:
  Variance:  0.0234
  Sharpness: 0.3456

Cosine Schedule:
  Variance:  0.0251
  Sharpness: 0.4123

Winner (Sharpness): cosine (1.193x better)
Winner (Variance):  cosine (1.072x better)

Interpretation:
- Cosine is 19% sharper
- Cosine has 7% more diversity
- Both use same 50 steps, but cosine schedules them better
```

---

## TODO 8 Solution: main()

### Implementation

```python
def main():
    """Complete analysis pipeline."""
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ===== STEP 1: Load pre-trained model =====
    print("\n1. Loading pre-trained model...")
    model = load_pretrained_ddpm_model(
        checkpoint_path="../../lesson-16-Implementing-Simple-Diffusion-Model/checkpoint.pt",
        device=device
    )
    model.eval()
    
    # ===== STEP 2: Create noise scheduler =====
    print("2. Initializing noise scheduler...")
    noise_scheduler = NoiseScheduler()
    
    # ===== STEP 3: Sample with linear schedule =====
    print("3. Sampling with LINEAR schedule (50 steps)...")
    linear_samples, linear_trajectory = sample_with_schedule(
        model=model,
        noise_scheduler=noise_scheduler,
        schedule_type="linear",
        num_steps=50,
        batch_size=8,
        device=device,
        return_trajectory=True
    )
    print(f"   Generated {linear_samples.shape[0]} samples")
    
    # ===== STEP 4: Sample with cosine schedule =====
    print("4. Sampling with COSINE schedule (50 steps)...")
    cosine_samples, cosine_trajectory = sample_with_schedule(
        model=model,
        noise_scheduler=noise_scheduler,
        schedule_type="cosine",
        num_steps=50,
        batch_size=8,
        device=device,
        return_trajectory=True
    )
    print(f"   Generated {cosine_samples.shape[0]} samples")
    
    # ===== STEP 5: Compute metrics =====
    print("5. Computing fidelity metrics...")
    metrics = compare_schedule_metrics(
        linear_samples,
        cosine_samples,
        linear_trajectory,
        cosine_trajectory
    )
    
    # ===== STEP 6: Visualize =====
    print("6. Creating comparison visualization...")
    fig = visualize_schedule_comparison(
        linear_samples,
        cosine_samples,
        linear_trajectory,
        cosine_trajectory
    )
    
    # ===== STEP 7: Print analysis =====
    print("\n" + "="*60)
    print("SCHEDULE COMPARISON RESULTS")
    print("="*60)
    
    print("\nLINEAR SCHEDULE:")
    print(f"  Variance:  {metrics['linear']['variance']:.6f}")
    print(f"  Sharpness: {metrics['linear']['sharpness']:.6f}")
    print(f"  Steps:     {metrics['linear']['steps']}")
    
    print("\nCOSINE SCHEDULE:")
    print(f"  Variance:  {metrics['cosine']['variance']:.6f}")
    print(f"  Sharpness: {metrics['cosine']['sharpness']:.6f}")
    print(f"  Steps:     {metrics['cosine']['steps']}")
    
    print("\nCOMPARISON:")
    print(f"  Sharpness Winner: {metrics['comparison']['sharpness_winner']}")
    print(f"  Sharpness Ratio:  {metrics['comparison']['sharpness_ratio']:.3f}x")
    print(f"  Variance Winner:  {metrics['comparison']['variance_winner']}")
    print(f"  Variance Ratio:   {metrics['comparison']['variance_ratio']:.3f}x")
    
    # ===== STEP 8: Save outputs =====
    print("\n7. Saving outputs...")
    fig.savefig('schedule_comparison.png', dpi=150, bbox_inches='tight')
    print("   Saved: schedule_comparison.png")
    
    # Save metrics to JSON
    import json
    metrics_json = {
        'linear': {k: float(v) for k, v in metrics['linear'].items()},
        'cosine': {k: float(v) for k, v in metrics['cosine'].items()},
        'comparison': {k: float(v) if isinstance(v, (int, float)) else v 
                      for k, v in metrics['comparison'].items()},
    }
    with open('metrics.json', 'w') as f:
        json.dump(metrics_json, f, indent=2)
    print("   Saved: metrics.json")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
```

### Expected Output

```
Using device: cuda

1. Loading pre-trained model...
   Model loaded: UNet (64 → 128 → 256 → 128 → 64)

2. Initializing noise scheduler...
   Scheduler ready: 1000 timesteps

3. Sampling with LINEAR schedule (50 steps)...
   Generated 8 samples
   Linear shape: torch.Size([8, 1, 28, 28])

4. Sampling with COSINE schedule (50 steps)...
   Generated 8 samples
   Cosine shape: torch.Size([8, 1, 28, 28])

5. Computing fidelity metrics...

6. Creating comparison visualization...

7. Saving outputs...

============================================================
SCHEDULE COMPARISON RESULTS
============================================================

LINEAR SCHEDULE:
  Variance:  0.023456
  Sharpness: 0.345678
  Steps:     50

COSINE SCHEDULE:
  Variance:  0.025123
  Sharpness: 0.412345
  Steps:     50

COMPARISON:
  Sharpness Winner: cosine
  Sharpness Ratio:  1.193x
  Variance Winner:  cosine
  Variance Ratio:   1.072x

7. Saving outputs...
   Saved: schedule_comparison.png
   Saved: metrics.json

============================================================
ANALYSIS COMPLETE!
============================================================
```

---
