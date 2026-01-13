# Exercise: Comparing Sampling Schedules

## Overview

In this exercise, you'll implement the complete reverse diffusion sampling pipeline and compare two distinct sampling schedules: **linear** and **cosine**.

**Learning Objectives:**
- Implement timestep scheduling strategies
- Understand reverse diffusion step-by-step
- Compare quality-speed tradeoffs
- Analyze sampling schedule impact on image generation


---

## Exercise Structure

This exercise has **8 TODOs** organized in 3 phases:

### Phase 1: Implement Schedules (TODOs 1-2)
Create two different strategies for spacing timesteps during sampling.

### Phase 2: Implement Sampling (TODOs 3-4)
Build the reverse diffusion loop and complete sampling pipeline.

### Phase 3: Evaluate & Compare (TODOs 5-8)
Compute metrics and create comprehensive visual comparisons.

---

## TODO 1: Implement LinearSchedule.__init__()


### Objective
Create a linear (uniform) timestep schedule.

### Background

A "schedule" defines which timesteps to visit during sampling.

If we train with 1000 noise steps but want to sample in 50 steps:
- **Linear:** Visit timesteps [1000, 980, 960, 940, ..., 20, 0]
- **Cosine:** Visit timesteps [1000, 970, 890, 750, ..., 10, 0]

Linear schedule uses uniform intervals.

### What You Need to Do

Complete the `__init__` method of `LinearSchedule` class:

```python
class LinearSchedule:
    def __init__(self, num_train_timesteps=1000, num_inference_steps=50):
        """
        Initialize linear timestep schedule.
        
        Args:
            num_train_timesteps: Total timesteps in training (usually 1000)
            num_inference_steps: Steps to use during sampling (50, 100, 1000, etc.)
        
        TODO: Set self.timesteps to linearly spaced values from num_train_timesteps-1 to 0
        Hint: Use torch.linspace(num_train_timesteps-1, 0, num_inference_steps)
        Then convert to long integer indices.
        """
```

### Expected Behavior

For 50 inference steps from 1000 training timesteps:

```python
schedule = LinearSchedule(1000, 50)
print(schedule.timesteps[:10])
# Expected: tensor([999, 978, 957, 936, 915, 894, 873, 852, 831, 810])

print(len(schedule.timesteps))
# Expected: 50

print(schedule.timesteps[-1])
# Expected: tensor(0)
```

### Hint

```python
# Uniformly spaced from end to beginning
values = torch.linspace(num_train_timesteps - 1, 0, num_inference_steps)
# Convert to integers (tensor indices must be Long)
self.timesteps = values.long()
```

### Why This Matters

Linear schedule is the baseline. It's simple and fast but not optimal for quality.

---

## TODO 2: Implement CosineSchedule.__init__()


### Objective
Create a cosine (non-linear) timestep schedule that emphasizes high-noise (difficult) steps.

### Background

Cosine schedule is designed to focus computational effort where it matters most:
- **Early steps (high noise):** Many sampling steps
- **Late steps (low noise):** Few sampling steps

Formula:
```
α_t = cos²(π * (t/N + s) / (1+s))
timestep = round(find_nearest_actual_timestep(α))
```

This creates a non-linear distribution that concentrates samples in the high-noise regime.

### What You Need to Do

Complete the `__init__` method of `CosineSchedule` class:

```python
class CosineSchedule:
    def __init__(self, num_train_timesteps=1000, num_inference_steps=50):
        """
        Initialize cosine timestep schedule (non-linear, quality-focused).
        
        Args:
            num_train_timesteps: Total timesteps in training (usually 1000)
            num_inference_steps: Steps to use during sampling (50, 100, 1000, etc.)
        
        TODO: Compute alpha values using cosine formula
        1. Create steps from 0 to N (N = num_inference_steps)
        2. Compute: alphas = cos²(π * (steps/N + s) / (1+s)) where s=0.008
        3. Find nearest timesteps for each alpha value
        4. Store in self.timesteps
        """
```

### Expected Behavior

For 50 inference steps from 1000 training timesteps:

```python
schedule = CosineSchedule(1000, 50)
print(schedule.timesteps[:10])
# Expected (approximately): tensor([999, 970, 890, 750, 550, 300, 150, 50, 10, 0])
#                           (NOT uniform like linear!)

print(len(schedule.timesteps))
# Expected: 50

# Check that steps are concentrated early (high noise)
diffs = torch.diff(schedule.timesteps)
print(diffs[:5])   # Large gaps (high noise)
print(diffs[-5:])  # Small gaps (low noise)
```

### Hint

```python
# Step 1: Create step indices
s = 0.008
steps = torch.arange(num_inference_steps + 1, dtype=torch.float32)

# Step 2: Compute alphas using cosine schedule
alphas = torch.cos(((steps / num_inference_steps + s) / (1 + s)) * math.pi / 2) ** 2

# Step 3: Convert alphas to timesteps
# Alpha values range [0, 1] where alpha_t = alpha_cumprod[t]
# You need to find which timestep index t gives each alpha value
# Hint: alpha_cumprod decreases from 1 to 0 as t increases from 0 to 1000
# Find nearest timestep by comparing with alpha_cumprod values

timesteps = []
for alpha in alphas:
    # Find closest timestep
    distances = torch.abs(alpha_cumprod - alpha)
    timestep = torch.argmin(distances).item()
    timesteps.append(timestep)
self.timesteps = torch.tensor(timesteps, dtype=torch.long)
```

### Why This Matters

Cosine schedule provides better quality with fewer steps because it focuses effort on hard parts.

---

## TODO 3: Implement reverse_diffusion_step()



### Objective
Perform one reverse diffusion step: x_t → x_{t-1}.

### Background

The core of sampling is the reverse step. Given:
- x_t (current noisy image)
- t (current timestep)
- ε_θ (predicted noise from U-Net)

We compute x_{t-1} (cleaner image) using the posterior distribution.

**Mathematical formula:**
```
1. Predict original: x_0 = (x_t - √(1-ᾱ_t) * ε) / √ᾱ_t
2. Posterior mean: mean = coef1 * x_0 + coef2 * x_t
3. Add variance: x_{t-1} = mean + σ_t * z
```

### What You Need to Do

Complete the `reverse_diffusion_step()` function:

```python
def reverse_diffusion_step(
    x_t,                    # Current image (batch, 1, 28, 28)
    t,                      # Current timestep (int)
    predicted_noise,        # U-Net output (batch, 1, 28, 28)
    alpha_cumprod,          # Pre-computed cumulative products
    alphas,                 # Pre-computed alphas
    betas,                  # Pre-computed betas
    device                  # cuda/cpu/mps
):
    """
    Perform one reverse diffusion step.
    
    Args:
        x_t: Current noisy image
        t: Current timestep index
        predicted_noise: Noise predicted by U-Net
        alpha_cumprod: Cumulative product of alphas [alpha_0, alpha_0*alpha_1, ...]
        alphas: Individual alpha values [alpha_1, alpha_2, ...]
        betas: Individual beta values [beta_1, beta_2, ...]
        device: Device to compute on
    
    Returns:
        x_{t-1}: Denoised image
    
    TODO: Implement reverse diffusion formula
    1. Get alpha values for timestep t (current and previous)
    2. Compute predicted original image
    3. Compute posterior mean
    4. Compute posterior variance
    5. Sample noise and apply to mean
    """
```

### Step-by-Step Implementation

**Step 1: Extract pre-computed coefficients**
```python
alpha_t = alpha_cumprod[t]
alpha_t_prev = alpha_cumprod[t-1] if t > 0 else torch.tensor(1.0, device=device)
beta_t = betas[t]

# Useful derived values
sqrt_alpha_t = torch.sqrt(alpha_t)
sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
```

**Step 2: Predict original image**
```python
# Rearrange x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
# to solve for x_0: x_0 = (x_t - √(1-ᾱ_t) * ε) / √ᾱ_t

pred_original = (x_t - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t
```

**Step 3: Compute posterior mean**
```python
# Posterior variance using conjugacy
variance_numerator = (1 - alpha_t_prev) * beta_t
variance_denominator = 1 - alpha_t
variance = variance_numerator / variance_denominator

# Coefficients for posterior mean
coef1 = sqrt_alpha_t_prev * beta_t / (1 - alpha_t)
coef2 = (1 - beta_t) * sqrt_alpha_t / (1 - alpha_t)

mean = coef1 * pred_original + coef2 * x_t
```

**Step 4: Add stochastic noise**
```python
# Sample noise from standard normal
z = torch.randn_like(x_t)

# Denoise step
x_t_minus_1 = mean + torch.sqrt(variance) * z
```

### Expected Behavior

```python
# Initialize a random noisy image
x_t = torch.randn(1, 1, 28, 28)

# Perform one step
x_t_minus_1 = reverse_diffusion_step(
    x_t=x_t,
    t=500,
    predicted_noise=torch.randn(1, 1, 28, 28),
    alpha_cumprod=alpha_cumprod,
    alphas=alphas,
    betas=betas,
    device='cpu'
)

# Output shape should match input
assert x_t_minus_1.shape == x_t.shape
# Output should have smaller noise level
assert x_t_minus_1.std() < x_t.std()
```

### Why This Matters

This is the core sampling operation. Everything else builds on this step.

---

## TODO 4: Implement sample_with_schedule()


### Objective
Build the complete sampling loop using a chosen schedule.

### Background

The complete sampling process:
```
1. Start with pure Gaussian noise
2. For each timestep (from high noise to clean):
   a. Use U-Net to predict the noise
   b. Remove noise using one reverse step
3. End with clean image
```

### What You Need to Do

Complete the `sample_with_schedule()` function:

```python
def sample_with_schedule(
    model,                  # U-Net model
    noise_scheduler,        # Object with alpha_cumprod, betas, etc.
    schedule_type,          # "linear" or "cosine"
    num_steps,              # Number of sampling steps
    batch_size,             # How many images to generate
    device,                 # cuda/cpu/mps
    return_trajectory       # Whether to save intermediate steps
):
    """
    Complete sampling pipeline with chosen schedule.
    
    Args:
        model: Trained U-Net
        noise_scheduler: Contains alpha_cumprod, betas, etc.
        schedule_type: "linear" or "cosine"
        num_steps: Number of sampling steps (50, 100, 1000)
        batch_size: Images to generate
        device: Compute device
        return_trajectory: Save all intermediate images
    
    Returns:
        samples: Final generated images (batch, 1, 28, 28)
        trajectory: (optional) All intermediate steps
    
    TODO: Implement complete sampling loop
    1. Create schedule object (LinearSchedule or CosineSchedule)
    2. Initialize with pure Gaussian noise
    3. For each timestep in reversed schedule:
       a. Use model to predict noise
       b. Call reverse_diffusion_step()
       c. Save intermediate if needed
    4. Return final samples
    """
```

### Step-by-Step Implementation

**Step 1: Create appropriate schedule**
```python
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
    raise ValueError("Unknown schedule type")
```

**Step 2: Initialize with noise**
```python
x_t = torch.randn(batch_size, 1, 28, 28, device=device)
```

**Step 3: Main sampling loop**
```python
trajectory = [] if return_trajectory else None

with torch.no_grad():
    for t in schedule.timesteps:
        # Get timestep index
        t_index = int(t.item()) if isinstance(t, torch.Tensor) else t
        
        # Model prediction
        model_output = model(x_t, t_index)
        
        # Reverse diffusion step
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
```

**Step 4: Return results**
```python
if return_trajectory:
    return x_t, trajectory
else:
    return x_t
```

### Expected Behavior

```python
# Sample using linear schedule, 50 steps
samples_linear = sample_with_schedule(
    model=model,
    noise_scheduler=noise_scheduler,
    schedule_type="linear",
    num_steps=50,
    batch_size=4,
    device='cpu',
    return_trajectory=True
)

samples, trajectory = samples_linear

# Check outputs
assert samples.shape == (4, 1, 28, 28)
assert len(trajectory) == 50
assert trajectory[0].shape == (4, 1, 28, 28)
# Early trajectory steps should be noisy
# Late trajectory steps should be clean
```

### Why This Matters

This ties everything together into a complete generative pipeline.

---

## TODO 5: Compute Fidelity Metrics


### Objective
Implement two metrics to assess generated image quality.

### Metrics to Implement

#### 5.1: Sample Variance

Measure diversity across generated samples.

```python
def compute_sample_variance(samples):
    """
    Compute variance across sample batch.
    
    Args:
        samples: (batch, 1, 28, 28) tensor
    
    Returns:
        variance: scalar indicating sample diversity
    
    TODO: Flatten samples, compute mean across batch,
    then compute variance of means.
    """
```

**Implementation:**
```python
# Flatten to (batch, 28*28)
flat = samples.reshape(samples.shape[0], -1)
# Compute mean across batch for each pixel
mean_sample = flat.mean(dim=0)
# Compute variance
variance = mean_sample.var().item()
return variance
```

**Interpretation:**
- High variance: Samples are diverse (good)
- Low variance: Samples are similar (potential mode collapse)

#### 5.2: Sample Sharpness

Estimate image clarity using edge detection.

```python
def compute_sharpness(samples):
    """
    Compute sharpness using Sobel edge detection.
    
    Args:
        samples: (batch, 1, 28, 28) tensor
    
    Returns:
        sharpness: scalar indicating edge strength
    
    TODO: Apply Sobel filters to detect edges,
    then compute average magnitude.
    """
```

**Implementation:**
```python
# Sobel kernels (pre-defined)
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

# Apply convolution
edges_x = F.conv2d(samples, sobel_x, padding=1)
edges_y = F.conv2d(samples, sobel_y, padding=1)

# Magnitude
edges = torch.sqrt(edges_x**2 + edges_y**2 + 1e-8)

# Average
sharpness = edges.mean().item()
return sharpness
```

**Interpretation:**
- High sharpness: Clear edges and details (good)
- Low sharpness: Blurry (bad training or poor schedule)

### Expected Behavior

```python
# Generate some samples
samples = torch.randn(4, 1, 28, 28)

variance = compute_sample_variance(samples)
sharpness = compute_sharpness(samples)

print(f"Variance: {variance:.4f}")  # Should be positive
print(f"Sharpness: {sharpness:.4f}")  # Should be positive

# Real images should have higher sharpness than random noise
real_variance = compute_sample_variance(real_images)
noise_variance = compute_sample_variance(torch.randn_like(real_images))
assert real_variance > noise_variance  # Real is more structured
```

### Why This Matters

Metrics let us objectively compare different sampling strategies.

---

## TODO 6: Visualize Schedule Comparison


### Objective
Create visual comparison of linear vs cosine schedules.

### What You Need to Do

```python
def visualize_schedule_comparison(
    linear_samples,
    cosine_samples,
    linear_trajectory,
    cosine_trajectory,
    num_display=4,
    figsize=(16, 8)
):
    """
    Create side-by-side comparison visualization.
    
    Args:
        linear_samples: Generated images (linear schedule)
        cosine_samples: Generated images (cosine schedule)
        linear_trajectory: Denoising steps (linear)
        cosine_trajectory: Denoising steps (cosine)
        num_display: How many trajectory steps to show
        figsize: Figure size
    
    Returns:
        fig: Matplotlib figure
    
    TODO: Create 2 main subplots:
    1. Top: Final samples (linear vs cosine, 4 images each)
    2. Bottom: Denoising trajectory (both schedules, 6 steps each)
    """
```

### Step-by-Step Implementation

**Step 1: Create figure**
```python
fig, axes = plt.subplots(2, 8, figsize=figsize)
fig.suptitle('Linear vs Cosine Schedule Comparison', fontsize=16)
```

**Step 2: Top row - final samples**
```python
# Linear samples (left 4)
for i in range(4):
    axes[0, i].imshow(
        linear_samples[i].squeeze(0).cpu().numpy(),
        cmap='gray'
    )
    axes[0, i].set_title('Linear' if i == 0 else '')
    axes[0, i].axis('off')

# Cosine samples (right 4)
for i in range(4):
    axes[0, i+4].imshow(
        cosine_samples[i].squeeze(0).cpu().numpy(),
        cmap='gray'
    )
    axes[0, i+4].set_title('Cosine' if i == 0 else '')
    axes[0, i+4].axis('off')
```

**Step 3: Bottom row - trajectories**
```python
# Select 6 evenly-spaced trajectory steps
indices = np.linspace(0, len(linear_trajectory)-1, num_display, dtype=int)

# Linear trajectory
for col, idx in enumerate(indices):
    axes[1, col].imshow(
        linear_trajectory[idx][0].squeeze(0).cpu().numpy(),
        cmap='gray'
    )
    axes[1, col].set_title(f'Step {idx}')
    axes[1, col].axis('off')

# Cosine trajectory
for col, idx in enumerate(indices):
    axes[1, col+4].imshow(
        cosine_trajectory[idx][0].squeeze(0).cpu().numpy(),
        cmap='gray'
    )
    axes[1, col+4].set_title(f'Step {idx}')
    axes[1, col+4].axis('off')
```

### Expected Output

A figure showing:
- Top-left: 4 samples from linear schedule
- Top-right: 4 samples from cosine schedule
- Bottom-left: Linear denoising progression
- Bottom-right: Cosine denoising progression

### Why This Matters

Visual comparison makes the quality differences obvious.

---

## TODO 7: Compare Quality Metrics


### Objective
Compute metrics for both schedules and analyze results.

### What You Need to Do

```python
def compare_schedule_metrics(
    linear_samples,
    cosine_samples,
    linear_trajectory,
    cosine_trajectory
):
    """
    Compute metrics for both schedules and compare.
    
    Args:
        linear_samples: Generated images (linear schedule)
        cosine_samples: Generated images (cosine schedule)
        linear_trajectory: Denoising trajectory (linear)
        cosine_trajectory: Denoising trajectory (cosine)
    
    Returns:
        results: Dictionary with all metrics
    
    TODO: Compute variance and sharpness for both schedules.
    Create a results dictionary with:
    - 'linear': {'variance': float, 'sharpness': float}
    - 'cosine': {'variance': float, 'sharpness': float}
    - 'comparison': Analysis of which is better and why
    """
```

### Implementation

```python
# Compute metrics
results = {
    'linear': {
        'variance': compute_sample_variance(linear_samples),
        'sharpness': compute_sharpness(linear_samples),
        'trajectory_lengths': len(linear_trajectory),
    },
    'cosine': {
        'variance': compute_sample_variance(cosine_samples),
        'sharpness': compute_sharpness(cosine_samples),
        'trajectory_lengths': len(cosine_trajectory),
    }
}

# Analysis
lin_var = results['linear']['variance']
cos_var = results['cosine']['variance']
lin_sharp = results['linear']['sharpness']
cos_sharp = results['cosine']['sharpness']

results['comparison'] = {
    'variance_winner': 'cosine' if cos_var > lin_var else 'linear',
    'sharpness_winner': 'cosine' if cos_sharp > lin_sharp else 'linear',
    'variance_ratio': cos_var / (lin_var + 1e-6),
    'sharpness_ratio': cos_sharp / (lin_sharp + 1e-6),
}

return results
```

### Expected Output

```
Linear Schedule:
  - Variance:  0.1234
  - Sharpness: 0.5678

Cosine Schedule:
  - Variance:  0.1456
  - Sharpness: 0.6234

Winner: Cosine has 1.18x better sharpness
```

### Why This Matters

Numbers back up visual impressions and guide schedule selection.

---

## TODO 8: Main Analysis Script

### Objective
Tie everything together and run complete comparison.

### What You Need to Do

```python
def main():
    """
    Complete analysis pipeline.
    
    TODO: Implement the following steps in order:
    1. Load pre-trained DDPM model from Module 16
    2. Create noise scheduler
    3. Generate samples with linear schedule (50 steps)
    4. Generate samples with cosine schedule (50 steps)
    5. Compute fidelity metrics for both
    6. Visualize comparison
    7. Print analysis
    8. Save outputs (images, metrics)
    """
```

### Step-by-Step

```python
# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Load model
model = load_pretrained_ddpm_model(
    checkpoint_path="path/to/model.pt",
    device=device
)

# 2. Create scheduler
scheduler = NoiseScheduler()

# 3. Sample with linear schedule
print("Sampling with linear schedule...")
linear_samples, linear_trajectory = sample_with_schedule(
    model=model,
    noise_scheduler=scheduler,
    schedule_type="linear",
    num_steps=50,
    batch_size=8,
    device=device,
    return_trajectory=True
)

# 4. Sample with cosine schedule
print("Sampling with cosine schedule...")
cosine_samples, cosine_trajectory = sample_with_schedule(
    model=model,
    noise_scheduler=scheduler,
    schedule_type="cosine",
    num_steps=50,
    batch_size=8,
    device=device,
    return_trajectory=True
)

# 5. Compute metrics
metrics = compare_schedule_metrics(
    linear_samples,
    cosine_samples,
    linear_trajectory,
    cosine_trajectory
)

# 6. Visualize
fig = visualize_schedule_comparison(
    linear_samples,
    cosine_samples,
    linear_trajectory,
    cosine_trajectory
)

# 7. Print results
print("\n" + "="*50)
print("SCHEDULE COMPARISON RESULTS")
print("="*50)
for schedule_type in ['linear', 'cosine']:
    print(f"\n{schedule_type.upper()}:")
    for metric, value in metrics[schedule_type].items():
        print(f"  {metric}: {value:.4f}")

print(f"\nBEST SHARPNESS: {metrics['comparison']['sharpness_winner']}")
print(f"BEST VARIANCE: {metrics['comparison']['variance_winner']}")

# 8. Save
plt.savefig('schedule_comparison.png', dpi=150, bbox_inches='tight')
print("\nSaved: schedule_comparison.png")
```

### Expected Output

When you run this:
```
Using device: cuda
Sampling with linear schedule...
100%|████| 50/50 [00:05<00:00,  9.43it/s]
Sampling with cosine schedule...
100%|████| 50/50 [00:05<00:00,  9.25it/s]

==================================================
SCHEDULE COMPARISON RESULTS
==================================================

LINEAR:
  variance: 0.1234
  sharpness: 0.5123
  trajectory_lengths: 50

COSINE:
  variance: 0.1456
  sharpness: 0.6234
  trajectory_lengths: 50

BEST SHARPNESS: cosine
BEST VARIANCE: cosine

Saved: schedule_comparison.png
```

---

## Common Issues & Debugging

### Issue 1: "ModuleNotFoundError: No module named 'exercise'"

**Solution:** Make sure you're in the right directory and paths are correct.
```bash
cd lesson-17-Sampling-and-Image-Generation-with-Diffusion/exercises/starter/
python exercise.py
```

### Issue 2: "Model not found" error

**Solution:** Pre-trained model path is incorrect. Check:
```python
# The model checkpoint should be at:
model_path = "../../lesson-16-Implementing-Simple-Diffusion-Model/checkpoint.pt"
# Adjust based on your actual path
```

### Issue 3: Generated images are all noise (not denoising)

**Solution:** Check that:
```python
# 1. Model is in eval mode
model.eval()

# 2. torch.no_grad() is used during sampling
with torch.no_grad():
    # sampling code

# 3. Timesteps are correct (should decrease from high to low)
print(schedule.timesteps)  # Should be [1000, 900, 800, ...] for linear
```

### Issue 4: Cosine and linear produce the same results

**Solution:** Check that schedules are truly different:
```python
linear_sched = LinearSchedule()
cosine_sched = CosineSchedule()

print("Linear:", linear_sched.timesteps[:10])
print("Cosine:", cosine_sched.timesteps[:10])
# These should be different!
```

### Issue 5: Out of memory (CUDA)

**Solution:** Reduce batch size:
```python
# Change from batch_size=8 to batch_size=4
samples = sample_with_schedule(..., batch_size=4, ...)
```

---

**Further Reading:**
1. **DDPM Paper:** [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
   - Section 3: Reverse Process
   - Equation 6: Posterior Mean & Variance

2. **DDIM Paper:** [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
   - More efficient sampling using fewer steps

3. **Improved DDPM:** [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)
   - Better variance scheduling

