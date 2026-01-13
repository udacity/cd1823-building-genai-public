# Demo: Diffusion Sampling and Image Generation

## Overview

This demo implements the complete reverse diffusion sampling process for generating Fashion MNIST images from pure Gaussian noise.

**What You'll Learn:**
- How reverse diffusion recovers images from noise
- Implementing different sampling schedules
- Visualizing the denoising trajectory
- Comparing quality across strategies

---

## Part 1: Understanding Reverse Diffusion

### The Core Idea

During training (forward process):
- Start with clean image x₀
- Add noise progressively: x₀ → x₁ → x₂ → ... → x_T (pure noise)

During sampling (reverse process):
- Start with pure noise x_T
- Remove noise progressively: x_T → x_{T-1} → ... → x₁ → x₀ (clean image)

The U-Net learned to predict the noise at each step. We use this to reverse the process.

### Mathematical Formula

**One reverse step:**
```
1. Use U-Net to predict noise: ε_θ(x_t, t)
2. Compute "denoised" version: x_pred = (x_t - √(1-ᾱ_t) * ε) / √ᾱ_t
3. Compute posterior mean: m_t = coef1 * x_pred + coef2 * x_t
4. Add variance: x_{t-1} = m_t + σ_t * z
```

**Key insight:** We're solving an inverse problem - given x_t and t, predict the noise added, then remove it.

---

## Part 2: DDPMSampler Class

### Purpose

The `DDPMSampler` class orchestrates the complete sampling process.

### Key Methods

#### `__init__(noise_scheduler, num_inference_steps)`

Initializes the sampler with:
- Noise scheduler (provides α_t, β_t, etc.)
- Number of inference steps (50, 100, 1000, etc.)

Pre-computes all coefficients needed for efficiency.

#### `set_timesteps(num_inference_steps, schedule)`

Configures which timesteps to use during sampling.

**Schedules:**
- `"linear"`: Uniform spacing (simplest)
- `"cosine"`: Non-linear spacing (best quality)

**Implementation detail:**
```python
# Linear: evenly spaced
timesteps = torch.linspace(0, T-1, N)

# Cosine: weighted toward high noise
t_i = T * cos²(π(i/N + s)/(1+s))
```

#### `step(model_output, timestep, sample)`

Performs one reverse diffusion step.

**Inputs:**
- `model_output`: Noise predicted by U-Net (batch, 1, 28, 28)
- `timestep`: Current timestep t (integer)
- `sample`: Current image x_t (batch, 1, 28, 28)

**Output:**
- Denoised image x_{t-1}

**Implementation:**
```python
# Get coefficients for this timestep
sqrt_alpha = α_cumprod[t]^0.5
sqrt_one_minus_alpha = (1 - α_cumprod[t])^0.5

# Predict "clean" image
pred_original = (x_t - sqrt(1-α) * ε) / sqrt(α)

# Posterior mean
mean = coef1 * pred_original + coef2 * x_t

# Add variance (stochastic sampling)
z ~ N(0, I)
x_{t-1} = mean + σ_t * z
```

#### `__call__(model, batch_size, num_inference_steps, schedule, device)`

Main entry point for sampling.

**Process:**
```python
# 1. Initialize with pure Gaussian noise
x_T ~ N(0, I)

# 2. For each timestep (T → 0):
for t in reversed(range(T)):
    ε_θ = model(x_t, t)  # Predict noise
    x_{t-1} = step(ε_θ, t, x_t)  # Reverse diffusion step

# 3. Return final image x_0
```

---

## Part 3: Sampling Schedules Explained

### Linear Schedule

**Code:**
```python
timesteps = torch.linspace(0, T-1, N)
```

**Resulting timestep sequence (N=10, T=1000):**
```
1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 0
     ← evenly spaced intervals →
```

**Characteristics:**
- Uniform spacing
- Same "effort" at each step
- Simple to understand and implement

**Pros:**
- Fastest (uniform steps)
- Predictable
- Easy to analyze

**Cons:**
- Wastes computation on easy steps
- Lower quality per step
- Not optimal for noise removal

### Cosine Schedule

**Code:**
```python
s = 0.008
steps = torch.arange(N + 1)
alphas = torch.cos(((steps / N + s) / (1 + s)) * π/2) ** 2
timesteps = find_nearest_actual_timesteps(alphas)
```

**Resulting timestep sequence (N=10, T=1000):**
```
1000, 970, 890, 750, 550, 300, 150, 50, 10, 0
  ↑    many steps    ↑    few steps    ↑
 high noise      medium noise       low noise
```

**Characteristics:**
- Non-linear spacing
- More steps at high noise (hard)
- Fewer steps at low noise (easy)
- Focuses computational budget on difficult steps

**Pros:**
- Better quality (focuses on hard parts)
- Empirically superior
- Uses budget more efficiently

**Cons:**
- More complex to implement
- Slightly slower per-step
- Requires finding nearest timesteps

### Visual Comparison

```
Noise level over time:

Linear Schedule:
  ▁ ▁ ▁ ▁ ▁ ▁ ▁ ▁ ▁ ▁ (uniform effort)
  │ │ │ │ │ │ │ │ │ │

Cosine Schedule:
  ▓ ▒ ▒ ░ ░ ░ ░ ░ ░ ░ (more effort early)
  │ │ │ │ │ │ │ │ │ │
```

---

## Part 4: Reverse Diffusion Step Detailed

### The step() Method

This is the heart of sampling. Let's trace through one step carefully.

**Inputs:**
- `model_output`: ε_θ (noise predicted by U-Net)
- `timestep`: t (current step)
- `sample`: x_t (current noisy image)

**Step-by-step:**

**1. Retrieve pre-computed coefficients:**
```python
alpha_t = α_cumprod[t]              # Signal retention
beta_t = β[t]                       # Noise added at this step
alpha_t_prev = α_cumprod[t-1]       # Signal retention at t-1
```

**2. Predict original image (reverse one step of forward process):**
```python
# Forward process: x_t = √α * x_0 + √(1-α) * ε
# Solving for x_0: x_0 = (x_t - √(1-α) * ε) / √α

pred_original = (sample - sqrt(1-α_t) * model_output) / sqrt(α_t)
```

This estimates what the original clean image was.

**3. Compute posterior mean (Bayes theorem application):**
```python
# The posterior distribution p(x_{t-1} | x_t, x_0) is Gaussian
# Mean: weighted combination of x_t and x_0

coef1 = β_t * √α_{t-1} / (1 - α_cumprod_t)
coef2 = (1 - β_t) * √(1 - α_{t-1}) / (1 - α_cumprod_t)

mean = coef1 * pred_original + coef2 * sample
```

**4. Add variance (stochastic sampling):**
```python
# Posterior variance
variance = β_t * (1 - α_{t-1}) / (1 - α_cumprod_t)

# Sample noise
z ~ N(0, I)

# Denoised sample
x_{t-1} = mean + sqrt(variance) * z
```

### Why This Works

The mathematical foundation is Bayes theorem applied to the diffusion process.

**Key insight:** If we know the noise added at step t, we can reverse it exactly:
- Forward: x_t = √α * x_0 + √(1-α) * ε
- Reverse: x_0 = (x_t - √(1-α) * ε) / √α

But we don't know ε. The U-Net predicts it. This prediction error propagates but is minimized by training.

---

## Part 5: FastSampler - Speed Optimization

### The Problem

Full sampling with 1000 steps takes ~20 seconds per image on GPU.

For interactive applications, we need faster generation.

### The Solution

Use only 50 steps instead of 1000.

```python
class FastSampler(DDPMSampler):
    def __call__(self, model, batch_size, device, schedule="linear"):
        return super().__call__(
            model,
            batch_size=batch_size,
            num_inference_steps=50,      # ← Only 50 steps!
            schedule=schedule,
            device=device,
            return_trajectory=True
        )
```

### Trade-off Analysis

| Metric | 50 Steps | 1000 Steps |
|--------|----------|-----------|
| Speed | 1 sec | 20 sec |
| Quality | Good | Excellent |
| Speedup | 20x | 1x |
| FID Score | ~28 | ~12 |

**Decision:**
- Interactive apps: Use 50 steps (1 sec acceptable)
- High quality: Use 100+ steps (better fidelity)
- Production: Use optimal tradeoff for your use case

---

## Part 6: SamplingComparison Utilities

### Comparing Schedules

```python
comparison = SamplingComparison(noise_scheduler)

results = comparison.compare_schedules(
    model,
    num_steps=50,
    device=device
)

# Results:
# {
#   'linear': {'samples': tensor, 'trajectory': [...]},
#   'cosine': {'samples': tensor, 'trajectory': [...]}
# }
```

### Comparing Step Counts

```python
results = comparison.compare_step_counts(
    model,
    steps_list=[10, 50, 100, 1000],
    device=device
)

# Show quality improvement with more steps
```

### Visualizing Trajectories

```python
fig = SamplingComparison.visualize_trajectory(
    trajectory,
    title="Denoising Process"
)
```

Shows the progression from noise to image.

---

## Part 7: Quality Metrics

### 1. Sample Variance

Measure diversity of generated samples.

```python
def compute_sample_variance(samples):
    flat = samples.reshape(batch, -1)
    return torch.var(flat.mean(dim=0))
```

**Interpretation:**
- High: Diverse samples (good)
- Low: Similar samples (potential mode collapse)

### 2. Sharpness

Estimate image clarity using edge detection.

```python
def compute_sharpness(sample):
    # Apply Sobel operator
    edges = sobel_filter(sample)
    # Average edge magnitude
    return edges.mean()
```

**Interpretation:**
- High: Sharp details (good)
- Low: Blurry (bad training or wrong schedule)

### 3. Generation Speed

Time per image.

```python
speed = total_time / batch_size
```

**Interpretation:**
- <1s: Interactive
- 1-5s: Acceptable
- >5s: Offline processing

---

## Summary

The demo teaches the complete sampling pipeline:

1. **Mathematical foundation:** Reverse diffusion theory
2. **Implementation:** DDPMSampler class
3. **Scheduling:** Linear vs Cosine comparison
4. **Optimization:** FastSampler for speed
5. **Evaluation:** Metrics for quality assessment
6. **Comparison:** Side-by-side analysis

**What You Can Now Do:**
-  Generate images from pure noise
-  Choose optimal sampling schedule
-  Trade off quality vs speed
-  Evaluate generated image quality


**Key Takeaway:** Sampling is the inverse of training - we use the learned noise predictor to systematically remove noise, transforming pure randomness into coherent images.
