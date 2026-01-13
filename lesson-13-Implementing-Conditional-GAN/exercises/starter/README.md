# Exercise: Implementing Conditional GANs



## Learning Objectives

 **Understand Conditioning:** How to add class information to neural networks

 **Implement cGAN Models:** Build Generator and Discriminator with class control

 **Train Conditional Models:** Implement training loop with class labels

 **Analyze Results:** Interpret 10×10 class grids to evaluate quality

 **Apply to Augmentation:** Use generated images for data augmentation

---

## TODO Structure

Both files have **10 TODO sections** organized by difficulty:

### Beginner TODOs (1-3): Setup

**TODO 1: Load CIFAR-10 Dataset**
```python
# What to do:
# - Define transforms (ToTensor + Normalize to [-1, 1])
# - Load CIFAR-10 with download=True
# - Create DataLoader with batch_size=64

# Why it matters:
# - Normalization affects training stability
# - Proper dataloader ensures batch consistency
# - Shuffling enables stochastic gradient descent
```

**TODO 2: Create cGAN Models**
```python
# What to do:
# - Use create_cgan_models() function
# - Set latent_dim=100 (noise vector)
# - Set num_classes=10 (CIFAR-10)
# - Initialize weights using DCGAN guidelines

# Why it matters:
# - Proper weight initialization speeds up convergence
# - Correct dimensions ensure clean tensor operations
```

**TODO 3: Create Trainer**
```python
# What to do:
# - Instantiate ConditionalGANTrainer
# - Pass generator and discriminator
# - Set learning rates to 0.0002
# - Configure Adam optimizer with beta1=0.5, beta2=0.999

# Why it matters:
# - Trainer manages loss computation and updates
# - These hyperparameters are DCGAN standards
```

### Intermediate TODOs (4-6): Training and Visualization

**TODO 4: Train the cGAN**
```python
# What to do:
# - Call trainer.train() for 20 epochs
# - Pass train_loader, latent_dim, num_classes
# - Let it run (will take 20-30 minutes)

# What to expect:
# - D loss should stabilize around 0.5-0.7
# - G loss should decrease over time
# - Loss curves show training progress

# Common questions:
# Q: Why so long?
# A: Training GANs requires many iterations for convergence

# Q: Can I interrupt?
# A: Yes, you can stop early and see partial results
```

**TODO 5: Plot Loss Curves**
```python
# What to do:
# - Create 1x2 subplot (D loss, G loss)
# - Plot results['d_losses'] on left
# - Plot results['g_losses'] on right
# - Add reference line at y=0.5 (ideal D loss)

# What to look for:
# ✓ D loss: Should be around 0.5 (random guessing)
# ✓ G loss: Should decrease over time
# ✗ Both losses high: Training instability
# ✗ D loss → 0: D overpowering G
```

**TODO 6: Generate 10×10 Class Grid**
```python
# What to do:
# - Use trainer.generate_all_classes_grid()
# - Generate 10 classes × 10 samples = 100 images
# - Same noise z, different class labels y
# - Denormalize from [-1, 1] to [0, 1]
# - Display as 10×10 subplot grid

# What to look for:
# ✓ Each row is clearly a different class
# ✓ Within rows: style variation visible
# ✓ Across rows: clear separation
# ✗ All rows look similar: class control failing
# ✗ Blurry/noisy: model underfitted

# This is the MOST IMPORTANT visualization!
```

### Advanced TODOs (7-8): Analysis

**TODO 7: Generate Single-Class Samples**
```python
# What to do:
# - Call trainer.generate_class_samples() for target_class=5 (dogs)
# - Generate 16 samples with different noise
# - Denormalize and display as 2×8 grid

# Key insight:
# - Same class, different noise = style variation
# - Shows the noise dimension creates diversity
# - All 16 should look like dogs, different appearances
```

**TODO 8: Analyze Class Disentanglement**
```python
# What to do:
# - Answer 5 analysis questions
# - Look at the 10×10 grid
# - Interpret what you see

# Questions:
# 1. Within-row variation: Do images in same row vary?
# 2. Across-row differences: Are rows clearly different?
# 3. Class control: Do labels strongly affect output?
# 4. Quality issues: Which classes work best?
# 5. Data augmentation: Would you use these images?

# This teaches critical thinking about GAN results
```

### Expert TODOs (9-10): Optional Extensions

**TODO 9: Compare Generated vs Real (Optional)**
```python
# What to do:
# - For each class, show:
#   * Generated sample
#   * Real sample
#   * Quality assessment
# - Visualize as 3 rows × 10 columns

# Purpose:
# - See if generated images look realistic
# - Identify which classes are easier/harder
# - Benchmark against real data
```

**TODO 10: CIFAR-100 Extension (Bonus Challenge)**
```python
# What to do:
# - Modify code to work with CIFAR-100
# - Change num_classes from 10 to 100
# - Increase label_dim to 100
# - Train and evaluate

# Challenge level: HIGH
# - Requires code modification
# - More training time
# - Harder learning problem

# Why it's interesting:
# - Tests scalability
# - Shows limitations at scale
# - Real-world datasets have many classes
```

---

## Step-by-Step Guide

### Step 1: Setup (5 minutes)

1. Open `exercise_notebook.ipynb` in Jupyter
2. Complete **TODO 1**: Load CIFAR-10
3. Run the cell - should print dataset info
4. **Checkpoint:** Dataset loaded successfully 

### Step 2: Build Models (10 minutes)

1. Complete **TODO 2**: Create models
2. Run the cell - should print parameter counts
3. **Checkpoint:** Models created successfully 

### Step 3: Create Trainer (5 minutes)

1. Complete **TODO 3**: Create trainer
2. Run the cell - should print trainer info
3. **Checkpoint:** Trainer ready 

### Step 4: Train (30 minutes)

1. Complete **TODO 4**: Train the model
2. **IMPORTANT:** This will take 20-30 minutes
3. Monitor loss values - should show progress
4. **Checkpoint:** Training complete 

### Step 5: Evaluate (15 minutes)

1. Complete **TODO 5**: Plot loss curves
   - Analyze the shapes
   - Both should be reasonable

2. Complete **TODO 6**: Generate 10×10 grid
   - **CRITICAL EVALUATION STEP**
   - Each row should be clearly one class
   - Within rows: different appearances of same class

3. Complete **TODO 7**: Generate single class
   - All images same class, different style
   - Verify class control works

4. **Checkpoint:** Evaluation complete 

### Step 6: Analyze (15 minutes)

1. Complete **TODO 8**: Answer analysis questions
   - Based on what you see in grids
   - Write detailed observations
   - Compare with theory

2. **Checkpoint:** Analysis complete 

### Step 7: Extensions (Optional, 15+ minutes)

1. Complete **TODO 9**: Generated vs real comparison
2. Attempt **TODO 10**: CIFAR-100 extension (if time)


## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Import error for cgan | Wrong path | Check sys.path.insert() in code |
| CUDA out of memory | Batch size too large | Reduce batch_size to 32 |
| Very slow training | CPU mode | Move model to GPU: `.to(device)` |
| Blurry images | Underfitting | Train more epochs (50+) |
| Class labels ignored | Poor conditioning | Increase label_dim to 100 |
| Both losses high | Bad initialization | Call initialize_weights() |
| Generated same image | Mode collapse | Reduce D learning rate |

---

## References

- **Original cGAN Paper:** Mirza & Osindero (2014)
- **DCGAN Guidelines:** Radford et al. (2015)
- **CIFAR-10 Dataset:** Krizhevsky (2009)
- **PyTorch Docs:** pytorch.org

---

## Submission Checklist

Before considering this exercise complete:

- [ ] TODO 1: Dataset loads successfully
- [ ] TODO 2: Models created with correct shapes
- [ ] TODO 3: Trainer initialized
- [ ] TODO 4: Model trained for 20 epochs
- [ ] TODO 5: Loss curves plotted and analyzed
- [ ] TODO 6: 10×10 grid generated and shows class control
- [ ] TODO 7: Single-class samples generated
- [ ] TODO 8: Analysis questions answered
- [ ] (Optional) TODO 9: Generated vs real comparison
- [ ] (Bonus) TODO 10: CIFAR-100 extension attempted

---

