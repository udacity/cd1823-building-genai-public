# Solution: Conditional GAN Implementation

---

## Implementation Walkthrough

### TODO 1: Load CIFAR-10 Dataset

**Solution Pattern:**

```python
# Step 1a: Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to [0, 1]
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    # Normalizes to [-1, 1]: (x - 0.5) / 0.5 = 2x - 1
])

# Step 1b: Load dataset
train_dataset = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,  # Auto-download if not present
    transform=transform
)

# Step 1c: Create DataLoader
batch_size = 64
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,  # Critical for SGD
    num_workers=0,  # MPS compatibility
)
```

**Key Points:**
- Normalize to [-1, 1] because Generator uses Tanh (outputs in [-1, 1])
- Shuffling ensures random sampling (helps with convergence)
- num_workers=0 for MPS/Apple Silicon compatibility

**Verification:**
```python
print(f\"Dataset: {len(train_dataset)} images\")
print(f\"Batches: {len(train_loader)}\")
print(f\"Batch size: {batch_size}\")
# Output: Dataset: 50000 images, Batches: 781, Batch size: 64
```

---

### TODO 2: Create cGAN Models

**Solution Pattern:**

```python
latent_dim = 100      # Noise vector size
num_classes = 10      # CIFAR-10 has 10 classes
label_dim = 50        # Embedding dimension for labels

# Factory function creates both
generator, discriminator = create_cgan_models(
    latent_dim=latent_dim,
    num_classes=num_classes,
    label_dim=label_dim,
    num_channels=3,  # RGB
    device=device,
)

# Initialize weights (DCGAN style)
initialize_weights(generator)
initialize_weights(discriminator)

# Verify
print(f\"Generator: {sum(p.numel() for p in generator.parameters()):,} params\")
print(f\"Discriminator: {sum(p.numel() for p in discriminator.parameters()):,} params\")
```

**What's Happening Inside:**

1. **Generator Creation:**
   - Label embedding: 10 classes → 50-dim continuous
   - FC layers: (z + label) → initial feature maps
   - ConvTranspose: Upsampling layers
   - Tanh activation: Output in [-1, 1]

2. **Discriminator Creation:**
   - Conv layers: Downsampling
   - Feature extraction: Image → 1024-dim features
   - Label embedding: Same as Generator
   - FC layers: Concatenate features + embedding → probability
   - Sigmoid activation: Output in [0, 1]

3. **Weight Initialization:**
   - Conv/ConvTranspose: Normal(mean=0, std=0.02)
   - BatchNorm: Normal(mean=1, std=0.02)
   - This is the DCGAN standard
   - Speeds up convergence significantly

---

### TODO 3: Create Trainer

**Solution Pattern:**

```python
trainer = ConditionalGANTrainer(
    generator=generator,
    discriminator=discriminator,
    device=device,
    lr_g=0.0002,    # Generator learning rate
    lr_d=0.0002,    # Discriminator learning rate
    beta1=0.5,      # Adam parameter (not default 0.9)
    beta2=0.999,    # Adam parameter (default)
)
```

**What This Does:**

1. Creates Adam optimizers for both networks
2. Sets up BCE loss function
3. Initializes loss tracking lists
4. Moves models to specified device

**Why These Hyperparameters:**

| Parameter | Value | Reasoning |
|---|---|---|
| lr_g | 0.0002 | DCGAN standard, slower learning = more stable |
| lr_d | 0.0002 | Match G, prevents D overpowering |
| beta1 | 0.5 | DCGAN standard (not default 0.9) |
| beta2 | 0.999 | Standard for GANs, momentum for gradients |

---

### TODO 4: Train the cGAN

**Solution Pattern:**

```python
num_epochs = 20

results = trainer.train(
    train_loader=train_loader,
    num_epochs=num_epochs,
    latent_dim=latent_dim,
    num_classes=num_classes,
    log_interval=50,  # Log every 50 batches
)

print(f\"Training complete after {num_epochs} epochs\")
```

**What Happens Internally:**

```python
# For each epoch:
for epoch in range(num_epochs):
    # For each batch:
    for real_images, real_labels in train_loader:
        # === Discriminator Step ===
        # Want: D(real_img, real_label) → 1.0
        #       D(fake_img, fake_label) → 0.0
        
        # === Generator Step ===
        # Want: D(G(z, target_label), target_label) → 1.0
        # (fool the discriminator)
        
        # Update both networks
```

**Expected Behavior:**

- Loss decreases initially
- D loss stabilizes around 0.5 (random guessing)
- G loss continues decreasing
- Progress printed every 50 batches

**Typical Output:**
```
Epoch 1/20:
  Batch 50: D Loss = 0.8532, G Loss = 2.1234
  Batch 100: D Loss = 0.6234, G Loss = 1.5678
  ...
  Epoch Loss: D = 0.6123, G = 1.2345

Epoch 2/20:
  ...
```

---

### TODO 5: Plot Loss Curves

**How to Interpret:**

**Good Training (What We Want):**
```
D Loss Graph:
- Starts high (~0.7)
- Decreases to ~0.5
- Stays around 0.5 (random guessing)
- Maybe oscillates slightly

G Loss Graph:
- Starts very high (~2.0)
- Steadily decreases
- After epoch 5+: maybe increases again
- Generally trending downward
```

**Bad Training (What We Don't Want):**
```
D Loss → 0: Discriminator overpowering
  Solution: Reduce D learning rate

Both losses → ∞: Exploding gradients
  Solution: Use smaller learning rates

D Loss oscillating wildly: Instability
  Solution: Add batch norm, reduce LR
```

---

### TODO 6: Generate 10×10 Class Grid


**Critical Analysis: What to Look For**

 **Strong Disentanglement (Good):**
- Row 0 (airplanes): All clearly airplanes
- Row 5 (dogs): All clearly dogs
- Different styles within rows (due to noise z)
- Columns show similar pose/style but different class

 **Weak Disentanglement (Bad):**
- All rows look similar
- Can't tell which class is which
- No variation within rows
- Class labels seem to have no effect

**Grid Structure:**
```
        Col0    Col1    Col2  ...  Col9
       (z1)    (z2)    (z3)      (z10)
Row0  [dog]  [dog]  [dog]  ...  [dog]    <- Same noise, all dogs
Row1  [cat]  [cat]  [cat]  ...  [cat]    <- Same noise, all cats
Row2  [bird] [bird] [bird] ...  [bird]   <- Same noise, all birds
...
Row9  [ship] [ship] [ship] ...  [ship]   <- Same noise, all ships
```

---

### TODO 7: Generate Single-Class Samples

**Solution Pattern:**

```python
target_class = 5  # dogs

class_samples = trainer.generate_class_samples(
    target_class=target_class,
    num_samples=16,
    latent_dim=latent_dim,
)

# Denormalize
class_samples_cpu = (class_samples.cpu() + 1) / 2
class_samples_cpu = torch.clamp(class_samples_cpu, 0, 1)

# Visualize as 2x8 grid


**What This Shows:**

- All 16 images: Same class (dogs)
- Different noise vectors z
- Results: Different dog appearances (pose, color, orientation)
- Validates that:
  - Class labels work (all are dogs)
  - Noise provides diversity (not identical images)
  - Generator learned class concept

---

### TODO 8: Class Disentanglement Analysis

**Solution Template:**

```python
print(\"\"\"
DISENTANGLEMENT ANALYSIS:

1. WITHIN-ROW VARIATION (Same class, different noise):
   ✓ Observation: Each row shows varied appearances
   ✓ Example: Dogs in row 5 have different poses, colors
   ✓ Cause: Different noise vectors z
   ✓ Meaning: Generator uses noise for style/appearance

2. ACROSS-ROW DIFFERENCES (Different classes):
   ✓ Observation: Rows are clearly distinct
   ✓ Example: Dogs (row 5) vs cats (row 3) obvious
   ✓ Cause: Different class labels y
   ✓ Meaning: Class labels strongly condition generation

3. CLASS DISENTANGLEMENT SUCCESS:
   ✓ Classes are clearly separated
   ✓ No confusion between classes
   ✓ Same z maintains consistency across classes
   ✓ Class identity is well-learned

4. QUALITY ASSESSMENT:
   ✓ Recognizable objects
   ✓ Correct colors for most classes
   ✓ Some blurriness (expected after 20 epochs)
   ✓ Classes: airplane, automobile, bird (easier)
   ✗ Classes: cat, horse, dog (harder, more variation)

5. DATA AUGMENTATION SUITABILITY:
   ✓ Images realistic enough for augmentation
   ✓ Class labels respected
   ✓ Would help with imbalanced datasets
   ✓ Could improve classifier performance
\"\"\")
```

**Scoring Rubric:**

| Aspect | Excellent | Good | Fair | Poor |
|--------|-----------|------|------|------|
| Class Clarity | Classes unmistakable | Clear distinction | Some confusion | Indistinguishable |
| Within-Class Diversity | High variation | Good variation | Limited variety | No variation |
| Image Quality | Sharp & detailed | Mostly clear | Somewhat blurry | Very blurry |
| Consistency | Always correct class | Rarely wrong | Often wrong | Never respects labels |
| Usability | Ready for augmentation | Usable with caution | Limited use | Not suitable |

---

### TODO 9: Generated vs Real Comparison

**Solution Pattern:**

```python
fig, axes = plt.subplots(3, 10, figsize=(18, 6))

for class_id in range(10):
    # Step 1: Generate one image
    generated = trainer.generate_class_samples(target_class=class_id, num_samples=1, latent_dim=latent_dim)
    generated_denorm = (generated.cpu() + 1) / 2
    generated_denorm = torch.clamp(generated_denorm, 0, 1)
    
    # Step 2: Find one real image
    idx = np.where(np.array(train_dataset.targets) == class_id)[0][0]
    real_img = (train_dataset[idx][0] + 1) / 2
    
    # Step 3: Display generated
    ax = axes[0, class_id]
    ax.imshow(generated_denorm[0].permute(1, 2, 0).numpy())
    ax.set_title(f'{CIFAR10_CLASSES[class_id]}', fontsize=9)
    ax.axis('off')
    if class_id == 0:
        ax.text(-0.3, 0.5, 'Generated', transform=ax.transAxes, fontsize=10, fontweight='bold', va='center')
    
    # Step 4: Display real
    ax = axes[1, class_id]
    ax.imshow(real_img.permute(1, 2, 0).numpy())
    ax.axis('off')
    if class_id == 0:
        ax.text(-0.3, 0.5, 'Real', transform=ax.transAxes, fontsize=10, fontweight='bold', va='center')
    
    # Step 5: Assessment
    ax = axes[2, class_id]
    ax.text(0.5, 0.5, f'Quality:\
Good✓', ha='center', va='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax.axis('off')
    if class_id == 0:
        ax.text(-0.3, 0.5, 'Assessment', transform=ax.transAxes, fontsize=10, fontweight='bold', va='center')

plt.suptitle('Generated vs Real Images (One per Class)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
```

**Comparison Insights:**

**Easy Classes (Good Match):**
- Airplane: Distinct shape, colors
- Car: Recognizable structure
- Ship: Large object, clear boundaries

**Hard Classes (More Blurry):**
- Cat/Dog: Complex textures, details
- Bird: Small, intricate features
- Horse: Complex structure

---

### TODO 10: CIFAR-100 Extension (Bonus)

**Solution Pattern:**

```python
# Step 1: Load CIFAR-100
transform_100 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

train_dataset_100 = datasets.CIFAR100(
    root='./data100',
    train=True,
    download=True,
    transform=transform_100
)

train_loader_100 = DataLoader(
    train_dataset_100,
    batch_size=64,
    shuffle=True,
    num_workers=0,
)

# Step 2: Update num_classes and label_dim
num_classes_100 = 100
label_dim_100 = 100  # Larger for more classes

# Step 3: Create models
generator_100, discriminator_100 = create_cgan_models(
    latent_dim=100,
    num_classes=num_classes_100,
    label_dim=label_dim_100,
    num_channels=3,
    device=device,
)

initialize_weights(generator_100)
initialize_weights(discriminator_100)

# Step 4: Create trainer and train (fewer epochs due to more classes)
trainer_100 = ConditionalGANTrainer(
    generator_100,
    discriminator_100,
    device=device,
    lr_g=0.0002,
    lr_d=0.0002,
)

results_100 = trainer_100.train(
    train_loader_100,
    num_epochs=10,  # Fewer due to more classes
    latent_dim=100,
    num_classes=num_classes_100,
)

# Step 5: Generate and analyze
grid_100 = trainer_100.generate_all_classes_grid(100, 10, 100)
# Display first 10 classes
```

**Why This Is Challenging:**

1. **More classes = harder learning**
   - Generator must learn 100 concepts
   - Takes more training time
   - Might need more capacity (bigger networks)

2. **Similar classes confuse the model**
   - Different dog breeds in separate classes
   - Different cat breeds in separate classes
   - Model must distinguish finer details

3. **Trade-offs:**
   - Fewer epochs for same time
   - Quality might be lower
   - But scalability insights valuable

**Expected Results:**
- After 10 epochs: Some class separation visible
- After 50 epochs: Much better quality
- Classes still somewhat blurry (harder problem)
- Worth exploring!

---

## Common Implementation Details

### Device Handling

```python
# Auto-detect best device
if torch.backends.mps.is_available():
    device = torch.device('mps')  # Apple Silicon
elif torch.cuda.is_available():
    device = torch.device('cuda')  # NVIDIA GPU
else:
    device = torch.device('cpu')   # Fallback
```

### Tensor Denormalization

```python
# Convert from [-1, 1] (generator output) to [0, 1] (for display)
denorm = (tensor + 1) / 2
denorm = torch.clamp(denorm, 0, 1)  # Safety clamp
```

### Label Handling

```python
# Convert class indices to embeddings
labels = torch.tensor([0, 5, 3, ...])  # Shape: (batch,)
# Inside model: embedded_labels = embedding(labels)  # Shape: (batch, label_dim)
```

---

## Key Differences from Solution to Theory

| Concept | Theory | Implementation | Why |
|---------|--------|---|---|
| Label Input | One-hot | Integer indices | More efficient, embedding learns relationships |
| Concatenation | Documented | Early (G), Late (D) | Proven to work better |
| Initialization | DCGAN std | Normal(0, 0.02) | Matches DCGAN paper |
| Loss Function | BCE | Binary cross-entropy | Standard for GANs |

---

## Debugging Tips

**If your code doesn't match:**

1. **Check tensor shapes**
   ```python
   print(z.shape)  # Should be (batch, latent_dim)
   print(labels.shape)  # Should be (batch,)
   print(fake_images.shape)  # Should be (batch, 3, 32, 32)
   ```

2. **Check device consistency**
   ```python
   print(z.device)  # Should match generator.device
   ```

3. **Verify loss computation**
   ```python
   d_loss_val = criterion(d_output, target)
   print(f\"D loss: {d_loss_val.item():.4f}\")  # Should be finite
   ```

4. **Monitor training progress**
   ```python
   print(f\"Epoch {e}, D: {d_loss:.4f}, G: {g_loss:.4f}\")
   ```

