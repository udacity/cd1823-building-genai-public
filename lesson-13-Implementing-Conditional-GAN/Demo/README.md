## Quick Start

### Running the Demo Notebook

```bash
jupyter notebook demo_notebook.ipynb
```

This will walk you through:
1. Loading CIFAR-10 dataset
2. Building the cGAN architecture
3. Training the model (20 epochs)
4. Visualizing results with 10×10 class grids
5. Analyzing class disentanglement

---

## Files Description

### `cgan.py` 
Core model implementations for Conditional GAN.

#### Classes

**`ConditionalGenerator`**
```python
class ConditionalGenerator(nn.Module):

    Generates 32x32 RGB images conditioned on class labels.
    
    Architecture:
    - Input: noise z (100-dim) + class label y (10 classes)
    - Label embedding: one-hot → continuous (50-dim)
    - Concatenation: [z + embedded_label] → (150-dim)
    - FC layers + ConvTranspose2D upsampling
    - Output: Image (3, 32, 32) with Tanh activation
    
    Key Design: Early concatenation (at FC input) for strong class influence

    def __init__(self, latent_dim, num_classes, label_dim, num_channels=3):
        # Creates label embedding and FC layers
        # Builds ConvTranspose2D blocks for upsampling
        
    def forward(self, z, labels):
        # z: (batch, latent_dim)
        # labels: (batch,) with class indices 0-9
        # Returns: (batch, 3, 32, 32) image in [-1, 1] range
```

**`ConditionalDiscriminator`**
```python
class ConditionalDiscriminator(nn.Module):

    Classifies images as real/fake and verifies class consistency.
    
    Architecture:
    - Input: image (3, 32, 32) + class label y (10 classes)
    - Conv2D downsampling: 32→16→8→4 spatial dims
    - Flatten features: (256, 2, 2) → (1024,)
    - Label embedding: one-hot → continuous (50-dim)
    - Concatenation: [flattened_features + embedded_label] → (1074-dim)
    - FC layers with LeakyReLU
    - Output: Probability [0, 1] with Sigmoid
    
    Key Design: Late concatenation (after features) to preserve image info

    def __init__(self, num_classes, label_dim, num_channels=3):
        # Creates Conv2D blocks for downsampling
        # Creates label embedding
        # Creates FC layers
        
    def forward(self, x, labels):
        # x: (batch, 3, 32, 32) image
        # labels: (batch,) with class indices 0-9
        # Returns: (batch, 1) probability in [0, 1]
```

#### Functions

**`create_cgan_models()`**
```python
def create_cgan_models(latent_dim=100, num_classes=10, label_dim=50, 
                       num_channels=3, device='cpu'):

    Factory function to create both generator and discriminator.
    
    Args:
        latent_dim: Dimension of noise vector (typically 100)
        num_classes: Number of classes (10 for CIFAR-10)
        label_dim: Dimension of class embedding (typically 50)
        num_channels: Number of image channels (3 for RGB)
        device: 'cpu', 'cuda', or 'mps'
    
    Returns:
        (generator, discriminator): Both on specified device

```

**`initialize_weights()`**
```python
def initialize_weights(model):

    DCGAN-style weight initialization.
    - Conv/ConvTranspose: Normal with mean=0, std=0.02
    - BatchNorm: Normal with mean=1, std=0.02
    
    Improves training stability and convergence speed.

```

---

### `cgan_training.py` (

High-level training interface for Conditional GANs.

#### Classes

**`ConditionalGANTrainer`**
```python
class ConditionalGANTrainer(nn.Module):

    Manages training loop, loss computation, and image generation.

    def __init__(self, generator, discriminator, device='cpu', 
                 lr_g=0.0002, lr_d=0.0002, beta1=0.5, beta2=0.999):
    
        Args:
            generator: ConditionalGenerator instance
            discriminator: ConditionalDiscriminator instance
            device: 'cpu', 'cuda', or 'mps'
            lr_g: Generator learning rate
            lr_d: Discriminator learning rate
            beta1, beta2: Adam optimizer parameters
    
        # Creates Adam optimizers
        # Sets up BCE loss
        # Initializes loss tracking
    
    def train_step(self, real_images, real_labels):
    
        Single training step (one batch).
        
        Args:
            real_images: (batch, 3, 32, 32) real images
            real_labels: (batch,) true class labels
        
        Returns:
            d_loss, g_loss: Scalar losses for this batch
        
        Process:
        1. Discriminator step:
           - Compute loss on real images with correct labels
           - Generate fake images with random labels
           - Compute loss on fake images
           - Backprop and update D
        
        2. Generator step:
           - Generate fake images with random target labels
           - Try to fool discriminator
           - Backprop and update G
    
    
    def train_epoch(self, dataloader, latent_dim, num_classes):
    
        Train for one epoch (all batches).
        
        Args:
            dataloader: PyTorch DataLoader
            latent_dim: Noise vector dimension
            num_classes: Number of classes
        
        Returns:
            epoch_d_loss, epoch_g_loss: Average losses for epoch
    
    
    def train(self, train_loader, num_epochs, latent_dim, num_classes, log_interval=50):
    
        Full training loop.
        
        Args:
            train_loader: PyTorch DataLoader
            num_epochs: Number of epochs to train
            latent_dim: Noise vector dimension
            num_classes: Number of classes
            log_interval: Log every N batches
        
        Returns:
            results: Dict with 'd_losses' and 'g_losses' lists
    
    
    def generate_class_samples(self, target_class, num_samples, latent_dim):
    
        Generate N images of a specific class.
        
        Args:
            target_class: Class index (0-9)
            num_samples: Number of images to generate
            latent_dim: Noise vector dimension
        
        Returns:
            images: (num_samples, 3, 32, 32) tensor in [-1, 1]
        
        Example:
            # Generate 16 dogs (class 5)
            dogs = trainer.generate_class_samples(5, 16, 100)
    
    
    def generate_all_classes_grid(self, num_classes=10, samples_per_class=10, 
                                   latent_dim=100, shared_z=None):
    
        Generate 10×10 grid showing class disentanglement.
        
        Args:
            num_classes: Number of classes (10)
            samples_per_class: Samples per class (10)
            latent_dim: Noise vector dimension
            shared_z: Use same noise for all classes (None = random)
        
        Returns:
            images: (100, 3, 32, 32) tensor in [-1, 1]
        
        Grid Layout:
            Rows = classes (0-9)
            Columns = different noise samples
            Same z, different y → shows class effect
        
        Example:
            grid = trainer.generate_all_classes_grid(10, 10, 100)
            # 10 rows × 10 columns = 100 images
            # Each row: same class, different style
            # Each column: different class, same noise
    
```

#### Key Training Concepts

**Conditional Training Loop**

Unlike standard GANs, cGANs require careful label handling:

1. **Discriminator:**
   - Real branch: Real images with correct labels → 1
   - Fake branch: Generated images with random labels → 0
   - Labels must match image reality (real dogs are real, fake anything is fake)

2. **Generator:**
   - Generate with random target labels
   - Try to fool discriminator
   - Discriminator will verify both realism AND class consistency

**Loss Functions**

Both use Binary Cross-Entropy:
```python
BCE_loss = nn.BCELoss()

# Discriminator
d_loss = BCE(D(real_img, real_label), 1.0) + \\
         BCE(D(fake_img, fake_label), 0.0)

# Generator  
g_loss = BCE(D(G(z, target_label), target_label), 1.0)
```

---


## Understanding Conditional Architectures

### Early vs Late Concatenation

**Generator: EARLY concatenation at FC layer**

```
Why:
- Class needs to influence generation from the start
- Early concatenation = class \"seed\" for entire upsampling process
- If you wait, the noise features have already formed
- Like telling a painter the subject before they start
```

**Discriminator: LATE concatenation after Conv blocks**

```
Why:
- First extract image features without class bias
- Then verify features match the claimed class
- Early concatenation would bias feature extraction
- Like a judge first examines evidence, then checks against claims
```

### Label Embedding Dimension

Why 50?

- Too small (5-10): Not enough expressiveness
  - Can't capture subtle class differences
  - Model underfits

- Too large (200+): Overfitting risk
  - Uses too many parameters
  - Less generalization

- Goldilocks zone (50): Balance
  - CIFAR-10 has 10 classes
  - 50-dim allows 5× expansion (good for learning)
  - Standard in literature

---

## Class Disentanglement Explained

### What is Disentanglement?

**In GANs:** Separating **class identity** from **style/appearance**

**Example with dogs:**
- \"Dog-ness\" = what makes it recognizable as a dog
- \"Style\" = pose, color, lighting, age, etc.

**Good disentanglement:** Change class, style stays similar
```
Same noise z with different labels y:
- (z, dog) → brown dog sitting
- (z, cat) → brown cat sitting (same pose due to z)
- (z, car) → brown car pointing same direction
```

**Poor disentanglement:** Class has no effect
```
Same noise z with different labels y:
- (z, dog) → random image
- (z, cat) → similar random image (labels ignored)
- (z, car) → similar random image (labels ignored)
```

### How to Evaluate Disentanglement

**Visual Inspection (Quick)**
1. Look at 10×10 grid
2. Do rows look like different classes? ✓
3. Does each row have style variation? ✓
4. Are columns coherent (similar style)? ✓

**Quantitative Metrics (Rigorous)**
1. **FID (Fréchet Inception Distance):** How realistic?
   - FID < 50: Good
   - FID < 30: Very good
   - FID < 10: Excellent

2. **Classification Accuracy:** Train classifier on generated images
   - Accuracy > 90%: Strong class control
   - Accuracy < 70%: Weak class control

3. **Per-Class Metrics:** Separate FID for each class
   - Similar across classes: Good balance
   - High variance: Some classes harder than others

---



## Key Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `latent_dim` | 100 | Size of noise vector z |
| `num_classes` | 10 | CIFAR-10 has 10 classes |
| `label_dim` | 50 | Embedding dimension for class labels |
| `batch_size` | 64 | Balance between gradient and speed |
| `lr_g` | 0.0002 | Generator learning rate |
| `lr_d` | 0.0002 | Discriminator learning rate |
| `beta1` | 0.5 | Adam exponential decay rate 1 |
| `beta2` | 0.999 | Adam exponential decay rate 2 |
| `num_epochs` | 20 | Number of training epochs |

---

## Extensions

### Using Generated Images for Augmentation

```python
# Generate synthetic dogs
synth_dogs = trainer.generate_class_samples(target_class=5, num_samples=1000, latent_dim=100)

# Add to training dataset
augmented_dataset = torch.cat([original_dataset, synth_dogs])

# Train classifier on augmented data
classifier.train(augmented_dataset)
```

### Extending to CIFAR-100

```python
# Change parameters
num_classes = 100
label_dim = 100  # Larger embedding for more classes

# Rest of code stays the same
generator, discriminator = create_cgan_models(
    num_classes=100,
    label_dim=100,
    ...
)
```

### Computing FID Metric

```python
from scipy.linalg import sqrtm

def compute_fid(real_images, fake_images):
Fréchet Inception Distance between real and fake.\"\"\"
    # Extract Inception features from both sets
    # Compute mean and covariance
    # Return FID score
    
fid_score = compute_fid(real_cifar10, generated_images)
print(f\"FID: {fid_score:.2f}\")  # Lower is better
```

---

## References

- **cGAN Paper:** Mirza & Osindero (2014) - \"Conditional Generative Adversarial Nets\"
- **DCGAN:** Radford et al. (2015) - Architecture guidelines
- **CIFAR-10:** Krizhevsky (2009) - Dataset description

