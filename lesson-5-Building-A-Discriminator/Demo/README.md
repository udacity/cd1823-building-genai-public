# Module 5 Demo: Building a Discriminator

## Overview

This demo introduces the **discriminator**, the second key component of a Generative Adversarial Network (GAN). While the generator creates fake images, the discriminator learns to distinguish real images from fake ones through binary classification.

**Key Learning Objectives:**
- Understand the discriminator architecture (fully-connected layers with LeakyReLU)
- Learn why certain activation functions work well for GANs
- See how Binary Cross-Entropy loss measures discrimination ability
- Understand the adversarial training dynamic (discriminator vs generator)



## How to Run

### Run the Demo Notebook
```bash
cd lesson-5-Building-A-Discriminator/Demo
jupyter notebook demo_notebook.ipynb
```

Then execute all cells in order.


## Key Concepts

### Binary Classification
The discriminator solves a **binary classification problem**:
- **Input**: Image (either real or fake)
- **Output**: Probability that image is real (0 = definitely fake, 1 = definitely real)

### Loss Function: Binary Cross-Entropy (BCE)
$$\text{BCE Loss} = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]$$

Where:
- $y_i$ = true label (1 for real, 0 for fake)
- $\hat{y}_i$ = predicted probability
- Loss is **high** when predictions are wrong
- Loss is **low** when predictions are confident and correct

### Untrained Discriminator Behavior
When randomly initialized:
- Makes ~50% accuracy (random guessing)
- Loss is ~0.693 (close to log(2), maximum for random binary classification)
- Produces predictions centered around 0.5

### Adversarial Training Dynamic
In a full GAN training loop:
1. **Discriminator update**: Learns to minimize loss on real/fake classification
2. **Generator update**: Learns to minimize discriminator loss on fake images
3. **Result**: Both networks improve iteratively

## Bridging to the Project

The project (`project/model/cgan.py`) uses a **conditional discriminator** that also receives class labels (0-9 for MNIST digits). This demo teaches the fundamentals:
- **Binary classification baseline** (this demo)
- **Label concatenation extension** (in project)

The discriminator in the project:
```python
# In project/model/cgan.py:
# Takes (image, label) → embedds label → concatenates with flattened image → discriminates
```

## Common Questions

**Q: Why LeakyReLU(0.2) instead of regular ReLU?**  
A: LeakyReLU allows small negative gradients, preventing neurons from "dying" (outputting zero) during adversarial training. Regular ReLU can cause training instability in GANs.

**Q: Why Sigmoid for output?**  
A: Sigmoid ensures output is in [0, 1], perfect for probability interpretation. It's the natural choice for binary classification.

**Q: What's the significance of 0.693 loss?**  
A: This is log(2) ≈ 0.693, the loss when a binary classifier randomly guesses (50/50). It's the "worst possible" discriminator.

**Q: How does this relate to real GAN training?**  
A: In practice, discriminator and generator train together in an alternating loop. The discriminator drives the generator to improve by providing meaningful gradients.

## Next Steps

1. Complete the **exercises** in `lesson-4-Building-A-Discriminator/exercises/`
2. Implement `create_mixed_batch()` and compute loss yourself
3. Explore the **conditional discriminator** in `project/model/cgan.py`
4. Train the full cGAN in `project/01_cGAN_training.ipynb`

## Resources

- PyTorch Binary Cross-Entropy: https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
- Goodfellow et al. "Generative Adversarial Nets": https://arxiv.org/abs/1406.2661
- Conditional GAN (cGAN): https://arxiv.org/abs/1411.1784
