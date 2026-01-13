# Lesson 3 Demos: Building a Simple Generator Network

This folder contains demo notebooks and code showcasing core concepts for building generative models.

## Contents

### `demo_notebook.ipynb`
A comprehensive walkthrough of building a basic generator for MNIST:
- Loads MNIST and visualizes real images
- Designs a simple 3-layer fully-connected generator
- Uses LeakyReLU and Tanh activations appropriately
- Samples noise vectors from standard Gaussian
- Generates and visualizes the output
- Compares real vs generated images

**Key Learning**: Understanding the basic architecture and how noise maps to image space.

### `generator_model.py`
A reusable `BasicGenerator` class with detailed docstrings explaining:
- Why we use specific activation functions
- What each layer does
- How to instantiate and use the generator


## Key Concepts

- **Latent Space**: We sample from a simple 100-dimensional Gaussian (noise)
- **Image Space**: The generator maps to 784-dimensional space (28×28 MNIST images)
- **Activation Functions**:
  - LeakyReLU in hidden layers prevents dead neurons
  - Tanh in output layer constrains values to [-1, 1] (normalized image range)

