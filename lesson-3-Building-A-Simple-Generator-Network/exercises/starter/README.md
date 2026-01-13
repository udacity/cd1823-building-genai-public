# Starter: Generate and Visualize MNIST-like Images

This folder contains the **starter template** for the exercise.

## What You'll Do

Complete all the **TODO** markers in these files:

1. **`exercise_notebook.ipynb`** - Interactive notebook version 
   - Sample noise vectors
   - Generate images
   - Reshape and visualize
   - Step-by-step guidance



## Structure

- `exercise_notebook.ipynb` - Interactive notebook with TODOs 
- `generator_model.py` - Pre-built generator (do not edit)

## Getting Started

### Interactive Notebook (Recommended)
```bash
jupyter notebook exercise_notebook.ipynb
```



## TODO Hints

1. **Sample Noise**: Use `torch.randn(16, latent_dim).to(device)`
2. **Generate**: Wrap in `torch.no_grad()`, call `generator(noise)`
3. **Reshape**: Use `.view(-1, 1, 28, 28)`
4. **Denormalize**: Apply `(tensor + 1) / 2`
5. **Grid**: Use `vutils.make_grid(images, nrow=4)`
6. **Visualize**: Permute, convert to numpy, use `plt.imshow()`

## Expected Output

When complete, you'll see:
- 16 generated images displayed in a 4×4 grid
- Console output showing tensor shapes at each step
-  Message confirming success

## Common Issues

- **Tensor on wrong device**: Add `.to(device)`
- **Wrong shape after reshape**: Use `.view(-1, 1, 28, 28)`
- **Pixel values out of range**: Remember to denormalize
- **ImportError**: Make sure `generator_model.py` is in the same directory

## Check Your Work

- Does your output shape match the hints?
- Are pixel values in the right range?
- Does the visualization display correctly?

--