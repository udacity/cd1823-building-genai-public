# Solution: Generate and Visualize MNIST-like Images

This folder contains the **complete solution** for the exercise.

## What to Do

1. Compare your implementation with `exercise_solution.ipynb`
2. Check if you:
   - Used `torch.randn()` correctly
   - Reshaped tensors properly
   - Denormalized correctly
   - Used `make_grid()` appropriately
3. Run this solution to see the expected output
4. Debug any differences in your implementation

## Key Points

- **Sample noise**: `torch.randn(16, 100).to(device)`
- **Reshape**: `generated_images.view(-1, 1, 28, 28)`
- **Denormalize**: `(tensor + 1) / 2`
- **Grid**: `vutils.make_grid(images, nrow=4)`
- **Display**: Permute to (H, W, C), convert to numpy, use imshow

## Running the Solution

Open and run the notebook:
```bash
jupyter notebook exercise_solution.ipynb
```

## Next Steps After Solving

Once you understand this exercise:
1. Move on to understanding **Discriminator** architecture
2. Implement the adversarial training loop
3. Train your first GAN!