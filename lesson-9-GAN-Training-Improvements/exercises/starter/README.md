# Module 9: GAN Training Improvements - Starter Exercises

## Objective

Train three GAN variants with different stabilization techniques and empirically determine which approach works best for ensuring stable training.

## What You'll Do

1. **Load Fashion MNIST dataset** with proper transforms
2. **Create three trainer instances**:
   - BaselineGAN (no modifications)
   - LabelSmoothingGAN (soft labels)
   - FeatureMatchingGAN (feature matching loss)
3. **Train all three for 50 epochs** on the same dataset
4. **Measure and compare stability** using multiple metrics
5. **Analyze results** to answer key questions about GAN stability


## Files


- `exercise_notebook.ipynb`: Interactive notebook version (fill in TODOs)

## Getting Started

Open the notebook in Jupyter:

```bash
jupyter notebook exercise_notebook.ipynb
```


## Exercise Structure

### Part 1: Setup
- Set device (MPS/CUDA/CPU)
- Set random seed

### Part 2: Load Data
- Create Fashion MNIST transforms
- Create dataset and dataloader

### Part 3: Create Trainer 
- Initialize ComparisonTrainer

### Part 4: Train Models 
- Call `train_all_variants()`
- Trains 3 models × 50 epochs

### Part 5: Visualize
- Plot loss curves
- Create stability comparison charts

### Part 6: Analyze 
- Compute metrics
- Print reports
- Answer analysis questions

## TODOs to Complete

### Beginner Level
- [ ] Import all libraries
- [ ] Set device and random seed
- [ ] Create transforms for Fashion MNIST
- [ ] Load dataset and dataloader

### Intermediate Level
- [ ] Create ComparisonTrainer
- [ ] Call train_all_variants()
- [ ] Plot loss curves

### Advanced Level
- [ ] Compute stability metrics
- [ ] Create comparison visualizations