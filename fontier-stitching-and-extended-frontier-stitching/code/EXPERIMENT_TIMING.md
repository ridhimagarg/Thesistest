# Complete Experiment Timing Estimates

## Overview

This document provides timing estimates for the complete watermarking experiment pipeline.

---

## Complete Experiment Pipeline

### Step 1: Train Original Model
- **Epochs**: 50 (default config)
- **Dataset**: CIFAR10 (or other)
- **Model**: CIFAR10_BASE_2 (~500K-1M parameters)

### Step 2: Generate Adversarial Examples (Watermarks)
- **eps_values**: [0.01, 0.015, 0.02] (3 values)
- **sample_sizes**: [10000]
- **attack_types**: ["fgsm"]

### Step 3: Watermark the Model (Fine-tuning)
- **finetuning_epochs**: 15
- **Method**: Fine-tuning on adversarial examples

### Step 4: Model Extraction Attack
- **query_budgets**: [250, 500, 1000, 5000, 10000, 20000] (6 values)
- **epochs_extract**: 50 per query budget

---

## Timing Estimates by Dataset

### CIFAR10 (32×32×3, 10 classes)

| Step | Description | Time (GPU) | Time (CPU) |
|------|-------------|------------|------------|
| **1. Train Original** | 50 epochs | ~10-15 min | ~60-90 min |
| **2. Generate Adversarial** | 3 eps values, 10K samples | ~15-30 min | ~60-120 min |
| **3. Watermark (Fine-tune)** | 15 epochs | ~3-5 min | ~15-25 min |
| **4. Attack** | 6 query budgets, 50 epochs each | ~60-90 min | ~4-6 hours |
| **TOTAL** | Complete experiment | **~1.5-2.5 hours** | **~6-9 hours** |

### MNIST (28×28×1, 10 classes)

| Step | Description | Time (GPU) | Time (CPU) |
|------|-------------|------------|------------|
| **1. Train Original** | 50 epochs | ~2-3 min | ~10-15 min |
| **2. Generate Adversarial** | 3 eps values, 10K samples | ~5-10 min | ~20-40 min |
| **3. Watermark (Fine-tune)** | 15 epochs | ~1-2 min | ~5-10 min |
| **4. Attack** | 6 query budgets, 50 epochs each | ~15-25 min | ~1-2 hours |
| **TOTAL** | Complete experiment | **~30-60 min** | **~2-3 hours** |

### CIFAR100 (32×32×3, 100 classes)

| Step | Description | Time (GPU) | Time (CPU) |
|------|-------------|------------|------------|
| **1. Train Original** | 50 epochs | ~12-18 min | ~70-100 min |
| **2. Generate Adversarial** | 3 eps values, 10K samples | ~18-35 min | ~70-130 min |
| **3. Watermark (Fine-tune)** | 15 epochs | ~4-6 min | ~18-28 min |
| **4. Attack** | 6 query budgets, 50 epochs each | ~70-100 min | ~5-7 hours |
| **TOTAL** | Complete experiment | **~2-3 hours** | **~7-10 hours** |

### STL10 (96×96×3, 10 classes)

| Step | Description | Time (GPU) | Time (CPU) |
|------|-------------|------------|------------|
| **1. Train Original** | 50 epochs | ~20-30 min | ~2-3 hours |
| **2. Generate Adversarial** | 3 eps values, 10K samples | ~30-60 min | ~2-4 hours |
| **3. Watermark (Fine-tune)** | 15 epochs | ~6-10 min | ~30-50 min |
| **4. Attack** | 6 query budgets, 50 epochs each | ~2-3 hours | ~8-12 hours |
| **TOTAL** | Complete experiment | **~3-4 hours** | **~12-19 hours** |

### EuroSAT (64×64×3, 10 classes)

| Step | Description | Time (GPU) | Time (CPU) |
|------|-------------|------------|------------|
| **1. Train Original** | 50 epochs | ~15-25 min | ~90-120 min |
| **2. Generate Adversarial** | 3 eps values, 10K samples | ~20-40 min | ~80-150 min |
| **3. Watermark (Fine-tune)** | 15 epochs | ~4-7 min | ~20-35 min |
| **4. Attack** | 6 query budgets, 50 epochs each | ~90-120 min | ~6-8 hours |
| **TOTAL** | Complete experiment | **~2.5-3.5 hours** | **~8-11 hours** |

---

## Factors Affecting Timing

### 1. **Hardware**
- **GPU (CUDA/Metal)**: 5-10x faster than CPU
- **CPU**: Slower but works for smaller datasets
- **Mixed Precision**: Can speed up by ~20-30% on GPU

### 2. **Configuration**
- **Epochs**: More epochs = longer training
- **Batch Size**: Larger batch = faster (if memory allows)
- **Query Budgets**: More query budgets = longer attack phase
- **Epsilon Values**: More eps values = longer adversarial generation

### 3. **Dataset Size**
- **Image Size**: Larger images = slower processing
- **Number of Classes**: More classes = slightly slower
- **Dataset Size**: Larger datasets = longer training

---

## Optimized Configurations for Faster Experiments

### Quick Test (Fast Iteration)
```yaml
training:
  epochs: 20  # Reduced from 50

watermark:
  eps_values: [0.01]  # Single epsilon
  sample_sizes: [5000]  # Smaller sample size
  finetuning_epochs: 10  # Reduced from 15

attack:
  query_budgets: [1000, 5000]  # Fewer query budgets
  epochs_extract: 30  # Reduced from 50
```
**Estimated Time**: ~30-45 min (GPU) for CIFAR10

### Standard Experiment (Recommended)
```yaml
training:
  epochs: 50

watermark:
  eps_values: [0.01, 0.015, 0.02]
  sample_sizes: [10000]
  finetuning_epochs: 15

attack:
  query_budgets: [250, 500, 1000, 5000, 10000, 20000]
  epochs_extract: 50
```
**Estimated Time**: ~1.5-2.5 hours (GPU) for CIFAR10

### Comprehensive Experiment (Full Results)
```yaml
training:
  epochs: 100  # More epochs for better accuracy

watermark:
  eps_values: [0.01, 0.015, 0.02, 0.025, 0.03]  # More eps values
  sample_sizes: [5000, 10000, 20000]  # Multiple sample sizes
  finetuning_epochs: 20  # More epochs

attack:
  query_budgets: [250, 500, 1000, 2500, 5000, 10000, 20000, 50000]  # More budgets
  epochs_extract: 100  # More epochs
```
**Estimated Time**: ~4-6 hours (GPU) for CIFAR10

---

## Running Multiple Datasets

If running all 5 RGB datasets sequentially:

| Dataset | Time (GPU) | Time (CPU) |
|---------|------------|------------|
| CIFAR10 | ~1.5-2.5 hours | ~6-9 hours |
| CIFAR100 | ~2-3 hours | ~7-10 hours |
| SVHN | ~1.5-2.5 hours | ~6-9 hours |
| STL10 | ~3-4 hours | ~12-19 hours |
| EuroSAT | ~2.5-3.5 hours | ~8-11 hours |
| **TOTAL** | **~10-15 hours** | **~40-60 hours** |

**Recommendation**: Run in parallel or overnight!

---

## Tips for Faster Experiments

1. **Use GPU**: 5-10x speedup
2. **Enable Mixed Precision**: 20-30% faster on GPU
3. **Reduce Epochs**: For quick tests, use 20-30 epochs
4. **Fewer Query Budgets**: Test with [1000, 5000, 10000] instead of all 6
5. **Single Epsilon**: Test with one epsilon value first
6. **Skip Steps**: Use `--skip-training` if model already exists
7. **Parallel Processing**: Enable `parallel_processing: true` for adversarial generation

---

## Example: Quick Test Run

```bash
# Quick test with reduced configuration
python run_complete_experiment.py \
  --config configs/quick_test.yaml \
  --method finetuning
```

**Expected Time**: ~30-45 minutes (GPU) for CIFAR10

---

## Summary

| Dataset | Complete Experiment (GPU) | Complete Experiment (CPU) |
|---------|---------------------------|---------------------------|
| **MNIST** | ~30-60 min | ~2-3 hours |
| **CIFAR10** | ~1.5-2.5 hours | ~6-9 hours |
| **CIFAR100** | ~2-3 hours | ~7-10 hours |
| **SVHN** | ~1.5-2.5 hours | ~6-9 hours |
| **STL10** | ~3-4 hours | ~12-19 hours |
| **EuroSAT** | ~2.5-3.5 hours | ~8-11 hours |

**All 5 RGB Datasets**: ~10-15 hours (GPU) or ~40-60 hours (CPU)

---

## Notes

- Times are estimates and may vary based on hardware
- GPU times assume CUDA/Metal GPU available
- CPU times assume modern multi-core CPU
- Actual times may be 20-30% faster or slower depending on system

