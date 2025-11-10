# Frontier Stitching and Extended Frontier Stitching - Complete Workflow Documentation

## Table of Contents
1. [Overview](#overview)
2. [Complete Process Flow](#complete-process-flow)
3. [Watermarking Mechanism](#watermarking-mechanism)
4. [What We Store](#what-we-store)
5. [Attacks Performed](#attacks-performed)
6. [Pipeline Details](#pipeline-details)
7. [File Structure](#file-structure)

---

## Overview

This project implements **Frontier Stitching** and **Extended Frontier Stitching** watermarking methods to protect neural network models from model stealing attacks. The watermarking mechanism embeds ownership information into models using adversarial examples.

**Key Papers:**
- Frontier Stitching: https://arxiv.org/abs/1711.01894
- Extended Frontier Stitching: https://arxiv.org/abs/2012.13628

---

## Complete Process Flow

### Step 1: Train Original Model (`train_original.py`)

**Purpose**: Train an unwatermarked model on the original dataset.

**Process**:
1. Load dataset (MNIST, CIFAR10, CIFAR100, SVHN, STL10, EuroSAT)
2. Preprocess data (normalize, one-hot encode)
3. Create model architecture:
   - **MNIST**: `MNIST_L2` (2-layer CNN, ~50K params)
   - **RGB Datasets**: `CIFAR10_BASE_2` (6-layer CNN with GlobalAveragePooling2D, ~500K-1M params)
4. Train model for specified epochs (default: 50)
5. Save best model checkpoint

**Output**:
- Trained model: `models/original_{date}/{dataset}_{epochs}_{model_name}/Original_checkpoint_best.keras`
- Training logs: `results/original_{date}/losses/{dataset}_{model_name}_logs.txt`
- Training plots: Loss and accuracy curves

**Time**: ~10-15 min (GPU) for CIFAR10, 50 epochs

---

### Step 2: Generate Adversarial Examples (Watermarks) (`frontier-stitching.py`)

**Purpose**: Generate adversarial examples that will serve as watermarks.

**Mechanism - Frontier Stitching**:
1. **Pre-filtering**: Find correctly classified images from test set
2. **FGSM Attack**: Generate adversarial examples using Fast Gradient Sign Method
   - Formula: `x_adv = x + eps * sign(∇_x L(x, y))`
   - `eps`: Perturbation level (default: 0.01, 0.015, 0.02)
3. **Classification**: Categorize adversarial examples into:
   - **True Adversaries**: Original correct, adversarial incorrect
     - Model predicts correctly on original
     - Model predicts incorrectly on adversarial
   - **False Adversaries**: Both original and adversarial correct
     - Model predicts correctly on both
   - **Full Set**: All adversarial examples (both true and false)
4. **Save**: Store adversarial examples in `.npz` format

**Process Details**:
- Uses FGSM (Fast Gradient Sign Method) attack
- Processes images in batches for efficiency
- Filters to find correctly classified images first
- Generates specified number of adversarial examples (default: 10,000)
- Supports multiple epsilon values and sample sizes

**Output**:
- **True Adversaries**: `data/fgsm/{dataset}/true/fgsm_{eps}_{size}_{model_name}.npz`
- **False Adversaries**: `data/fgsm/{dataset}/false/fgsm_{eps}_{size}_{model_name}.npz`
- **Full Set**: `data/fgsm/{dataset}/full/fgsm_{eps}_{size}_{model_name}.npz`

**Time**: ~15-30 min (GPU) for 3 epsilon values, 10K samples each

**Supported Attacks**:
- **FGSM** (Fast Gradient Sign Method) - Default
- **PGD** (Projected Gradient Descent)
- **BIM** (Basic Iterative Method)
- **Carlini-L2**
- **DeepFool**
- And more via ART library

---

### Step 3: Watermark the Model (`watermarking_finetuning.py` or `watermarking_retraining.py`)

**Purpose**: Embed watermark into the model using adversarial examples.

**Two Methods**:

#### Method 1: Fine-tuning (Extended Frontier Stitching)
**File**: `watermarking_finetuning.py`

**Process**:
1. Load pre-trained original model
2. Load adversarial examples (watermark set)
3. **Fine-tune** model on combination of:
   - Original training data
   - Adversarial examples (watermark set)
4. **Unfreeze layers**: Only unfreeze last N layers (default: 1 layer)
5. Train for specified epochs (default: 15)
6. Save watermarked model

**Key Features**:
- Lower learning rate (0.0001) for fine-tuning
- Learning rate scheduling (decay at epochs 8, 12)
- Preserves original model performance
- Embeds watermark in model weights

**Output**:
- Watermarked model: `models/finetuned_finetuning_{date}/{which_adv}/{dataset}_{epochs}_{model_name}/Victim_checkpoint_best.keras`
- Training logs: `results/finetuned_finetuning_{date}/losses/{which_adv}/{dataset}_{epochs}_{model_name}_logs.txt`
- Training plots: Loss and accuracy curves

**Time**: ~3-5 min (GPU) for 15 epochs

#### Method 2: Retraining
**File**: `watermarking_retraining.py`

**Process**:
1. Create new model with same architecture
2. Load adversarial examples (watermark set)
3. **Train from scratch** on combination of:
   - Original training data
   - Adversarial examples (watermark set)
4. Train for specified epochs
5. Save watermarked model

**Key Features**:
- Trains from scratch (not fine-tuning)
- May achieve better watermark retention
- Takes longer than fine-tuning

**Output**:
- Watermarked model: `models/finetuned_retraining_{date}/{which_adv}/{dataset}_{epochs}_{model_name}/Victim_checkpoint_best.keras`
- Training logs: `results/finetuned_retraining_{date}/losses/{which_adv}/{dataset}_{epochs}_{model_name}_logs.txt`

**Time**: ~10-15 min (GPU) for 15 epochs

---

### Step 4: Model Extraction Attack (`real_model_stealing.py`)

**Purpose**: Simulate model stealing attack and verify watermark retention.

**Process**:
1. Load watermarked model (victim model)
2. Load watermark set (adversarial examples)
3. **Evaluate victim model** on watermark set (baseline accuracy)
4. For each query budget (250, 500, 1000, 5000, 10000, 20000):
   - Sample query data from test set
   - **Extract model** using KnockoffNets attack:
     - Query victim model to get predictions
     - Train attacker model on query data + predictions
   - **Evaluate stolen model** on:
     - Test set (test accuracy)
     - Watermark set (watermark accuracy)
5. **Ownership Verification**:
   - If stolen model has high watermark accuracy → Model is stolen
   - Victim can claim ownership

**Attack Details**:
- **KnockoffNets Attack**: Model extraction via querying
- **Query Budgets**: Different amounts of queries (250 to 20,000)
- **Attacker Model**: Same or different architecture
- **Training**: Attacker trains on query data + victim predictions

**Output**:
- Attack results: `results/attack_{date}/losses_acc/{which_adv}/{dataset}_{epochs}_{model_name}_logs.txt`
- Plots: Test accuracy vs Watermark accuracy curves
- CSV files: Detailed results per query budget

**Time**: ~60-90 min (GPU) for 6 query budgets, 50 epochs each

**Metrics Calculated**:
- **Test Accuracy**: How well stolen model performs on test data
- **Watermark Accuracy**: How well stolen model performs on watermark set
- **Watermark Retention**: Ratio of watermark accuracy in stolen vs victim model
- **Fidelity**: How well stolen model matches victim predictions

---

### Step 5: Watermark Robustness Testing (`test_watermark_robustness.py`)

**Purpose**: Test if watermarks survive removal attacks.

**Removal Attacks Tested**:

#### 1. Fine-tuning Attack
- Fine-tune watermarked model on clean data
- Test if watermark is removed

#### 2. Pruning Attack
- Prune watermarked model (remove low-magnitude weights)
- Test if watermark is removed

#### 3. Fine-pruning Attack
- Combine fine-tuning + pruning
- Test if watermark survives

#### 4. Distillation Attack
- Train student model using teacher's soft labels
- Test if watermark transfers to student

**Process**:
1. Load watermarked model
2. Load watermark set
3. For each removal attack:
   - Apply attack to watermarked model
   - Evaluate attacked model on:
     - Test set (test accuracy)
     - Watermark set (watermark accuracy)
4. Calculate robustness metrics:
   - **Watermark Retention**: Ratio of watermark accuracy after attack
   - **Test Accuracy Change**: Change in test accuracy
   - **Robustness Score**: Composite score

**Output**:
- Robustness results: JSON files with metrics per attack
- Summary: Robustness assessment (STRONG/MODERATE/WEAK)

**Time**: ~10-20 min (GPU) per attack type

---

## Watermarking Mechanism

### Frontier Stitching (Original Method)

**Core Idea**: Use adversarial examples that lie on the decision boundary as watermarks.

**How It Works**:
1. **Generate Adversarial Examples**: Use FGSM to create adversarial examples
2. **Classify Adversaries**: 
   - **True Adversaries**: Cause misclassification (original correct → adversarial incorrect)
   - **False Adversaries**: Don't cause misclassification (both correct)
3. **Embed Watermark**: Fine-tune model on adversarial examples
4. **Verify Ownership**: If stolen model has high accuracy on watermark set → stolen

**Key Insight**: 
- Adversarial examples are model-specific
- Stolen models trained on query data will also have high accuracy on watermark set
- This proves the model was stolen

### Extended Frontier Stitching

**Enhancement**: Fine-tune only last layers instead of full retraining.

**Benefits**:
- Faster watermarking
- Better preserves original model performance
- More robust watermark embedding

**Process**:
1. Load pre-trained model
2. Freeze all layers except last N layers (default: 1 layer)
3. Fine-tune on adversarial examples
4. Watermark is embedded in unfrozen layers

---

## What We Store

### 1. Models

**Original Models**:
- Path: `models/original_{date}/{dataset}_{epochs}_{model_name}/`
- File: `Original_checkpoint_best.keras`
- Format: Keras SavedModel format
- Contains: Model architecture + trained weights

**Watermarked Models**:
- Path: `models/finetuned_finetuning_{date}/{which_adv}/{dataset}_{epochs}_{model_name}/`
- File: `Victim_checkpoint_best.keras`
- Format: Keras SavedModel format
- Contains: Watermarked model weights

**Attacker Models** (optional):
- Path: `models/attack_{date}/{which_adv}/`
- File: Stolen model checkpoints
- Format: Keras SavedModel format

### 2. Adversarial Examples (Watermarks)

**True Adversaries**:
- Path: `data/fgsm/{dataset}/true/fgsm_{eps}_{size}_{model_name}.npz`
- Format: NumPy compressed format
- Contains: `x_adv` (adversarial images), `y_adv` (labels)
- Use: Watermark set for ownership verification

**False Adversaries**:
- Path: `data/fgsm/{dataset}/false/fgsm_{eps}_{size}_{model_name}.npz`
- Format: NumPy compressed format
- Contains: Adversarial examples that don't cause misclassification

**Full Set**:
- Path: `data/fgsm/{dataset}/full/fgsm_{eps}_{size}_{model_name}.npz`
- Format: NumPy compressed format
- Contains: All adversarial examples (true + false)

### 3. Results and Logs

**Training Logs**:
- Path: `results/original_{date}/losses/{dataset}_{model_name}_logs.txt`
- Contains: Per-epoch training metrics (loss, accuracy)

**Watermarking Logs**:
- Path: `results/finetuned_finetuning_{date}/losses/{which_adv}/{dataset}_{epochs}_{model_name}_logs.txt`
- Contains: Fine-tuning metrics, watermark accuracy

**Attack Results**:
- Path: `results/attack_{date}/losses_acc/{which_adv}/{dataset}_{epochs}_{model_name}_logs.txt`
- Contains: Test accuracy and watermark accuracy per query budget

**CSV Files**:
- Path: `experiments/{timestamp}/model_extraction_attack/{timestamp}/`
- Files: `results.csv`, `metrics.csv`
- Contains: Structured results for analysis

**JSON Files**:
- Path: `experiments/{timestamp}/model_extraction_attack/{timestamp}/`
- Files: `experiment_config.json`, `summary.json`
- Contains: Configuration and summary metrics

**LaTeX Tables**:
- Path: `experiments/{timestamp}/tables/`
- Files: `training_results.tex`, `attack_results.tex`
- Contains: Formatted tables for papers

### 4. Visualizations

**Training Plots**:
- Loss curves: `results/original_{date}/losses/{dataset}OriginalLoss.png`
- Accuracy curves: `results/original_{date}/losses/{dataset}OriginalAcc.png`

**Watermarking Plots**:
- Loss curves: `results/finetuned_finetuning_{date}/losses/{which_adv}/...`
- Accuracy curves: Similar paths

**Attack Plots**:
- Test vs Watermark Accuracy: `results/attack_{date}/losses_acc/{which_adv}/...TestandWatermarkAcc.png`
- Shows relationship between test accuracy and watermark accuracy

### 5. Experiment Metadata

**Experiment Config**:
- Path: `experiments/{timestamp}/experiment_config.json`
- Contains: Complete experiment configuration

**MLflow Tracking** (optional):
- Database: `mlflow.db`
- Contains: Experiment tracking, metrics, artifacts
- Access: MLflow UI

---

## Attacks Performed

### 1. Adversarial Generation Attacks (For Watermark Creation)

#### FGSM (Fast Gradient Sign Method)
- **Purpose**: Generate adversarial examples for watermarks
- **Method**: Single-step gradient-based attack
- **Formula**: `x_adv = x + eps * sign(∇_x L(x, y))`
- **Parameters**: `eps` (perturbation level)
- **Use**: Create watermark set

#### PGD (Projected Gradient Descent)
- **Purpose**: Generate stronger adversarial examples
- **Method**: Iterative gradient-based attack
- **Parameters**: `eps`, `eps_step`, `max_iter`
- **Use**: Alternative to FGSM for stronger watermarks

#### BIM (Basic Iterative Method)
- **Purpose**: Iterative version of FGSM
- **Method**: Multiple FGSM steps
- **Parameters**: `eps`, `eps_step`, `max_iter`
- **Use**: Alternative to FGSM

#### Carlini-L2
- **Purpose**: Optimization-based attack
- **Method**: Minimize L2 distance while causing misclassification
- **Parameters**: `confidence`, `learning_rate`, `max_iter`
- **Use**: Stronger adversarial examples

### 2. Model Extraction Attacks (For Testing)

#### KnockoffNets Attack
- **Purpose**: Simulate model stealing
- **Method**: 
  1. Query victim model with data
  2. Get predictions from victim model
  3. Train attacker model on query data + predictions
- **Parameters**: 
  - `number_of_queries`: Query budget (250, 500, 1000, 5000, 10000, 20000)
  - `num_epochs_to_steal`: Training epochs (default: 50)
- **Use**: Test if watermark survives model extraction

**Process**:
1. Attacker queries victim model with test data
2. Attacker gets predictions (soft labels)
3. Attacker trains own model on query data + predictions
4. Attacker evaluates stolen model on watermark set
5. If watermark accuracy is high → Model is stolen

### 3. Watermark Removal Attacks (For Robustness Testing)

#### Fine-tuning Attack
- **Purpose**: Remove watermark by fine-tuning on clean data
- **Method**: Fine-tune watermarked model on original training data
- **Parameters**: `epochs` (default: 10), `lr` (default: 0.001)
- **Test**: Does watermark survive fine-tuning?

#### Pruning Attack
- **Purpose**: Remove watermark by pruning model
- **Method**: Remove low-magnitude weights
- **Parameters**: `prune_ratio` (default: 0.3 = 30% pruning)
- **Test**: Does watermark survive pruning?

#### Fine-pruning Attack
- **Purpose**: Combine fine-tuning + pruning
- **Method**: First prune, then fine-tune
- **Parameters**: `prune_ratio`, `epochs`
- **Test**: Does watermark survive combined attack?

#### Distillation Attack
- **Purpose**: Remove watermark via knowledge distillation
- **Method**: 
  1. Get soft labels from watermarked model (teacher)
  2. Train student model on soft labels
  3. Test if watermark transfers to student
- **Parameters**: `temperature` (default: 5.0), `epochs` (default: 20)
- **Test**: Does watermark transfer to student model?

---

## Pipeline Details

### Complete Experiment Pipeline (`run_complete_experiment.py`)

**Entry Point**: `run_complete_experiment.py`

**Steps**:
1. **Step 1**: Train Original Model
2. **Step 2**: Generate Adversarial Examples
3. **Step 3**: Watermark Model (Fine-tuning or Retraining)
4. **Step 4**: Model Extraction Attack
5. **Step 5**: (Optional) Watermark Robustness Testing

**Usage**:
```bash
# Run complete experiment
python run_complete_experiment.py --config configs/default.yaml

# Skip steps if already done
python run_complete_experiment.py --skip-training --skip-adversarial

# Run on multiple datasets
python run_complete_experiment.py --all-datasets
```

**Configuration**:
- YAML config file: `configs/default.yaml`
- Contains: All hyperparameters, paths, settings

### Individual Scripts

**Train Original**:
```bash
python train_original.py --config configs/default.yaml
```

**Generate Watermarks**:
```bash
python frontier-stitching.py
# Or via CLI
python cli.py generate-watermark --model-path <path> --dataset cifar10
```

**Watermark Model**:
```bash
python watermarking_finetuning.py
# Or via CLI
python cli.py watermark --model-path <path> --watermark-path <path> --method finetuning
```

**Test Attack**:
```bash
python real_model_stealing.py --config configs/knockoffattack_original.yaml
```

**Test Robustness**:
```bash
python test_watermark_robustness.py
```

---

## File Structure

```
fontier-stitching-and-extended-frontier-stitching/
├── code/
│   ├── train_original.py              # Step 1: Train original model
│   ├── frontier-stitching.py          # Step 2: Generate adversarial watermarks
│   ├── watermarking_finetuning.py     # Step 3a: Watermark via fine-tuning
│   ├── watermarking_retraining.py     # Step 3b: Watermark via retraining
│   ├── real_model_stealing.py         # Step 4: Model extraction attack
│   ├── test_watermark_robustness.py    # Step 5: Robustness testing
│   ├── run_complete_experiment.py     # Complete pipeline orchestrator
│   ├── models.py                       # Model architectures
│   ├── config.py                       # Configuration management
│   ├── cli.py                          # Command-line interface
│   ├── adversarial_attacks.py         # Adversarial attack implementations
│   └── utils/
│       ├── data_utils.py               # Data loading and preprocessing
│       ├── experiment_logger.py        # Comprehensive logging
│       ├── watermark_metrics.py        # Watermark evaluation metrics
│       └── watermark_verifier.py       # Ownership verification
├── models/
│   ├── original_{date}/               # Original trained models
│   ├── finetuned_finetuning_{date}/    # Watermarked models (fine-tuning)
│   ├── finetuned_retraining_{date}/    # Watermarked models (retraining)
│   └── attack_{date}/                  # Attacker models
├── data/
│   └── fgsm/
│       └── {dataset}/
│           ├── true/                   # True adversaries
│           ├── false/                  # False adversaries
│           └── full/                   # Full set
├── results/
│   ├── original_{date}/                # Training results
│   ├── finetuned_finetuning_{date}/    # Watermarking results
│   └── attack_{date}/                  # Attack results
├── experiments/
│   └── {timestamp}/                    # Complete experiment data
│       ├── experiment_config.json      # Experiment configuration
│       ├── losses_acc/                 # Training metrics
│       ├── model_extraction_attack/     # Attack results
│       └── tables/                     # LaTeX tables
└── configs/
    └── default.yaml                     # Default configuration
```

---

## Key Concepts

### True vs False Adversaries

**True Adversaries**:
- Original image: Model predicts correctly
- Adversarial image: Model predicts incorrectly
- **Use**: Stronger watermark signal
- **Mechanism**: Model learns to misclassify these examples

**False Adversaries**:
- Original image: Model predicts correctly
- Adversarial image: Model still predicts correctly
- **Use**: Weaker watermark signal
- **Mechanism**: Model learns to maintain correct classification

**Full Set**:
- Contains both true and false adversaries
- **Use**: Comprehensive watermark set

### Watermark Verification

**Ownership Claim Process**:
1. Victim has watermark set (adversarial examples)
2. Victim evaluates suspected stolen model on watermark set
3. If watermark accuracy is high (>80%) → Model is stolen
4. Victim can claim ownership

**Key Metrics**:
- **Watermark Accuracy**: Accuracy on watermark set
- **Watermark Retention**: Ratio of watermark accuracy in stolen vs victim
- **Fidelity**: How well stolen model matches victim predictions

### Model Architectures

**MNIST Models**:
- `MNIST_L2`: 2-layer CNN (32, 64 filters)
- ~50K parameters
- Input: 28×28×1

**RGB Models**:
- `CIFAR10_BASE_2`: 6-layer CNN with GlobalAveragePooling2D
- ~500K-1M parameters
- Input: Flexible (32×32×3, 64×64×3, 96×96×3)
- Works for: CIFAR10, CIFAR100, SVHN, STL10, EuroSAT

**Why GlobalAveragePooling2D?**
- Allows model to work with different input sizes
- One model architecture for all RGB datasets
- Adapts automatically to input shape

---

## Complete Experiment Timeline

### For CIFAR10 (Default Config)

| Step | Description | Time (GPU) | Time (CPU) |
|------|-------------|------------|------------|
| **1. Train Original** | 50 epochs | ~10-15 min | ~60-90 min |
| **2. Generate Adversarial** | 3 eps values, 10K samples | ~15-30 min | ~60-120 min |
| **3. Watermark (Fine-tune)** | 15 epochs | ~3-5 min | ~15-25 min |
| **4. Attack** | 6 query budgets, 50 epochs each | ~60-90 min | ~4-6 hours |
| **5. Robustness Test** | 4 attack types | ~10-20 min | ~40-60 min |
| **TOTAL** | Complete experiment | **~1.5-2.5 hours** | **~6-9 hours** |

### For All 5 RGB Datasets

| Dataset | Time (GPU) |
|---------|------------|
| CIFAR10 | ~1.5-2.5 hours |
| CIFAR100 | ~2-3 hours |
| SVHN | ~1.5-2.5 hours |
| STL10 | ~3-4 hours |
| EuroSAT | ~2.5-3.5 hours |
| **TOTAL** | **~10-15 hours** |

---

## Summary

**What We Do**:
1. Train original model
2. Generate adversarial examples (watermarks)
3. Embed watermark into model (fine-tuning or retraining)
4. Test watermark via model extraction attack
5. Test watermark robustness against removal attacks

**What We Store**:
- Models (original, watermarked, attacker)
- Adversarial examples (watermark sets)
- Results and logs (training, watermarking, attacks)
- Visualizations (plots, curves)
- Experiment metadata (configs, metrics)

**Our Mechanism**:
- **Frontier Stitching**: Use adversarial examples on decision boundary as watermarks
- **Extended Frontier Stitching**: Fine-tune only last layers for better performance
- **Ownership Verification**: High watermark accuracy in stolen model = proof of theft

**Attacks We Perform**:
- **For Watermark Creation**: FGSM, PGD, BIM, Carlini-L2
- **For Testing**: KnockoffNets (model extraction)
- **For Robustness**: Fine-tuning, Pruning, Fine-pruning, Distillation

This complete workflow enables comprehensive watermarking research and evaluation! 🎯

