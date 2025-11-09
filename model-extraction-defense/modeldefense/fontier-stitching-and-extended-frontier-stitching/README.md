# Frontier Stitching and Extended Frontier Stitching

Watermarking as Defense against Model Extraction Attacks on Deep Learning Models

## Overview

This implementation provides a comprehensive watermarking framework for protecting deep learning models against model extraction attacks. The framework uses **Frontier Stitching** and **Extended Frontier Stitching** methods to embed watermarks into models, enabling ownership verification and theft detection.

## Features

- ✅ **Adversarial Watermarking**: Generate FGSM-based adversarial examples as watermarks
- ✅ **Model Watermarking**: Embed watermarks via fine-tuning or retraining
- ✅ **Theft Detection**: Statistical verification of model theft using watermark accuracy
- ✅ **Comprehensive Metrics**: Fidelity, watermark retention, KL divergence, and detectability scores
- ✅ **Robustness Testing**: Test watermark survival against removal attacks (fine-tuning, pruning, distillation)
- ✅ **Centralized Configuration**: YAML-based configuration management
- ✅ **GPU Support**: Automatic GPU detection and configuration (Metal on macOS, CUDA on Linux/Windows)

## Installation

### Prerequisites

- Python 3.9+
- Poetry (for dependency management)

### Setup

1. **Install dependencies**:
   ```bash
   poetry install
   ```

2. **Activate the environment**:
   ```bash
   poetry shell
   ```

3. **Optional: Install tensorflow-model-optimization** (for pruning attacks):
   ```bash
   pip install tensorflow-model-optimization
   ```
   Note: This package has dependency conflicts with TensorFlow 2.19.0, but the code handles its absence gracefully.

## Workflow

The watermarking pipeline consists of 4 main steps. You can use either the **CLI interface** (recommended) or run scripts directly.

### Using CLI (Recommended)

The CLI provides a convenient interface for all operations:

```bash
cd code

# Create default configuration
poetry run python cli.py create-config

# Train original model
poetry run python cli.py train --dataset cifar10 --epochs 30

# Generate watermarks
poetry run python cli.py generate-watermark --model-path ../models/original_*/cifar10_*/Original_checkpoint_best.keras

# Watermark the model
poetry run python cli.py watermark --model-path ../models/original_*/cifar10_*/Original_checkpoint_best.keras --watermark-path ../data/fgsm/cifar10/true/*.npz

# Test watermark (attack)
poetry run python cli.py attack --victim-model ../models/finetuned_*/true/*/Victim_checkpoint_best.keras --watermark-path ../data/fgsm/cifar10/true/*.npz

# Verify theft
poetry run python cli.py verify --victim-model <victim_model> --suspected-model <suspected_model> --watermark-path <watermark_path>

# Test robustness
poetry run python cli.py test-robustness --watermarked-model <model> --watermark-path <watermark_path>
```

### Using Scripts Directly

#### Step 1: Train Original Model

Train an unwatermarked model on your dataset:

```bash
cd code
poetry run python train_original.py
```

**Output**: 
- Model saved to: `../models/original_<date>/<dataset>_<epochs>_<architecture>/Original_checkpoint_best.keras`

#### Step 2: Generate Adversarial Examples (Watermarks)

Generate FGSM-based adversarial examples to use as watermarks:

```bash
# Sequential processing (default)
poetry run python frontier-stitching.py

# Parallel processing (faster for multiple combinations)
USE_PARALLEL=true MAX_WORKERS=4 poetry run python frontier-stitching.py
```

**Output**:
- Adversarial examples saved to: `../data/fgsm/<dataset>/true/` and `../data/fgsm/<dataset>/false/`

#### Step 3: Watermark the Model

Embed watermarks into the model via fine-tuning:

```bash
poetry run python watermarking_finetuning.py
```

**Output**:
- Watermarked model saved to: `../models/finetuned_finetuning_<date>/true/<config>/Victim_checkpoint_best.keras`

#### Step 4: Test Watermark (Model Stealing Attack)

Perform model extraction attack and verify watermark transfer:

```bash
poetry run python real_model_stealing_watermark_single.py
```

**Output**:
- Stolen models saved to: `../models/attack_finetuned<date>/true/`
- Comprehensive metrics and theft verification results logged

## Configuration

### Using Configuration Files

The framework supports YAML-based configuration:

```python
from config import ConfigManager

# Load configuration from YAML
config = ConfigManager.load_from_yaml('configs/default.yaml')

# Access configuration
dataset_name = config.training.dataset_name
epochs = config.training.epochs
lr = config.training.lr
```

### Example Configuration

See `configs/default.yaml` for a complete example configuration file.

### Hardcoded Configuration (Current)

Most scripts currently use hardcoded configuration values. To use the config system, update scripts to load from YAML files.

## Project Structure

```
code/
├── train_original.py              # Step 1: Train original model
├── frontier-stitching.py          # Step 2: Generate adversarial watermarks
├── watermarking_finetuning.py    # Step 3: Watermark model via fine-tuning
├── watermarking_retraining.py    # Step 3 (alternative): Watermark via retraining
├── real_model_stealing_watermark_single.py  # Step 4: Test watermark
├── config.py                     # Configuration management
├── models.py                      # Model architectures
├── utils/
│   ├── data_utils.py              # Centralized data loading
│   ├── watermark_verifier.py      # Statistical theft verification
│   ├── watermark_metrics.py       # Comprehensive metrics
│   └── performance_utils.py        # Performance optimizations
├── test_watermark_robustness.py   # Test watermark against removal attacks
└── example_watermark_evaluation.py # Example usage of utilities
```

## Utilities

### Data Management

Centralized data loading via `DataManager`:

```python
from utils.data_utils import DataManager

# Load and preprocess dataset
x_train, y_train, x_test, y_test, input_shape, num_classes = \
    DataManager.load_and_preprocess('cifar10')

# Load adversarial data
x_adv, y_adv = DataManager.load_adversarial_data('path/to/adv_data.npz')
```

### Watermark Verification

Statistical theft detection:

```python
from utils.watermark_verifier import WatermarkVerifier

verifier = WatermarkVerifier(
    victim_acc=0.6842,  # Victim model's watermark accuracy
    num_classes=10,
    watermark_size=10000
)

result = verifier.verify_theft(
    suspected_acc=0.4470,  # Suspected model's watermark accuracy
    threshold_ratio=0.5,
    confidence=0.99
)

print(f"Is stolen: {result['is_stolen']}")
print(f"Confidence: {result['confidence']:.4f}")
print(f"P-value: {result['p_value']:.6f}")
```

### Comprehensive Metrics

Calculate detailed watermark metrics:

```python
from utils.watermark_metrics import WatermarkMetrics

metrics = WatermarkMetrics.calculate_all_metrics(
    victim_model=victim_model,
    stolen_model=stolen_model,
    x_test=x_test,
    y_test=y_test,
    x_watermark=x_watermark,
    y_watermark=y_watermark
)

WatermarkMetrics.print_metrics_summary(metrics)
```

## Performance Optimizations

### Mixed Precision Training

Enable mixed precision for faster training on GPU (2-3x speedup):

```bash
# Using CLI
poetry run python cli.py train --mixed-precision

# Using environment variable
USE_MIXED_PRECISION=true poetry run python train_original.py
```

### Parallel Processing

Enable parallel processing for generating multiple watermark combinations:

```bash
# Using environment variables
USE_PARALLEL=true MAX_WORKERS=4 poetry run python frontier-stitching.py

# Or set in config YAML
parallel_processing: true
max_workers: 4
```

### GPU Configuration

The framework automatically detects and configures GPUs:
- **macOS**: Metal GPU (via tensorflow-metal)
- **Linux/Windows**: CUDA GPU

## Testing

### Unit Tests

Run unit tests to verify functionality:

```bash
cd code

# Run all tests
poetry run python tests/run_tests.py

# Run specific test
poetry run python tests/run_tests.py data_utils

# Verbose output
poetry run python tests/run_tests.py -v

# Using pytest directly
poetry run pytest tests/ -v
```

### Testing Watermark Robustness

Test if watermarks survive removal attacks:

```bash
# Using CLI
poetry run python cli.py test-robustness --watermarked-model <model> --watermark-path <watermark_path>

# Or using Python
python -c "
from test_watermark_robustness import test_removal_robustness, print_robustness_summary
# ... your code ...
"
```

## Research Paper Logging

### Comprehensive Experiment Logging

For research paper reproducibility, the framework includes `ExperimentLogger` that logs:

- ✅ **Hyperparameters**: All training parameters (lr, batch_size, epochs, etc.)
- ✅ **Metrics**: All accuracy/loss metrics (train, val, test, watermark)
- ✅ **Model Information**: Size, parameters, architecture details
- ✅ **Timing**: Training time, inference time, per-epoch time
- ✅ **Statistical Significance**: P-values, confidence intervals, standard deviations
- ✅ **Reproducibility**: Seeds, versions, environment info, git commit
- ✅ **Structured Data**: CSV/JSON for easy analysis and plotting
- ✅ **LaTeX Tables**: Ready-to-use table format for paper inclusion
- ✅ **Summary Statistics**: Mean, std, min, max across runs

**Usage**:
```python
from utils.experiment_logger import ExperimentLogger, log_reproducibility_info

# Initialize logger
logger = ExperimentLogger("train_original", output_dir="../results")

# Log hyperparameters
logger.log_hyperparameters(dataset_name="cifar10", epochs=30, lr=0.001)

# Log per-epoch metrics
logger.log_training_epoch(epoch=0, train_loss=1.2, train_acc=0.5, ...)

# Log model metrics
logger.log_model_metrics(model=model, x_test=x_test, y_test=y_test)

# Save all data
logger.save_all()

# Create LaTeX table
logger.create_latex_table("results.tex")
```

**Output Structure**:
```
results/
└── train_original/
    └── 20250109_143022/
        ├── hyperparameters.json
        ├── experiment_data.json
        ├── reproducibility_info.json
        ├── metrics/
        │   ├── metrics.json
        │   ├── training_history.csv
        │   ├── results.csv
        │   └── summary_statistics.csv
        └── tables/
            └── results_table.tex
```

See `RESEARCH_LOGGING_ANALYSIS.md` for detailed analysis of what's logged and what's missing.

## Results Interpretation

### Watermark Accuracy

- **Victim Model**: High watermark accuracy (e.g., 68.42%) indicates watermark is embedded
- **Stolen Model**: High watermark accuracy (e.g., 44.70%) indicates watermark transferred → **Evidence of theft**

### Theft Verification

- **Is Stolen**: `True` if suspected model exceeds threshold AND passes statistical test
- **Confidence**: Higher confidence (0.0-1.0) = stronger evidence
- **P-value**: Lower p-value (< 0.01) = stronger statistical significance

### Comprehensive Metrics

- **Fidelity**: How well stolen model matches victim predictions (higher = more similar)
- **Watermark Retention**: Ratio of watermark accuracy in stolen vs victim (higher = better transfer)
- **Test Accuracy Gap**: Difference in test accuracy (lower = more similar)
- **KL Divergence**: Distribution similarity (lower = more similar)
- **Detectability Score**: Composite score (higher = easier to detect theft)

## Troubleshooting

### Common Issues

1. **FileNotFoundError: Adversarial examples not found**
   - **Solution**: Run `frontier-stitching.py` first to generate adversarial examples

2. **ImportError: tensorflow_model_optimization not available**
   - **Solution**: This is optional. Pruning attacks will be skipped if not installed.

3. **GPU not detected**
   - **Solution**: Check TensorFlow GPU installation. On macOS, ensure `tensorflow-metal` is installed.

4. **Memory errors during training**
   - **Solution**: Reduce batch size or enable mixed precision training

## Citation

If you use this code in your research, please cite:

```bibtex
@article{frontier_stitching,
  title={Frontier Stitching: Watermarking as Defense against Model Extraction},
  author={Garg, Ridhima},
  journal={arXiv preprint},
  year={2023}
}
```

## License

[Add your license here]

## Contact

For questions or issues, please contact:
- **Author**: Ridhima Garg
- **Email**: garg.ridhima72@gmail.com

## Acknowledgments

- Adversarial Robustness Toolbox (ART) for attack implementations
- TensorFlow/Keras for deep learning framework
- MLflow for experiment tracking

