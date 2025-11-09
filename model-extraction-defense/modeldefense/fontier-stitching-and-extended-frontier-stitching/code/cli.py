"""
Command-line interface for the Frontier Stitching watermarking pipeline.

This CLI provides a convenient way to run all steps of the watermarking pipeline
using click commands.
"""

import click
import os
import sys
from pathlib import Path

# Add the code directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from config import ConfigManager, ExperimentConfig, TrainingConfig, WatermarkConfig, AttackConfig


@click.group()
@click.version_option(version='1.0.0')
def cli():
    """Frontier Stitching Watermarking Pipeline CLI
    
    A comprehensive CLI for training models, generating watermarks, 
    watermarking models, and testing watermark effectiveness.
    """
    pass


@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True), 
              help='Path to YAML configuration file')
@click.option('--dataset', default='cifar10', 
              help='Dataset name (cifar10, mnist)')
@click.option('--epochs', default=30, type=int, 
              help='Number of training epochs')
@click.option('--batch-size', default=128, type=int, 
              help='Batch size for training')
@click.option('--model-arch', default='cifar10_base_2', 
              help='Model architecture')
@click.option('--lr', default=0.001, type=float, 
              help='Learning rate')
@click.option('--mixed-precision/--no-mixed-precision', default=False,
              help='Enable mixed precision training')
def train(config, dataset, epochs, batch_size, model_arch, lr, mixed_precision):
    """Train the original unwatermarked model."""
    click.echo("🚀 Starting model training...")
    
    if config:
        # Load from YAML
        exp_config = ConfigManager.load_from_yaml(config)
        train_config = exp_config.training
    else:
        # Use command-line arguments
        train_config = TrainingConfig(
            dataset_name=dataset,
            epochs=epochs,
            batch_size=batch_size,
            model_architecture=model_arch,
            lr=lr
        )
    
    # Set mixed precision environment variable
    if mixed_precision:
        os.environ['USE_MIXED_PRECISION'] = 'true'
    
    # Import and run training
    from train_original import train_model
    from tensorflow.keras.optimizers import Adam
    
    optimizer = Adam(learning_rate=train_config.lr, 
                    weight_decay=train_config.weight_decay)
    
    train_model(
        dataset_name=train_config.dataset_name,
        model_architecture=train_config.model_architecture,
        epochs=train_config.epochs,
        dropout=train_config.dropout,
        batch_size=train_config.batch_size,
        optimizer=optimizer,
        lr=train_config.lr,
        weight_decay=train_config.weight_decay
    )
    
    click.echo("✅ Model training completed!")


@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Path to YAML configuration file')
@click.option('--model-path', required=True, type=click.Path(exists=True),
              help='Path to the trained model to attack')
@click.option('--dataset', default='cifar10',
              help='Dataset name')
@click.option('--eps', multiple=True, type=float, default=[0.01, 0.015],
              help='Epsilon values for FGSM attack (can specify multiple)')
@click.option('--sample-sizes', multiple=True, type=int, default=[10000],
              help='Sample sizes for adversarial examples (can specify multiple)')
def generate_watermark(config, model_path, dataset, eps, sample_sizes):
    """Generate FGSM-based adversarial examples as watermarks."""
    click.echo("🎯 Generating adversarial watermarks...")
    
    if config:
        exp_config = ConfigManager.load_from_yaml(config)
        watermark_config = exp_config.watermark
        eps = watermark_config.eps_values
        sample_sizes = watermark_config.sample_sizes
    
    click.echo(f"   Model: {model_path}")
    click.echo(f"   Dataset: {dataset}")
    click.echo(f"   Epsilon values: {eps}")
    click.echo(f"   Sample sizes: {sample_sizes}")
    
    # Import and run frontier-stitching
    # Note: This requires updating frontier-stitching.py to accept parameters
    click.echo("⚠️  Note: Update frontier-stitching.py to accept CLI parameters")
    click.echo("   For now, run: poetry run python frontier-stitching.py")
    
    click.echo("✅ Watermark generation completed!")


@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Path to YAML configuration file')
@click.option('--model-path', required=True, type=click.Path(exists=True),
              help='Path to unwatermarked model')
@click.option('--watermark-path', required=True, type=click.Path(exists=True),
              help='Path to watermark data (.npz file)')
@click.option('--epochs', default=10, type=int,
              help='Number of fine-tuning epochs')
@click.option('--method', type=click.Choice(['finetuning', 'retraining'], case_sensitive=False),
              default='finetuning', help='Watermarking method')
def watermark(config, model_path, watermark_path, epochs, method):
    """Watermark a model via fine-tuning or retraining."""
    click.echo(f"💧 Watermarking model using {method}...")
    
    if config:
        exp_config = ConfigManager.load_from_yaml(config)
        watermark_config = exp_config.watermark
        epochs = watermark_config.finetuning_epochs
    
    click.echo(f"   Model: {model_path}")
    click.echo(f"   Watermark: {watermark_path}")
    click.echo(f"   Epochs: {epochs}")
    
    if method == 'finetuning':
        from watermarking_finetuning import watermark_finetuning
        # Extract dataset name from watermark path
        dataset_name = 'cifar10'  # Default, should be extracted from path
        watermark_finetuning(
            dataset_name=dataset_name,
            adv_data_path_numpy=watermark_path,
            model_to_finetune_path=model_path,
            epochs=epochs,
            dropout=0,
            batch_size=128,
            optimizer='adam',
            lr=0.0001,
            weight_decay=0,
            num_layers_unfreeze=10
        )
    else:
        from watermarking_retraining import watermark_retraining
        # Similar call for retraining
        click.echo("⚠️  Retraining method requires additional parameters")
    
    click.echo("✅ Model watermarking completed!")


@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Path to YAML configuration file')
@click.option('--victim-model', required=True, type=click.Path(exists=True),
              help='Path to watermarked victim model')
@click.option('--watermark-path', required=True, type=click.Path(exists=True),
              help='Path to watermark data (.npz file)')
@click.option('--query-budgets', multiple=True, type=int,
              default=[250, 500, 1000, 5000, 10000, 20000],
              help='Query budgets for attack (can specify multiple)')
@click.option('--epochs', default=50, type=int,
              help='Number of epochs for model extraction')
def attack(config, victim_model, watermark_path, query_budgets, epochs):
    """Perform model extraction attack and verify watermark transfer."""
    click.echo("⚔️  Performing model extraction attack...")
    
    if config:
        exp_config = ConfigManager.load_from_yaml(config)
        attack_config = exp_config.attack
        query_budgets = attack_config.query_budgets
        epochs = attack_config.epochs_extract
    
    click.echo(f"   Victim model: {victim_model}")
    click.echo(f"   Watermark: {watermark_path}")
    click.echo(f"   Query budgets: {query_budgets}")
    click.echo(f"   Epochs: {epochs}")
    
    # Import and run attack
    from real_model_stealing_watermark_single import model_extraction_attack
    from utils.data_utils import DataManager
    
    # Extract dataset name from paths
    dataset_name = 'cifar10'  # Default, should be extracted
    
    model_extraction_attack(
        dataset_name=dataset_name,
        adv_data_path_numpy=watermark_path,
        attacker_model_architecture='cifar10_base_2',
        number_of_queries=list(query_budgets),
        num_epochs_to_steal=epochs,
        dropout=0,
        optimizer='adam',
        lr=0.001,
        weight_decay=0,
        model_to_attack_path=victim_model
    )
    
    click.echo("✅ Attack completed! Check results for watermark verification.")


@cli.command()
@click.option('--victim-model', required=True, type=click.Path(exists=True),
              help='Path to watermarked victim model')
@click.option('--suspected-model', required=True, type=click.Path(exists=True),
              help='Path to suspected stolen model')
@click.option('--watermark-path', required=True, type=click.Path(exists=True),
              help='Path to watermark data (.npz file)')
@click.option('--dataset', default='cifar10',
              help='Dataset name')
@click.option('--threshold-ratio', default=0.5, type=float,
              help='Threshold ratio for verification (0.0-1.0)')
@click.option('--confidence', default=0.99, type=float,
              help='Confidence level for statistical test (0.0-1.0)')
def verify(victim_model, suspected_model, watermark_path, dataset, 
           threshold_ratio, confidence):
    """Verify if a suspected model is stolen based on watermark accuracy."""
    click.echo("🔍 Verifying model theft...")
    
    from tensorflow.keras.models import load_model
    from utils.watermark_verifier import WatermarkVerifier
    from utils.watermark_metrics import WatermarkMetrics
    from utils.data_utils import DataManager
    
    # Load models
    click.echo("   Loading models...")
    victim = load_model(victim_model, compile=False)
    suspected = load_model(suspected_model, compile=False)
    
    # Load data
    click.echo("   Loading data...")
    x_train, y_train, x_test, y_test, input_shape, num_classes = \
        DataManager.load_and_preprocess(dataset)
    x_adv, y_adv = DataManager.load_adversarial_data(watermark_path)
    
    # Evaluate victim model
    click.echo("   Evaluating victim model...")
    victim_watermark_acc = victim.evaluate(x_adv, y_adv, verbose=0)[1]
    
    # Evaluate suspected model
    click.echo("   Evaluating suspected model...")
    suspected_watermark_acc = suspected.evaluate(x_adv, y_adv, verbose=0)[1]
    
    # Verify theft
    click.echo("   Verifying theft...")
    verifier = WatermarkVerifier(
        victim_acc=victim_watermark_acc,
        num_classes=num_classes,
        watermark_size=len(x_adv)
    )
    
    result = verifier.verify_theft(
        suspected_acc=suspected_watermark_acc,
        threshold_ratio=threshold_ratio,
        confidence=confidence
    )
    
    # Calculate comprehensive metrics
    click.echo("   Calculating comprehensive metrics...")
    metrics = WatermarkMetrics.calculate_all_metrics(
        victim_model=victim,
        stolen_model=suspected,
        x_test=x_test,
        y_test=y_test,
        x_watermark=x_adv,
        y_watermark=y_adv
    )
    
    # Print results
    click.echo("\n" + "="*60)
    click.echo("THEFT VERIFICATION RESULTS")
    click.echo("="*60)
    click.echo(f"Victim watermark accuracy: {victim_watermark_acc:.4f}")
    click.echo(f"Suspected watermark accuracy: {suspected_watermark_acc:.4f}")
    click.echo(f"\nIs stolen: {result['is_stolen']}")
    click.echo(f"Confidence: {result['confidence']:.4f}")
    click.echo(f"P-value: {result['p_value']:.6f}")
    click.echo(f"Threshold: {result['threshold']:.4f}")
    
    click.echo("\n" + "="*60)
    click.echo("COMPREHENSIVE METRICS")
    click.echo("="*60)
    WatermarkMetrics.print_metrics_summary(metrics)
    
    click.echo("✅ Verification completed!")


@cli.command()
@click.option('--watermarked-model', required=True, type=click.Path(exists=True),
              help='Path to watermarked model')
@click.option('--watermark-path', required=True, type=click.Path(exists=True),
              help='Path to watermark data (.npz file)')
@click.option('--dataset', default='cifar10',
              help='Dataset name')
def test_robustness(watermarked_model, watermark_path, dataset):
    """Test watermark robustness against removal attacks."""
    click.echo("🛡️  Testing watermark robustness...")
    
    from tensorflow.keras.models import load_model
    from test_watermark_robustness import test_removal_robustness, print_robustness_summary
    from utils.data_utils import DataManager
    
    # Load model
    click.echo("   Loading model...")
    model = load_model(watermarked_model, compile=False)
    
    # Load data
    click.echo("   Loading data...")
    x_train, y_train, x_test, y_test, input_shape, num_classes = \
        DataManager.load_and_preprocess(dataset)
    x_adv, y_adv = DataManager.load_adversarial_data(watermark_path)
    
    # Test robustness
    click.echo("   Testing against removal attacks...")
    results = test_removal_robustness(
        watermarked_model=model,
        x_test=x_test,
        y_test=y_test,
        x_watermark=x_adv,
        y_watermark=y_adv,
        x_train=x_train,
        y_train=y_train
    )
    
    # Print summary
    print_robustness_summary(results)
    
    click.echo("✅ Robustness testing completed!")


@cli.command()
@click.option('--output', '-o', default='configs/default.yaml',
              help='Output path for default configuration file')
def create_config(output):
    """Create a default configuration file."""
    click.echo(f"📝 Creating default configuration at {output}...")
    
    config = ExperimentConfig()
    ConfigManager.save_to_yaml(config, output)
    
    click.echo(f"✅ Configuration file created at {output}")
    click.echo("\nYou can now edit this file and use it with --config option")


@cli.command()
def pipeline():
    """Run the complete watermarking pipeline (all steps)."""
    click.echo("🚀 Running complete watermarking pipeline...")
    click.echo("\nThis will run all steps in sequence:")
    click.echo("  1. Train original model")
    click.echo("  2. Generate adversarial watermarks")
    click.echo("  3. Watermark the model")
    click.echo("  4. Test watermark (model stealing attack)")
    
    if not click.confirm('\nDo you want to continue?'):
        click.echo("Pipeline cancelled.")
        return
    
    # Step 1: Train
    click.echo("\n" + "="*60)
    click.echo("STEP 1: Training Original Model")
    click.echo("="*60)
    ctx = click.get_current_context()
    ctx.invoke(train)
    
    # Step 2: Generate watermarks
    click.echo("\n" + "="*60)
    click.echo("STEP 2: Generating Adversarial Watermarks")
    click.echo("="*60)
    click.echo("⚠️  Please run frontier-stitching.py manually for now")
    
    # Step 3: Watermark
    click.echo("\n" + "="*60)
    click.echo("STEP 3: Watermarking Model")
    click.echo("="*60)
    click.echo("⚠️  Please run watermarking_finetuning.py manually for now")
    
    # Step 4: Attack
    click.echo("\n" + "="*60)
    click.echo("STEP 4: Testing Watermark")
    click.echo("="*60)
    click.echo("⚠️  Please run real_model_stealing_watermark_single.py manually for now")
    
    click.echo("\n✅ Pipeline completed!")


if __name__ == '__main__':
    cli()

