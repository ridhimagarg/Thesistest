"""
Test watermark robustness against removal attacks.

This module provides functionality to test if watermarks survive
various removal attacks such as fine-tuning, pruning, fine-pruning, and distillation.
"""

import os
import logging
import numpy as np
import tensorflow as tf
from typing import Dict, Callable, Tuple, Optional
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# Make tensorflow_model_optimization optional
try:
    import tensorflow_model_optimization as tfmot
    TFMOT_AVAILABLE = True
except ImportError:
    TFMOT_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("tensorflow_model_optimization not available. Pruning attacks will be disabled.")

logger = logging.getLogger(__name__)


def fine_tune_attack(model: Model, x_train: np.ndarray, y_train: np.ndarray,
                    epochs: int = 10, batch_size: int = 128,
                    lr: float = 0.001) -> Model:
    """
    Fine-tuning attack: Fine-tune the model on clean data.
    
    Args:
        model: Watermarked model to attack.
        x_train: Training data (clean, non-watermarked).
        y_train: Training labels.
        epochs: Number of fine-tuning epochs.
        batch_size: Batch size for fine-tuning.
        lr: Learning rate for fine-tuning.
        
    Returns:
        Fine-tuned model.
    """
    logger.info(f"Performing fine-tuning attack: {epochs} epochs")
    
    # Clone the model to avoid modifying the original
    attacked_model = tf.keras.models.clone_model(model)
    attacked_model.set_weights(model.get_weights())
    
    # Compile with lower learning rate
    attacked_model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Fine-tune on clean data
    attacked_model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )
    
    return attacked_model


def pruning_attack(model: Model, prune_ratio: float = 0.3) -> Model:
    """
    Pruning attack: Prune the model to remove watermark-related neurons.
    
    Args:
        model: Watermarked model to attack.
        prune_ratio: Ratio of weights to prune (0.0 to 1.0).
        
    Returns:
        Pruned model.
        
    Raises:
        ImportError: If tensorflow_model_optimization is not available.
    """
    if not TFMOT_AVAILABLE:
        raise ImportError(
            "tensorflow_model_optimization is required for pruning attacks. "
            "Install it with: pip install tensorflow-model-optimization"
        )
    
    logger.info(f"Performing pruning attack: {prune_ratio*100}% pruning")
    
    # Clone the model
    attacked_model = tf.keras.models.clone_model(model)
    attacked_model.set_weights(model.get_weights())
    
    # Apply pruning
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(
            target_sparsity=prune_ratio,
            begin_step=0,
            end_step=0,
            frequency=1
        )
    }
    
    # Prune the model
    attacked_model = tfmot.sparsity.keras.prune_low_magnitude(
        attacked_model, **pruning_params
    )
    
    # Compile
    attacked_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Strip pruning wrapper
    attacked_model = tfmot.sparsity.keras.strip_pruning(attacked_model)
    
    return attacked_model


def fine_pruning_attack(model: Model, x_train: np.ndarray, y_train: np.ndarray,
                       prune_ratio: float = 0.3, epochs: int = 10,
                       batch_size: int = 128) -> Model:
    """
    Fine-pruning attack: Combine fine-tuning and pruning.
    
    Args:
        model: Watermarked model to attack.
        x_train: Training data.
        y_train: Training labels.
        prune_ratio: Ratio of weights to prune.
        epochs: Number of fine-tuning epochs.
        batch_size: Batch size.
        
    Returns:
        Fine-pruned model.
    """
    logger.info(f"Performing fine-pruning attack: {prune_ratio*100}% pruning, {epochs} epochs")
    
    # First prune
    pruned_model = pruning_attack(model, prune_ratio)
    
    # Then fine-tune
    attacked_model = fine_tune_attack(pruned_model, x_train, y_train, epochs, batch_size)
    
    return attacked_model


def distillation_attack(model: Model, x_train: np.ndarray, y_train: np.ndarray,
                       temperature: float = 5.0, epochs: int = 20,
                       batch_size: int = 128) -> Model:
    """
    Knowledge distillation attack: Train a student model using teacher's soft labels.
    
    Args:
        model: Watermarked teacher model.
        x_train: Training data.
        y_train: Training labels.
        temperature: Temperature for softmax (higher = softer probabilities).
        epochs: Number of training epochs.
        batch_size: Batch size.
        
    Returns:
        Distilled student model.
    """
    logger.info(f"Performing distillation attack: temperature={temperature}, {epochs} epochs")
    
    # Get teacher's soft predictions
    teacher_logits = model.predict(x_train, verbose=0)
    teacher_probs = tf.nn.softmax(teacher_logits / temperature)
    
    # Create student model (same architecture)
    student_model = tf.keras.models.clone_model(model)
    student_model.set_weights(model.get_weights())  # Initialize with teacher weights
    
    # Custom loss: combination of hard labels and soft labels
    def distillation_loss(y_true, y_pred):
        # Hard loss
        hard_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        
        # Soft loss
        soft_loss = tf.keras.losses.kl_divergence(
            teacher_probs, tf.nn.softmax(y_pred / temperature)
        ) * (temperature ** 2)
        
        return 0.5 * hard_loss + 0.5 * soft_loss
    
    student_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=distillation_loss,
        metrics=['accuracy']
    )
    
    # Train student model
    student_model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )
    
    return student_model


def test_removal_robustness(watermarked_model: Model,
                            x_test: np.ndarray, y_test: np.ndarray,
                            x_watermark: np.ndarray, y_watermark: np.ndarray,
                            x_train: Optional[np.ndarray] = None,
                            y_train: Optional[np.ndarray] = None) -> Dict[str, Dict[str, float]]:
    """
    Test if watermark survives removal attacks.
    
    Args:
        watermarked_model: The watermarked model to test.
        x_test: Test data (clean).
        y_test: Test labels.
        x_watermark: Watermark data.
        y_watermark: Watermark labels.
        x_train: Training data for attacks (if None, uses x_test).
        y_train: Training labels for attacks (if None, uses y_test).
        
    Returns:
        Dictionary with results for each attack:
        {
            'attack_name': {
                'watermark_acc': float,
                'test_acc': float,
                'watermark_retention': float,
                'test_acc_change': float
            }
        }
    """
    logger.info("Testing watermark robustness against removal attacks")
    
    # Use test data if training data not provided
    if x_train is None:
        x_train = x_test
        y_train = y_test
    
    # Get original watermark accuracy
    original_watermark_acc = watermarked_model.evaluate(x_watermark, y_watermark, verbose=0)[1]
    original_test_acc = watermarked_model.evaluate(x_test, y_test, verbose=0)[1]
    
    logger.info(f"Original watermark accuracy: {original_watermark_acc:.4f}")
    logger.info(f"Original test accuracy: {original_test_acc:.4f}")
    
    # Define attacks (only include pruning if TFMOT is available)
    attacks = {
        'fine_tuning': lambda m: fine_tune_attack(m, x_train, y_train, epochs=10),
        'distillation': lambda m: distillation_attack(m, x_train, y_train, temperature=5.0, epochs=20),
    }
    
    # Add pruning attacks only if tensorflow_model_optimization is available
    if TFMOT_AVAILABLE:
        attacks['pruning'] = lambda m: pruning_attack(m, prune_ratio=0.3)
        attacks['fine_pruning'] = lambda m: fine_pruning_attack(m, x_train, y_train, prune_ratio=0.3, epochs=10)
    else:
        logger.warning("Skipping pruning attacks (tensorflow_model_optimization not available)")
    
    results = {}
    
    for attack_name, attack_fn in attacks.items():
        logger.info(f"Testing against {attack_name} attack...")
        
        try:
            # Apply attack
            attacked_model = attack_fn(watermarked_model)
            
            # Evaluate on watermark set
            watermark_acc = attacked_model.evaluate(x_watermark, y_watermark, verbose=0)[1]
            
            # Evaluate on test set
            test_acc = attacked_model.evaluate(x_test, y_test, verbose=0)[1]
            
            # Calculate metrics
            watermark_retention = watermark_acc / original_watermark_acc if original_watermark_acc > 0 else 0.0
            test_acc_change = test_acc - original_test_acc
            
            results[attack_name] = {
                'watermark_acc': float(watermark_acc),
                'test_acc': float(test_acc),
                'watermark_retention': float(watermark_retention),
                'test_acc_change': float(test_acc_change),
                'original_watermark_acc': float(original_watermark_acc),
                'original_test_acc': float(original_test_acc)
            }
            
            logger.info(f"  {attack_name}: watermark_acc={watermark_acc:.4f}, "
                       f"retention={watermark_retention:.4f}, test_acc={test_acc:.4f}")
            
        except Exception as e:
            logger.error(f"Error in {attack_name} attack: {e}")
            results[attack_name] = {
                'error': str(e),
                'watermark_acc': 0.0,
                'test_acc': 0.0,
                'watermark_retention': 0.0,
                'test_acc_change': 0.0
            }
    
    return results


def print_robustness_summary(results: Dict[str, Dict[str, float]]) -> None:
    """
    Print a summary of robustness test results.
    
    Args:
        results: Results dictionary from test_removal_robustness().
    """
    print("\n" + "=" * 60)
    print("Watermark Robustness Test Summary")
    print("=" * 60)
    
    for attack_name, metrics in results.items():
        if 'error' in metrics:
            print(f"\n{attack_name.upper()}: ERROR - {metrics['error']}")
            continue
        
        print(f"\n{attack_name.upper()}:")
        print(f"  Watermark Accuracy: {metrics['watermark_acc']:.4f}")
        print(f"  Watermark Retention: {metrics['watermark_retention']:.4f} ({metrics['watermark_retention']*100:.2f}%)")
        print(f"  Test Accuracy: {metrics['test_acc']:.4f}")
        print(f"  Test Accuracy Change: {metrics['test_acc_change']:+.4f}")
        
        # Determine robustness
        if metrics['watermark_retention'] > 0.8:
            robustness = "STRONG"
        elif metrics['watermark_retention'] > 0.5:
            robustness = "MODERATE"
        else:
            robustness = "WEAK"
        
        print(f"  Robustness: {robustness}")
    
    print("\n" + "=" * 60)

