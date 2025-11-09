"""
@author: Ridhima Garg

Introduction:
    This file contains the implementation of model finepruning using keras tmot library.

"""

import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow_model_optimization as tfmot
import tempfile
import models
import random
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import logging
from utils.data_utils import DataManager

# GPU Configuration
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    try:
        for gpu in physical_devices:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except AttributeError:
                # Metal GPU doesn't support memory_growth, which is fine
                pass
        print(f"✅ GPU detected: {len(physical_devices)} GPU(s) available")
        print(f"   Using: {physical_devices[0].name}")
    except RuntimeError as e:
        print(f"⚠️  GPU configuration error: {e}")
        print("   Falling back to CPU")
else:
    print("ℹ️  No GPU detected, using CPU")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
NORMALIZATION_FACTOR = 255.0
VISUALIZATION_SIZE = 24
PRUNING_FREQUENCY = 100
VALIDATION_SPLIT = 0.2

now = datetime.now().strftime("%d-%m-%Y")
seed = 0
random.seed(seed)
np.random.seed(seed)
tf.compat.v1.random.set_random_seed(seed)


# Data preprocessing is now handled by utils.data_utils.DataManager


def plot_separation_lines(height: int, width: int) -> None:
    """
    Helper function to plot separation lines for pruned neurons visualization.

    Args:
        height: Height of the visualization grid.
        width: Width of the visualization grid.
    """
    block_size = [1, 4]

    # Add separation lines to the figure
    num_hlines = int((height - 1) / block_size[0])
    num_vlines = int((width - 1) / block_size[1])
    line_y_pos = [y * block_size[0] for y in range(1, num_hlines + 1)]
    line_x_pos = [x * block_size[1] for x in range(1, num_vlines + 1)]

    for y_pos in line_y_pos:
        plt.plot([-0.5, width], [y_pos - 0.5, y_pos - 0.5], color='w')

    for x_pos in line_x_pos:
        plt.plot([x_pos - 0.5, x_pos - 0.5], [-0.5, height], color='w')


def extract_steal_size_from_path(attacker_model_path: str) -> int:
    """
    Extract the steal size from the attacker model path.

    Args:
        attacker_model_path: Path to the attacker model file.

    Returns:
        The steal size extracted from the filename.

    Raises:
        ValueError: If the steal size cannot be extracted from the path.
    """
    try:
        filename = Path(attacker_model_path).stem
        parts = filename.split('_')
        if len(parts) < 2:
            raise ValueError(f"Cannot extract steal size from path: {attacker_model_path}")
        return int(parts[1])
    except (ValueError, IndexError) as e:
        raise ValueError(f"Error extracting steal size from path {attacker_model_path}: {e}")


def evaluate_model_metrics(model: tf.keras.Model, x_adv: np.ndarray, y_adv: np.ndarray,
                          x_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """
    Evaluate model on adversarial and test datasets.

    Args:
        model: The model to evaluate.
        x_adv: Adversarial images.
        y_adv: Adversarial labels.
        x_test: Test images.
        y_test: Test labels.

    Returns:
        Dictionary containing accuracy metrics.
    """
    watermark_accuracy = model.evaluate(x_adv, y_adv, verbose=0)[1]
    test_accuracy = model.evaluate(x_test, y_test, verbose=0)[1]

    return {
        'watermark_accuracy': watermark_accuracy,
        'test_accuracy': test_accuracy
    }


def create_output_paths(results_path: str, loss_acc_folder: str, adv_data_path: str,
                        dataset_name: str, pruning_level: float, epochs_pruning: int,
                        attacker_model_path: str, suffix: str = "logs.txt") -> Tuple[str, str]:
    """
    Create standardized output paths for logs and images.

    Args:
        results_path: Base results directory.
        loss_acc_folder: Subfolder for losses and accuracies.
        adv_data_path: Path to adversarial data (used to extract subfolder name).
        dataset_name: Name of the dataset.
        pruning_level: Pruning level.
        epochs_pruning: Number of pruning epochs.
        attacker_model_path: Path to attacker model.
        suffix: File suffix (default: "logs.txt").

    Returns:
        Tuple of (log_file_path, image_save_path).
    """
    # Normalize paths
    adv_data_path_normalized = str(Path(adv_data_path).as_posix())
    attacker_model_path_normalized = str(Path(attacker_model_path).as_posix())

    # Extract subfolder from adversarial data path
    adv_subfolder = Path(adv_data_path_normalized).parent.name

    # Extract model filename
    model_filename = Path(attacker_model_path_normalized).stem

    # Create filename components
    filename_parts = [
        dataset_name,
        str(pruning_level),
        str(epochs_pruning),
        model_filename
    ]

    if suffix.endswith('.png'):
        filename_parts.append('pruning')
        filename = "_".join(filename_parts) + ".png"
    else:
        filename = "_".join(filename_parts) + "_" + suffix

    # Create full paths
    log_file_path = os.path.join(results_path, loss_acc_folder, adv_subfolder, filename)
    image_save_path = os.path.join(results_path, loss_acc_folder, adv_subfolder,
                                   "_".join(filename_parts) + "_pruning.png")

    return log_file_path, image_save_path


def visualize_pruned_weights(interpreter: tf.lite.Interpreter, results_path: str,
                            loss_acc_folder: str, adv_data_path: str, dataset_name: str,
                            pruning_level: float, epochs_pruning: int,
                            attacker_model_path: str) -> None:
    """
    Visualize pruned weights from a TFLite model.

    Args:
        interpreter: TFLite interpreter with loaded model.
        results_path: Base results directory.
        loss_acc_folder: Subfolder for losses and accuracies.
        adv_data_path: Path to adversarial data.
        dataset_name: Name of the dataset.
        pruning_level: Pruning level used.
        epochs_pruning: Number of pruning epochs.
        attacker_model_path: Path to attacker model.
    """
    details = interpreter.get_tensor_details()
    tensor_names = ["pruning_sparsity_0_5_1"]  # Can be extended for more visualizations

    matching_layers = [x for x in details for t in tensor_names if t in x["name"]]

    for layer in matching_layers:
        try:
            tensor_data = interpreter.tensor(layer["index"])()
            logger.info(f"Visualizing layer: {layer['name']}, Shape: {tensor_data.shape}")

            # Reshape weights for visualization
            width = height = VISUALIZATION_SIZE
            weights_to_display = tf.reshape(
                tensor_data,
                [tf.reduce_prod(tensor_data.shape[:-1]), -1]
            )
            weights_to_display = weights_to_display[0:width, 0:height]

            # Create binary mask for non-zero weights
            val_zeros = np.zeros([height, width])
            subset_values_to_display = np.where(
                abs(weights_to_display) > 0,
                abs(weights_to_display),
                val_zeros
            )

            # Create visualization
            plt.figure(figsize=(10, 8))
            plot_separation_lines(height, width)
            plt.axis('off')
            plt.imshow(subset_values_to_display, cmap='viridis')
            plt.colorbar()

            layer_name = str(layer['name'].split("/")[1])
            plt.title(f"Structurally pruned weights for {layer_name} layer")

            # Save image
            _, image_save_path = create_output_paths(
                results_path, loss_acc_folder, adv_data_path, dataset_name,
                pruning_level, epochs_pruning, attacker_model_path, suffix="pruning.png"
            )
            # Override with layer-specific name
            image_save_path = os.path.join(
                os.path.dirname(image_save_path),
                f"{Path(image_save_path).stem}_{layer_name}.png"
            )

            plt.savefig(image_save_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved visualization to: {image_save_path}")

        except Exception as e:
            logger.warning(f"Failed to visualize layer {layer.get('name', 'unknown')}: {e}")


def perform_finepruning(dataset_name: str, attacker_model_path: str, model_finetune_name: str,
                        batch_size: int, epochs_pruning: int, optimizer: tf.keras.optimizers.Optimizer,
                        lr: float, weight_decay: float, dropout: Optional[float],
                        adv_data_path_numpy: str, pruning_level: float,
                        results_path: str, loss_acc_folder: str) -> None:
    """
    Perform finepruning using the steal dataset and evaluate watermark accuracy.

    Args:
        dataset_name: Name of the dataset.
        attacker_model_path: Path to attacker model trained via model stealing attack.
        model_finetune_name: Model architecture name for finetuning.
        batch_size: Batch size for training.
        epochs_pruning: Number of epochs to prune and finetune the model.
        optimizer: Optimizer to use for the model.
        lr: Learning rate (used for logging).
        weight_decay: Weight decay parameter (used for logging).
        dropout: Dropout rate (optional).
        adv_data_path_numpy: Path to watermark data samples.
        pruning_level: Pruning level (sparsity target).
        results_path: Base results directory.
        loss_acc_folder: Subfolder for losses and accuracies.
    """
    # Use centralized DataManager for data loading
    x_train, y_train, x_test, y_test, x_adv, y_adv, input_shape = DataManager.load_and_preprocess_with_adversarial(
        dataset_name=dataset_name, 
        adv_data_path=adv_data_path_numpy
    )

    # Model mapping
    models_mapping = {
        "mnist_l2_prune": models.MNIST_L2_Prune,
        "CIFAR10_BASE_2_Prune": models.CIFAR10_BASE_2_Prune,
        "cifar10_wideresnet_prune": models.wide_residual_network_prune
    }

    if model_finetune_name not in models_mapping:
        raise ValueError(f"Unknown model_finetune_name: {model_finetune_name}. "
                        f"Supported: {list(models_mapping.keys())}")

    # Load attacker model
    if not os.path.exists(attacker_model_path):
        raise FileNotFoundError(f"Attacker model not found: {attacker_model_path}")

    try:
        attacker_model = tf.keras.models.load_model(attacker_model_path)
        logger.info(f"Loaded attacker model from: {attacker_model_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading attacker model: {e}")

    # Create output paths
    log_file_path, image_save_path = create_output_paths(
        results_path, loss_acc_folder, adv_data_path_numpy, dataset_name,
        pruning_level, epochs_pruning, attacker_model_path
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # Evaluate model before pruning
    logger.info("Evaluating model before pruning...")
    metrics_before = evaluate_model_metrics(attacker_model, x_adv, y_adv, x_test, y_test)

    with open(log_file_path, 'w') as log_file:
        log_file.write("Before Pruning:\n")
        log_file.write(f"Accuracy on watermark samples: {metrics_before['watermark_accuracy'] * 100:.2f}%\n")
        log_file.write(f"Accuracy on test set: {metrics_before['test_accuracy'] * 100:.2f}%\n")
        logger.info(f"Watermark accuracy before pruning: {metrics_before['watermark_accuracy'] * 100:.2f}%")
        logger.info(f"Test accuracy before pruning: {metrics_before['test_accuracy'] * 100:.2f}%")

        # Configure pruning parameters
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(
                target_sparsity=pruning_level,
                begin_step=0,
                frequency=PRUNING_FREQUENCY
            )
        }

        # Create prunable model
        if dropout is not None:
            model_name, finetune_model = models_mapping[model_finetune_name](
                pruning_params, input_shape, dropout
            )
        else:
            model_name, finetune_model = models_mapping[model_finetune_name](pruning_params)

        finetune_model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=optimizer,
            metrics=['accuracy']
        )

        # Copy weights from attacker model
        logger.info("Copying weights from attacker model...")
        for layer_finetune, layer_original in zip(finetune_model.layers, attacker_model.layers):
            try:
                weight_layer = layer_original.get_weights()
                if weight_layer:  # Only set weights if layer has weights
                    layer_finetune.set_weights(weight_layer)
            except Exception as e:
                logger.warning(f"Could not copy weights for layer {layer_finetune.name}: {e}")

        # Prepare steal dataset
        len_steal = extract_steal_size_from_path(attacker_model_path)
        indices = np.random.permutation(len(x_test))
        x_steal = x_test[indices[:len_steal]]
        y_steal = y_test[indices[:len_steal]]

        logger.info(f"Training with steal dataset size: {len_steal}")

        # Fine-tune with pruning
        history = finetune_model.fit(
            x_steal, y_steal,
            batch_size=batch_size,
            shuffle=True,
            epochs=epochs_pruning,
            verbose=1,
            validation_split=VALIDATION_SPLIT,
            callbacks=[tfmot.sparsity.keras.UpdatePruningStep()]
        )

        # Evaluate model after pruning
        logger.info("Evaluating model after pruning...")
        metrics_after = evaluate_model_metrics(finetune_model, x_adv, y_adv, x_test, y_test)

        log_file.write("\nAfter Pruning:\n")
        log_file.write(f"Finetuned model test acc: {metrics_after['test_accuracy']:.4f}\n")
        log_file.write(f"Finetuned model watermark acc: {metrics_after['watermark_accuracy']:.4f}\n")
        logger.info(f"Test accuracy after pruning: {metrics_after['test_accuracy'] * 100:.2f}%")
        logger.info(f"Watermark accuracy after pruning: {metrics_after['watermark_accuracy'] * 100:.2f}%")

    # Visualize pruning
    logger.info("Visualizing pruned weights...")
    finetune_model = tfmot.sparsity.keras.strip_pruning(finetune_model)

    # Convert to TFLite for visualization
    converter = tf.lite.TFLiteConverter.from_keras_model(finetune_model)
    tflite_model = converter.convert()

    with tempfile.NamedTemporaryFile(suffix='.tflite', delete=False) as tmp_file:
        tflite_file = tmp_file.name
        tmp_file.write(tflite_model)

    logger.info(f'Saved converted pruned model to: {tflite_file}')

    try:
        # Load TFLite model for visualization
        interpreter = tf.lite.Interpreter(model_path=tflite_file)
        interpreter.allocate_tensors()

        visualize_pruned_weights(
            interpreter, results_path, loss_acc_folder, adv_data_path_numpy,
            dataset_name, pruning_level, epochs_pruning, attacker_model_path
        )
    finally:
        # Clean up temporary file
        try:
            os.unlink(tflite_file)
        except Exception as e:
            logger.warning(f"Could not delete temporary file {tflite_file}: {e}")


def create_optimizer(optimizer_name: str, lr: float, weight_decay: float) -> tf.keras.optimizers.Optimizer:
    """
    Create optimizer based on configuration.

    Args:
        optimizer_name: Name of the optimizer ('adam' or 'sgd').
        lr: Learning rate.
        weight_decay: Weight decay parameter.

    Returns:
        Configured optimizer instance.
    """
    if optimizer_name.lower() == "adam":
        # Note: 'decay' parameter is deprecated in newer TensorFlow versions
        # Consider using weight_decay parameter or learning_rate_schedule
        return tf.keras.optimizers.Adam(learning_rate=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}. Supported: 'adam', 'sgd'")


if __name__ == "__main__":
    # Configuration values (previously from configs/pruning.yaml)
    dataset = "cifar10resnet_255_preprocess"
    attacker_model_path = "../models/attack_finetuned06-11-2023/true/cifar10resnet_255_preprocess_10000_50_fgsm_0.025_10000_cifar10_250_WideResNet_255_preprocess_Original_checkpoint_best.h5"
    model_finetune_name = "cifar10_wideresnet_prune"
    dropout = 0
    epochs_pruning = 10
    optimizer_name = "adam"
    lr = 0.001
    weight_decay = 0
    adv_key_set_path = "../data/fgsm/cifar10/true/fgsm_0.035_10000_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best.npz"
    batch_size = 128
    pruning_levels = [0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.75, 0.9]

    # Setup paths
    RESULTS_PATH = f"../results/finepruning_{now}"
    LOSS_Acc_FOLDER = "losses_acc"

    # Create results directory
    results_dir = os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER, "true")
    os.makedirs(results_dir, exist_ok=True)

    # Create optimizer
    optimizer = create_optimizer(optimizer_name, lr, weight_decay)

    # Process each pruning level
    logger.info(f"Processing {len(pruning_levels)} pruning levels...")
    for pruning_level in pruning_levels:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing pruning level: {pruning_level}")
        logger.info(f"{'='*60}")

        try:
            perform_finepruning(
                dataset,
                attacker_model_path,
                model_finetune_name,
                batch_size,
                epochs_pruning,
                optimizer,
                lr,
                weight_decay,
                dropout,
                adv_key_set_path,
                pruning_level,
                RESULTS_PATH,
                LOSS_Acc_FOLDER
            )
        except Exception as e:
            logger.error(f"Error processing pruning level {pruning_level}: {e}", exc_info=True)
            continue

    logger.info("Finepruning process completed.")
