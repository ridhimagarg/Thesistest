"""
@author: Ridhima Garg

Introduction:
    This file contains the code to "create" the fgsm based adversarial (watermarked) samples 
    to finetune the model in order to make it watermarked.
    But please make sure to run the current code, use "frontier-stitching.py" to get the all 
    adversaries: true, false (to understand this).
    Please refer to the paper: https://arxiv.org/abs/1711.01894

"""
import argparse
import logging
import os
import warnings
import random

warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.models import load_model

from art.estimators.classification import KerasClassifier
from art.attacks.evasion import FastGradientMethod

# Note: ART library may require eager execution disabled for some operations
# Uncomment if needed: tf.compat.v1.disable_eager_execution()
from utils.data_utils import DataManager

# Constants
SEED = 0
DEFAULT_TEST_SAMPLE_SIZE = 1000
DATA_PATH = "../data/fgsm"

# Set random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
tf.compat.v1.random.set_random_seed(SEED)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Data preprocessing is now handled by utils.data_utils.DataManager


def evaluate_model_accuracy(model, x_test, y_test, sample_size=DEFAULT_TEST_SAMPLE_SIZE):
    """
    Evaluate model accuracy on test data.

    Args:
        model: Keras model or ART classifier to evaluate.
        x_test (np.ndarray): Test images.
        y_test (np.ndarray): Test labels (one-hot encoded).
        sample_size (int): Number of samples to evaluate.

    Returns:
        tuple: (num_correct, num_incorrect, accuracy)
    """
    actual_size = min(sample_size, len(x_test))
    x_test_subset = x_test[:actual_size]
    y_test_subset = y_test[:actual_size]

    predictions = np.argmax(model.predict(x_test_subset), axis=1)
    true_labels = np.argmax(y_test_subset, axis=1)
    num_correct = np.sum(predictions == true_labels)
    num_incorrect = actual_size - num_correct
    accuracy = num_correct / actual_size

    return num_correct, num_incorrect, accuracy


def fgsm_attack(dataset_name, model_path, clip_values=(0., 1.), eps=0.3, 
                adversarial_sample_size=1000, npz_file_name='mnist_cnn_adv.npz',
                test_sample_size=DEFAULT_TEST_SAMPLE_SIZE):
    """
    Perform FGSM (Fast Gradient Sign Method) attack to generate adversarial samples.

    Reference: https://arxiv.org/abs/1412.6572

    Args:
        dataset_name (str): Name of the dataset ('mnist' or 'cifar10').
        model_path (str): Path to the model file (.h5) to use for attack generation.
        clip_values (tuple): Tuple of (min, max) values for clipping adversarial samples.
        eps (float): Perturbation level (epsilon) for the attack.
        adversarial_sample_size (int): Number of adversarial samples to generate.
        npz_file_name (str): Output file path for saving adversarial samples.
        test_sample_size (int): Number of test samples to use for evaluation.

    Raises:
        FileNotFoundError: If model_path does not exist.
        ValueError: If adversarial_sample_size exceeds available test samples.
    """
    # Validate model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info(f"Loading model from: {model_path}")
    classifier_model = load_model(model_path)

    # Use centralized DataManager for data loading
    x_train, y_train, x_test, y_test, input_shape, num_classes = DataManager.load_and_preprocess(dataset_name)

    # Validate adversarial sample size
    if adversarial_sample_size > len(x_test):
        raise ValueError(
            f"adversarial_sample_size ({adversarial_sample_size}) exceeds "
            f"available test samples ({len(x_test)})"
        )

    # Evaluate original model accuracy
    logger.info("Evaluating original model accuracy...")
    num_correct, num_incorrect, accuracy = evaluate_model_accuracy(
        classifier_model, x_test, y_test, test_sample_size
    )
    logger.info(f"Original test data (first {min(test_sample_size, len(x_test))} images):")
    logger.info(f"  Correctly classified: {num_correct}")
    logger.info(f"  Incorrectly classified: {num_incorrect}")
    logger.info(f"  Accuracy: {accuracy:.4f}")

    # Create ART classifier wrapper
    logger.info("Creating ART KerasClassifier wrapper...")
    classifier = KerasClassifier(
        clip_values=clip_values, 
        model=classifier_model, 
        use_logits=False
    )
    
    logger.info('Original model architecture:')
    classifier_model.summary(print_fn=logger.info)

    # Verify classifier wrapper works correctly
    logger.info('Verifying classifier wrapper...')
    num_correct_wrapper, num_incorrect_wrapper, accuracy_wrapper = evaluate_model_accuracy(
        classifier, x_test, y_test, test_sample_size
    )
    logger.info(f"Prediction after creating keras classifier model from original model:")
    logger.info(f"  Correctly classified: {num_correct_wrapper}")
    logger.info(f"  Incorrectly classified: {num_incorrect_wrapper}")
    logger.info(f"  Accuracy: {accuracy_wrapper:.4f}")

    # Generate adversarial samples using FGSM
    logger.info(f"Generating {adversarial_sample_size} adversarial samples with eps={eps}...")
    attacker = FastGradientMethod(classifier, eps=eps)
    x_test_adv = attacker.generate(x_test[:adversarial_sample_size])
    
    # Save adversarial samples
    logger.info(f"Saving adversarial samples to: {npz_file_name}")
    np.savez(npz_file_name, x_test_adv, y_test[:adversarial_sample_size])

    logger.info("FGSM attack completed successfully.")


def create_output_filename(data_path, dataset_name, eps, adversarial_sample_size, model_path):
    """
    Create standardized output filename for adversarial samples.

    Args:
        data_path (str): Base directory for output files.
        dataset_name (str): Name of the dataset.
        eps (float): Epsilon value used in attack.
        adversarial_sample_size (int): Number of adversarial samples.
        model_path (str): Path to the model file.

    Returns:
        str: Full path to output file.
    """
    # Extract model name from path (handle both / and \ separators)
    model_path_normalized = model_path.replace("\\", "/")
    model_parts = model_path_normalized.split("/")[-2:]
    model_name = "_".join(model_parts).replace(".h5", "")
    
    filename = f"fgsm_{eps}_{adversarial_sample_size}_{model_name}.npz"
    return os.path.join(data_path, dataset_name, filename)


if __name__ == '__main__':
    # Configuration values (previously from configs/fgsm.yaml)
    dataset_name = "cifar10"
    model_to_attack_path = "../models/original_09-11-2025/cifar10_30_CIFAR10_BASE_2/Original_checkpoint_best.keras"
    adversarial_sample_size_list = [10000]
    eps_list = [0.01, 0.015]

    # Create output directories
    os.makedirs(DATA_PATH, exist_ok=True)
    dataset_output_path = os.path.join(DATA_PATH, dataset_name)
    os.makedirs(dataset_output_path, exist_ok=True)

    # Generate adversarial samples for each combination of parameters
    for adversarial_sample_size in adversarial_sample_size_list:
        for eps in eps_list:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: eps={eps}, sample_size={adversarial_sample_size}")
            logger.info(f"{'='*60}")
            
            numpy_array_file_name = create_output_filename(
                DATA_PATH, dataset_name, eps, adversarial_sample_size, model_to_attack_path
            )
            
            try:
                fgsm_attack(
                    dataset_name=dataset_name,
                    model_path=model_to_attack_path,
                    clip_values=(0., 1.),
                    eps=eps,
                    adversarial_sample_size=adversarial_sample_size,
                    npz_file_name=numpy_array_file_name
                )
                
                logger.info(f"Successfully generated adversarial samples: {numpy_array_file_name}")
                
            except Exception as e:
                logger.error(
                    f"Failed to generate adversarial samples for "
                    f"eps={eps}, sample_size={adversarial_sample_size}: {e}",
                    exc_info=True
                )
                # Continue with next combination instead of failing completely
                continue

    logger.info("All FGSM attack generations completed.")
