"""
@author: Ridhima Garg

Introduction:
    This file contains the code to "create" the FGSM-based adversarial (watermarked) samples 
    to finetune the model in order to make it watermarked.
    Please refer to the paper: https://arxiv.org/abs/1711.01894

"""

import argparse
import logging
import os
import sys
import warnings
from typing import Tuple, List, Optional, Dict

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.models import load_model
from utils.data_utils import DataManager
from utils.performance_utils import run_parallel_processing, create_config_list
from config import ConfigManager, ExperimentConfig
from adversarial_attacks import (
    create_art_classifier,
    create_attack_instance,
    get_supported_attacks
)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
SEED = 0
np.random.seed(SEED)
tf.random.set_seed(SEED)

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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DATA_PATH = "../data/fgsm"
BATCH_SIZE = 64  # Batch size for processing images

# Data preprocessing is now handled by utils.data_utils.DataManager

@tf.function(reduce_retracing=True)
def fast_gradient_signed_batch(x: tf.Tensor, y: tf.Tensor, model, eps: float, 
                                clip_min: float = 0.0, clip_max: float = 1.0) -> tf.Tensor:
    """
    Perform FGSM (Fast Gradient Sign Method) attack on a batch of images.
    
    Reference: https://arxiv.org/abs/1412.6572
    
    Optimized with @tf.function for graph compilation and better performance.

    Args:
        x: Input data tensor (batch of images).
        y: True labels tensor (one-hot encoded).
        model: Keras model to attack.
        eps: Perturbation level (epsilon).
        clip_min: Minimum value for clipping adversarial examples.
        clip_max: Maximum value for clipping adversarial examples.

    Returns:
        Perturbed images tensor clipped to valid range.
    """
    x = tf.cast(x, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(x)
        y_pred = model(x, training=False)  # Explicitly set training=False
        # Use categorical crossentropy loss for multi-class classification
        loss_fn = tf.keras.losses.CategoricalCrossentropy()
        loss = loss_fn(y, y_pred)
    
    gradient = tape.gradient(loss, x)
    sign = tf.sign(gradient)
    x_adv = x + eps * sign
    
    # Clip to valid pixel range
    x_adv = tf.clip_by_value(x_adv, clip_min, clip_max)
    
    return x_adv

def gen_adversaries(
    model, 
    num_adversaries: int, 
    images: np.ndarray, 
    labels: np.ndarray, 
    eps: float,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    batch_size: int = BATCH_SIZE,
    attack_type: str = 'fgsm',
    attack_params: Optional[Dict] = None
) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Generate and separate true and false adversaries using specified attack type with batch processing.

    True adversaries: Original prediction is correct, adversarial prediction is incorrect.
    False adversaries: Both original and adversarial predictions are correct.

    Args:
        model: Keras model to attack.
        num_adversaries: Total number of adversarial examples to generate.
        images: Array of input images.
        labels: Array of one-hot encoded labels.
        eps: Perturbation level (epsilon).
        clip_min: Minimum value for clipping adversarial examples.
        clip_max: Maximum value for clipping adversarial examples.
        batch_size: Batch size for processing images.
        attack_type: Type of attack to use ('fgsm', 'pgd', 'bim', etc.). Default: 'fgsm'.
        attack_params: Optional dict of attack-specific parameters.

    Returns:
        Tuple of (true_adversaries, false_adversaries) where each is a list of
        tuples (original_image, adversarial_image, label).
    """
    true_advs = []
    false_advs = []
    max_true_advs = max_false_advs = num_adversaries // 2
    
    logger.info(f"Generating {num_adversaries} adversaries ({max_true_advs} true, {max_false_advs} false) using {attack_type.upper()}")
    logger.info(f"Using batch size: {batch_size}")
    
    # Create ART classifier if using non-FGSM attack
    use_art_attack = attack_type.lower() != 'fgsm'
    classifier = None
    attacker = None
    
    if use_art_attack:
        try:
            logger.info(f"Using ART attack: {attack_type}")
            from art.estimators.classification import KerasClassifier
            classifier = KerasClassifier(
                clip_values=(clip_min, clip_max),
                model=model,
                use_logits=False
            )
            attack_params = attack_params or {}
            attacker = create_attack_instance(attack_type, classifier, eps, **attack_params)
        except Exception as e:
            logger.warning(f"Failed to create ART attack {attack_type}: {e}. Falling back to FGSM.")
            use_art_attack = False
            attack_type = 'fgsm'
    
    # Pre-filter: Check enough images to find what we need
    # We need at least num_adversaries correctly classified images
    # With ~82% accuracy, we need to check roughly num_adversaries / 0.82 images
    # Add a safety margin of 2x to ensure we have enough
    max_images_to_check = min(len(images), int(num_adversaries / 0.82 * 2))
    
    print(f"🔍 Pre-filtering correctly classified images...", flush=True)
    print(f"   Checking up to {max_images_to_check} images (need {num_adversaries} correctly classified)", flush=True)
    print(f"   Processing in batches of {batch_size}...", flush=True)
    logger.info(f"Pre-filtering correctly classified images...")
    logger.info(f"Checking up to {max_images_to_check} images (need {num_adversaries} correctly classified)")
    logger.info(f"Processing in batches of {batch_size}...")
    
    # Process in batches to find correctly classified images
    correct_indices = []
    num_batches_to_check = (max_images_to_check + batch_size - 1) // batch_size
    
    print(f"   Total batches to process: {num_batches_to_check}", flush=True)
    logger.info(f"Total batches to process: {num_batches_to_check}")
    
    for i in range(num_batches_to_check):
        batch_start = i * batch_size
        batch_end = min(batch_start + batch_size, max_images_to_check)
        x_batch = images[batch_start:batch_end]
        y_batch = labels[batch_start:batch_end]
        
        # Predict on batch
        y_pred_batch = model.predict(x_batch, batch_size=batch_size, verbose=0)
        y_pred_classes = np.argmax(y_pred_batch, axis=1)
        y_true_classes = np.argmax(y_batch, axis=1)
        
        # Find correctly classified images in this batch
        correct_mask = y_pred_classes == y_true_classes
        batch_correct_indices = np.where(correct_mask)[0] + batch_start
        correct_indices.extend(batch_correct_indices.tolist())
        
        # Progress logging (every 10 batches or at start/end)
        if (i + 1) % 10 == 0 or i == 0 or i == num_batches_to_check - 1:
            progress_pct = (batch_end / max_images_to_check) * 100
            msg = f"   Progress: {batch_end}/{max_images_to_check} images ({progress_pct:.1f}%) | Found {len(correct_indices)} correctly classified"
            print(msg, flush=True)
            logger.info(f"Progress: {batch_end}/{max_images_to_check} images ({progress_pct:.1f}%) | Found {len(correct_indices)} correctly classified")
        
        # Early exit if we have enough correctly classified images
        if len(correct_indices) >= num_adversaries * 1.5:  # 1.5x safety margin
            msg = f"✅ Found enough correctly classified images ({len(correct_indices)}), stopping pre-filtering"
            print(msg, flush=True)
            logger.info(msg)
            break
    
    correct_indices = np.array(correct_indices)
    msg = f"✅ Found {len(correct_indices)} correctly classified images out of {max_images_to_check} checked"
    print(msg, flush=True)
    logger.info(f"Found {len(correct_indices)} correctly classified images out of {max_images_to_check} checked")
    
    # Adjust target counts based on available images
    if len(correct_indices) < num_adversaries:
        logger.warning(f"Only {len(correct_indices)} correctly classified images found, but need {num_adversaries}")
        logger.warning(f"Adjusting target: will try to generate up to {len(correct_indices)} adversaries")
        # Adjust targets proportionally
        max_true_advs = min(max_true_advs, len(correct_indices) // 2)
        max_false_advs = min(max_false_advs, len(correct_indices) // 2)
        logger.info(f"Adjusted targets: {max_true_advs} true, {max_false_advs} false adversaries")
    
    # Process images in batches - process all available correctly classified images
    # to maximize chances of finding false adversaries (especially with small eps)
    processed = 0
    max_images_to_process = len(correct_indices)  # Process all available images
    
    logger.info(f"Processing up to {max_images_to_process} correctly classified images to find adversaries...")
    
    for batch_start in range(0, max_images_to_process, batch_size):
        if len(true_advs) >= max_true_advs and len(false_advs) >= max_false_advs:
            break
        
        batch_end = min(batch_start + batch_size, max_images_to_process)
        batch_indices = correct_indices[batch_start:batch_end]
        
        if len(batch_indices) == 0:
            break
        
        # Get batch data
        x_batch = images[batch_indices]
        y_batch = labels[batch_indices]
        
        # Generate adversarial examples for the batch
        if use_art_attack and attacker is not None:
            # Use ART attack
            x_adv_batch = attacker.generate(x_batch)
            x_adv_batch = np.clip(x_adv_batch, clip_min, clip_max)
            x_adv_np = x_adv_batch  # Already numpy array
            
            # Get predictions using direct model calls (faster than predict())
            x_batch_tensor = tf.constant(x_batch, dtype=tf.float32)
            x_adv_tensor = tf.constant(x_adv_batch, dtype=tf.float32)
            y_pred_orig = model(x_batch_tensor, training=False)
            y_pred_adv = model(x_adv_tensor, training=False)
        else:
            # Use original FGSM method - create tensors once and reuse
            x_tensor = tf.constant(x_batch, dtype=tf.float32)
            y_tensor = tf.constant(y_batch, dtype=tf.float32)
            
            # Generate adversarial examples
            x_adv_tensor = fast_gradient_signed_batch(x_tensor, y_tensor, model, eps, clip_min, clip_max)
            
            # Get predictions using tensors directly (no numpy conversion needed yet)
            y_pred_orig = model(x_tensor, training=False)
            y_pred_adv = model(x_adv_tensor, training=False)
            
            # Only convert to numpy when needed for storage/classification
            x_adv_np = x_adv_tensor.numpy()
            x_adv_batch = x_adv_np
        
        # Convert predictions to numpy for classification (batch conversion)
        if isinstance(y_pred_orig, tf.Tensor):
            y_pred_orig = y_pred_orig.numpy()
        if isinstance(y_pred_adv, tf.Tensor):
            y_pred_adv = y_pred_adv.numpy()
        
        y_pred_classes = np.argmax(y_pred_orig, axis=1)
        y_pred_adv_classes = np.argmax(y_pred_adv, axis=1)
        y_true_classes = np.argmax(y_batch, axis=1)
        
        # Classify and store adversaries
        for i in range(len(batch_indices)):
            if len(true_advs) >= max_true_advs and len(false_advs) >= max_false_advs:
                break
            
            y_pred = y_pred_classes[i]
            y_pred_adv = y_pred_adv_classes[i]
            y_true = y_true_classes[i]
            
            # True adversary: original correct, adversarial incorrect
            if (y_pred == y_true and 
                y_pred_adv != y_true and 
                len(true_advs) < max_true_advs):
                true_advs.append((x_batch[i], x_adv_np[i], y_batch[i]))
            
            # False adversary: both original and adversarial correct
            if (y_pred == y_true and 
                y_pred_adv == y_true and 
                len(false_advs) < max_false_advs):
                false_advs.append((x_batch[i], x_adv_np[i], y_batch[i]))
        
        processed += len(batch_indices)
        
        # Progress logging
        if processed % (batch_size * 10) == 0 or len(true_advs) >= max_true_advs or len(false_advs) >= max_false_advs:
            logger.info(f"Processed {processed}/{max_images_to_process} images | True: {len(true_advs)}/{max_true_advs} | False: {len(false_advs)}/{max_false_advs}")
    
    logger.info(f"Generation complete: {len(true_advs)} true, {len(false_advs)} false adversaries")
    
    # More informative warning
    if len(true_advs) < max_true_advs or len(false_advs) < max_false_advs:
        logger.warning(
            f"Could not generate enough adversaries. "
            f"True: {len(true_advs)}/{max_true_advs}, "
            f"False: {len(false_advs)}/{max_false_advs}"
        )
        if len(false_advs) < max_false_advs:
            logger.info(
                f"Note: With small epsilon (eps={eps}), most adversarial examples remain correctly classified. "
                f"This makes it harder to find 'false adversaries' (adversarial examples that are still correct). "
                f"Consider using a larger epsilon or accepting fewer false adversaries."
            )
    
    return true_advs, false_advs

def fgsm_attack(
    dataset_name: str,
    model_path: str,
    eps: float = 0.3,
    adversarial_sample_size: int = 1000,
    npz_full_file_name: str = 'mnist_cnn_adv.npz',
    npz_true_file_name: str = 'mnist_cnn_adv.npz',
    npz_false_file_name: str = 'mnist_cnn_adv.npz',
    attack_type: str = 'fgsm',
    attack_types: Optional[List[str]] = None,
    attack_params: Optional[Dict[str, Dict]] = None
) -> None:
    """
    Generate adversarial examples using specified attack type(s) and save them to NPZ files.

    This function combines data preprocessing, model loading, and adversary generation
    to create and save true adversaries, false adversaries, and combined adversaries.
    Supports multiple attack types via ART library.

    Args:
        dataset_name: Name of the dataset.
        model_path: Path to the Keras model file.
        eps: Perturbation level (epsilon) for attack.
        adversarial_sample_size: Total number of adversarial examples to generate.
        npz_full_file_name: Path to save combined (true + false) adversaries.
        npz_true_file_name: Path to save true adversaries only.
        npz_false_file_name: Path to save false adversaries only.
        attack_type: Single attack type to use ('fgsm', 'pgd', 'bim', etc.). Default: 'fgsm'.
        attack_types: List of attack types to use. If provided, generates for each attack type.
        attack_params: Dict mapping attack_type to its specific parameters.

    Raises:
        FileNotFoundError: If model_path does not exist.
        ValueError: If adversarial_sample_size is not positive or even.
    """
    # Validate inputs
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if adversarial_sample_size <= 0:
        raise ValueError(f"adversarial_sample_size must be positive, got {adversarial_sample_size}")
    
    if adversarial_sample_size % 2 != 0:
        logger.warning(f"adversarial_sample_size ({adversarial_sample_size}) is not even. "
                      f"True and false adversaries will be {adversarial_sample_size // 2} each.")

    print(f"📦 Loading model from {model_path}...", flush=True)
    logger.info(f"Loading model from {model_path}")
    classifier_model = load_model(model_path)
    print("✅ Model loaded successfully", flush=True)
    logger.info("✅ Model loaded successfully")

    print(f"📊 Preprocessing {dataset_name} dataset...", flush=True)
    logger.info(f"Preprocessing {dataset_name} dataset")
    # Use centralized DataManager for data loading
    x_train, y_train, x_test, y_test, input_shape, num_classes = DataManager.load_and_preprocess(dataset_name)
    print(f"✅ Data preprocessing complete: train={len(x_train)}, test={len(x_test)}", flush=True)
    logger.info(f"✅ Data preprocessing complete: train={len(x_train)}, test={len(x_test)}")

    # Model prediction on original test data
    logger.info("Evaluating model on original test data (first 1000 images)")
    x_test_pred = np.argmax(classifier_model.predict(x_test[:1000], verbose=0, batch_size=BATCH_SIZE), axis=1)
    nb_correct_pred = np.sum(x_test_pred == np.argmax(y_test[:1000], axis=1))

    logger.info(f"Original test data (first 1000 images):")
    logger.info(f"  Correctly classified: {nb_correct_pred}")
    logger.info(f"  Incorrectly classified: {1000 - nb_correct_pred}")

    # Determine clip values based on dataset
    clip_min, clip_max = 0.0, 1.0
    if dataset_name == 'cifar10resnet':
        # For normalized CIFAR-10 ResNet, use approximate bounds
        clip_min, clip_max = -2.0, 2.0

    # Use training set if we need more images (training set is larger)
    # For CIFAR-10: train=50k, test=10k
    # For MNIST: train=60k, test=10k
    use_training_set = adversarial_sample_size > len(x_test) * 0.8  # Use training set if we need >80% of test set
    
    if use_training_set:
        msg = f"📚 Using training set ({len(x_train)} images) instead of test set ({len(x_test)} images) for more data"
        print(msg, flush=True)
        logger.info(msg)
        images_to_use = x_train
        labels_to_use = y_train
    else:
        msg = f"📚 Using test set ({len(x_test)} images)"
        print(msg, flush=True)
        logger.info(msg)
        images_to_use = x_test
        labels_to_use = y_test

    # Determine attack types to use
    if attack_types and len(attack_types) > 1:
        # Multi-attack mode: generate for each attack type
        logger.info(f"Multi-attack mode: Generating adversaries for {len(attack_types)} attack types: {attack_types}")
        all_results = {}
        
        for at_type in attack_types:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing attack type: {at_type.upper()}")
            logger.info(f"{'='*60}")
            
            # Generate file names for this attack type
            base_name = os.path.splitext(npz_full_file_name)[0]
            at_full = f"{base_name}_{at_type}.npz"
            at_true = f"{os.path.splitext(npz_true_file_name)[0]}_{at_type}.npz"
            at_false = f"{os.path.splitext(npz_false_file_name)[0]}_{at_type}.npz"
            
            try:
                # Get attack-specific parameters
                at_params = (attack_params or {}).get(at_type, {})
                
                # Generate adversaries
                msg = f"🎯 Generating {adversarial_sample_size} adversarial examples using {at_type.upper()} with eps={eps}"
                print(msg, flush=True)
                logger.info(msg)
                
                true_advs, false_advs = gen_adversaries(
                    classifier_model, 
                    adversarial_sample_size, 
                    images_to_use, 
                    labels_to_use, 
                    eps,
                    clip_min=clip_min,
                    clip_max=clip_max,
                    batch_size=BATCH_SIZE,
                    attack_type=at_type,
                    attack_params=at_params
                )
                
                # Save results for this attack type
                full_advs = true_advs + false_advs
                x_test_adv_orig = np.array([x for x, x_adv, y in full_advs])
                x_test_adv = np.array([x_adv for x, x_adv, y in full_advs])
                y_test_adv = np.array([y for x, x_adv, y in full_advs])
                
                x_true_adv_orig = np.array([x for x, x_adv, y in true_advs])
                x_true_adv = np.array([x_adv for x, x_adv, y in true_advs])
                y_true_adv = np.array([y for x, x_adv, y in true_advs])
                
                x_false_adv_orig = np.array([x for x, x_adv, y in false_advs])
                x_false_adv = np.array([x_adv for x, x_adv, y in false_advs])
                y_false_adv = np.array([y for x, x_adv, y in false_advs])
                
                # Create directories if they don't exist
                os.makedirs(os.path.dirname(at_full) if os.path.dirname(at_full) else '.', exist_ok=True)
                os.makedirs(os.path.dirname(at_true) if os.path.dirname(at_true) else '.', exist_ok=True)
                os.makedirs(os.path.dirname(at_false) if os.path.dirname(at_false) else '.', exist_ok=True)
                
                # Save to NPZ files
                logger.info(f"Saving {at_type.upper()} full adversaries to {at_full}")
                np.savez(at_full, x_test_adv_orig, x_test_adv, y_test_adv)
                
                logger.info(f"Saving {at_type.upper()} true adversaries to {at_true}")
                np.savez(at_true, x_true_adv_orig, x_true_adv, y_true_adv)
                
                logger.info(f"Saving {at_type.upper()} false adversaries to {at_false}")
                np.savez(at_false, x_false_adv_orig, x_false_adv, y_false_adv)
                
                all_results[at_type] = {
                    'full': at_full,
                    'true': at_true,
                    'false': at_false,
                    'true_count': len(true_advs),
                    'false_count': len(false_advs)
                }
                
                logger.info(f"✅ Successfully generated {at_type.upper()} adversaries: {len(true_advs)} true, {len(false_advs)} false")
                
            except Exception as e:
                logger.error(f"❌ Failed to generate {at_type.upper()} adversaries: {e}")
                continue
        
        # Use first attack type results for backward compatibility
        if all_results:
            first_at = list(all_results.keys())[0]
            logger.info(f"Using {first_at.upper()} results as primary output")
            # Load first attack type results for backward compatibility
            data = np.load(all_results[first_at]['full'])
            x_test_adv_orig = data['arr_0']
            x_test_adv = data['arr_1']
            y_test_adv = data['arr_2']
            
            data_true = np.load(all_results[first_at]['true'])
            x_true_adv_orig = data_true['arr_0']
            x_true_adv = data_true['arr_1']
            y_true_adv = data_true['arr_2']
            
            data_false = np.load(all_results[first_at]['false'])
            x_false_adv_orig = data_false['arr_0']
            x_false_adv = data_false['arr_1']
            y_false_adv = data_false['arr_2']
        else:
            raise RuntimeError("Failed to generate adversaries for any attack type")
    else:
        # Single attack mode (original behavior)
        single_attack = attack_type if not attack_types else attack_types[0]
        at_params = (attack_params or {}).get(single_attack, {})
        
        # Generate adversaries
        msg = f"🎯 Generating {adversarial_sample_size} adversarial examples using {single_attack.upper()} with eps={eps}"
        print(msg, flush=True)
        logger.info(msg)
        true_advs, false_advs = gen_adversaries(
            classifier_model, 
            adversarial_sample_size, 
            images_to_use, 
            labels_to_use, 
            eps,
            clip_min=clip_min,
            clip_max=clip_max,
            batch_size=BATCH_SIZE,
            attack_type=single_attack,
            attack_params=at_params
        )
        full_advs = true_advs + false_advs

        # Extract arrays from tuples (optimized - single pass)
        logger.info("Extracting arrays from adversaries...")
        x_test_adv_orig = np.array([x for x, x_adv, y in full_advs])
        x_test_adv = np.array([x_adv for x, x_adv, y in full_advs])
        y_test_adv = np.array([y for x, x_adv, y in full_advs])

        x_true_adv_orig = np.array([x for x, x_adv, y in true_advs])
        x_true_adv = np.array([x_adv for x, x_adv, y in true_advs])
        y_true_adv = np.array([y for x, x_adv, y in true_advs])

        x_false_adv_orig = np.array([x for x, x_adv, y in false_advs])
        x_false_adv = np.array([x_adv for x, x_adv, y in false_advs])
        y_false_adv = np.array([y for x, x_adv, y in false_advs])

        # Create directories if they don't exist
        os.makedirs(os.path.dirname(npz_full_file_name) if os.path.dirname(npz_full_file_name) else '.', exist_ok=True)
        os.makedirs(os.path.dirname(npz_true_file_name) if os.path.dirname(npz_true_file_name) else '.', exist_ok=True)
        os.makedirs(os.path.dirname(npz_false_file_name) if os.path.dirname(npz_false_file_name) else '.', exist_ok=True)
        
        # Save to NPZ files
        logger.info(f"Saving full adversaries to {npz_full_file_name}")
        np.savez(npz_full_file_name, x_test_adv_orig, x_test_adv, y_test_adv)
        
        logger.info(f"Saving true adversaries to {npz_true_file_name}")
        np.savez(npz_true_file_name, x_true_adv_orig, x_true_adv, y_true_adv)
        
        logger.info(f"Saving false adversaries to {npz_false_file_name}")
        np.savez(npz_false_file_name, x_false_adv_orig, x_false_adv, y_false_adv)
    
    logger.info("Adversarial example generation complete")

def process_single_fgsm_config(eps: float, sample_size: int, 
                               model_path: str, dataset_name: str,
                               output_dir: str) -> dict:
    """
    Process a single epsilon-sample_size combination for FGSM attack.
    
    This is a wrapper function for parallel processing.
    
    Args:
        eps: Epsilon value for FGSM attack.
        sample_size: Sample size for adversarial examples.
        model_path: Path to the model.
        dataset_name: Name of the dataset.
        output_dir: Output directory for results.
        
    Returns:
        Dictionary with results.
    """
    try:
        # Generate file names
        model_path_normalized = model_path.replace("\\", "/")
        model_parts = model_path_normalized.split("/")[-2:]
        model_name = "_".join(model_parts).replace(".h5", "").replace(".keras", "")
        base_filename = f"fgsm_{eps}_{sample_size}_{model_name}.npz"
        
        numpy_array_full_file_name = os.path.join(
            output_dir, dataset_name, "full", base_filename
        )
        numpy_array_true_file_name = os.path.join(
            output_dir, dataset_name, "true", base_filename
        )
        numpy_array_false_file_name = os.path.join(
            output_dir, dataset_name, "false", base_filename
        )
        
        # Run FGSM attack
        fgsm_attack(
            dataset_name,
            model_path=model_path,
            eps=eps,
            adversarial_sample_size=sample_size,
            npz_full_file_name=numpy_array_full_file_name,
            npz_true_file_name=numpy_array_true_file_name,
            npz_false_file_name=numpy_array_false_file_name
        )
        
        return {
            'eps': eps,
            'sample_size': sample_size,
            'success': True,
            'full_file': numpy_array_full_file_name,
            'true_file': numpy_array_true_file_name,
            'false_file': numpy_array_false_file_name
        }
    except Exception as e:
        logger.error(f"Error processing eps={eps}, sample_size={sample_size}: {e}")
        return {
            'eps': eps,
            'sample_size': sample_size,
            'success': False,
            'error': str(e)
        }


if __name__ == '__main__':
    """
    Main entry point for FGSM adversarial example generation.
    
    Loads configuration and generates adversarial examples
    for all combinations of epsilon values and sample sizes.
    Supports both sequential and parallel processing.
    """
    # Force stdout to be unbuffered for immediate output
    import sys
    sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
    
    print("=" * 60, flush=True)
    print("🚀 Starting FGSM Adversarial Example Generation", flush=True)
    print("=" * 60, flush=True)
    
    # Optimized configuration values for adversarial generation
    dataset_name = "cifar10"
    model_to_attack_path = "../models/original_09-11-2025/cifar10_30_CIFAR10_BASE_2/Original_checkpoint_best.keras"
    adversarial_sample_size_list = [10000]
    eps_list = [0.01, 0.015, 0.02]  # Added 0.02 for stronger watermarks
    
    # Parallel processing configuration
    use_parallel = os.getenv('USE_PARALLEL', 'false').lower() == 'true'
    max_workers = int(os.getenv('MAX_WORKERS', '0')) or None

    print(f"📋 Configuration:", flush=True)
    print(f"   Dataset: {dataset_name}", flush=True)
    print(f"   Model: {model_to_attack_path}", flush=True)
    print(f"   Sample sizes: {adversarial_sample_size_list}", flush=True)
    print(f"   Epsilon values: {eps_list}", flush=True)
    print(f"   Parallel processing: {use_parallel}", flush=True)
    if use_parallel and max_workers:
        print(f"   Max workers: {max_workers}", flush=True)
    print("", flush=True)

    # Validate model path
    if not os.path.exists(model_to_attack_path):
        print(f"❌ ERROR: Model file not found: {model_to_attack_path}", flush=True)
        logger.error(f"Model file not found: {model_to_attack_path}")
        sys.exit(1)

    # Create directory structure
    dirs_to_create = [
        DATA_PATH,
        os.path.join(DATA_PATH, dataset_name),
        os.path.join(DATA_PATH, dataset_name, "full"),
        os.path.join(DATA_PATH, dataset_name, "true"),
        os.path.join(DATA_PATH, dataset_name, "false")
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)

    total_combinations = len(adversarial_sample_size_list) * len(eps_list)
    
    if use_parallel and total_combinations > 1:
        # Parallel processing
        print("🚀 Using parallel processing", flush=True)
        logger.info("Using parallel processing")
        
        # Create configuration list
        configs = create_config_list(
            eps_list, adversarial_sample_size_list,
            model_to_attack_path, dataset_name, DATA_PATH
        )
        
        # Run parallel processing
        results = run_parallel_processing(
            configs,
            process_single_fgsm_config,
            max_workers=max_workers
        )
        
        # Process results
        for result in results:
            if result['success']:
                print(f"✅ Successfully generated adversaries for eps={result['eps']}, sample_size={result['sample_size']}", flush=True)
            else:
                print(f"❌ ERROR generating adversaries for eps={result['eps']}, sample_size={result['sample_size']}: {result.get('error', 'Unknown error')}", flush=True)
                logger.error(
                    f"Error generating adversaries for eps={result['eps']}, "
                    f"sample_size={result['sample_size']}: {result.get('error', 'Unknown error')}"
                )
    else:
        # Sequential processing (original behavior)
        current_combination = 0
        for adversarial_sample_size in adversarial_sample_size_list:
            for eps in eps_list:
                current_combination += 1
                print("", flush=True)
                print("=" * 60, flush=True)
                print(f"🔄 Processing combination {current_combination}/{total_combinations}", flush=True)
                print(f"   Epsilon: {eps}", flush=True)
                print(f"   Sample size: {adversarial_sample_size}", flush=True)
                print("=" * 60, flush=True)
                logger.info(
                    f"Processing combination {current_combination}/{total_combinations}: "
                    f"eps={eps}, sample_size={adversarial_sample_size}"
                )

                # Generate file names - Fixed to handle .keras extension
                model_path_normalized = model_to_attack_path.replace("\\", "/")
                model_parts = model_path_normalized.split("/")[-2:]
                model_name = "_".join(model_parts).replace(".h5", "").replace(".keras", "")
                base_filename = f"fgsm_{eps}_{adversarial_sample_size}_{model_name}.npz"
                
                numpy_array_full_file_name = os.path.join(
                    DATA_PATH, dataset_name, "full", base_filename
                )
                numpy_array_true_file_name = os.path.join(
                    DATA_PATH, dataset_name, "true", base_filename
                )
                numpy_array_false_file_name = os.path.join(
                    DATA_PATH, dataset_name, "false", base_filename
                )

                try:
                    fgsm_attack(
                        dataset_name,
                        model_path=model_to_attack_path,
                        eps=eps,
                        adversarial_sample_size=adversarial_sample_size,
                        npz_full_file_name=numpy_array_full_file_name,
                        npz_true_file_name=numpy_array_true_file_name,
                        npz_false_file_name=numpy_array_false_file_name
                    )
                    
                    print(f"✅ Successfully generated adversaries for eps={eps}, sample_size={adversarial_sample_size}", flush=True)
                    
                except Exception as e:
                    print(f"❌ ERROR generating adversaries for eps={eps}, sample_size={adversarial_sample_size}: {e}", flush=True)
                    logger.error(
                        f"Error generating adversaries for eps={eps}, "
                        f"sample_size={adversarial_sample_size}: {e}",
                        exc_info=True
                    )
                    continue

    print("", flush=True)
    print("=" * 60, flush=True)
    print("✅ All adversarial example generation complete!", flush=True)
    print("=" * 60, flush=True)
    logger.info("All adversarial example generation complete")
