"""
Multi-Attack Adversarial Example Generation for Watermarking

This module provides support for multiple attack types using the Adversarial Robustness Toolbox (ART).
Supports: FGSM, PGD, BIM, C&W, DeepFool, JSMA, and more.

Reference: https://github.com/Trusted-AI/adversarial-robustness-toolbox
"""

import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from typing import Tuple, List, Optional, Dict, Any

from art.estimators.classification import KerasClassifier
from art.attacks.evasion import (
    FastGradientMethod,
    ProjectedGradientDescent,
    BasicIterativeMethod,
    CarliniLInfMethod,
    CarliniL2Method,
    DeepFool,
    SaliencyMapMethod,
)

# Try to import optional attacks (may not be available in all ART versions)
try:
    from art.attacks.evasion import AutoAttack, HopSkipJump
    AUTOATTACK_AVAILABLE = True
    HOPSKIPJUMP_AVAILABLE = True
except ImportError:
    AUTOATTACK_AVAILABLE = False
    HOPSKIPJUMP_AVAILABLE = False

from utils.data_utils import DataManager

logger = logging.getLogger(__name__)

# Attack type registry
ATTACK_REGISTRY = {
    'fgsm': FastGradientMethod,
    'pgd': ProjectedGradientDescent,
    'bim': BasicIterativeMethod,
    'carlini_l2': CarliniL2Method,
    'carlini_linf': CarliniLInfMethod,
    'deepfool': DeepFool,
    'jsma': SaliencyMapMethod,
}

# Add optional attacks if available
if AUTOATTACK_AVAILABLE:
    ATTACK_REGISTRY['autoattack'] = AutoAttack
if HOPSKIPJUMP_AVAILABLE:
    ATTACK_REGISTRY['hopskipjump'] = HopSkipJump


def create_art_classifier(model_path: str, clip_values: Tuple[float, float] = (0., 1.)) -> KerasClassifier:
    """
    Create ART KerasClassifier wrapper from model file.
    
    Args:
        model_path: Path to Keras model file.
        clip_values: Tuple of (min, max) values for input clipping.
        
    Returns:
        KerasClassifier instance.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    logger.info(f"Loading model from: {model_path}")
    model = load_model(model_path)
    
    classifier = KerasClassifier(
        clip_values=clip_values,
        model=model,
        use_logits=False
    )
    
    return classifier


def create_attack_instance(attack_type: str, classifier: KerasClassifier, 
                           eps: float = 0.3, **attack_params) -> Any:
    """
    Create an attack instance from ART.
    
    Args:
        attack_type: Type of attack ('fgsm', 'pgd', 'bim', etc.).
        classifier: ART classifier to attack.
        eps: Perturbation level (epsilon).
        **attack_params: Additional attack-specific parameters.
        
    Returns:
        Attack instance.
        
    Raises:
        ValueError: If attack_type is not supported.
    """
    attack_type_lower = attack_type.lower()
    
    if attack_type_lower not in ATTACK_REGISTRY:
        raise ValueError(
            f"Unsupported attack type: {attack_type}. "
            f"Supported types: {list(ATTACK_REGISTRY.keys())}"
        )
    
    attack_class = ATTACK_REGISTRY[attack_type_lower]
    
    # Default parameters for each attack type
    default_params = {
        'fgsm': {'eps': eps},
        'pgd': {'eps': eps, 'eps_step': eps / 10, 'max_iter': 10},
        'bim': {'eps': eps, 'eps_step': eps / 10, 'max_iter': 10},
        'carlini_l2': {'confidence': 0.0, 'learning_rate': 0.01, 'max_iter': 10},
        'carlini_linf': {'eps': eps, 'max_iter': 10},
        'deepfool': {'max_iter': 50, 'epsilon': eps},
        'jsma': {'theta': eps, 'gamma': 1.0},
    }
    
    # Add optional attacks if available
    if AUTOATTACK_AVAILABLE and attack_type_lower == 'autoattack':
        default_params['autoattack'] = {'eps': eps, 'n_classes': 10}
    if HOPSKIPJUMP_AVAILABLE and attack_type_lower == 'hopskipjump':
        default_params['hopskipjump'] = {'max_iter': 50, 'max_eval': 10000, 'init_eval': 100}
    
    # Merge default params with provided params
    params = default_params.get(attack_type_lower, {})
    params.update(attack_params)
    
    # Override eps for attacks that support it
    if 'eps' in params:
        params['eps'] = eps
    
    try:
        attack = attack_class(classifier, **params)
        logger.info(f"Created {attack_type.upper()} attack with params: {params}")
        return attack
    except Exception as e:
        logger.error(f"Failed to create {attack_type} attack: {e}")
        raise


def generate_adversarial_examples(
    dataset_name: str,
    model_path: str,
    attack_type: str = 'fgsm',
    eps: float = 0.3,
    adversarial_sample_size: int = 1000,
    clip_values: Tuple[float, float] = (0., 1.),
    attack_params: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate adversarial examples using specified attack type.
    
    Args:
        dataset_name: Name of the dataset ('mnist' or 'cifar10').
        model_path: Path to the model file.
        attack_type: Type of attack ('fgsm', 'pgd', 'bim', etc.).
        eps: Perturbation level (epsilon).
        adversarial_sample_size: Number of adversarial samples to generate.
        clip_values: Tuple of (min, max) values for clipping.
        attack_params: Additional attack-specific parameters.
        output_path: Optional path to save adversarial examples.
        
    Returns:
        Tuple of (x_original, x_adversarial, y_labels).
    """
    # Load data
    x_train, y_train, x_test, y_test, input_shape, num_classes = DataManager.load_and_preprocess(dataset_name)
    
    # Validate sample size
    if adversarial_sample_size > len(x_test):
        raise ValueError(
            f"adversarial_sample_size ({adversarial_sample_size}) exceeds "
            f"available test samples ({len(x_test)})"
        )
    
    # Create ART classifier
    classifier = create_art_classifier(model_path, clip_values)
    
    # Prepare data
    x_original = x_test[:adversarial_sample_size]
    y_labels = y_test[:adversarial_sample_size]
    
    # Create attack instance
    attack_params = attack_params or {}
    attacker = create_attack_instance(attack_type, classifier, eps, **attack_params)
    
    # Generate adversarial examples
    logger.info(f"Generating {adversarial_sample_size} adversarial examples using {attack_type.upper()} with eps={eps}...")
    try:
        x_adversarial = attacker.generate(x_original)
        logger.info(f"Successfully generated {len(x_adversarial)} adversarial examples")
    except Exception as e:
        logger.error(f"Failed to generate adversarial examples: {e}")
        raise
    
    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        np.savez(output_path, x_original, x_adversarial, y_labels)
        logger.info(f"Saved adversarial examples to: {output_path}")
    
    return x_original, x_adversarial, y_labels


def generate_multiple_attacks(
    dataset_name: str,
    model_path: str,
    attack_types: List[str],
    eps: float = 0.3,
    adversarial_sample_size: int = 1000,
    clip_values: Tuple[float, float] = (0., 1.),
    attack_params: Optional[Dict[str, Dict[str, Any]]] = None,
    output_dir: str = "../data/adversarial"
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Generate adversarial examples using multiple attack types.
    
    Args:
        dataset_name: Name of the dataset.
        model_path: Path to the model file.
        attack_types: List of attack types to use.
        eps: Perturbation level.
        adversarial_sample_size: Number of samples to generate.
        clip_values: Tuple of (min, max) values.
        attack_params: Dict mapping attack_type to its specific parameters.
        output_dir: Directory to save adversarial examples.
        
    Returns:
        Dict mapping attack_type to (x_original, x_adversarial, y_labels).
    """
    results = {}
    attack_params = attack_params or {}
    
    for attack_type in attack_types:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing attack: {attack_type.upper()}")
        logger.info(f"{'='*60}")
        
        try:
            # Get attack-specific parameters
            params = attack_params.get(attack_type, {})
            
            # Generate output path
            model_name = os.path.basename(model_path).replace('.h5', '').replace('.keras', '')
            output_filename = f"{attack_type}_{eps}_{adversarial_sample_size}_{model_name}.npz"
            output_path = os.path.join(output_dir, dataset_name, output_filename)
            
            # Generate adversarial examples
            x_orig, x_adv, y_labels = generate_adversarial_examples(
                dataset_name=dataset_name,
                model_path=model_path,
                attack_type=attack_type,
                eps=eps,
                adversarial_sample_size=adversarial_sample_size,
                clip_values=clip_values,
                attack_params=params,
                output_path=output_path
            )
            
            results[attack_type] = (x_orig, x_adv, y_labels)
            logger.info(f"✅ Successfully generated {attack_type.upper()} adversarial examples")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate {attack_type.upper()} adversarial examples: {e}")
            continue
    
    return results


def get_supported_attacks() -> List[str]:
    """Get list of supported attack types."""
    return list(ATTACK_REGISTRY.keys())


def get_attack_info(attack_type: str) -> Dict[str, Any]:
    """
    Get information about a specific attack type.
    
    Args:
        attack_type: Type of attack.
        
    Returns:
        Dict with attack information.
    """
    if attack_type.lower() not in ATTACK_REGISTRY:
        return {'error': f'Unknown attack type: {attack_type}'}
    
    info = {
        'fgsm': {
            'name': 'Fast Gradient Sign Method',
            'paper': 'https://arxiv.org/abs/1412.6572',
            'description': 'Single-step gradient-based attack',
            'params': ['eps']
        },
        'pgd': {
            'name': 'Projected Gradient Descent',
            'paper': 'https://arxiv.org/abs/1706.06083',
            'description': 'Multi-step iterative attack with projection',
            'params': ['eps', 'eps_step', 'max_iter']
        },
        'bim': {
            'name': 'Basic Iterative Method',
            'paper': 'https://arxiv.org/abs/1607.02533',
            'description': 'Iterative version of FGSM',
            'params': ['eps', 'eps_step', 'max_iter']
        },
        'carlini_l2': {
            'name': 'Carlini & Wagner L2',
            'paper': 'https://arxiv.org/abs/1608.04644',
            'description': 'Optimization-based attack (L2 norm)',
            'params': ['confidence', 'learning_rate', 'max_iter']
        },
        'carlini_linf': {
            'name': 'Carlini & Wagner L∞',
            'paper': 'https://arxiv.org/abs/1608.04644',
            'description': 'Optimization-based attack (L∞ norm)',
            'params': ['eps', 'max_iter']
        },
        'deepfool': {
            'name': 'DeepFool',
            'paper': 'https://arxiv.org/abs/1511.04599',
            'description': 'Minimal perturbation attack',
            'params': ['max_iter', 'epsilon']
        },
        'jsma': {
            'name': 'Jacobian-based Saliency Map Attack',
            'paper': 'https://arxiv.org/abs/1511.07528',
            'description': 'Sparse attack modifying few pixels',
            'params': ['theta', 'gamma']
        },
    }
    
    # Add optional attacks if available
    if AUTOATTACK_AVAILABLE:
        info['autoattack'] = {
            'name': 'AutoAttack',
            'paper': 'https://arxiv.org/abs/2003.01690',
            'description': 'Ensemble of attacks for robustness evaluation',
            'params': ['eps', 'n_classes']
        }
    if HOPSKIPJUMP_AVAILABLE:
        info['hopskipjump'] = {
            'name': 'HopSkipJump',
            'paper': 'https://arxiv.org/abs/1904.02144',
            'description': 'Query-efficient black-box attack',
            'params': ['max_iter', 'max_eval', 'init_eval']
        }
    
    return info.get(attack_type.lower(), {'name': attack_type, 'description': 'Unknown attack'})

