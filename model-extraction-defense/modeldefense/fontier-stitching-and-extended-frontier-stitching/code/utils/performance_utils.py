"""
Performance optimization utilities for training and data processing.

This module provides utilities for parallel processing, memory-efficient
data loading, and mixed precision training.
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, Optional, Callable, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import logging

logger = logging.getLogger(__name__)


def create_data_generator(x: np.ndarray, y: np.ndarray, 
                         batch_size: int = 64,
                         shuffle: bool = True,
                         buffer_size: Optional[int] = None) -> tf.data.Dataset:
    """
    Create memory-efficient data generator using tf.data.
    
    Args:
        x: Input features.
        y: Labels.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data.
        buffer_size: Buffer size for shuffling. If None, uses 10x batch_size.
        
    Returns:
        tf.data.Dataset configured for efficient loading.
    """
    if buffer_size is None:
        buffer_size = min(10000, len(x))
    
    # Create dataset from tensors
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    
    # Shuffle if requested
    if shuffle:
        dataset = dataset.shuffle(buffer_size, seed=42)
    
    # Batch and prefetch for performance
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def create_data_generator_with_validation(
        x_train: np.ndarray, y_train: np.ndarray,
        x_val: np.ndarray, y_val: np.ndarray,
        batch_size: int = 64,
        shuffle_train: bool = True) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Create train and validation data generators.
    
    Args:
        x_train: Training features.
        y_train: Training labels.
        x_val: Validation features.
        y_val: Validation labels.
        batch_size: Batch size.
        shuffle_train: Whether to shuffle training data.
        
    Returns:
        Tuple of (train_dataset, val_dataset).
    """
    train_dataset = create_data_generator(
        x_train, y_train, batch_size, shuffle=shuffle_train
    )
    val_dataset = create_data_generator(
        x_val, y_val, batch_size, shuffle=False
    )
    
    return train_dataset, val_dataset


def enable_mixed_precision(policy_name: str = 'mixed_float16') -> None:
    """
    Enable mixed precision training for faster training on GPUs.
    
    This can provide 2-3x speedup on modern GPUs (V100, A100, etc.)
    while maintaining numerical stability.
    
    Args:
        policy_name: Precision policy ('mixed_float16' or 'mixed_bfloat16').
    """
    try:
        policy = tf.keras.mixed_precision.Policy(policy_name)
        tf.keras.mixed_precision.set_global_policy(policy)
        logger.info(f"✅ Mixed precision enabled: {policy_name}")
        print(f"✅ Mixed precision enabled: {policy_name}", flush=True)
    except Exception as e:
        logger.warning(f"Failed to enable mixed precision: {e}")
        print(f"⚠️  Warning: Failed to enable mixed precision: {e}", flush=True)


def disable_mixed_precision() -> None:
    """Disable mixed precision training."""
    try:
        tf.keras.mixed_precision.set_global_policy('float32')
        logger.info("Mixed precision disabled")
    except Exception as e:
        logger.warning(f"Failed to disable mixed precision: {e}")


def process_single_config(
        eps: float,
        sample_size: int,
        model_path: str,
        dataset_name: str,
        output_dir: str,
        process_func: Callable) -> dict:
    """
    Process a single epsilon-sample_size combination.
    
    This is a wrapper function for parallel processing.
    
    Args:
        eps: Epsilon value for FGSM attack.
        sample_size: Sample size for adversarial examples.
        model_path: Path to the model.
        dataset_name: Name of the dataset.
        output_dir: Output directory for results.
        process_func: Function to process the configuration.
        
    Returns:
        Dictionary with results.
    """
    try:
        result = process_func(eps, sample_size, model_path, dataset_name, output_dir)
        return {
            'eps': eps,
            'sample_size': sample_size,
            'success': True,
            'result': result
        }
    except Exception as e:
        logger.error(f"Error processing eps={eps}, sample_size={sample_size}: {e}")
        return {
            'eps': eps,
            'sample_size': sample_size,
            'success': False,
            'error': str(e)
        }


def run_parallel_processing(
        configs: list,
        process_func: Callable,
        max_workers: Optional[int] = None,
        timeout: Optional[float] = None) -> list:
    """
    Run parallel processing of configurations.
    
    Args:
        configs: List of configuration dictionaries, each containing:
                 {'eps': float, 'sample_size': int, 'model_path': str, 
                  'dataset_name': str, 'output_dir': str}
        process_func: Function to process each configuration.
        max_workers: Maximum number of worker processes. If None, uses CPU count - 1.
        timeout: Timeout in seconds for each task. If None, no timeout.
        
    Returns:
        List of results from processing.
    """
    if max_workers is None:
        max_workers = max(1, mp.cpu_count() - 1)
    
    results = []
    
    logger.info(f"Starting parallel processing with {max_workers} workers")
    print(f"🚀 Starting parallel processing with {max_workers} workers", flush=True)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {}
        for config in configs:
            future = executor.submit(
                process_single_config,
                config['eps'],
                config['sample_size'],
                config['model_path'],
                config['dataset_name'],
                config.get('output_dir', '../results'),
                process_func
            )
            futures[future] = config
        
        # Collect results as they complete
        for future in as_completed(futures, timeout=timeout):
            config = futures[future]
            try:
                result = future.result()
                results.append(result)
                logger.info(f"Completed: eps={config['eps']}, sample_size={config['sample_size']}")
            except Exception as e:
                logger.error(f"Error in parallel processing: {e}")
                results.append({
                    'eps': config['eps'],
                    'sample_size': config['sample_size'],
                    'success': False,
                    'error': str(e)
                })
    
    logger.info(f"Parallel processing completed: {len(results)} results")
    print(f"✅ Parallel processing completed: {len(results)} results", flush=True)
    
    return results


def create_config_list(eps_list: list, sample_size_list: list,
                      model_path: str, dataset_name: str,
                      output_dir: str = '../results') -> list:
    """
    Create list of configurations for parallel processing.
    
    Args:
        eps_list: List of epsilon values.
        sample_size_list: List of sample sizes.
        model_path: Path to the model.
        dataset_name: Name of the dataset.
        output_dir: Output directory.
        
    Returns:
        List of configuration dictionaries.
    """
    configs = []
    for eps in eps_list:
        for sample_size in sample_size_list:
            configs.append({
                'eps': eps,
                'sample_size': sample_size,
                'model_path': model_path,
                'dataset_name': dataset_name,
                'output_dir': output_dir
            })
    return configs


def optimize_gpu_memory() -> None:
    """
    Optimize GPU memory settings for TensorFlow.
    
    Enables memory growth to prevent TensorFlow from allocating all GPU memory.
    """
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"GPU memory growth enabled for {gpu}")
                except RuntimeError as e:
                    logger.warning(f"Could not set GPU memory growth: {e}")
    except Exception as e:
        logger.warning(f"Error optimizing GPU memory: {e}")


def get_optimal_batch_size(base_batch_size: int = 128,
                           available_memory_gb: Optional[float] = None) -> int:
    """
    Calculate optimal batch size based on available memory.
    
    Args:
        base_batch_size: Base batch size to use.
        available_memory_gb: Available GPU memory in GB. If None, uses base_batch_size.
        
    Returns:
        Optimal batch size.
    """
    if available_memory_gb is None:
        return base_batch_size
    
    # Rough estimate: 1GB per 1000 samples for CIFAR-10
    # Adjust based on dataset and model complexity
    samples_per_gb = 1000
    max_batch_size = int(available_memory_gb * samples_per_gb)
    
    return min(base_batch_size, max_batch_size)

