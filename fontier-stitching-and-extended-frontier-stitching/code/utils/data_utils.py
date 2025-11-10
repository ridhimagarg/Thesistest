"""
Centralized data management utilities.

This module provides a single source of truth for data loading and preprocessing,
eliminating the duplication of data_preprocessing() across multiple files.
"""

import os
import logging
import shutil
from typing import Tuple, Optional, Dict, Any
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10, cifar100

# Configure logger
logger = logging.getLogger(__name__)

# Try to import tensorflow_datasets for STL-10, EuroSAT, and SVHN
try:
    import tensorflow_datasets as tfds
    TFDS_AVAILABLE = True
except ImportError:
    TFDS_AVAILABLE = False
    logger.warning("tensorflow_datasets not available. STL-10, EuroSAT, and SVHN will not work.")


def _check_disk_space(path: str, required_gb: float = 5.0) -> Tuple[bool, float, float]:
    """
    Check available disk space at the given path.
    
    Args:
        path: Path to check disk space for.
        required_gb: Required space in GB (default: 5GB for large datasets).
        
    Returns:
        Tuple of (has_enough_space, available_gb, total_gb).
    """
    try:
        stat = shutil.disk_usage(path)
        available_gb = stat.free / (1024 ** 3)  # Convert bytes to GB
        total_gb = stat.total / (1024 ** 3)
        has_enough = available_gb >= required_gb
        return has_enough, available_gb, total_gb
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")
        return True, 0.0, 0.0  # Assume enough space if check fails


def _get_tfds_dir() -> str:
    """Get TensorFlow Datasets directory path."""
    try:
        import tensorflow_datasets as tfds
        # Try to get the actual TFDS data directory
        tfds_dir = os.path.expanduser("~/tensorflow_datasets")
        # Also check common alternative locations
        alt_paths = [
            os.path.expanduser("~/tensorflow_datasets"),
            "/Users/akshatgupta/tensorflow_datasets",  # From error message
            os.path.join(os.path.expanduser("~"), "tensorflow_datasets"),
        ]
        for path in alt_paths:
            if os.path.exists(path):
                return path
        return tfds_dir  # Return default even if it doesn't exist yet
    except Exception:
        return os.path.expanduser("~/tensorflow_datasets")


def _cleanup_incomplete_datasets(dataset_name: str):
    """Clean up incomplete TensorFlow Datasets downloads."""
    try:
        tfds_dir = _get_tfds_dir()
        incomplete_dir = os.path.join(tfds_dir, dataset_name)
        
        # Look for incomplete directories
        if os.path.exists(incomplete_dir):
            for item in os.listdir(incomplete_dir):
                if item.startswith("incomplete."):
                    incomplete_path = os.path.join(incomplete_dir, item)
                    try:
                        logger.info(f"Cleaning up incomplete dataset: {incomplete_path}")
                        shutil.rmtree(incomplete_path)
                        logger.info(f"Successfully removed incomplete dataset: {incomplete_path}")
                    except Exception as e:
                        logger.warning(f"Could not remove incomplete dataset {incomplete_path}: {e}")
    except Exception as e:
        logger.warning(f"Could not cleanup incomplete datasets: {e}")


def _load_svhn_tfds():
    """Load SVHN dataset from TensorFlow Datasets."""
    if not TFDS_AVAILABLE:
        raise ImportError("tensorflow_datasets is required for SVHN. Install with: pip install tensorflow-datasets")
    
    # Check disk space before loading (SVHN needs ~2GB)
    import tensorflow_datasets as tfds
    tfds_dir = _get_tfds_dir()
    has_space, available_gb, total_gb = _check_disk_space(tfds_dir, required_gb=2.0)
    
    if not has_space:
        _cleanup_incomplete_datasets('svhn_cropped')
        has_space, available_gb, total_gb = _check_disk_space(tfds_dir, required_gb=2.0)
        
        if not has_space:
            error_msg = (
                f"Insufficient disk space to load SVHN dataset.\n"
                f"  Required: ~2 GB\n"
                f"  Available: {available_gb:.2f} GB\n"
                f"  Please free up disk space or clean up incomplete downloads."
            )
            raise RuntimeError(error_msg)
    
    logger.info(f"Loading SVHN dataset from TensorFlow Datasets... (Available space: {available_gb:.2f} GB)")
    
    try:
        train_ds = tfds.load('svhn_cropped', split='train', as_supervised=True, shuffle_files=True)
        test_ds = tfds.load('svhn_cropped', split='test', as_supervised=True, shuffle_files=False)
    except Exception as e:
        if "No space left on device" in str(e) or "ResourceExhaustedError" in str(type(e).__name__):
            _cleanup_incomplete_datasets('svhn_cropped')
            error_msg = (
                f"Failed to load SVHN dataset due to insufficient disk space.\n"
                f"  Error: {str(e)}\n"
                f"  Please free up at least 2 GB of disk space."
            )
            raise RuntimeError(error_msg) from e
        raise
    
    # Optimized: Use batch conversion instead of per-item loops
    def _process_batch(batch):
        """Process a batch of images and labels."""
        imgs, labels = batch
        # Convert to numpy in batch
        if hasattr(imgs, 'numpy'):
            imgs_np = imgs.numpy()
        else:
            imgs_np = np.array(imgs)
        
        # Ensure uint8 format
        if imgs_np.dtype != np.uint8:
            imgs_np = (imgs_np * 255).astype(np.uint8) if imgs_np.max() <= 1.0 else imgs_np.astype(np.uint8)
        
        if hasattr(labels, 'numpy'):
            labels_np = labels.numpy()
        else:
            labels_np = np.array(labels)
        
        return imgs_np, labels_np.astype(np.int32)
    
    # Batch processing for efficiency
    x_train, y_train = [], []
    batch_size = 1000
    train_ds_batched = train_ds.batch(batch_size)
    for batch in train_ds_batched:
        imgs_batch, labels_batch = _process_batch(batch)
        x_train.append(imgs_batch)
        y_train.append(labels_batch)
    
    x_test, y_test = [], []
    test_ds_batched = test_ds.batch(batch_size)
    for batch in test_ds_batched:
        imgs_batch, labels_batch = _process_batch(batch)
        x_test.append(imgs_batch)
        y_test.append(labels_batch)
    
    # Concatenate batches
    x_train = np.concatenate(x_train, axis=0) if len(x_train) > 1 else x_train[0]
    y_train = np.concatenate(y_train, axis=0) if len(y_train) > 1 else y_train[0]
    x_test = np.concatenate(x_test, axis=0) if len(x_test) > 1 else x_test[0]
    y_test = np.concatenate(y_test, axis=0) if len(y_test) > 1 else y_test[0]
    
    return (x_train, y_train), (x_test, y_test)


def _load_stl10():
    """Load STL-10 dataset from TensorFlow Datasets."""
    if not TFDS_AVAILABLE:
        raise ImportError("tensorflow_datasets is required for STL-10. Install with: pip install tensorflow-datasets")
    
    # Check disk space before loading (STL-10 needs ~5GB for download + processing)
    import tensorflow_datasets as tfds
    tfds_dir = _get_tfds_dir()
    has_space, available_gb, total_gb = _check_disk_space(tfds_dir, required_gb=5.0)
    
    if not has_space:
        # Try to clean up incomplete downloads
        _cleanup_incomplete_datasets('stl10')
        has_space, available_gb, total_gb = _check_disk_space(tfds_dir, required_gb=5.0)
        
        if not has_space:
            error_msg = (
                f"Insufficient disk space to load STL-10 dataset.\n"
                f"  Required: ~5 GB\n"
                f"  Available: {available_gb:.2f} GB\n"
                f"  Total: {total_gb:.2f} GB\n"
                f"  Path: {tfds_dir}\n\n"
                f"Please free up disk space or clean up incomplete dataset downloads:\n"
                f"  rm -rf {tfds_dir}/stl10/incomplete.*"
            )
            raise RuntimeError(error_msg)
    
    logger.info(f"Loading STL-10 dataset from TensorFlow Datasets... (Available space: {available_gb:.2f} GB)")
    
    # Disable shuffling to reduce temporary file requirements when disk space is tight
    shuffle_files = available_gb > 3.0  # Only shuffle if we have >3GB free
    if not shuffle_files:
        logger.warning("Disk space is limited. Disabling file shuffling to reduce temporary file requirements.")
    
    try:
        train_ds = tfds.load('stl10', split='train', as_supervised=True, shuffle_files=shuffle_files)
        test_ds = tfds.load('stl10', split='test', as_supervised=True, shuffle_files=False)
    except Exception as e:
        if "No space left on device" in str(e) or "ResourceExhaustedError" in str(type(e).__name__):
            # Clean up and provide helpful error message
            _cleanup_incomplete_datasets('stl10')
            error_msg = (
                f"Failed to load STL-10 dataset due to insufficient disk space.\n"
                f"  Error: {str(e)}\n\n"
                f"Please free up disk space. You can:\n"
                f"  1. Clean up incomplete downloads: rm -rf {tfds_dir}/stl10/incomplete.*\n"
                f"  2. Free up at least 5 GB of disk space\n"
                f"  3. Check available space: df -h {tfds_dir}"
            )
            raise RuntimeError(error_msg) from e
        raise
    
    # Optimized: Use batch conversion instead of per-item loops
    def _process_batch(batch):
        """Process a batch of images and labels."""
        imgs, labels = batch
        # Convert to numpy in batch
        if hasattr(imgs, 'numpy'):
            imgs_np = imgs.numpy()
        else:
            imgs_np = np.array(imgs)
        
        # Ensure uint8 format
        if imgs_np.dtype != np.uint8:
            imgs_np = (imgs_np * 255).astype(np.uint8) if imgs_np.max() <= 1.0 else imgs_np.astype(np.uint8)
        
        if hasattr(labels, 'numpy'):
            labels_np = labels.numpy()
        else:
            labels_np = np.array(labels)
        
        return imgs_np, labels_np.astype(np.int32)
    
    # Batch processing for efficiency
    x_train, y_train = [], []
    batch_size = 1000
    train_ds_batched = train_ds.batch(batch_size)
    for batch in train_ds_batched:
        imgs_batch, labels_batch = _process_batch(batch)
        x_train.append(imgs_batch)
        y_train.append(labels_batch)
    
    x_test, y_test = [], []
    test_ds_batched = test_ds.batch(batch_size)
    for batch in test_ds_batched:
        imgs_batch, labels_batch = _process_batch(batch)
        x_test.append(imgs_batch)
        y_test.append(labels_batch)
    
    # Concatenate batches
    x_train = np.concatenate(x_train, axis=0) if len(x_train) > 1 else x_train[0]
    y_train = np.concatenate(y_train, axis=0) if len(y_train) > 1 else y_train[0]
    x_test = np.concatenate(x_test, axis=0) if len(x_test) > 1 else x_test[0]
    y_test = np.concatenate(y_test, axis=0) if len(y_test) > 1 else y_test[0]
    
    return (x_train, y_train), (x_test, y_test)


def _load_eurosat():
    """Load EuroSAT dataset from TensorFlow Datasets."""
    if not TFDS_AVAILABLE:
        raise ImportError("tensorflow_datasets is required for EuroSAT. Install with: pip install tensorflow-datasets")
    
    # Check disk space before loading (EuroSAT needs ~3GB)
    import tensorflow_datasets as tfds
    tfds_dir = _get_tfds_dir()
    has_space, available_gb, total_gb = _check_disk_space(tfds_dir, required_gb=3.0)
    
    if not has_space:
        _cleanup_incomplete_datasets('eurosat')
        has_space, available_gb, total_gb = _check_disk_space(tfds_dir, required_gb=3.0)
        
        if not has_space:
            error_msg = (
                f"Insufficient disk space to load EuroSAT dataset.\n"
                f"  Required: ~3 GB\n"
                f"  Available: {available_gb:.2f} GB\n"
                f"  Please free up disk space or clean up incomplete downloads."
            )
            raise RuntimeError(error_msg)
    
    logger.info(f"Loading EuroSAT dataset from TensorFlow Datasets... (Available space: {available_gb:.2f} GB)")
    
    # Disable shuffling to reduce temporary file requirements when disk space is tight
    shuffle_files = available_gb > 2.0
    if not shuffle_files:
        logger.warning("Disk space is limited. Disabling file shuffling to reduce temporary file requirements.")
    
    try:
        # EuroSAT only has train split, so we split it 80/20
        full_ds = tfds.load('eurosat', split='train', as_supervised=True, shuffle_files=shuffle_files)
    except Exception as e:
        if "No space left on device" in str(e) or "ResourceExhaustedError" in str(type(e).__name__):
            _cleanup_incomplete_datasets('eurosat')
            error_msg = (
                f"Failed to load EuroSAT dataset due to insufficient disk space.\n"
                f"  Error: {str(e)}\n"
                f"  Please free up at least 3 GB of disk space."
            )
            raise RuntimeError(error_msg) from e
        raise
    
    # Optimized: Use batch conversion instead of per-item loops
    def _process_batch(batch):
        """Process a batch of images and labels."""
        imgs, labels = batch
        # Convert to numpy in batch
        if hasattr(imgs, 'numpy'):
            imgs_np = imgs.numpy()
        else:
            imgs_np = np.array(imgs)
        
        # Ensure uint8 format
        if imgs_np.dtype != np.uint8:
            imgs_np = (imgs_np * 255).astype(np.uint8) if imgs_np.max() <= 1.0 else imgs_np.astype(np.uint8)
        
        if hasattr(labels, 'numpy'):
            labels_np = labels.numpy()
        else:
            labels_np = np.array(labels)
        
        return imgs_np, labels_np.astype(np.int32)
    
    # Batch processing for efficiency
    all_imgs, all_labels = [], []
    batch_size = 1000
    full_ds_batched = full_ds.batch(batch_size)
    for batch in full_ds_batched:
        imgs_batch, labels_batch = _process_batch(batch)
        all_imgs.append(imgs_batch)
        all_labels.append(labels_batch)
    
    # Concatenate all batches
    all_imgs = np.concatenate(all_imgs, axis=0) if len(all_imgs) > 1 else all_imgs[0]
    all_labels = np.concatenate(all_labels, axis=0) if len(all_labels) > 1 else all_labels[0]
    
    # Split 80/20
    split_idx = int(len(all_imgs) * 0.8)
    x_train = all_imgs[:split_idx]
    y_train = all_labels[:split_idx]
    x_test = all_imgs[split_idx:]
    y_test = all_labels[split_idx:]
    
    return (x_train, y_train), (x_test, y_test)


class DataManager:
    """Centralized data management for model training and evaluation."""
    
    # Dataset configuration mapping
    DATASET_CONFIG = {
        'mnist': {
            'img_rows': 28, 
            'img_cols': 28, 
            'num_channels': 1, 
            'num_classes': 10,
            'loader': mnist.load_data
        },
        'cifar10': {
            'img_rows': 32, 
            'img_cols': 32, 
            'num_channels': 3, 
            'num_classes': 10,
            'loader': cifar10.load_data
        },
        'cifar100': {
            'img_rows': 32, 
            'img_cols': 32, 
            'num_channels': 3, 
            'num_classes': 100,
            'loader': cifar100.load_data
        },
        'svhn': {
            'img_rows': 32, 
            'img_cols': 32, 
            'num_channels': 3, 
            'num_classes': 10,
            'loader': _load_svhn_tfds
        },
        'stl10': {
            'img_rows': 96, 
            'img_cols': 96, 
            'num_channels': 3, 
            'num_classes': 10,
            'loader': _load_stl10
        },
        'eurosat': {
            'img_rows': 64, 
            'img_cols': 64, 
            'num_channels': 3, 
            'num_classes': 10,
            'loader': _load_eurosat
        },
        'cifar10resnet': {
            'img_rows': 32, 
            'img_cols': 32, 
            'num_channels': 3, 
            'num_classes': 10,
            'loader': cifar10.load_data,
            'normalize_method': 'resnet'
        },
        'cifar10resnet_255_preprocess': {
            'img_rows': 32, 
            'img_cols': 32, 
            'num_channels': 3, 
            'num_classes': 10,
            'loader': cifar10.load_data,
            'normalize_method': 'standard'
        }
    }
    
    # ResNet normalization parameters for CIFAR-10
    RESNET_MEAN = np.array([125.3, 123.0, 113.9])
    RESNET_STD = np.array([63.0, 62.1, 66.7])
    NORMALIZATION_FACTOR = 255.0
    
    @staticmethod
    def _validate_dataset(dataset_name: str) -> None:
        """
        Validate if dataset name is supported.
        
        Args:
            dataset_name: Name of the dataset to validate.
            
        Raises:
            ValueError: If dataset_name is not supported.
        """
        if dataset_name not in DataManager.DATASET_CONFIG:
            supported = list(DataManager.DATASET_CONFIG.keys())
            raise ValueError(
                f'Invalid dataset name: {dataset_name}. '
                f'Supported datasets: {supported}'
            )
    
    @staticmethod
    def _load_dataset(dataset_name: str) -> Tuple[Tuple[np.ndarray, np.ndarray], 
                                                    Tuple[np.ndarray, np.ndarray]]:
        """
        Load raw dataset from Keras.
        
        Args:
            dataset_name: Name of the dataset to load.
            
        Returns:
            Tuple of ((x_train, y_train), (x_test, y_test))
        """
        DataManager._validate_dataset(dataset_name)
        loader = DataManager.DATASET_CONFIG[dataset_name]['loader']
        return loader()
    
    @staticmethod
    def _normalize_data(x_train: np.ndarray, 
                       x_test: np.ndarray,
                       dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply dataset-specific normalization.
        
        Args:
            x_train: Training images.
            x_test: Test images.
            dataset_name: Name of the dataset (determines normalization method).
            
        Returns:
            Tuple of (normalized_x_train, normalized_x_test)
        """
        config = DataManager.DATASET_CONFIG[dataset_name]
        normalize_method = config.get('normalize_method', 'standard')
        
        # Convert to float32
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        
        if normalize_method == 'resnet':
            # ResNet-specific normalization with mean/std
            x_train = (x_train - DataManager.RESNET_MEAN) / DataManager.RESNET_STD
            x_test = (x_test - DataManager.RESNET_MEAN) / DataManager.RESNET_STD
        else:
            # Standard normalization: divide by 255
            x_train /= DataManager.NORMALIZATION_FACTOR
            x_test /= DataManager.NORMALIZATION_FACTOR
        
        return x_train, x_test
    
    @staticmethod
    def load_and_preprocess(dataset_name: str,
                           normalize: bool = True,
                           shuffle: bool = True,
                           one_hot_encode: bool = True) -> Tuple[np.ndarray, 
                                                                  np.ndarray, 
                                                                  np.ndarray, 
                                                                  np.ndarray, 
                                                                  tuple, 
                                                                  int]:
        """
        Single source of truth for data loading and preprocessing.
        
        This method loads a dataset, preprocesses it (reshaping, normalization, 
        one-hot encoding), and returns it ready for model training/evaluation.
        
        Args:
            dataset_name: Name of the dataset ('mnist', 'cifar10', 'cifar100', 'svhn', 
                         'stl10', 'eurosat', 'cifar10resnet', 'cifar10resnet_255_preprocess').
            normalize: Whether to normalize the data. Default is True.
            shuffle: Whether to shuffle the training data. Default is True.
            one_hot_encode: Whether to one-hot encode labels. Default is True.
            
        Returns:
            Tuple containing:
                - x_train: Preprocessed training images
                - y_train: Training labels (one-hot encoded if one_hot_encode=True)
                - x_test: Preprocessed test images
                - y_test: Test labels (one-hot encoded if one_hot_encode=True)
                - input_shape: Shape of input images (rows, cols, channels)
                - num_classes: Number of classes
                
        Raises:
            ValueError: If dataset_name is not supported.
            
        Example:
            >>> x_train, y_train, x_test, y_test, input_shape, num_classes = \\
            ...     DataManager.load_and_preprocess('cifar10')
        """
        # Validate and get config
        DataManager._validate_dataset(dataset_name)
        config = DataManager.DATASET_CONFIG[dataset_name]
        
        # Load dataset
        (x_train, y_train), (x_test, y_test) = DataManager._load_dataset(dataset_name)
        
        # Shuffle training data
        if shuffle:
            indices = np.random.permutation(len(x_train))
            x_train = x_train[indices]
            y_train = y_train[indices]
        
        # Get image dimensions
        img_rows = config['img_rows']
        img_cols = config['img_cols']
        num_channels = config['num_channels']
        num_classes = config['num_classes']
        input_shape = (img_rows, img_cols, num_channels)
        
        # Reshape data (only if needed - TensorFlow Datasets may already be in correct shape)
        if len(x_train.shape) == 4 and x_train.shape[1:] == (img_rows, img_cols, num_channels):
            # Already in correct shape
            pass
        elif len(x_train.shape) == 4:
            # Different shape, resize if needed (for STL-10, EuroSAT which might be different sizes)
            if x_train.shape[1] != img_rows or x_train.shape[2] != img_cols:
                # Resize images to expected size - batch conversion for efficiency
                x_train_t = tf.image.resize(tf.constant(x_train, dtype=tf.float32), [img_rows, img_cols])
                x_test_t = tf.image.resize(tf.constant(x_test, dtype=tf.float32), [img_rows, img_cols])
                x_train = x_train_t.numpy()
                x_test = x_test_t.numpy()
        else:
            # Need to reshape
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, num_channels)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, num_channels)
        
        # One-hot encode labels
        if one_hot_encode:
            y_train = tf.keras.utils.to_categorical(y_train, num_classes)
            y_test = tf.keras.utils.to_categorical(y_test, num_classes)
        
        # Normalize data
        if normalize:
            x_train, x_test = DataManager._normalize_data(x_train, x_test, dataset_name)
        
        logger.info(f"Loaded {dataset_name}: train={x_train.shape}, test={x_test.shape}")
        
        return x_train, y_train, x_test, y_test, input_shape, num_classes
    
    @staticmethod
    def load_adversarial_data(adv_path: str, 
                             validate: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Standardized adversarial data loading.
        
        Args:
            adv_path: Path to the numpy file containing adversarial data.
            validate: Whether to validate the file exists before loading.
            
        Returns:
            Tuple of (x_adv, y_adv) where:
                - x_adv: Adversarial images
                - y_adv: Adversarial labels
                
        Raises:
            FileNotFoundError: If adversarial data file is not found (when validate=True).
            ValueError: If adversarial data file is missing required keys.
            RuntimeError: If there's an error loading the data.
            
        Example:
            >>> x_adv, y_adv = DataManager.load_adversarial_data('path/to/adv_data.npz')
        """
        if validate and not os.path.exists(adv_path):
            raise FileNotFoundError(f'Adversarial data file not found: {adv_path}')
        
        try:
            logger.info(f"Loading adversarial data from: {adv_path}")
            print(f"📦 Loading adversarial data from: {adv_path}", flush=True)
            
            adv = np.load(adv_path)
            x_adv, y_adv = adv['arr_1'], adv['arr_2']
            
            logger.info(f"Adversarial data shape: {x_adv.shape}, Labels shape: {y_adv.shape}")
            print(f"   Adversarial data shape: {x_adv.shape}", flush=True)
            
            return x_adv, y_adv
            
        except KeyError as e:
            raise ValueError(
                f'Adversarial data file missing required keys. '
                f'Expected "arr_1" and "arr_2". Error: {e}'
            )
        except Exception as e:
            raise RuntimeError(f'Error loading adversarial data: {e}')
    
    @staticmethod
    def load_and_preprocess_with_adversarial(
            dataset_name: str,
            adv_data_path: str,
            normalize: bool = True,
            shuffle: bool = True,
            one_hot_encode: bool = True) -> Tuple[np.ndarray, 
                                                   np.ndarray, 
                                                   np.ndarray, 
                                                   np.ndarray, 
                                                   np.ndarray, 
                                                   np.ndarray, 
                                                   tuple]:
        """
        Load and preprocess data along with adversarial examples.
        
        This is a convenience method that combines load_and_preprocess() with
        load_adversarial_data() for workflows that need both.
        
        Args:
            dataset_name: Name of the dataset.
            adv_data_path: Path to the numpy file containing adversarial data.
            normalize: Whether to normalize the data. Default is True.
            shuffle: Whether to shuffle the training data. Default is True.
            one_hot_encode: Whether to one-hot encode labels. Default is True.
            
        Returns:
            Tuple containing:
                - x_train: Preprocessed training images
                - y_train: Training labels
                - x_test: Preprocessed test images
                - y_test: Test labels
                - x_adv: Adversarial images
                - y_adv: Adversarial labels
                - input_shape: Shape of input images
                
        Raises:
            ValueError: If dataset_name is invalid.
            FileNotFoundError: If adversarial data file is not found.
            
        Example:
            >>> x_train, y_train, x_test, y_test, x_adv, y_adv, input_shape = \\
            ...     DataManager.load_and_preprocess_with_adversarial(
            ...         'cifar10', 'path/to/adv_data.npz')
        """
        # Load and preprocess standard data
        x_train, y_train, x_test, y_test, input_shape, _ = DataManager.load_and_preprocess(
            dataset_name=dataset_name,
            normalize=normalize,
            shuffle=shuffle,
            one_hot_encode=one_hot_encode
        )
        
        # Load adversarial data
        x_adv, y_adv = DataManager.load_adversarial_data(adv_data_path)
        
        return x_train, y_train, x_test, y_test, x_adv, y_adv, input_shape
    
    @staticmethod
    def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
        """
        Get configuration information for a dataset.
        
        Args:
            dataset_name: Name of the dataset.
            
        Returns:
            Dictionary containing dataset configuration.
            
        Raises:
            ValueError: If dataset_name is not supported.
            
        Example:
            >>> info = DataManager.get_dataset_info('mnist')
            >>> print(info['num_classes'])  # 10
        """
        DataManager._validate_dataset(dataset_name)
        return DataManager.DATASET_CONFIG[dataset_name].copy()

