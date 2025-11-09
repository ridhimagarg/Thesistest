"""
Centralized data management utilities.

This module provides a single source of truth for data loading and preprocessing,
eliminating the duplication of data_preprocessing() across multiple files.
"""

import os
import logging
from typing import Tuple, Optional, Dict, Any
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10

# Configure logger
logger = logging.getLogger(__name__)


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
            dataset_name: Name of the dataset ('mnist', 'cifar10', 'cifar10resnet', 
                         'cifar10resnet_255_preprocess').
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
        
        # Reshape data
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

