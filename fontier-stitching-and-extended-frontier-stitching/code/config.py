"""
Configuration management for model training, watermarking, and attacks.

This module provides centralized configuration management using dataclasses
and YAML support for easy configuration management.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import yaml
import os
from pathlib import Path


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    dataset_name: str = "cifar10"
    epochs: int = 50  # Optimized: increased from 30 for better accuracy
    batch_size: int = 128
    lr: float = 0.001
    weight_decay: float = 0.0001  # L2 regularization to prevent overfitting
    model_architecture: str = "cifar10_base_2"
    dropout: float = 0.0
    optimizer: str = "adam"
    validation_split: float = 0.2  # Optimized: increased from 0.1
    early_stopping_patience: int = 10
    save_best_only: bool = True
    # Learning rate schedule
    lr_schedule_enabled: bool = True
    lr_decay_factor: float = 0.1
    lr_decay_epochs: List[int] = field(default_factory=lambda: [30, 40])
    # Optimizer hyperparameters
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-7
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary."""
        return cls(**data)


@dataclass
class WatermarkConfig:
    """Configuration for watermarking."""
    eps_values: List[float] = field(default_factory=lambda: [0.01, 0.015, 0.02])  # Optimized: added 0.02
    sample_sizes: List[int] = field(default_factory=lambda: [10000])
    true_adv_ratio: float = 0.5
    finetuning_epochs: int = 15  # Optimized: increased from 10 for better watermark retention
    num_layers_unfreeze: int = 1  # Optimized: changed from 10 to 1
    watermark_batch_size: int = 128
    watermark_lr: float = 0.0001
    watermark_optimizer: str = "adam"
    watermark_weight_decay: float = 0.0
    # Learning rate schedule for watermarking
    watermark_lr_schedule_enabled: bool = True
    watermark_lr_decay_factor: float = 0.5
    watermark_lr_decay_epochs: List[int] = field(default_factory=lambda: [8, 12])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WatermarkConfig':
        """Create config from dictionary."""
        return cls(**data)


@dataclass
class AttackConfig:
    """Configuration for model extraction attacks."""
    query_budgets: List[int] = field(default_factory=lambda: [250, 500, 1000, 5000, 10000, 20000])
    attacker_architecture: str = "cifar10_base_2"
    epochs_extract: int = 50
    attack_batch_size: int = 128
    attack_lr: float = 0.001
    attack_optimizer: str = "adam"
    attack_weight_decay: float = 0.0
    dropout: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AttackConfig':
        """Create config from dictionary."""
        return cls(**data)


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    training: TrainingConfig = field(default_factory=TrainingConfig)
    watermark: WatermarkConfig = field(default_factory=WatermarkConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    output_dir: str = "../results"
    model_dir: str = "../models"
    data_dir: str = "../data"
    seed: int = 42  # Optimized: changed from 0 to 42 (more standard)
    use_gpu: bool = True
    gpu_memory_growth: bool = True
    mixed_precision: bool = False
    parallel_processing: bool = False
    max_workers: Optional[int] = None
    deterministic: bool = False  # Set to True for full reproducibility (slower)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'training': self.training.to_dict(),
            'watermark': self.watermark.to_dict(),
            'attack': self.attack.to_dict(),
            'output_dir': self.output_dir,
            'model_dir': self.model_dir,
            'data_dir': self.data_dir,
            'seed': self.seed,
            'use_gpu': self.use_gpu,
            'gpu_memory_growth': self.gpu_memory_growth,
            'mixed_precision': self.mixed_precision,
            'parallel_processing': self.parallel_processing,
            'max_workers': self.max_workers,
            'deterministic': self.deterministic
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Create config from dictionary."""
        return cls(
            training=TrainingConfig.from_dict(data.get('training', {})),
            watermark=WatermarkConfig.from_dict(data.get('watermark', {})),
            attack=AttackConfig.from_dict(data.get('attack', {})),
            output_dir=data.get('output_dir', '../results'),
            model_dir=data.get('model_dir', '../models'),
            data_dir=data.get('data_dir', '../data'),
            seed=data.get('seed', 0),
            use_gpu=data.get('use_gpu', True),
            gpu_memory_growth=data.get('gpu_memory_growth', True),
            mixed_precision=data.get('mixed_precision', False),
            parallel_processing=data.get('parallel_processing', False),
            max_workers=data.get('max_workers', None),
            deterministic=data.get('deterministic', False)
        )


class ConfigManager:
    """Manager for loading and saving configurations."""
    
    @staticmethod
    def load_from_yaml(path: str) -> ExperimentConfig:
        """
        Load configuration from YAML file.
        
        Args:
            path: Path to YAML configuration file.
            
        Returns:
            ExperimentConfig object.
            
        Raises:
            FileNotFoundError: If the file doesn't exist.
            yaml.YAMLError: If the YAML is invalid.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return ExperimentConfig.from_dict(data)
    
    @staticmethod
    def save_to_yaml(config: ExperimentConfig, path: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config: ExperimentConfig object to save.
            path: Path where to save the YAML file.
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    @staticmethod
    def create_default_config(path: Optional[str] = None) -> ExperimentConfig:
        """
        Create and optionally save a default configuration.
        
        Args:
            path: Optional path to save the default config. If None, config is not saved.
            
        Returns:
            ExperimentConfig with default values.
        """
        config = ExperimentConfig()
        
        if path:
            ConfigManager.save_to_yaml(config, path)
        
        return config
    
    @staticmethod
    def merge_configs(base: ExperimentConfig, override: Dict[str, Any]) -> ExperimentConfig:
        """
        Merge override values into base configuration.
        
        Args:
            base: Base configuration.
            override: Dictionary with override values (can be nested).
            
        Returns:
            New ExperimentConfig with merged values.
        """
        base_dict = base.to_dict()
        
        def deep_merge(base_dict: Dict, override_dict: Dict) -> Dict:
            """Recursively merge dictionaries."""
            result = base_dict.copy()
            for key, value in override_dict.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        merged = deep_merge(base_dict, override)
        return ExperimentConfig.from_dict(merged)


# Convenience function for quick config access
def get_default_config() -> ExperimentConfig:
    """Get default configuration."""
    return ExperimentConfig()

