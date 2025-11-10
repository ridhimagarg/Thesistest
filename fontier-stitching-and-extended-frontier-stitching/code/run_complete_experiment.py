"""
Complete Experiment Runner for Frontier Stitching Watermarking

This script orchestrates the complete watermarking experiment pipeline:
1. Train original unwatermarked model
2. Generate adversarial examples (watermarks)
3. Watermark the model via finetuning
4. Run model extraction attack and verify watermark

Usage:
    python run_complete_experiment.py
    python run_complete_experiment.py --skip-training  # Skip step 1 if model already exists
    python run_complete_experiment.py --skip-adversarial  # Skip step 2 if adversarial examples exist
"""

import os
import sys
import argparse
import time
from datetime import datetime
from pathlib import Path
import subprocess
import logging
import json
from config import ConfigManager, ExperimentConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Orchestrates the complete watermarking experiment."""
    
    def __init__(self, config=None, config_file=None):
        """
        Initialize experiment runner.
        
        Args:
            config: Optional configuration dict. If None, uses default values.
            config_file: Optional path to YAML or JSON config file. Takes precedence over config dict.
        """
        # Load configuration from file if provided
        if config_file:
            if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                exp_config = ConfigManager.load_from_yaml(config_file)
                config = self._config_to_dict(exp_config)
            elif config_file.endswith('.json'):
                with open(config_file, 'r') as f:
                    config = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_file}. Use .yaml or .json")
        
        # Use provided config dict or load from default YAML
        if config is None:
            try:
                # Try to load from default.yaml
                default_yaml = Path(__file__).parent / "configs" / "default.yaml"
                if default_yaml.exists():
                    exp_config = ConfigManager.load_from_yaml(str(default_yaml))
                    config = self._config_to_dict(exp_config)
                else:
                    # Fall back to hardcoded defaults
                    config = self.get_default_config()
            except Exception as e:
                logger.warning(f"Could not load default.yaml: {e}. Using hardcoded defaults.")
                config = self.get_default_config()
        
        self.config = config
        self.start_time = time.time()
        self.step_times = {}
        
        # Create experiment directory
        self.experiment_dir = Path(f"../experiments/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🚀 Starting complete experiment")
        logger.info(f"   Experiment directory: {self.experiment_dir}")
        logger.info(f"   Configuration: {self.config}")
    
    @staticmethod
    def _config_to_dict(exp_config: ExperimentConfig) -> dict:
        """Convert ExperimentConfig to flat dict for compatibility."""
        config_dict = {
            # Dataset and model
            'dataset_name': exp_config.training.dataset_name,
            'model_architecture': exp_config.training.model_architecture,
            
            # Training
            'epochs_original': exp_config.training.epochs,
            'batch_size': exp_config.training.batch_size,
            'lr': exp_config.training.lr,
            'weight_decay': exp_config.training.weight_decay,
            'dropout': exp_config.training.dropout,
            'optimizer': exp_config.training.optimizer,
            'validation_split': exp_config.training.validation_split,
            'lr_schedule_enabled': exp_config.training.lr_schedule_enabled,
            'lr_decay_factor': exp_config.training.lr_decay_factor,
            'lr_decay_epochs': exp_config.training.lr_decay_epochs,
            'beta_1': exp_config.training.beta_1,
            'beta_2': exp_config.training.beta_2,
            'epsilon': exp_config.training.epsilon,
            
            # Watermarking
            'epochs_watermarking': exp_config.watermark.finetuning_epochs,
            'watermark_batch_size': exp_config.watermark.watermark_batch_size,
            'watermark_lr': exp_config.watermark.watermark_lr,
            'watermark_weight_decay': exp_config.watermark.watermark_weight_decay,
            'watermark_dropout': 0.0,  # Not in watermark config, use training default
            'watermark_optimizer': exp_config.watermark.watermark_optimizer,
            'num_layers_unfreeze': exp_config.watermark.num_layers_unfreeze,
            'watermark_lr_schedule_enabled': exp_config.watermark.watermark_lr_schedule_enabled,
            'watermark_lr_decay_factor': exp_config.watermark.watermark_lr_decay_factor,
            'watermark_lr_decay_epochs': exp_config.watermark.watermark_lr_decay_epochs,
            
            # Adversarial generation
            'eps_list': exp_config.watermark.eps_values,
            'adversarial_sample_size_list': exp_config.watermark.sample_sizes,
            'which_adv': 'true',  # Default
            'adversarial_batch_size': 64,  # Not in config, use default
            'clip_min': 0.0,
            'clip_max': 1.0,
            # Multi-attack support
            'attack_types': exp_config.watermark.attack_types,
            'use_multiple_attacks': exp_config.watermark.use_multiple_attacks,
            'attack_params': exp_config.watermark.attack_params,
            
            # Attack
            'attacker_model_architecture': exp_config.attack.attacker_architecture,
            'epochs_attack': exp_config.attack.epochs_extract,
            'attack_batch_size': exp_config.attack.attack_batch_size,
            'attack_lr': exp_config.attack.attack_lr,
            'attack_weight_decay': exp_config.attack.attack_weight_decay,
            'attack_dropout': exp_config.attack.dropout,
            'attack_optimizer': exp_config.attack.attack_optimizer,
            'query_budgets': exp_config.attack.query_budgets,
            'use_probability': False,
            
            # Performance
            'use_mixed_precision': exp_config.mixed_precision,
            'use_parallel_processing': exp_config.parallel_processing,
            'max_workers': exp_config.max_workers,
            'gpu_memory_growth': exp_config.gpu_memory_growth,
            
            # Reproducibility
            'seed': exp_config.seed,
            'deterministic': exp_config.deterministic,
            
            # Paths (will be auto-generated)
            'model_path': None,
            'adversarial_path': None,
            'watermarked_model_path': None,
        }
        return config_dict
    
    @staticmethod
    def get_default_config():
        """Get optimized default experiment configuration."""
        return {
            # Dataset and model configuration
            'dataset_name': 'cifar10',
            'model_architecture': 'cifar10_base_2',
            
            # Training configuration (optimized for CIFAR-10)
            'epochs_original': 50,  # Increased from 30 for better accuracy
            'batch_size': 128,
            'lr': 0.001,
            'lr_schedule_enabled': True,  # Enable learning rate decay
            'lr_decay_factor': 0.1,
            'lr_decay_epochs': [30, 40],  # Decay at epochs 30 and 40
            'weight_decay': 0.0001,  # L2 regularization to prevent overfitting
            'dropout': 0.0,
            'optimizer': 'adam',
            'beta_1': 0.9,
            'beta_2': 0.999,
            'epsilon': 1e-7,
            'validation_split': 0.2,
            
            # Watermarking configuration (optimized for watermark retention)
            'epochs_watermarking': 15,  # Increased from 10 for better watermark retention
            'watermark_batch_size': 128,
            'watermark_lr': 0.0001,  # Lower LR for fine-tuning
            'watermark_lr_schedule_enabled': True,
            'watermark_lr_decay_factor': 0.5,
            'watermark_lr_decay_epochs': [8, 12],
            'watermark_weight_decay': 0.0,
            'watermark_dropout': 0.0,
            'watermark_optimizer': 'adam',
            'num_layers_unfreeze': 1,
            
            # Adversarial example generation (optimized for watermark effectiveness)
            'eps_list': [0.01, 0.015, 0.02],  # Added 0.02 for stronger watermarks
            'adversarial_sample_size_list': [10000],
            'which_adv': 'true',  # 'true' or 'false'
            'adversarial_batch_size': 64,
            'clip_min': 0.0,
            'clip_max': 1.0,
            
            # Attack configuration (optimized for comprehensive evaluation)
            'attacker_model_architecture': 'cifar10_base_2',
            'epochs_attack': 50,
            'attack_batch_size': 128,
            'attack_lr': 0.001,
            'attack_weight_decay': 0.0,
            'attack_dropout': 0.0,
            'attack_optimizer': 'adam',
            'query_budgets': [250, 500, 1000, 5000, 10000, 20000],
            'use_probability': False,
            
            # Performance optimizations
            'use_mixed_precision': False,  # Set to True for faster training on GPU
            'use_parallel_processing': False,  # Set to True for parallel adversarial generation
            'max_workers': None,  # Auto-detect CPU cores
            'gpu_memory_growth': True,
            
            # Reproducibility
            'seed': 42,  # Changed from 0 to 42 (more standard)
            'deterministic': False,  # Set to True for full reproducibility (slower)
            
            # Paths (will be auto-generated)
            'model_path': None,
            'adversarial_path': None,
            'watermarked_model_path': None,
        }
    
    def run_step(self, step_name, step_func, *args, **kwargs):
        """Run a single experiment step with timing and error handling."""
        logger.info("=" * 80)
        logger.info(f"📋 Step: {step_name}")
        logger.info("=" * 80)
        
        step_start = time.time()
        try:
            result = step_func(*args, **kwargs)
            step_time = time.time() - step_start
            self.step_times[step_name] = step_time
            logger.info(f"✅ {step_name} completed in {step_time:.2f} seconds ({step_time/60:.2f} minutes)")
            return result
        except Exception as e:
            step_time = time.time() - step_start
            logger.error(f"❌ {step_name} failed after {step_time:.2f} seconds")
            logger.error(f"   Error: {str(e)}", exc_info=True)
            raise
    
    def step1_train_original(self):
        """Step 1: Train the original unwatermarked model."""
        logger.info("Step 1: Training original model...")
        
        # Import and run training
        from train_original import train_model
        
        model = train_model(
            dataset_name=self.config['dataset_name'],
            model_architecture=self.config['model_architecture'],
            epochs=self.config['epochs_original'],
            dropout=self.config['dropout'],
            batch_size=self.config['batch_size'],
            optimizer=self.config['optimizer'],
            lr=self.config['lr'],
            weight_decay=self.config.get('weight_decay', 0.0001),
            lr_schedule_enabled=self.config.get('lr_schedule_enabled', True),
            lr_decay_factor=self.config.get('lr_decay_factor', 0.1),
            lr_decay_epochs=self.config.get('lr_decay_epochs', [30, 40]),
            beta_1=self.config.get('beta_1', 0.9),
            beta_2=self.config.get('beta_2', 0.999),
            epsilon=self.config.get('epsilon', 1e-7),
            use_mixed_precision=self.config.get('use_mixed_precision', False)
        )
        
        # Determine model path
        now = datetime.now().strftime("%d-%m-%Y")
        model_name = f"{self.config['dataset_name']}_{self.config['epochs_original']}_{self.config['model_architecture']}"
        model_path = f"../models/original_{now}/{model_name}/Original_checkpoint_best.keras"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at expected path: {model_path}")
        
        self.config['model_path'] = model_path
        logger.info(f"   Model saved to: {model_path}")
        
        return model_path
    
    def step2_generate_adversarial(self):
        """Step 2: Generate adversarial examples (watermarks)."""
        logger.info("Step 2: Generating adversarial examples...")
        
        if not self.config['model_path']:
            raise ValueError("Model path not set. Run step1_train_original first.")
        
        # Check if multi-attack is enabled
        use_multi_attack = self.config.get('use_multiple_attacks', False)
        attack_types = self.config.get('attack_types', ['fgsm'])
        
        if use_multi_attack and len(attack_types) > 1:
            # Use new multi-attack module
            logger.info(f"   Using multi-attack mode with: {attack_types}")
            from adversarial_attacks import generate_multiple_attacks
            
            # Generate for each epsilon and sample size combination
            generated_files = []
            for eps in self.config['eps_list']:
                for sample_size in self.config['adversarial_sample_size_list']:
                    logger.info(f"   Generating: eps={eps}, size={sample_size}, attacks={attack_types}")
                    
                    try:
                        results = generate_multiple_attacks(
                            dataset_name=self.config['dataset_name'],
                            model_path=self.config['model_path'],
                            attack_types=attack_types,
                            eps=eps,
                            adversarial_sample_size=sample_size,
                            clip_values=(0., 1.),
                            attack_params=self.config.get('attack_params', {}),
                            output_dir="../data/adversarial"
                        )
                        generated_files.append((eps, sample_size, attack_types))
                        logger.info(f"   Successfully generated with {len(results)} attack types")
                    except Exception as e:
                        logger.warning(f"   Failed to generate for eps={eps}, size={sample_size}: {e}")
                        continue
            
            # Check if any files were successfully generated
            if not generated_files:
                raise RuntimeError(
                    f"Failed to generate any adversarial examples. "
                    f"Tried {len(self.config['eps_list']) * len(self.config['adversarial_sample_size_list'])} combinations."
                )
            
            # Use the first successfully generated combination for watermarking
            # If the first combination failed, use the first successful one
            eps, size, _ = generated_files[0]
            attack_type = attack_types[0]
            model_path_normalized = self.config['model_path'].replace("\\", "/")
            model_parts = model_path_normalized.split("/")[-2:]
            model_name = "_".join(model_parts).replace(".h5", "").replace(".keras", "")
            base_filename = f"{attack_type}_{eps}_{size}_{model_name}.npz"
            adv_path = f"../data/adversarial/{self.config['dataset_name']}/{base_filename}"
            
        else:
            # Use original FGSM-based frontier-stitching
            logger.info("   Using FGSM-based frontier-stitching (original method)")
            import importlib.util
            spec = importlib.util.spec_from_file_location("frontier_stitching", "frontier-stitching.py")
            frontier_stitching = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(frontier_stitching)
            fgsm_attack = frontier_stitching.fgsm_attack
            
            # Generate adversarial examples for each epsilon and sample size combination
            generated_files = []
            
            for eps in self.config['eps_list']:
                for sample_size in self.config['adversarial_sample_size_list']:
                    logger.info(f"   Generating: eps={eps}, size={sample_size}")
                    
                    try:
                        # Generate file names
                        model_path_normalized = self.config['model_path'].replace("\\", "/")
                        model_parts = model_path_normalized.split("/")[-2:]
                        model_name = "_".join(model_parts).replace(".h5", "").replace(".keras", "")
                        base_filename = f"fgsm_{eps}_{sample_size}_{model_name}.npz"
                        
                        numpy_array_full_file_name = f"../data/fgsm/{self.config['dataset_name']}/full/{base_filename}"
                        numpy_array_true_file_name = f"../data/fgsm/{self.config['dataset_name']}/{self.config.get('which_adv', 'true')}/{base_filename}"
                        numpy_array_false_file_name = f"../data/fgsm/{self.config['dataset_name']}/false/{base_filename}"
                        
                        # Call the adversarial generation function
                        fgsm_attack(
                            dataset_name=self.config['dataset_name'],
                            model_path=self.config['model_path'],
                            eps=eps,
                            adversarial_sample_size=sample_size,
                            npz_full_file_name=numpy_array_full_file_name,
                            npz_true_file_name=numpy_array_true_file_name,
                            npz_false_file_name=numpy_array_false_file_name
                        )
                        generated_files.append((eps, sample_size))
                    except Exception as e:
                        logger.warning(f"   Failed to generate for eps={eps}, size={sample_size}: {e}")
                        continue
            
            # Check if any files were successfully generated
            if not generated_files:
                raise RuntimeError(
                    f"Failed to generate any adversarial examples. "
                    f"Tried {len(self.config['eps_list']) * len(self.config['adversarial_sample_size_list'])} combinations."
                )
            
            # Find the generated adversarial file
            # The path structure is: ../data/fgsm/{dataset_name}/{which_adv}/fgsm_{eps}_{size}_{model_name}.npz
            model_path_normalized = self.config['model_path'].replace("\\", "/")
            model_parts = model_path_normalized.split("/")[-2:]
            model_name = "_".join(model_parts).replace(".h5", "").replace(".keras", "")
            
            # Use the first successfully generated combination for watermarking
            # If the first combination failed, use the first successful one
            eps, size = generated_files[0]
            base_filename = f"fgsm_{eps}_{size}_{model_name}.npz"
            adv_path = f"../data/fgsm/{self.config['dataset_name']}/{self.config.get('which_adv', 'true')}/{base_filename}"
        
        # Verify the file exists
        if not os.path.exists(adv_path):
            # Try to provide helpful error message
            error_msg = f"Adversarial file not found. Expected: {adv_path}"
            if 'generated_files' in locals():
                error_msg += f"\nSuccessfully generated {len(generated_files)} file(s): {generated_files}"
            raise FileNotFoundError(error_msg)
        
        self.config['adversarial_path'] = adv_path
        logger.info(f"   Adversarial examples saved to: {adv_path}")
        
        return adv_path
    
    def step3_watermark_model(self, method='finetuning'):
        """Step 3: Watermark the model via finetuning or retraining.
        
        Args:
            method: 'finetuning' or 'retraining'
        """
        logger.info(f"Step 3: Watermarking model via {method}...")
        
        if method == 'retraining':
            if not self.config['adversarial_path']:
                raise ValueError("Adversarial path not set. Run step2_generate_adversarial first.")
            if 'model_architecture' not in self.config:
                raise ValueError("Model architecture not set in config.")
        else:  # finetuning
            if not self.config['model_path']:
                raise ValueError("Model path not set. Run step1_train_original first.")
            if not self.config['adversarial_path']:
                raise ValueError("Adversarial path not set. Run step2_generate_adversarial first.")
        
        # Set results path for watermarking
        now = datetime.now().strftime("%d-%m-%Y")
        watermark_results_path = f"../results/finetuned_{method}_{now}"
        
        if method == 'finetuning':
            from watermarking_finetuning import watermark_finetuning
            
            model = watermark_finetuning(
                dataset_name=self.config['dataset_name'],
                adv_data_path_numpy=self.config['adversarial_path'],
                model_to_finetune_path=self.config['model_path'],
                epochs=self.config['epochs_watermarking'],
                dropout=self.config.get('watermark_dropout', self.config['dropout']),
                batch_size=self.config.get('watermark_batch_size', self.config['batch_size']),
                optimizer=self.config.get('watermark_optimizer', self.config['optimizer']),
                lr=self.config.get('watermark_lr', 0.0001),
                weight_decay=self.config.get('watermark_weight_decay', 0.0),
                num_layers_unfreeze=self.config.get('num_layers_unfreeze', 1),
                lr_schedule_enabled=self.config.get('watermark_lr_schedule_enabled', True),
                lr_decay_factor=self.config.get('watermark_lr_decay_factor', 0.5),
                lr_decay_epochs=self.config.get('watermark_lr_decay_epochs', [8, 12]),
                results_path=watermark_results_path
            )
            
            # Determine watermarked model path
            model_name = self.config['model_path'].replace("\\", "/").split("/")[-2]
            adv_name = self.config['adversarial_path'].replace("\\", "/").split("/")[-1].split(".npz")[0]
            
            watermarked_path = f"../models/finetuned_finetuning_{now}/{self.config['which_adv']}/{self.config['dataset_name']}_{self.config['epochs_watermarking']}_{self.config['epochs_watermarking']}_{model_name}{adv_name}/Victim_checkpoint_best.keras"
            
        else:  # retraining
            from watermarking_retraining import watermark_retraining
            import watermarking_retraining as wm_retraining_module
            
            # Set global variables that watermark_retraining expects
            wm_retraining_module.RESULTS_PATH = watermark_results_path
            wm_retraining_module.DATA_PATH = "../data"
            wm_retraining_module.MODEL_PATH = f"../models/finetuned_retraining_{now}"
            wm_retraining_module.LOSS_FOLDER = "losses"
            
            # Create directories
            os.makedirs(os.path.join(wm_retraining_module.RESULTS_PATH, wm_retraining_module.LOSS_FOLDER, self.config['which_adv']), exist_ok=True)
            os.makedirs(os.path.join(wm_retraining_module.MODEL_PATH, self.config['which_adv']), exist_ok=True)
            
            model = watermark_retraining(
                dataset_name=self.config['dataset_name'],
                adv_data_path_numpy=self.config['adversarial_path'],
                model_architecture=self.config['model_architecture'],
                epochs=self.config['epochs_watermarking'],
                dropout=self.config.get('watermark_dropout', self.config['dropout']),
                batch_size=self.config.get('watermark_batch_size', self.config['batch_size']),
                optimizer=self.config.get('watermark_optimizer', 'adam'),
                lr=self.config.get('watermark_lr', 0.001),  # Higher LR for retraining
                weight_decay=self.config.get('watermark_weight_decay', 0.0)
            )
            
            # Determine watermarked model path for retraining
            adv_name = self.config['adversarial_path'].replace("\\", "/").split("/")[-1].split(".npz")[0]
            model_arch_short = self.config['model_architecture'].replace('_', '').upper()
            
            watermarked_path = f"../models/finetuned_retraining_{now}/{self.config['which_adv']}/{self.config['dataset_name']}_{self.config['epochs_watermarking']}_{model_arch_short}{adv_name}/Victim_checkpoint_best.keras"
        
        if not os.path.exists(watermarked_path):
            raise FileNotFoundError(f"Watermarked model not found at expected path: {watermarked_path}")
        
        self.config['watermarked_model_path'] = watermarked_path
        logger.info(f"   Watermarked model saved to: {watermarked_path}")
        
        return watermarked_path
    
    def step4_model_extraction_attack(self):
        """Step 4: Run model extraction attack and verify watermark."""
        logger.info("Step 4: Running model extraction attack...")
        
        if not self.config['watermarked_model_path']:
            raise ValueError("Watermarked model path not set. Run step3_watermark_model first.")
        if not self.config['adversarial_path']:
            raise ValueError("Adversarial path not set. Run step2_generate_adversarial first.")
        
        # Import and run attack
        from real_model_stealing_watermark_single import model_extraction_attack
        
        df, df_adv = model_extraction_attack(
            dataset_name=self.config['dataset_name'],
            adv_data_path_numpy=self.config['adversarial_path'],
            attacker_model_architecture=self.config['attacker_model_architecture'],
            number_of_queries=self.config['query_budgets'],
            num_epochs_to_steal=self.config['epochs_attack'],
            dropout=self.config.get('attack_dropout', self.config['dropout']),
            optimizer=self.config.get('attack_optimizer', self.config['optimizer']),
            lr=self.config.get('attack_lr', self.config['lr']),
            weight_decay=self.config.get('attack_weight_decay', self.config['weight_decay']),
            model_to_attack_path=self.config['watermarked_model_path'],
            results_path=self.experiment_dir
        )
        
        logger.info("   Model extraction attack completed")
        logger.info(f"   Results: {len(df)} test accuracy results, {len(df_adv)} watermark accuracy results")
        
        return df, df_adv
    
    def run_complete_experiment(self, skip_training=False, skip_adversarial=False, 
                                skip_watermarking=False, skip_attack=False,
                                watermark_method='finetuning', run_both_methods=False):
        """
        Run the complete experiment pipeline.
        
        Args:
            skip_training: Skip step 1 if model already exists
            skip_adversarial: Skip step 2 if adversarial examples exist
            skip_watermarking: Skip step 3 if watermarked model exists
            skip_attack: Skip step 4 if attack results exist
            watermark_method: 'finetuning' or 'retraining'
            run_both_methods: If True, run both finetuning and retraining for comparison
        """
        try:
            # Step 1: Train original model
            if not skip_training:
                self.run_step("Train Original Model", self.step1_train_original)
            else:
                logger.info("⏭️  Skipping Step 1: Training (using existing model)")
                if not self.config['model_path']:
                    raise ValueError("skip_training=True but model_path not provided in config")
            
            # Step 2: Generate adversarial examples
            if not skip_adversarial:
                self.run_step("Generate Adversarial Examples", self.step2_generate_adversarial)
            else:
                logger.info("⏭️  Skipping Step 2: Adversarial Generation (using existing examples)")
                if not self.config['adversarial_path']:
                    raise ValueError("skip_adversarial=True but adversarial_path not provided in config")
            
            # Step 3: Watermark the model
            if not skip_watermarking:
                if run_both_methods:
                    logger.info("🔄 Running both watermarking methods for comparison...")
                    
                    # Run finetuning
                    logger.info("\n" + "="*60)
                    logger.info("Method 1: Fine-tuning")
                    logger.info("="*60)
                    self.run_step("Watermark Model (Fine-tuning)", 
                                lambda: self.step3_watermark_model(method='finetuning'))
                    finetuned_path = self.config['watermarked_model_path']
                    
                    # Run retraining
                    logger.info("\n" + "="*60)
                    logger.info("Method 2: Retraining")
                    logger.info("="*60)
                    self.run_step("Watermark Model (Retraining)", 
                                lambda: self.step3_watermark_model(method='retraining'))
                    retrained_path = self.config['watermarked_model_path']
                    
                    logger.info("\n" + "="*60)
                    logger.info("✅ Both methods completed!")
                    logger.info(f"   Fine-tuned model: {finetuned_path}")
                    logger.info(f"   Retrained model: {retrained_path}")
                    logger.info("="*60)
                else:
                    self.run_step("Watermark Model", 
                                lambda: self.step3_watermark_model(method=watermark_method))
            else:
                logger.info("⏭️  Skipping Step 3: Watermarking (using existing watermarked model)")
                if not self.config['watermarked_model_path']:
                    raise ValueError("skip_watermarking=True but watermarked_model_path not provided in config")
            
            # Step 4: Model extraction attack
            if not skip_attack:
                self.run_step("Model Extraction Attack", self.step4_model_extraction_attack)
            else:
                logger.info("⏭️  Skipping Step 4: Model Extraction Attack")
            
            # Summary
            total_time = time.time() - self.start_time
            logger.info("=" * 80)
            logger.info("🎉 Complete Experiment Finished Successfully!")
            logger.info("=" * 80)
            logger.info(f"Total experiment time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
            logger.info("\nStep timings:")
            for step, step_time in self.step_times.items():
                logger.info(f"   {step}: {step_time:.2f} seconds ({step_time/60:.2f} minutes)")
            logger.info(f"\nExperiment directory: {self.experiment_dir}")
            logger.info(f"Results saved to: ../results/")
            logger.info(f"Models saved to: ../models/")
            logger.info(f"Data saved to: ../data/")
            
            # Save configuration
            import json
            config_file = self.experiment_dir / "experiment_config.json"
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to: {config_file}")
            
        except Exception as e:
            total_time = time.time() - self.start_time
            logger.error("=" * 80)
            logger.error("❌ Experiment Failed!")
            logger.error("=" * 80)
            logger.error(f"Total time before failure: {total_time:.2f} seconds")
            logger.error(f"Error: {str(e)}", exc_info=True)
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run complete watermarking experiment pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete experiment from scratch
  python run_complete_experiment.py
  
  # Skip training if model already exists
  python run_complete_experiment.py --skip-training
  
  # Skip multiple steps
  python run_complete_experiment.py --skip-training --skip-adversarial
  
  # Custom configuration (modify script or use config file)
  python run_complete_experiment.py --config config.json
  
  # Run on all RGB datasets
  python run_complete_experiment.py --all-datasets
        """
    )
    
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip step 1: Training original model')
    parser.add_argument('--skip-adversarial', action='store_true',
                        help='Skip step 2: Generating adversarial examples')
    parser.add_argument('--skip-watermarking', action='store_true',
                        help='Skip step 3: Watermarking model')
    parser.add_argument('--skip-attack', action='store_true',
                        help='Skip step 4: Model extraction attack')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML or JSON configuration file (default: configs/default.yaml)')
    parser.add_argument('--method', choices=['finetuning', 'retraining'], 
                       default='finetuning', help='Watermarking method (default: finetuning)')
    parser.add_argument('--run-both', action='store_true',
                       help='Run both finetuning and retraining for comparison')
    parser.add_argument('--all-datasets', action='store_true',
                        help='Run experiment on all RGB datasets (cifar10, cifar100, svhn, stl10, eurosat)')
    parser.add_argument('--datasets', nargs='+', type=str, default=None,
                        help='List of specific datasets to run (e.g., --datasets cifar10 cifar100)')
    
    args = parser.parse_args()
    
    # Determine which datasets to run
    if args.all_datasets:
        # All RGB datasets
        datasets = ['cifar10', 'cifar100', 'svhn', 'stl10', 'eurosat']
        logger.info(f"🚀 Running experiments on all RGB datasets: {datasets}")
    elif args.datasets:
        datasets = args.datasets
        logger.info(f"🚀 Running experiments on specified datasets: {datasets}")
    else:
        # Single dataset from config
        datasets = None
    
    if datasets:
        # Run on multiple datasets
        total_datasets = len(datasets)
        for idx, dataset_name in enumerate(datasets, 1):
            logger.info("=" * 80)
            logger.info(f"📊 Dataset {idx}/{total_datasets}: {dataset_name.upper()}")
            logger.info("=" * 80)
            
            try:
                # Load config and update dataset name
                if args.config:
                    runner = ExperimentRunner(config_file=args.config)
                else:
                    runner = ExperimentRunner()
                
                # Update dataset name in config
                runner.config['dataset_name'] = dataset_name
                
                # Update model architecture based on dataset if needed
                if dataset_name in ['cifar10', 'cifar100', 'svhn']:
                    runner.config['model_architecture'] = 'cifar10_base_2'
                elif dataset_name in ['stl10', 'eurosat']:
                    # STL-10 and EuroSAT might need different architectures
                    runner.config['model_architecture'] = 'cifar10_base_2'  # Default for now
                
                runner.run_complete_experiment(
                    skip_training=args.skip_training,
                    skip_adversarial=args.skip_adversarial,
                    skip_watermarking=args.skip_watermarking,
                    skip_attack=args.skip_attack,
                    watermark_method=args.method,
                    run_both_methods=args.run_both
                )
                logger.info(f"✅ Completed experiment for {dataset_name}")
            except Exception as e:
                logger.error(f"❌ Failed experiment for {dataset_name}: {e}")
                continue
        
        logger.info("=" * 80)
        logger.info(f"🎉 Completed experiments on {total_datasets} dataset(s)")
        logger.info("=" * 80)
    else:
        # Single dataset from config
        runner = ExperimentRunner(config_file=args.config)
        runner.run_complete_experiment(
            skip_training=args.skip_training,
            skip_adversarial=args.skip_adversarial,
            skip_watermarking=args.skip_watermarking,
            skip_attack=args.skip_attack,
            watermark_method=args.method,
            run_both_methods=args.run_both
        )


if __name__ == "__main__":
    main()

