"""
Comprehensive experiment logging for research paper reproducibility.

This module provides structured logging of all metrics, hyperparameters,
and results in formats suitable for research papers (CSV, JSON, LaTeX tables).
"""

import os
import json
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model

logger = logging.getLogger(__name__)


class ExperimentLogger:
    """Comprehensive experiment logger for research paper metrics."""
    
    def __init__(self, experiment_name: str, output_dir: str = "../results"):
        """
        Initialize experiment logger.
        
        Args:
            experiment_name: Name of the experiment (e.g., 'train_original', 'watermark_finetuning')
            output_dir: Base output directory for results
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.experiment_dir = self.output_dir / experiment_name / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data structures
        self.metrics = {}
        self.hyperparameters = {}
        self.training_history = []
        self.results = []
        
        # Create subdirectories
        (self.experiment_dir / "metrics").mkdir(exist_ok=True)
        (self.experiment_dir / "models").mkdir(exist_ok=True)
        (self.experiment_dir / "plots").mkdir(exist_ok=True)
        (self.experiment_dir / "tables").mkdir(exist_ok=True)
        
        logger.info(f"Experiment logger initialized: {self.experiment_dir}")
    
    def log_hyperparameters(self, **kwargs):
        """Log all hyperparameters."""
        self.hyperparameters.update(kwargs)
        self.hyperparameters['timestamp'] = datetime.now().isoformat()
        self.hyperparameters['tensorflow_version'] = tf.__version__
        self.hyperparameters['python_version'] = f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
        
        # Log GPU info
        gpus = tf.config.list_physical_devices('GPU')
        self.hyperparameters['gpu_count'] = len(gpus)
        self.hyperparameters['gpu_names'] = [gpu.name for gpu in gpus] if gpus else []
        
        logger.info(f"Logged hyperparameters: {len(kwargs)} parameters")
    
    def log_training_epoch(self, epoch: int, train_loss: float, train_acc: float,
                          val_loss: float, val_acc: float, lr: Optional[float] = None,
                          epoch_time: Optional[float] = None, **kwargs):
        """Log metrics for a single training epoch."""
        epoch_data = {
            'epoch': epoch,
            'train_loss': float(train_loss),
            'train_acc': float(train_acc),
            'val_loss': float(val_loss),
            'val_acc': float(val_acc),
        }
        
        if lr is not None:
            epoch_data['learning_rate'] = float(lr)
        if epoch_time is not None:
            epoch_data['epoch_time'] = float(epoch_time)
        
        epoch_data.update({k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                          for k, v in kwargs.items()})
        
        self.training_history.append(epoch_data)
    
    def log_model_metrics(self, model: Model, x_test: np.ndarray, y_test: np.ndarray,
                          x_watermark: Optional[np.ndarray] = None,
                          y_watermark: Optional[np.ndarray] = None,
                          prefix: str = ""):
        """Log comprehensive model metrics."""
        # Test accuracy
        test_results = model.evaluate(x_test, y_test, verbose=0)
        self.metrics[f'{prefix}test_loss'] = float(test_results[0])
        self.metrics[f'{prefix}test_accuracy'] = float(test_results[1])
        
        # Watermark accuracy if provided
        if x_watermark is not None and y_watermark is not None:
            watermark_results = model.evaluate(x_watermark, y_watermark, verbose=0)
            self.metrics[f'{prefix}watermark_loss'] = float(watermark_results[0])
            self.metrics[f'{prefix}watermark_accuracy'] = float(watermark_results[1])
        
        # Model size and parameters
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        self.metrics[f'{prefix}total_parameters'] = int(total_params)
        self.metrics[f'{prefix}trainable_parameters'] = int(trainable_params)
        self.metrics[f'{prefix}non_trainable_parameters'] = int(non_trainable_params)
        
        # Model size in MB (approximate)
        model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
        self.metrics[f'{prefix}model_size_mb'] = float(model_size_mb)
        
        logger.info(f"Logged model metrics: test_acc={self.metrics.get(f'{prefix}test_accuracy', 'N/A'):.4f}")
    
    def log_attack_results(self, query_budget: int, test_acc: float, watermark_acc: float,
                          verification_result: Optional[Dict] = None,
                          comprehensive_metrics: Optional[Dict] = None,
                          **kwargs):
        """Log results from model extraction attack."""
        result = {
            'query_budget': int(query_budget),
            'test_accuracy': float(test_acc),
            'watermark_accuracy': float(watermark_acc),
        }
        
        if verification_result:
            result.update({
                'is_stolen': bool(verification_result.get('is_stolen', False)),
                'verification_confidence': float(verification_result.get('confidence', 0.0)),
                'p_value': float(verification_result.get('p_value', 0.0)),
                'threshold': float(verification_result.get('threshold', 0.0)),
            })
        
        if comprehensive_metrics:
            result.update({
                'fidelity': float(comprehensive_metrics.get('fidelity', 0.0)),
                'watermark_retention': float(comprehensive_metrics.get('watermark_retention', 0.0)),
                'test_acc_gap': float(comprehensive_metrics.get('test_acc_gap', 0.0)),
                'kl_divergence': float(comprehensive_metrics.get('kl_divergence', 0.0)),
                'detectability_score': float(comprehensive_metrics.get('detectability', 0.0)),
            })
        
        result.update({k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                      for k, v in kwargs.items()})
        
        self.results.append(result)
    
    def log_training_time(self, total_time: float, **kwargs):
        """Log total training time and resource usage."""
        self.metrics['total_training_time_seconds'] = float(total_time)
        self.metrics['total_training_time_minutes'] = float(total_time / 60)
        self.metrics['total_training_time_hours'] = float(total_time / 3600)
        self.metrics.update({k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                            for k, v in kwargs.items()})
    
    def save_all(self):
        """Save all logged data to files."""
        # Save hyperparameters
        with open(self.experiment_dir / "hyperparameters.json", 'w') as f:
            json.dump(self.hyperparameters, f, indent=2)
        
        # Save metrics
        with open(self.experiment_dir / "metrics" / "metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Save training history as CSV
        if self.training_history:
            df_history = pd.DataFrame(self.training_history)
            df_history.to_csv(self.experiment_dir / "metrics" / "training_history.csv", index=False)
        
        # Save results as CSV
        if self.results:
            df_results = pd.DataFrame(self.results)
            df_results.to_csv(self.experiment_dir / "metrics" / "results.csv", index=False)
            
            # Create summary statistics
            summary = self._create_summary(df_results)
            summary.to_csv(self.experiment_dir / "metrics" / "summary_statistics.csv", index=True)
        
        # Save complete experiment data
        experiment_data = {
            'experiment_name': self.experiment_name,
            'hyperparameters': self.hyperparameters,
            'metrics': self.metrics,
            'training_history': self.training_history,
            'results': self.results,
        }
        
        with open(self.experiment_dir / "experiment_data.json", 'w') as f:
            json.dump(experiment_data, f, indent=2, default=str)
        
        logger.info(f"Saved all experiment data to {self.experiment_dir}")
    
    def _create_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create summary statistics from results DataFrame."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        summary = df[numeric_cols].describe()
        
        # Add additional statistics
        for col in numeric_cols:
            summary.loc['median', col] = df[col].median()
            summary.loc['std', col] = df[col].std()
            summary.loc['min', col] = df[col].min()
            summary.loc['max', col] = df[col].max()
        
        return summary
    
    def create_latex_table(self, output_file: Optional[str] = None) -> str:
        """Create LaTeX table from results for paper."""
        if not self.results:
            logger.warning("No results to create LaTeX table")
            return ""
        
        df = pd.DataFrame(self.results)
        
        # Create LaTeX table
        latex_table = "\\begin{table}[h]\n"
        latex_table += "\\centering\n"
        latex_table += "\\caption{Experimental Results}\n"
        latex_table += "\\label{tab:results}\n"
        latex_table += "\\begin{tabular}{" + "c" * len(df.columns) + "}\n"
        latex_table += "\\toprule\n"
        
        # Header
        latex_table += " & ".join([col.replace("_", " ").title() for col in df.columns]) + " \\\\\n"
        latex_table += "\\midrule\n"
        
        # Data rows
        for _, row in df.iterrows():
            latex_table += " & ".join([f"{val:.4f}" if isinstance(val, (int, float)) else str(val) 
                                      for val in row.values]) + " \\\\\n"
        
        latex_table += "\\bottomrule\n"
        latex_table += "\\end{tabular}\n"
        latex_table += "\\end{table}\n"
        
        if output_file:
            output_path = self.experiment_dir / "tables" / output_file
            with open(output_path, 'w') as f:
                f.write(latex_table)
            logger.info(f"LaTeX table saved to {output_path}")
        
        return latex_table
    
    def get_experiment_path(self) -> Path:
        """Get the experiment directory path."""
        return self.experiment_dir


def log_reproducibility_info(output_dir: str, seed: int = 0):
    """
    Log reproducibility information (seeds, versions, etc.).
    
    Args:
        output_dir: Output directory for reproducibility info
        seed: Random seed used
    """
    info = {
        'timestamp': datetime.now().isoformat(),
        'random_seed': seed,
        'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        'tensorflow_version': tf.__version__,
        'numpy_version': np.__version__,
    }
    
    # GPU info
    gpus = tf.config.list_physical_devices('GPU')
    info['gpu_count'] = len(gpus)
    info['gpu_names'] = [gpu.name for gpu in gpus] if gpus else []
    
    # Try to get git commit hash
    try:
        import subprocess
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        info['git_commit'] = git_hash
    except:
        info['git_commit'] = 'unknown'
    
    output_path = Path(output_dir) / "reproducibility_info.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    logger.info(f"Reproducibility info saved to {output_path}")
    return info

