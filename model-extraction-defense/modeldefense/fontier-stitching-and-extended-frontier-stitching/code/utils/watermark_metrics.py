"""
Comprehensive watermark evaluation metrics.

This module provides comprehensive metrics for evaluating watermark effectiveness,
including fidelity, watermark retention, test accuracy gap, KL divergence, and detectability.
"""

import numpy as np
import tensorflow as tf
from typing import Dict, Tuple, Optional
from tensorflow.keras.models import Model
import logging

logger = logging.getLogger(__name__)


class WatermarkMetrics:
    """Comprehensive watermark evaluation metrics."""
    
    @staticmethod
    def calculate_all_metrics(victim_model: Model, stolen_model: Model,
                              x_test: np.ndarray, y_test: np.ndarray,
                              x_watermark: np.ndarray, y_watermark: np.ndarray,
                              batch_size: int = 128) -> Dict[str, float]:
        """
        Calculate all watermark evaluation metrics.
        
        Args:
            victim_model: Original victim model.
            stolen_model: Suspected stolen model.
            x_test: Test data.
            y_test: Test labels.
            x_watermark: Watermark data.
            y_watermark: Watermark labels.
            batch_size: Batch size for predictions.
            
        Returns:
            Dictionary with all metrics:
            - fidelity: How well does stolen model match victim predictions
            - watermark_retention: Ratio of watermark accuracy in stolen vs victim
            - test_acc_gap: Absolute difference in test accuracy
            - kl_divergence: KL divergence between model predictions
            - detectability: Composite detectability score
        """
        logger.info("Calculating comprehensive watermark metrics")
        
        metrics = {}
        
        # 1. Fidelity: How well does stolen model match victim?
        logger.info("Calculating fidelity...")
        victim_preds = victim_model.predict(x_test, batch_size=batch_size, verbose=0)
        stolen_preds = stolen_model.predict(x_test, batch_size=batch_size, verbose=0)
        
        victim_preds_classes = np.argmax(victim_preds, axis=1)
        stolen_preds_classes = np.argmax(stolen_preds, axis=1)
        
        metrics['fidelity'] = float(np.mean(victim_preds_classes == stolen_preds_classes))
        logger.info(f"  Fidelity: {metrics['fidelity']:.4f}")
        
        # 2. Watermark Retention Rate
        logger.info("Calculating watermark retention...")
        victim_watermark_acc = victim_model.evaluate(x_watermark, y_watermark, 
                                                     batch_size=batch_size, verbose=0)[1]
        stolen_watermark_acc = stolen_model.evaluate(x_watermark, y_watermark,
                                                    batch_size=batch_size, verbose=0)[1]
        
        metrics['watermark_retention'] = float(stolen_watermark_acc / victim_watermark_acc) if victim_watermark_acc > 0 else 0.0
        metrics['victim_watermark_acc'] = float(victim_watermark_acc)
        metrics['stolen_watermark_acc'] = float(stolen_watermark_acc)
        logger.info(f"  Watermark retention: {metrics['watermark_retention']:.4f}")
        logger.info(f"  Victim watermark acc: {victim_watermark_acc:.4f}")
        logger.info(f"  Stolen watermark acc: {stolen_watermark_acc:.4f}")
        
        # 3. Test Accuracy Gap
        logger.info("Calculating test accuracy gap...")
        victim_test_acc = victim_model.evaluate(x_test, y_test, 
                                               batch_size=batch_size, verbose=0)[1]
        stolen_test_acc = stolen_model.evaluate(x_test, y_test,
                                               batch_size=batch_size, verbose=0)[1]
        
        metrics['test_acc_gap'] = float(abs(victim_test_acc - stolen_test_acc))
        metrics['victim_test_acc'] = float(victim_test_acc)
        metrics['stolen_test_acc'] = float(stolen_test_acc)
        logger.info(f"  Test accuracy gap: {metrics['test_acc_gap']:.4f}")
        logger.info(f"  Victim test acc: {victim_test_acc:.4f}")
        logger.info(f"  Stolen test acc: {stolen_test_acc:.4f}")
        
        # 4. KL Divergence (distribution similarity)
        logger.info("Calculating KL divergence...")
        # Clip predictions to avoid log(0)
        epsilon = 1e-10
        victim_preds_clipped = np.clip(victim_preds, epsilon, 1.0 - epsilon)
        stolen_preds_clipped = np.clip(stolen_preds, epsilon, 1.0 - epsilon)
        
        # Normalize to ensure they sum to 1
        victim_preds_clipped = victim_preds_clipped / victim_preds_clipped.sum(axis=1, keepdims=True)
        stolen_preds_clipped = stolen_preds_clipped / stolen_preds_clipped.sum(axis=1, keepdims=True)
        
        # Calculate KL divergence for each sample
        kl_divs = []
        for i in range(len(victim_preds_clipped)):
            kl = np.sum(victim_preds_clipped[i] * np.log(
                victim_preds_clipped[i] / (stolen_preds_clipped[i] + epsilon) + epsilon
            ))
            kl_divs.append(kl)
        
        metrics['kl_divergence'] = float(np.mean(kl_divs))
        logger.info(f"  KL divergence: {metrics['kl_divergence']:.4f}")
        
        # 5. Detectability Score (composite)
        # Higher watermark retention = better detectability
        # Lower test acc gap = better detectability (model is similar)
        # Higher fidelity = better detectability (model is similar)
        metrics['detectability'] = float(
            0.5 * metrics['watermark_retention'] +
            0.3 * (1 - min(metrics['test_acc_gap'], 1.0)) +  # Normalize gap to [0, 1]
            0.2 * metrics['fidelity']
        )
        logger.info(f"  Detectability score: {metrics['detectability']:.4f}")
        
        return metrics
    
    @staticmethod
    def calculate_watermark_accuracy(model: Model, x_watermark: np.ndarray,
                                    y_watermark: np.ndarray,
                                    batch_size: int = 128) -> float:
        """
        Calculate watermark accuracy for a model.
        
        Args:
            model: Model to evaluate.
            x_watermark: Watermark data.
            y_watermark: Watermark labels.
            batch_size: Batch size.
            
        Returns:
            Watermark accuracy.
        """
        acc = model.evaluate(x_watermark, y_watermark, batch_size=batch_size, verbose=0)[1]
        return float(acc)
    
    @staticmethod
    def calculate_test_accuracy(model: Model, x_test: np.ndarray, y_test: np.ndarray,
                               batch_size: int = 128) -> float:
        """
        Calculate test accuracy for a model.
        
        Args:
            model: Model to evaluate.
            x_test: Test data.
            y_test: Test labels.
            batch_size: Batch size.
            
        Returns:
            Test accuracy.
        """
        acc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)[1]
        return float(acc)
    
    @staticmethod
    def print_metrics_summary(metrics: Dict[str, float]) -> None:
        """
        Print a formatted summary of metrics.
        
        Args:
            metrics: Dictionary of metrics from calculate_all_metrics().
        """
        print("\n" + "=" * 60)
        print("Watermark Evaluation Metrics Summary")
        print("=" * 60)
        print(f"Fidelity: {metrics['fidelity']:.4f}")
        print(f"Watermark Retention: {metrics['watermark_retention']:.4f} ({metrics['watermark_retention']*100:.2f}%)")
        print(f"  - Victim watermark acc: {metrics['victim_watermark_acc']:.4f}")
        print(f"  - Stolen watermark acc: {metrics['stolen_watermark_acc']:.4f}")
        print(f"Test Accuracy Gap: {metrics['test_acc_gap']:.4f}")
        print(f"  - Victim test acc: {metrics['victim_test_acc']:.4f}")
        print(f"  - Stolen test acc: {metrics['stolen_test_acc']:.4f}")
        print(f"KL Divergence: {metrics['kl_divergence']:.4f}")
        print(f"Detectability Score: {metrics['detectability']:.4f}")
        print("=" * 60 + "\n")


def plot_roc_curve(victim_watermark_acc: float,
                   stolen_models_watermark_accs: list,
                   innocent_models_watermark_accs: list,
                   save_path: Optional[str] = None) -> Tuple[float, float]:
    """
    Plot ROC curve for theft detection threshold selection.
    
    Args:
        victim_watermark_acc: Watermark accuracy of the victim model.
        stolen_models_watermark_accs: List of watermark accuracies from known stolen models.
        innocent_models_watermark_accs: List of watermark accuracies from independent models.
        save_path: Optional path to save the plot. If None, plot is not saved.
        
    Returns:
        Tuple of (optimal_threshold, auc_score).
    """
    try:
        from sklearn.metrics import roc_curve, auc
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("sklearn and matplotlib required for ROC curve plotting")
        return None, None
    
    # Create labels: 1 for stolen, 0 for innocent
    y_true = [1] * len(stolen_models_watermark_accs) + [0] * len(innocent_models_watermark_accs)
    y_scores = stolen_models_watermark_accs + innocent_models_watermark_accs
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Find optimal threshold (max Youden's J statistic: TPR - FPR)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], c='red', s=100,
                label=f'Optimal threshold = {optimal_threshold:.3f}', zorder=5)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve for Watermark-based Theft Detection', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curve saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    logger.info(f"ROC AUC: {roc_auc:.4f}")
    logger.info(f"Optimal threshold: {optimal_threshold:.4f}")
    logger.info(f"  TPR at optimal: {tpr[optimal_idx]:.4f}")
    logger.info(f"  FPR at optimal: {fpr[optimal_idx]:.4f}")
    
    return float(optimal_threshold), float(roc_auc)

