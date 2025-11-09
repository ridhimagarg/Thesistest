"""
Example script demonstrating watermark evaluation and robustness testing.

This script shows how to use:
1. WatermarkVerifier for statistical theft detection
2. WatermarkMetrics for comprehensive evaluation
3. test_removal_robustness for robustness testing
4. plot_roc_curve for threshold selection
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from utils.watermark_verifier import WatermarkVerifier
from utils.watermark_metrics import WatermarkMetrics, plot_roc_curve
from utils.data_utils import DataManager
from test_watermark_robustness import test_removal_robustness, print_robustness_summary


def example_watermark_verification():
    """Example: Verify if a model is stolen using WatermarkVerifier."""
    print("\n" + "=" * 60)
    print("Example 1: Watermark Verification")
    print("=" * 60)
    
    # Initialize verifier
    # Assume victim model has 95% accuracy on watermark set
    verifier = WatermarkVerifier(
        victim_acc=0.95,
        num_classes=10,
        watermark_size=1000
    )
    
    # Test suspected model
    suspected_acc = 0.85  # 85% accuracy on watermark set
    
    result = verifier.verify_theft(
        suspected_acc=suspected_acc,
        threshold_ratio=0.5,
        confidence=0.99
    )
    
    print(f"\nVerification Results:")
    print(f"  Suspected accuracy: {result['suspected_acc']:.4f}")
    print(f"  Victim accuracy: {result['victim_acc']:.4f}")
    print(f"  Threshold: {result['threshold']:.4f}")
    print(f"  P-value: {result['p_value']:.4f}")
    print(f"  Is stolen: {result['is_stolen']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    
    return result


def example_comprehensive_metrics(victim_model_path: str, stolen_model_path: str,
                                  dataset_name: str = "cifar10"):
    """Example: Calculate comprehensive watermark metrics."""
    print("\n" + "=" * 60)
    print("Example 2: Comprehensive Metrics")
    print("=" * 60)
    
    # Load models
    print(f"\nLoading models...")
    victim_model = load_model(victim_model_path)
    stolen_model = load_model(stolen_model_path)
    
    # Load data
    print(f"Loading {dataset_name} dataset...")
    x_train, y_train, x_test, y_test, input_shape, num_classes = DataManager.load_and_preprocess(dataset_name)
    
    # Load watermark data (example: use first 1000 test samples as watermark)
    # In practice, watermark data should be adversarial examples
    x_watermark = x_test[:1000]
    y_watermark = y_test[:1000]
    
    # Calculate all metrics
    metrics = WatermarkMetrics.calculate_all_metrics(
        victim_model=victim_model,
        stolen_model=stolen_model,
        x_test=x_test,
        y_test=y_test,
        x_watermark=x_watermark,
        y_watermark=y_watermark
    )
    
    # Print summary
    WatermarkMetrics.print_metrics_summary(metrics)
    
    return metrics


def example_robustness_testing(watermarked_model_path: str, dataset_name: str = "cifar10"):
    """Example: Test watermark robustness against removal attacks."""
    print("\n" + "=" * 60)
    print("Example 3: Robustness Testing")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading watermarked model...")
    watermarked_model = load_model(watermarked_model_path)
    
    # Load data
    print(f"Loading {dataset_name} dataset...")
    x_train, y_train, x_test, y_test, input_shape, num_classes = DataManager.load_and_preprocess(dataset_name)
    
    # Use first 1000 test samples as watermark (in practice, use adversarial examples)
    x_watermark = x_test[:1000]
    y_watermark = y_test[:1000]
    
    # Test robustness
    results = test_removal_robustness(
        watermarked_model=watermarked_model,
        x_test=x_test,
        y_test=y_test,
        x_watermark=x_watermark,
        y_watermark=y_watermark,
        x_train=x_train[:5000],  # Use subset for faster testing
        y_train=y_train[:5000]
    )
    
    # Print summary
    print_robustness_summary(results)
    
    return results


def example_roc_curve():
    """Example: Plot ROC curve for threshold selection."""
    print("\n" + "=" * 60)
    print("Example 4: ROC Curve for Threshold Selection")
    print("=" * 60)
    
    # Simulate data
    # In practice, collect these from multiple experiments
    victim_watermark_acc = 0.95
    
    # Known stolen models (should have high watermark accuracy)
    stolen_models_watermark_accs = [0.88, 0.92, 0.85, 0.90, 0.87, 0.91, 0.89]
    
    # Independent/innocent models (should have low watermark accuracy)
    innocent_models_watermark_accs = [0.12, 0.15, 0.10, 0.13, 0.11, 0.14, 0.12]
    
    # Plot ROC curve
    optimal_threshold, auc_score = plot_roc_curve(
        victim_watermark_acc=victim_watermark_acc,
        stolen_models_watermark_accs=stolen_models_watermark_accs,
        innocent_models_watermark_accs=innocent_models_watermark_accs,
        save_path='../results/roc_curve.png'
    )
    
    print(f"\nROC Curve Results:")
    print(f"  AUC Score: {auc_score:.4f}")
    print(f"  Optimal Threshold: {optimal_threshold:.4f}")
    print(f"  ROC curve saved to: ../results/roc_curve.png")
    
    return optimal_threshold, auc_score


if __name__ == '__main__':
    """
    Example usage of watermark evaluation tools.
    
    Usage:
        python example_watermark_evaluation.py [verification|metrics|robustness|roc|all]
    """
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = 'all'
    
    print("=" * 60)
    print("Watermark Evaluation Examples")
    print("=" * 60)
    
    if mode in ['verification', 'all']:
        example_watermark_verification()
    
    if mode in ['metrics', 'all']:
        # Example paths - update these to your actual model paths
        victim_model_path = "../models/original_09-11-2025/cifar10_30_CIFAR10_BASE_2/Original_checkpoint_best.keras"
        stolen_model_path = "../models/stolen_model.keras"
        
        if os.path.exists(victim_model_path) and os.path.exists(stolen_model_path):
            example_comprehensive_metrics(victim_model_path, stolen_model_path)
        else:
            print(f"\n⚠️  Skipping metrics example: Model files not found")
            print(f"   Victim model: {victim_model_path}")
            print(f"   Stolen model: {stolen_model_path}")
    
    if mode in ['robustness', 'all']:
        # Example path - update this to your actual model path
        watermarked_model_path = "../models/finetuned_finetuning_XX-XX-XXXX/true/model.keras"
        
        if os.path.exists(watermarked_model_path):
            example_robustness_testing(watermarked_model_path)
        else:
            print(f"\n⚠️  Skipping robustness example: Model file not found")
            print(f"   Watermarked model: {watermarked_model_path}")
    
    if mode in ['roc', 'all']:
        example_roc_curve()
    
    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)

