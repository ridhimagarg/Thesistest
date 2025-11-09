"""
@author: Ridhima Garg

Introduction:
    Unified visualization script for:
    1. Watermark vs Test Accuracy vs Stealing Dataset Size (from CSV)
    2. Learning Rate over Epochs (from JSON)
    3. Training/Validation/Watermark/Test Accuracies over Epochs (from JSON)

Usage:
    python visualization_unified.py --mode <mode> [options]
    
    Modes:
    - stealing_accuracy: Plot watermark vs test accuracy vs stealing dataset size
    - learning_rate: Plot learning rate over epochs
    - training_accuracy: Plot training/validation/watermark/test accuracies over epochs
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Set seaborn style
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 1.5})


def plot_stealing_accuracy(
    test_acc_file: str,
    watermark_acc_file: str,
    output_path: str,
    show_plot: bool = False
) -> None:
    """
    Plot watermark accuracy vs test accuracy vs stealing dataset size.
    
    Args:
        test_acc_file: Path to CSV file with test accuracy data
        watermark_acc_file: Path to CSV file with watermark accuracy data
        output_path: Path to save the output image
        show_plot: Whether to display the plot
    """
    if not os.path.exists(test_acc_file):
        raise FileNotFoundError(f"Test accuracy file not found: {test_acc_file}")
    if not os.path.exists(watermark_acc_file):
        raise FileNotFoundError(f"Watermark accuracy file not found: {watermark_acc_file}")
    
    test_df = pd.read_csv(test_acc_file)
    watermark_df = pd.read_csv(watermark_acc_file)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot watermark accuracy
    ax = sns.lineplot(
        x="Stealing Dataset Size", 
        y="Accuracy", 
        estimator=np.median,
        data=watermark_df,
        ci=None, 
        color="tab:red", 
        marker='s', 
        label="Watermark Accuracy",
        ax=ax
    )
    
    # Add confidence bounds for watermark accuracy
    bounds = watermark_df.groupby('Stealing Dataset Size')['Accuracy'].quantile((0.20, 0.90)).unstack()
    ax.fill_between(
        x=bounds.index,
        y1=bounds.iloc[:, 0],
        y2=bounds.iloc[:, 1],
        alpha=0.3, 
        color="tomato"
    )
    ax.set(yticks=np.arange(0, 1, 0.1))
    
    # Plot test accuracy
    ax1 = sns.lineplot(
        x="Stealing Dataset Size", 
        y="Accuracy", 
        estimator=np.median,
        data=test_df,
        ci=None, 
        color="tab:blue", 
        linestyle='--', 
        marker='s', 
        label='Test Accuracy',
        ax=ax
    )
    
    # Add confidence bounds for test accuracy
    bounds = test_df.groupby('Stealing Dataset Size')['Accuracy'].quantile((0.20, 0.90)).unstack()
    ax1.fill_between(
        x=bounds.index,
        y1=bounds.iloc[:, 0],
        y2=bounds.iloc[:, 1],
        alpha=0.2, 
        color="tab:blue"
    )
    
    # Format x-axis labels
    plt.gca().set_xticklabels(plt.gca().get_xticks(), fontsize=12)
    ax.set_xticklabels([f'{int(label)}' for label in ax.get_xticks()])
    
    plt.xlabel("Stealing Dataset Size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved plot to: {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_learning_rate(
    json_path: str,
    output_path: str,
    show_plot: bool = False
) -> None:
    """
    Plot learning rate over epochs.
    
    Args:
        json_path: Path to JSON file with learning rate data
        output_path: Path to save the output image
        show_plot: Whether to display the plot
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    with open(json_path, 'r') as file:
        loaded_dict = json.load(file)
    
    if "lr" not in loaded_dict:
        raise KeyError(f"'lr' key not found in JSON file: {json_path}")
    
    lr = loaded_dict["lr"]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax1 = sns.lineplot(
        x=list(range(len(lr))), 
        y=lr,
        ci=None, 
        color="tab:purple", 
        marker='o', 
        label='Learning rate', 
        markersize=7,
        ax=ax
    )
    ax1.set_xlim(1, len(lr))
    plt.xlabel("Epochs")
    plt.ylabel("Learning rate")
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved plot to: {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_training_accuracy(
    json_path: str,
    output_paths: Dict[str, str],
    show_plot: bool = False
) -> None:
    """
    Plot training/validation/watermark/test accuracies over epochs.
    
    Args:
        json_path: Path to JSON file with accuracy data
        output_paths: Dictionary with keys:
            - combined: Path for combined accuracy plot
            - normal_test: Path for normal test accuracy plot
            - watermark: Path for watermark accuracy plot
        show_plot: Whether to display the plot
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    with open(json_path, 'r') as file:
        loaded_dict = json.load(file)
    
    # Extract data
    train_acc = loaded_dict.get("train_acc", [])
    val_acc = loaded_dict.get("val_acc", [])
    adv_test_acc = loaded_dict.get("adv_test_acc", [])
    normal_test_acc = loaded_dict.get("normal_test_acc", [])
    
    # Plot 1: Combined accuracies
    if "combined" in output_paths:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if train_acc:
            ax1 = sns.lineplot(
                x=list(range(len(train_acc))), 
                y=train_acc,
                ci=None, 
                color="tab:purple", 
                marker='o', 
                label='Combined train set accuracy', 
                markersize=7,
                ax=ax
            )
            ax1.set(yticks=np.arange(0, 1.1, 0.1))
        
        if val_acc:
            ax2 = sns.lineplot(
                x=list(range(len(val_acc))), 
                y=val_acc,
                ci=None, 
                color="tab:orange", 
                linestyle='--', 
                marker='o', 
                label='Combined validation set accuracy', 
                markersize=7,
                ax=ax
            )
            ax2.set(yticks=np.arange(0, 1.2, 0.1))
        
        if adv_test_acc:
            ax3 = sns.lineplot(
                x=list(range(len(adv_test_acc))), 
                y=adv_test_acc,
                ci=None, 
                color="tab:blue", 
                linestyle='--', 
                marker='o', 
                label='Watermark set accuracy', 
                markersize=7,
                ax=ax
            )
            ax3.set(yticks=np.arange(0, 1.2, 0.1))
        
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(output_paths["combined"]), exist_ok=True)
        plt.savefig(output_paths["combined"], bbox_inches='tight', dpi=300)
        print(f"Saved combined accuracy plot to: {output_paths['combined']}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    # Plot 2: Normal test accuracy
    if "normal_test" in output_paths and normal_test_acc:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax3 = sns.lineplot(
            x=list(range(len(normal_test_acc))), 
            y=normal_test_acc,
            ci=None, 
            color="tab:purple", 
            linestyle='--', 
            marker='o', 
            label='Legitimate test set accuracy', 
            markersize=7,
            ax=ax
        )
        ax3.set(yticks=np.arange(0, 1.2, 0.1))
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(output_paths["normal_test"]), exist_ok=True)
        plt.savefig(output_paths["normal_test"], bbox_inches='tight', dpi=300)
        print(f"Saved normal test accuracy plot to: {output_paths['normal_test']}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    # Plot 3: Watermark accuracy only
    if "watermark" in output_paths and adv_test_acc:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax3 = sns.lineplot(
            x=list(range(len(adv_test_acc))), 
            y=adv_test_acc,
            ci=None, 
            color="tab:purple", 
            linestyle='--', 
            marker='o', 
            label='Watermark set accuracy', 
            markersize=7,
            ax=ax
        )
        ax3.set(yticks=np.arange(0, 1.2, 0.1))
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(output_paths["watermark"]), exist_ok=True)
        plt.savefig(output_paths["watermark"], bbox_inches='tight', dpi=300)
        print(f"Saved watermark accuracy plot to: {output_paths['watermark']}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()


def main():
    """Main entry point for unified visualization script."""
    parser = argparse.ArgumentParser(
        description='Unified visualization tool for watermarking experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot stealing accuracy
  python visualization_unified.py --mode stealing_accuracy \\
      --test-acc results/test_acc.csv \\
      --watermark-acc results/watermark_acc.csv \\
      --output results/images/stealing_accuracy.png
  
  # Plot learning rate
  python visualization_unified.py --mode learning_rate \\
      --json results/acc_loss.json \\
      --output results/images/learning_rate.png
  
  # Plot training accuracies
  python visualization_unified.py --mode training_accuracy \\
      --json results/acc_loss.json \\
      --output-combined results/images/combined.png \\
      --output-normal results/images/normal.png \\
      --output-watermark results/images/watermark.png
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['stealing_accuracy', 'learning_rate', 'training_accuracy'],
        help='Visualization mode to use'
    )
    
    # Common arguments
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display plots interactively'
    )
    
    # Arguments for stealing_accuracy mode
    parser.add_argument(
        '--test-acc',
        type=str,
        help='Path to test accuracy CSV file (for stealing_accuracy mode)'
    )
    parser.add_argument(
        '--watermark-acc',
        type=str,
        help='Path to watermark accuracy CSV file (for stealing_accuracy mode)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output path for the plot (for stealing_accuracy and learning_rate modes)'
    )
    
    # Arguments for learning_rate mode
    parser.add_argument(
        '--json',
        type=str,
        help='Path to JSON file with data (for learning_rate and training_accuracy modes)'
    )
    
    # Arguments for training_accuracy mode
    parser.add_argument(
        '--output-combined',
        type=str,
        help='Output path for combined accuracy plot (for training_accuracy mode)'
    )
    parser.add_argument(
        '--output-normal',
        type=str,
        help='Output path for normal test accuracy plot (for training_accuracy mode)'
    )
    parser.add_argument(
        '--output-watermark',
        type=str,
        help='Output path for watermark accuracy plot (for training_accuracy mode)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'stealing_accuracy':
            if not args.test_acc or not args.watermark_acc or not args.output:
                parser.error("--mode stealing_accuracy requires --test-acc, --watermark-acc, and --output")
            plot_stealing_accuracy(
                test_acc_file=args.test_acc,
                watermark_acc_file=args.watermark_acc,
                output_path=args.output,
                show_plot=args.show
            )
        
        elif args.mode == 'learning_rate':
            if not args.json or not args.output:
                parser.error("--mode learning_rate requires --json and --output")
            plot_learning_rate(
                json_path=args.json,
                output_path=args.output,
                show_plot=args.show
            )
        
        elif args.mode == 'training_accuracy':
            if not args.json:
                parser.error("--mode training_accuracy requires --json")
            
            output_paths = {}
            if args.output_combined:
                output_paths["combined"] = args.output_combined
            if args.output_normal:
                output_paths["normal_test"] = args.output_normal
            if args.output_watermark:
                output_paths["watermark"] = args.output_watermark
            
            if not output_paths:
                parser.error("--mode training_accuracy requires at least one of --output-combined, --output-normal, or --output-watermark")
            
            plot_training_accuracy(
                json_path=args.json,
                output_paths=output_paths,
                show_plot=args.show
            )
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

