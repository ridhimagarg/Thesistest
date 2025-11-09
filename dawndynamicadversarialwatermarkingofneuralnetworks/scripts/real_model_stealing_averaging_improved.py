"""
Model extraction attack script using KnockoffNets.

This script performs model stealing attacks on victim models and evaluates
the stolen models on test sets and watermark sets.
"""

import argparse
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import mlconfig
import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
import torchvision as tv
from art.attacks.extraction import KnockoffNets
from art.estimators.classification.pytorch import PyTorchClassifier

from utils import logger
import models
from environment import download_victim, setup_transformations

# Constants
NUM_CLASSES = 10
BATCH_SIZE_FIT = 64
BATCH_SIZE_QUERY = 64
DEFAULT_LR = 0.001
WATERMARK_THRESHOLDS = {
    "small": (0, 500, "watermark_set_50"),
    "medium": (500, 5000, "watermark_set_100"),
    "large": (5000, 20000, "watermark_set_250"),
}

log = logger.Logger(prefix=">>>")


def get_input_shape(dataset_name: str, attacker_model_architecture: str) -> Tuple[int, ...]:
    """
    Determine the input shape based on dataset and model architecture.
    
    Args:
        dataset_name: Name of the dataset (MNIST or CIFAR10)
        attacker_model_architecture: Architecture of the attacker model
        
    Returns:
        Input shape tuple (channels, height, width)
    """
    if dataset_name == "MNIST":
        return (1, 28, 28)
    elif dataset_name == "CIFAR10":
        if attacker_model_architecture == "RN34":
            return (3, 224, 224)
        else:
            return (3, 32, 32)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def get_watermark_set(
    len_steal: int, 
    watermark_50: Dict, 
    watermark_100: Dict, 
    watermark_250: Dict
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select appropriate watermark set based on stealing dataset size.
    
    Args:
        len_steal: Size of the stealing dataset
        watermark_50: Watermark set for small datasets
        watermark_100: Watermark set for medium datasets
        watermark_250: Watermark set for large datasets
        
    Returns:
        Tuple of (watermark features, watermark labels)
    """
    if len_steal <= WATERMARK_THRESHOLDS["small"][1]:
        return watermark_50["arr_0"], watermark_50["arr_1"]
    elif len_steal <= WATERMARK_THRESHOLDS["medium"][1]:
        return watermark_100["arr_0"], watermark_100["arr_1"]
    else:
        return watermark_250["arr_0"], watermark_250["arr_1"]


def load_test_set(data_path: str, dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load test set from saved numpy file.
    
    Args:
        data_path: Base path to data directory
        dataset_name: Name of the dataset
        
    Returns:
        Tuple of (test features, test labels)
        
    Raises:
        FileNotFoundError: If test set file doesn't exist
    """
    test_set_path = os.path.join(data_path, "test_set", dataset_name, "test_set.npz")
    if not os.path.exists(test_set_path):
        raise FileNotFoundError(f"Test set not found at {test_set_path}")
    
    test_set_array = np.load(test_set_path)
    return test_set_array["arr_0"], test_set_array["arr_1"]


def load_watermark_sets(data_path: str, dataset_name: str) -> Tuple[Dict, Dict, Dict]:
    """
    Load all watermark sets for a dataset.
    
    Args:
        data_path: Base path to data directory
        dataset_name: Name of the dataset
        
    Returns:
        Tuple of (watermark_50, watermark_100, watermark_250) dictionaries
        
    Raises:
        FileNotFoundError: If watermark set files don't exist
    """
    watermark_base_path = os.path.join(data_path, "watermark_set", dataset_name)
    
    watermark_paths = {
        "50": os.path.join(watermark_base_path, "watermark_set_50.npz"),
        "100": os.path.join(watermark_base_path, "watermark_set_100.npz"),
        "250": os.path.join(watermark_base_path, "watermark_set_250.npz"),
    }
    
    for name, path in watermark_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Watermark set {name} not found at {path}")
    
    return (
        np.load(watermark_paths["50"]),
        np.load(watermark_paths["100"]),
        np.load(watermark_paths["250"]),
    )


def get_available_models() -> Dict[str, type]:
    """Get dictionary of available model architectures."""
    return {
        "MNIST_L2": models.MNIST_L2,
        "MNIST_L2_DRP03": models.MNIST_L2_DRP03,
        "MNIST_L2_DRP05": models.MNIST_L2_DRP05,
        "MNIST_L5": models.MNIST_L5,
        "MNIST_L5_Latent": models.MNIST_L5_with_latent,
        "MNIST_L5_DRP03": models.MNIST_L5_DRP03,
        "MNIST_L5_DRP05": models.MNIST_L5_DRP05,
        "CIFAR10_BASE": models.CIFAR10_BASE,
        "CIFAR10_BASE_2": models.CIFAR10_BASE_2,
        "CIFAR10_BASE_LATENT": models.CIFAR10_BASE_LATENT,
        "CIFAR10_BASE_DRP03": models.CIFAR10_BASE_DRP03,
        "CIFAR10_BASE_DRP05": models.CIFAR10_BASE_DRP05,
        "RN34": tv.models.resnet34,
    }


def setup_victim_attacker(
    dataset_name: str,
    victim_model_architecture: str,
    attacker_model_architecture: str,
    model_to_attack_path: str,
    data_path: str,
) -> Tuple[nn.Module, PyTorchClassifier, nn.Module, np.ndarray, np.ndarray, Dict, Dict, Dict, Tuple[int, ...]]:
    """
    Setup victim and attacker models with test and watermark sets.
    
    Args:
        dataset_name: Name of the dataset
        victim_model_architecture: Architecture of victim model
        attacker_model_architecture: Architecture of attacker model
        model_to_attack_path: Path to victim model weights
        data_path: Base path to data directory
        
    Returns:
        Tuple containing:
        - victim_model: PyTorch victim model
        - classifier_victim: ART PyTorchClassifier wrapper for victim
        - attacker_model: PyTorch attacker model (uninitialized)
        - x_test: Test set features
        - y_test: Test set labels
        - watermark_50: Small watermark set
        - watermark_100: Medium watermark set
        - watermark_250: Large watermark set
        - input_shape: Input shape tuple
    """
    # Load test set
    x_test, y_test = load_test_set(data_path, dataset_name)
    
    # Determine input shape
    input_shape = get_input_shape(dataset_name, attacker_model_architecture)
    
    # Get available models
    available_models = get_available_models()
    
    # Validate model architectures
    if victim_model_architecture not in available_models:
        raise ValueError(f"Unknown victim model architecture: {victim_model_architecture}")
    if attacker_model_architecture not in available_models:
        raise ValueError(f"Unknown attacker model architecture: {attacker_model_architecture}")
    
    # Initialize models
    victim_model = available_models[victim_model_architecture]()
    attacker_model = available_models[attacker_model_architecture]()
    
    # Load victim model weights
    if not os.path.exists(model_to_attack_path):
        raise FileNotFoundError(f"Model file not found: {model_to_attack_path}")
    
    models.load_state(victim_model, model_to_attack_path)
    log.info(f"Loaded victim model from: {model_to_attack_path}")
    
    # Convert victim model to ART classifier
    classifier_victim = PyTorchClassifier(
        victim_model,
        loss=nn.CrossEntropyLoss(),
        input_shape=input_shape,
        nb_classes=NUM_CLASSES,
        device_type="gpu",
        optimizer=t.optim.Adam(victim_model.parameters(), lr=DEFAULT_LR),
    )
    
    # Load watermark sets
    watermark_50, watermark_100, watermark_250 = load_watermark_sets(data_path, dataset_name)
    
    return (
        victim_model,
        classifier_victim,
        attacker_model,
        x_test,
        y_test,
        watermark_50,
        watermark_100,
        watermark_250,
        input_shape,
    )


def calculate_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate classification accuracy.
    
    Args:
        predictions: Model predictions (logits or probabilities)
        labels: True labels
        
    Returns:
        Accuracy as a float between 0 and 1
    """
    pred_labels = np.argmax(predictions, axis=1)
    return np.mean(pred_labels == labels)


def model_extraction_attack(
    dataset_name: str,
    victim_model_architecture: str,
    attacker_model_architecture: str,
    number_of_queries: List[int],
    model_to_attack_path: str,
    num_epochs_to_steal: int,
    results_path: str,
    data_path: str,
    optimizer: str = "adam",
    lr: float = DEFAULT_LR,
    weight_decay: float = 0.0,
    batch_size_fit: int = BATCH_SIZE_FIT,
    batch_size_query: int = BATCH_SIZE_QUERY,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform model extraction attack using KnockoffNets.
    
    Args:
        dataset_name: Name of the dataset
        victim_model_architecture: Architecture of victim model
        attacker_model_architecture: Architecture of attacker model
        number_of_queries: List of query sizes to test
        model_to_attack_path: Path to victim model weights
        num_epochs_to_steal: Number of epochs for model stealing
        results_path: Path to save results
        data_path: Base path to data directory
        optimizer: Optimizer name (currently unused)
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        batch_size_fit: Batch size for fitting
        batch_size_query: Batch size for querying
        
    Returns:
        Tuple of (test accuracy DataFrame, watermark accuracy DataFrame)
    """
    # Setup models and data
    (
        victim_model,
        classifier_victim,
        attacker_model,
        x_test,
        y_test,
        watermark_50,
        watermark_100,
        watermark_250,
        input_shape,
    ) = setup_victim_attacker(
        dataset_name, victim_model_architecture, attacker_model_architecture, model_to_attack_path, data_path
    )
    
    # Create log file path
    model_name = Path(model_to_attack_path).stem
    log_file_path = os.path.join(
        results_path,
        "_".join((dataset_name, str(victim_model_architecture), model_name + "_logs.txt")),
    )
    
    # Evaluate victim model
    victim_predictions = classifier_victim.predict(x_test)
    victim_acc = calculate_accuracy(victim_predictions, y_test)
    log.info(f"Victim model test accuracy: {victim_acc:.4f}")
    
    results = []
    results_adv = []
    
    # Perform attacks with different query sizes
    for len_steal in number_of_queries:
        if len_steal > len(x_test):
            log.warning(f"Query size {len_steal} exceeds test set size {len(x_test)}, skipping")
            continue
        
        # Randomly sample stealing dataset
        indices = np.random.permutation(len(x_test))
        x_steal = x_test[indices[:len_steal]]
        y_steal = y_test[indices[:len_steal]]
        
        # Get appropriate watermark set
        x_watermark_numpy, y_watermark_numpy = get_watermark_set(
            len_steal, watermark_50, watermark_100, watermark_250
        )
        
        # Create attack
        attack = KnockoffNets(
            classifier=classifier_victim,
            batch_size_fit=batch_size_fit,
            batch_size_query=batch_size_query,
            nb_epochs=num_epochs_to_steal,
            nb_stolen=len_steal,
            use_probability=False,
        )
        
        log.info(f"Performing attack with {len_steal} queries")
        
        # Initialize stolen classifier
        classifier_stolen = PyTorchClassifier(
            attacker_model,
            loss=nn.CrossEntropyLoss(),
            input_shape=input_shape,
            nb_classes=NUM_CLASSES,
            device_type="gpu",
            optimizer=t.optim.Adam(attacker_model.parameters(), lr=lr),
        )
        
        # Extract model
        classifier_stolen = attack.extract(
            x_steal,
            y_steal,
            thieved_classifier=classifier_stolen,
            x_watermark=x_watermark_numpy,
            y_watermark=y_watermark_numpy,
        )
        
        # Evaluate on test set
        stolen_predictions = classifier_stolen.predict(x_test)
        test_acc = calculate_accuracy(stolen_predictions, y_test)
        log.info(f"Stolen model test accuracy with {len_steal} queries: {test_acc:.4f}")
        results.append(("argmax_knockoffNets", len_steal, test_acc))
        
        # Evaluate on watermark set
        watermark_predictions = classifier_stolen.predict(x_watermark_numpy)
        watermark_acc = calculate_accuracy(watermark_predictions, y_watermark_numpy)
        log.info(f"Stolen model watermark accuracy: {watermark_acc:.4f}")
        results_adv.append(("argmax_knockoffNets", len_steal, watermark_acc))
    
    # Save results to log file
    os.makedirs(results_path, exist_ok=True)
    with open(log_file_path, "w") as log_file:
        log_file.write(f"Victim model path: {model_to_attack_path}\n")
        log_file.write(f"Victim model test accuracy: {victim_acc:.4f}\n\n")
        for name, len_steal, acc in results:
            log_file.write(f"Method: {name}, Queries: {len_steal}, Test Accuracy: {acc:.4f}\n")
        log_file.write("\n")
        for name, len_steal, acc in results_adv:
            log_file.write(f"Method: {name}, Queries: {len_steal}, Watermark Accuracy: {acc:.4f}\n")
    
    # Create DataFrames
    df_test = pd.DataFrame(results, columns=("Method Name", "Stealing Dataset Size", "Accuracy"))
    df_watermark = pd.DataFrame(
        results_adv, columns=("Method Name", "Stealing Dataset Size", "Accuracy")
    )
    
    # Create and save plot
    image_save_name = os.path.join(
        results_path,
        "_".join((dataset_name, str(victim_model_architecture), model_name + "TestandWatermarkAcc.png")),
    )
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot test accuracy
    for name, group in df_test.groupby("Method Name"):
        group.plot(
            x="Stealing Dataset Size",
            y="Accuracy",
            ax=ax,
            label="Test acc",
            linestyle="--",
            marker="o",
            color="tab:purple",
        )
    
    # Plot watermark accuracy
    for name, group in df_watermark.groupby("Method Name"):
        group.plot(
            x="Stealing Dataset Size",
            y="Accuracy",
            ax=ax,
            label="Watermark acc",
            linestyle="--",
            marker="o",
            color="tab:orange",
        )
    
    ax.set_xlabel("Stealing Dataset Size")
    ax.set_ylabel("Stolen Model Test and Adversarial Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(image_save_name, dpi=300, bbox_inches="tight")
    plt.close()
    
    log.info(f"Results saved to {results_path}")
    
    return df_test, df_watermark


def main():
    """Main function to run model extraction attacks."""
    parser = argparse.ArgumentParser(description="Model extraction attack using KnockoffNets")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="../configurations/knockoffnet/attack_original.yaml",
        help="Path to configuration file",
    )
    
    args = parser.parse_args()
    config = mlconfig.load(args.config)
    
    # Set random seeds for reproducibility
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)
    
    # Setup paths
    now = datetime.now().strftime("%d-%m-%Y")
    results_path = f"../results/attack_original_{now}"
    model_path = f"../models/attack_original_{now}"
    data_path = "../data"
    
    # Create directories
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    
    # Extract configuration
    dataset_name = config.dataset_name
    model_to_attack_path = config.model_to_attack_path
    number_of_queries = getattr(config, "number_of_queries", [250, 500, 1000, 5000, 10000, 20000])
    
    # Initialize result DataFrames
    final_df_test = pd.DataFrame(columns=("Method Name", "Stealing Dataset Size", "Accuracy"))
    final_df_adv = pd.DataFrame(columns=("Method Name", "Stealing Dataset Size", "Accuracy"))
    
    # Run attacks (can be modified to run multiple times for averaging)
    num_runs = getattr(config, "num_runs", 1)
    for run in range(num_runs):
        log.info(f"Starting run {run + 1}/{num_runs}")
        
        df_test, df_watermark = model_extraction_attack(
            dataset_name=dataset_name,
            victim_model_architecture=config.victim_model_architecture,
            attacker_model_architecture=config.attacker_model_architecture,
            number_of_queries=number_of_queries,
            model_to_attack_path=model_to_attack_path,
            num_epochs_to_steal=config.epochs_extract,
            results_path=results_path,
            data_path=data_path,
            optimizer=getattr(config, "optimizer", "adam"),
            lr=getattr(config, "lr", DEFAULT_LR),
            weight_decay=getattr(config, "weight_decay", 0.0),
        )
        
        final_df_test = pd.concat([final_df_test, df_test], ignore_index=True)
        final_df_adv = pd.concat([final_df_adv, df_watermark], ignore_index=True)
    
    # Save final results
    model_name = Path(model_to_attack_path).stem
    test_acc_path = os.path.join(
        results_path,
        "_".join((dataset_name, str(config.victim_model_architecture), model_name + "df_test_acc.csv")),
    )
    watermark_acc_path = os.path.join(
        results_path,
        "_".join((dataset_name, str(config.victim_model_architecture), model_name + "df_watermark_acc.csv")),
    )
    
    final_df_test.to_csv(test_acc_path, index=False)
    final_df_adv.to_csv(watermark_acc_path, index=False)
    
    log.info(f"Final results saved to {results_path}")


if __name__ == "__main__":
    main()

