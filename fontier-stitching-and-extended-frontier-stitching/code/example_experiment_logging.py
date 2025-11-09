"""
Example: How to integrate ExperimentLogger for research paper logging.

This example shows how to add comprehensive logging to your scripts
for research paper reproducibility and analysis.
"""

import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from utils.experiment_logger import ExperimentLogger, log_reproducibility_info
from utils.data_utils import DataManager


def example_training_with_logging():
    """Example: Training with comprehensive logging."""
    
    # Initialize logger
    logger = ExperimentLogger("train_original", output_dir="../results")
    
    # Log reproducibility info
    log_reproducibility_info(output_dir=str(logger.get_experiment_path()), seed=0)
    
    # Configuration
    dataset_name = "cifar10"
    epochs = 30
    batch_size = 128
    lr = 0.001
    optimizer_name = "adam"
    model_architecture = "cifar10_base_2"
    
    # Log hyperparameters
    logger.log_hyperparameters(
        dataset_name=dataset_name,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        optimizer=optimizer_name,
        model_architecture=model_architecture,
        weight_decay=0.0001,
        dropout=0.0
    )
    
    # Load data
    x_train, y_train, x_test, y_test, input_shape, num_classes = \
        DataManager.load_and_preprocess(dataset_name)
    
    # Create model (simplified example)
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Training with logging
    start_time = time.time()
    epoch_times = []
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Train one epoch (simplified)
        history = model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=1,
            verbose=0,
            validation_split=0.2
        )
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        # Log epoch metrics
        logger.log_training_epoch(
            epoch=epoch,
            train_loss=history.history['loss'][0],
            train_acc=history.history['accuracy'][0],
            val_loss=history.history['val_loss'][0],
            val_acc=history.history['val_accuracy'][0],
            lr=lr,
            epoch_time=epoch_time
        )
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"train_acc={history.history['accuracy'][0]:.4f}, "
              f"val_acc={history.history['val_accuracy'][0]:.4f}")
    
    total_time = time.time() - start_time
    
    # Log model metrics
    logger.log_model_metrics(
        model=model,
        x_test=x_test,
        y_test=y_test,
        prefix="final_"
    )
    
    # Log training time
    logger.log_training_time(
        total_time=total_time,
        avg_epoch_time=np.mean(epoch_times),
        min_epoch_time=np.min(epoch_times),
        max_epoch_time=np.max(epoch_times)
    )
    
    # Save all data
    logger.save_all()
    
    # Create LaTeX table
    logger.create_latex_table("training_results.tex")
    
    print(f"\n✅ All metrics logged to: {logger.get_experiment_path()}")
    return logger


def example_attack_with_logging():
    """Example: Model extraction attack with comprehensive logging."""
    
    # Initialize logger
    logger = ExperimentLogger("model_extraction_attack", output_dir="../results")
    
    # Log hyperparameters
    logger.log_hyperparameters(
        dataset_name="cifar10",
        attacker_architecture="cifar10_base_2",
        query_budgets=[250, 500, 1000, 5000, 10000, 20000],
        epochs_extract=50,
        lr=0.001,
        optimizer="adam"
    )
    
    # Simulate attack results for each query budget
    query_budgets = [250, 500, 1000, 5000, 10000, 20000]
    victim_watermark_acc = 0.6842
    
    for query_budget in query_budgets:
        # Simulate results (replace with actual attack)
        test_acc = 0.1 + (query_budget / 20000) * 0.6  # Simulated
        watermark_acc = 0.1 + (query_budget / 20000) * 0.5  # Simulated
        
        # Verification result
        from utils.watermark_verifier import WatermarkVerifier
        verifier = WatermarkVerifier(
            victim_acc=victim_watermark_acc,
            num_classes=10,
            watermark_size=10000
        )
        verification_result = verifier.verify_theft(
            suspected_acc=watermark_acc,
            threshold_ratio=0.5,
            confidence=0.99
        )
        
        # Comprehensive metrics (simulated)
        comprehensive_metrics = {
            'fidelity': 0.7 + (query_budget / 20000) * 0.2,
            'watermark_retention': watermark_acc / victim_watermark_acc,
            'test_acc_gap': abs(0.7 - test_acc),
            'kl_divergence': 0.5 - (query_budget / 20000) * 0.3,
            'detectability': 0.6 + (query_budget / 20000) * 0.2
        }
        
        # Log attack results
        logger.log_attack_results(
            query_budget=query_budget,
            test_acc=test_acc,
            watermark_acc=watermark_acc,
            verification_result=verification_result,
            comprehensive_metrics=comprehensive_metrics
        )
        
        print(f"Query budget {query_budget}: "
              f"test_acc={test_acc:.4f}, "
              f"watermark_acc={watermark_acc:.4f}, "
              f"is_stolen={verification_result['is_stolen']}")
    
    # Save all data
    logger.save_all()
    
    # Create LaTeX table
    logger.create_latex_table("attack_results.tex")
    
    print(f"\n✅ All attack results logged to: {logger.get_experiment_path()}")
    return logger


if __name__ == "__main__":
    print("=" * 60)
    print("Example: Training with Comprehensive Logging")
    print("=" * 60)
    example_training_with_logging()
    
    print("\n" + "=" * 60)
    print("Example: Attack with Comprehensive Logging")
    print("=" * 60)
    example_attack_with_logging()
    
    print("\n" + "=" * 60)
    print("✅ Examples completed!")
    print("=" * 60)
    print("\nCheck the results/ directory for:")
    print("  - CSV files with all metrics")
    print("  - JSON files with hyperparameters and results")
    print("  - LaTeX tables ready for paper inclusion")
    print("  - Summary statistics")

