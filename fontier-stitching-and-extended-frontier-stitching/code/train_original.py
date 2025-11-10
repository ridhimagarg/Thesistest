"""
@author: Ridhima Garg

Introduction:
    This file contains the implementation for training the original unwatermarked model for the particular dataset

"""

import argparse
import warnings

warnings.filterwarnings('ignore')

import numpy as np
from tensorflow.keras.datasets import mnist, cifar10
import os
import models
import tensorflow as tf
import matplotlib.pyplot as plt
# Optional MLflow import
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

from datetime import datetime
from utils.data_utils import DataManager
from utils.performance_utils import enable_mixed_precision, optimize_gpu_memory, create_data_generator_with_validation
from utils.experiment_logger import ExperimentLogger, log_reproducibility_info
import time
import logging

# Set up logger for MLflow warnings
mlflow_logger = logging.getLogger("mlflow")

# GPU Configuration - uncomment to disable GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Configure GPU (Metal on macOS, CUDA on Linux/Windows)
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    try:
        # For Metal GPU on macOS, memory growth is not needed but we can still configure it
        # Metal handles memory management automatically
        for gpu in physical_devices:
            # Memory growth is mainly for CUDA GPUs, but safe to set for Metal too
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except AttributeError:
                # Metal GPU doesn't support memory_growth, which is fine
                pass
        print(f"✅ GPU detected: {len(physical_devices)} GPU(s) available")
        print(f"   Using: {physical_devices[0].name}")
    except RuntimeError as e:
        print(f"⚠️  GPU configuration error: {e}")
        print("   Falling back to CPU")
else:
    print("ℹ️  No GPU detected, using CPU")

# Initialize MLflow with error handling
mlflow_enabled = False
if MLFLOW_AVAILABLE:
    try:
        mlflow.set_tracking_uri("sqlite:///../mlflow.db")
        # mlflow.set_tracking_uri("file:///../mlruns")
        mlflow.set_experiment("frontier-stiching-original")
        mlflow_enabled = True
    except Exception as e:
        print(f"⚠️  Warning: MLflow initialization failed: {e}")
        print("   Continuing without MLflow logging...")
        mlflow_enabled = False
else:
    print("ℹ️  MLflow not available, continuing without MLflow logging...")

now = datetime.now().strftime("%d-%m-%Y")


## creating the folders.

RESULTS_PATH = f"../results/original_{now}"
RESULTS_FOLDER_TRIGGERS = "triggers"
LOSS_FOLDER = "losses"
MODEL_PATH = f"../models/original_{now}"

if not os.path.exists(os.path.join(RESULTS_PATH, LOSS_FOLDER)):
    os.makedirs(os.path.join(RESULTS_PATH, LOSS_FOLDER))
if not os.path.exists(os.path.join(RESULTS_PATH, RESULTS_FOLDER_TRIGGERS)):
    os.makedirs(os.path.join(RESULTS_PATH, RESULTS_FOLDER_TRIGGERS))


# Data preprocessing is now handled by utils.data_utils.DataManager


def lr_schedule(epoch, initial_lr=0.001, decay_factor=0.1, decay_epochs=[30, 40]):
    """Optimized Learning Rate Schedule for CIFAR-10

    Learning rate is scheduled to be reduced at specified epochs.
    This helps the model converge better and achieve higher accuracy.

    # Arguments
        epoch (int): The number of epochs
        initial_lr (float): Initial learning rate
        decay_factor (float): Factor to multiply LR by at decay epochs
        decay_epochs (list): Epochs at which to decay learning rate

    # Returns
        lr (float32): learning rate
    """
    lr = initial_lr
    # Apply decay at specified epochs
    for decay_epoch in sorted(decay_epochs, reverse=True):
        if epoch >= decay_epoch:
            lr *= decay_factor
            break
    return lr


def train_model(dataset_name, model_architecture, epochs, dropout, batch_size=128, optimizer="adam", lr=0.001,
                weight_decay=0.0001, lr_schedule_enabled=True, lr_decay_factor=0.1, lr_decay_epochs=[30, 40],
                beta_1=0.9, beta_2=0.999, epsilon=1e-7, use_mixed_precision=False):

    """
    Main idea
    ---------
    Training the model

    Args:
        dataset_name: name of the dataset
        model_architecture: architecture for the model
        epochs: number of epochs
        dropout: dropout for training
        batch_size: batch size
        optimizer: optimizer to use for training
        lr: learning rate
        weight_decay: learning rate decay

    """

    experiment_name = dataset_name + "Frontier_Orignal"
    
    # Initialize ExperimentLogger for comprehensive logging
    exp_logger = ExperimentLogger("train_original", output_dir=RESULTS_PATH)
    
    # Log reproducibility info
    log_reproducibility_info(output_dir=str(exp_logger.experiment_dir), seed=0)
    
    # Enable mixed precision if requested
    if use_mixed_precision:
        enable_mixed_precision('mixed_float16')
        print("✅ Mixed precision training enabled", flush=True)
    
    # Helper function to safely log to MLflow
    def safe_mlflow_log(func, *args, **kwargs):
        """Safely execute MLflow logging functions with error handling."""
        if not mlflow_enabled or mlflow is None:
            return
        try:
            func(*args, **kwargs)
        except Exception as e:
            print(f"⚠️  Warning: MLflow logging failed: {e}")
            print("   Continuing without MLflow logging...")
    
    # Context manager for MLflow run with error handling
    class SafeMLflowRun:
        """Context manager for MLflow runs that handles errors gracefully."""
        def __init__(self, run_name):
            self.run_name = run_name
            self.run_active = False
            
        def __enter__(self):
            if mlflow_enabled and mlflow is not None:
                try:
                    mlflow.start_run(run_name=self.run_name)
                    self.run_active = True
                except Exception as e:
                    print(f"⚠️  Warning: Failed to start MLflow run: {e}")
                    print("   Continuing without MLflow logging...")
                    self.run_active = False
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.run_active and mlflow_enabled and mlflow is not None:
                try:
                    mlflow.end_run()
                except Exception as e:
                    print(f"⚠️  Warning: Failed to end MLflow run: {e}")
            return False  # Don't suppress exceptions
    
    # Use safe MLflow run context manager
    with SafeMLflowRun(experiment_name):
        params = {"dataset_name": dataset_name, "epochs_pretrain": epochs,
                  "model_architecture": model_architecture, "optimizer": str(optimizer), "lr": lr,
                  "weight_decay": weight_decay, "dropout": dropout, "mixed_precision": use_mixed_precision}
        
        # Log hyperparameters to ExperimentLogger
        exp_logger.log_hyperparameters(**params)

        #    models_mapping = {"resnet34": models.ResNet34, "conv_2": models.Plain_2_conv_Keras, "small": models.Small,
        #                      "mnist_l2": models.MNIST_L2,
        #                      "mnist_l2_drp02": models.MNIST_L2, "mnist_l2_drp03": models.MNIST_L2, "mnist_l5": models.MNIST_L5,
        #                      "mnist_l5_drp02": models.MNIST_L5, "mnist_l5_drp03": models.MNIST_L5,
        #                      "cifar10_base": models.CIFAR10_BASE, "cifar10_base_drp02": models.CIFAR10_BASE,
        #                      "cifar10_base_drp03": models.CIFAR10_BASE,
        #                      "cifar10_base_2": models.CIFAR10_BASE_2}

        # Auto-select model based on dataset if model_architecture is 'auto'
        if model_architecture == "auto":
            if dataset_name == "mnist":
                model_architecture = "mnist_l2"
            elif dataset_name in ["cifar10", "cifar100", "svhn", "stl10", "eurosat"]:
                # All RGB datasets use the same flexible model
                # CIFAR10_BASE_2 uses GlobalAveragePooling2D which adapts to different input sizes
                model_architecture = "cifar10_base_2"
            else:
                raise ValueError(f"Auto model selection not supported for dataset: {dataset_name}")
            print(f"Auto-selected model architecture: {model_architecture} for dataset: {dataset_name}")

        models_mapping = {"mnist_l2": models.MNIST_L2, "cifar10_base_2": models.CIFAR10_BASE_2,
                          "cifar10_base_3": models.CIFAR10_BASE_3, "resnet34": models.ResNet34}

        # Use centralized DataManager for data loading
        x_train, y_train, x_test, y_test, input_shape, num_classes = DataManager.load_and_preprocess(dataset_name)

        print(x_train.shape, y_train.shape, x_test.shape)

        if model_architecture == "resnet34":
            model_name, model = models_mapping[model_architecture]().call(input_shape)
        else:
            if dropout:
                model_name, model = models_mapping[model_architecture](input_shape, dropout, num_classes=num_classes)
            else:
                model_name, model = models_mapping[model_architecture](num_classes=num_classes)

        params["model_detail_architecture_name"] = model_name

        for param, param_val in params.items():
            if mlflow is not None:
                safe_mlflow_log(mlflow.log_param, param, param_val)

        print(model.summary())
        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=optimizer, metrics=['accuracy'])
        
        # Log model size before training
        total_params = model.count_params()
        model_size_mb = total_params * 4 / (1024 * 1024)  # float32 = 4 bytes
        print(f"📊 Model size: {total_params:,} parameters ({model_size_mb:.2f} MB)")
        exp_logger.metrics['initial_model_size_params'] = int(total_params)
        exp_logger.metrics['initial_model_size_mb'] = float(model_size_mb)

        CHECKPOINT_FOLDER = os.path.join(MODEL_PATH, dataset_name + "_" + str(epochs) + "_" + model_name)

        CHECKPOINT_FILEPATH = os.path.join(MODEL_PATH, dataset_name + "_" + str(epochs) + "_" + model_name,
                                           "Original_checkpoint_best.keras")
        if not os.path.exists(CHECKPOINT_FOLDER):
            os.makedirs(CHECKPOINT_FOLDER)

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=CHECKPOINT_FILEPATH,
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            mode='max',
            save_weights_only=False)

        # Track training time
        training_start_time = time.time()
        
        # Setup callbacks
        callbacks = [model_checkpoint_callback]
        
        # Add learning rate scheduler if enabled
        if lr_schedule_enabled:
            lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
                lambda e: lr_schedule(e, initial_lr=lr, decay_factor=lr_decay_factor, decay_epochs=lr_decay_epochs),
                verbose=1
            )
            callbacks.append(lr_scheduler)
        
        # Optional: Add ReduceLROnPlateau for adaptive learning rate
        # lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
        #     factor=np.sqrt(0.1),
        #     cooldown=0,
        #     patience=5,
        #     min_lr=0.5e-6,
        #     verbose=1
        # )
        # callbacks.append(lr_reducer)

        # Optimized: Use tf.data.Dataset for better performance
        # Split validation data manually since we're using tf.data
        val_split = 0.2
        val_size = int(len(x_train) * val_split)
        x_val = x_train[:val_size]
        y_val = y_train[:val_size]
        x_train_split = x_train[val_size:]
        y_train_split = y_train[val_size:]
        
        # Create tf.data.Dataset with prefetching and optimization
        train_ds, val_ds = create_data_generator_with_validation(
            x_train_split, y_train_split,
            x_val, y_val,
            batch_size=batch_size,
            shuffle_train=True
        )
        
        history = model.fit(train_ds, epochs=epochs, verbose=1, validation_data=val_ds,
                            callbacks=callbacks)
        
        training_time = time.time() - training_start_time
        print(f"⏱️  Total training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

        train_acc_pretrain = history.history["accuracy"]
        val_acc_pretrain = history.history["val_accuracy"]
        train_loss_pretrain = history.history["loss"]
        val_loss_pretrain = history.history["val_loss"]
        
        # Log training history to ExperimentLogger
        epoch_times = []
        for idx in range(epochs):
            # Estimate epoch time (total time / epochs)
            epoch_time = training_time / epochs if epochs > 0 else 0
            epoch_times.append(epoch_time)
            exp_logger.log_training_epoch(
                epoch=idx,
                train_loss=train_loss_pretrain[idx],
                train_acc=train_acc_pretrain[idx],
                val_loss=val_loss_pretrain[idx],
                val_acc=val_acc_pretrain[idx],
                epoch_time=epoch_time
            )

        file = open(os.path.join(RESULTS_PATH, LOSS_FOLDER, dataset_name + "_" + model_name + "_logs.txt"), "w")
        file.write(f"Model size: {total_params:,} parameters ({model_size_mb:.2f} MB)\n")
        file.write(f"Total training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)\n\n")
        
        for idx, (train_loss, train_acc, val_loss, val_acc) in enumerate(
                zip(train_loss_pretrain, train_acc_pretrain, val_loss_pretrain, val_acc_pretrain)):
            file.write(
                f'Epoch: {idx + 1}, Train Loss: {train_loss:.3f} Train Acc: {train_acc:.3f} Val Loss: {val_loss:.3f} Val Acc: {val_acc:.3f}')
            file.write("\n")

        test_results = model.evaluate(x_test, y_test)
        print(f"Test results: {test_results}")
        file.write(f'\nTest Loss: {test_results[0]:.4f}, Test Accuracy: {test_results[1]:.4f}\n')
        
        # Log model metrics and training time
        exp_logger.log_model_metrics(model=model, x_test=x_test, y_test=y_test, prefix="final_")
        exp_logger.log_training_time(total_time=training_time, avg_epoch_time=np.mean(epoch_times))

        ## --------------------------------- Plotting the graphs --------------------------------- ##
        plt.figure()
        plt.plot(list(range(epochs)), train_loss_pretrain, label="Train loss_" + dataset_name, marker='o',
                 color='tab:purple')
        plt.plot(list(range(epochs)), val_loss_pretrain, label="Val loss_" + dataset_name, linestyle='--',
                 marker='o', color='tab:orange')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(RESULTS_PATH, LOSS_FOLDER, dataset_name + "OriginalLoss.png"))
        if mlflow is not None:
            safe_mlflow_log(mlflow.log_artifact, 
                           os.path.join(RESULTS_PATH, LOSS_FOLDER, dataset_name + "OriginalLoss.png"),
                           "OriginalinLoss.png")

        plt.figure()
        plt.plot(list(range(epochs)), train_acc_pretrain, label="Train acc_" + dataset_name, marker='o',
                 color='tab:purple')
        plt.plot(list(range(epochs)), val_acc_pretrain, label="Val acc_" + dataset_name, linestyle='--',
                 marker='o', color='tab:orange')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(RESULTS_PATH, LOSS_FOLDER, dataset_name + "OriginalAcc.png"))
        if mlflow is not None:
            safe_mlflow_log(mlflow.log_artifact,
                           os.path.join(RESULTS_PATH, LOSS_FOLDER, dataset_name + "OriginalAcc.png"),
                           "OriginalAcc.png")
            safe_mlflow_log(mlflow.log_artifact,
                           os.path.join(RESULTS_PATH, LOSS_FOLDER, dataset_name + "_" + model_name + "_logs.txt"),
                           "logs.txt")
        
        # Save all ExperimentLogger data
        exp_logger.save_all()
        
        # Create LaTeX table from training history if available
        if exp_logger.training_history:
            latex_table = exp_logger.create_latex_table("training_results.tex")
            if latex_table:
                print(f"✅ LaTeX table saved to: {exp_logger.experiment_dir / 'tables' / 'training_results.tex'}")
        
        print(f"✅ Comprehensive experiment data saved to: {exp_logger.experiment_dir}")

        return model


if __name__ == '__main__':
    # Optimized configuration values for CIFAR-10
    dataset_name = "cifar10"
    epochs = 50  # Increased from 30 for better accuracy
    batch_size = 128
    model_architecture = "cifar10_base_2"
    dropout = 0
    optimizer_name = "adam"
    lr = 0.001
    weight_decay = 0.0001  # L2 regularization to prevent overfitting
    lr_schedule_enabled = True  # Enable learning rate decay
    lr_decay_factor = 0.1
    lr_decay_epochs = [30, 40]  # Decay at epochs 30 and 40
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-7
    
    # Performance optimizations
    use_mixed_precision = os.getenv('USE_MIXED_PRECISION', 'false').lower() == 'true'
    
    # Optimize GPU memory
    optimize_gpu_memory()
    
    # Enable mixed precision if requested
    if use_mixed_precision:
        enable_mixed_precision('mixed_float16')

    # Setup optimizer
    if model_architecture == "resnet34":
        # Use learning rate schedule for ResNet
        if lr_schedule_enabled:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr,
                beta_1=beta_1,
                beta_2=beta_2,
                epsilon=epsilon,
                weight_decay=weight_decay
            )
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule(0))
    else:
        if optimizer_name == "adam":
            # Optimized Adam with proper hyperparameters
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr,
                beta_1=beta_1,
                beta_2=beta_2,
                epsilon=epsilon,
                weight_decay=weight_decay
            )
        else:
            optimizer = None

    train_model(dataset_name, model_architecture, epochs, dropout, batch_size,
                optimizer=optimizer, lr=lr, weight_decay=weight_decay,
                lr_schedule_enabled=lr_schedule_enabled, lr_decay_factor=lr_decay_factor,
                lr_decay_epochs=lr_decay_epochs, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
