"""
@author: Ridhima Garg

Introduction:
    This is the main file for performing the watermarking using extended frontoer stitching method.

"""

import warnings
import argparse
import os
import sys
import math
import json
import time
import random
from datetime import datetime
from typing import Tuple, List, Dict, Any

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, Callback
import matplotlib.pyplot as plt
import models
from utils.data_utils import DataManager
from utils.performance_utils import enable_mixed_precision, optimize_gpu_memory
from utils.experiment_logger import ExperimentLogger, log_reproducibility_info

warnings.filterwarnings('ignore')

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

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def data_preprocessing(dataset_name, adv_data_path_numpy):
    """
    Preprocess data for watermarking with extended frontier stitching.
    
    This function uses the centralized DataManager for basic loading, 
    then performs custom splitting and combination for watermarking.
    
    Args:
        dataset_name: Name of the dataset.
        adv_data_path_numpy: Path to adversarial data file.
        
    Returns:
        Tuple containing multiple data splits for watermarking training.
    """
    # Use centralized DataManager for basic data loading
    x_train, y_train, x_test, y_test, x_adv, y_adv, input_shape = DataManager.load_and_preprocess_with_adversarial(
        dataset_name=dataset_name,
        adv_data_path=adv_data_path_numpy
    )
    
    num_classes = DataManager.get_dataset_info(dataset_name)['num_classes']
    
    print(x_adv.shape)
    print(y_adv.shape)

    # Split adversarial data into train/test (90/10 split)
    idx = np.random.randint(x_adv.shape[0], size=x_adv.shape[0])
    x_train_adv = x_adv[idx[:int(len(idx) * 0.9)]]
    y_train_adv = y_adv[idx[:int(len(idx) * 0.9)]]
    x_test_adv = x_adv[idx[int(len(idx) * 0.9):]]
    y_test_adv = y_adv[idx[int(len(idx) * 0.9):]]

    # Select matching amount of regular training data
    x_train_selected = x_train[:int(len(idx) * 0.9)]
    y_train_selected = y_train[:int(len(idx) * 0.9)]

    # Combine regular and adversarial data
    x_combined = np.concatenate((x_train_selected, x_train_adv), axis=0)
    y_combined = np.concatenate((y_train_selected, y_train_adv), axis=0)

    # Create combined train/val splits
    x_combined_train = np.concatenate(
        (x_train[: int(len(x_train_selected) * 0.9)], x_train_adv[: int(len(x_train_adv) * 0.9)]),
        axis=0)
    y_combined_train = np.concatenate(
        (y_train[: int(len(y_train_selected) * 0.9)], y_train_adv[: int(len(y_train_adv) * 0.9)]),
        axis=0)

    x_combined_val = np.concatenate(
        (x_train[int(len(x_train_selected) * 0.9):], x_train_adv[int(len(x_train_adv) * 0.9):]),
        axis=0)
    y_combined_val = np.concatenate(
        (y_train[int(len(y_train_selected) * 0.9):], y_train_adv[int(len(y_train_adv) * 0.9):]),
        axis=0)

    return x_train, y_train, x_test, y_test, x_adv, y_adv, x_train_adv, y_train_adv, x_test_adv, y_test_adv, x_combined, y_combined, x_combined_train, y_combined_train, x_combined_val, y_combined_val, input_shape, num_classes


def learning_rate_schedule(epoch):
    """
    learning rate schedule for finetuning over the watermark set.
    """
    # return epoch * 0.0001
    # epoch += 1
    if epoch <= 5:
        print("epoch", epoch)
        # return 0.001
        return (epoch) * 0.0001 ## return (epoch) * 0.001 for mnist 
    if epoch > 5 and epoch <= 10:
        print("epoch", epoch)
        print("decay", math.pow(2, ((epoch) - 5)))
        return 0.0005 / math.pow(2, ((epoch) - 5)) ## 0.005 / math.pow(2, ((epoch) - 5)) for mnist
    # if epoch <= 15:
    #     return 0.005 / math.pow(2, ((10) - 5))


def scheduler(epoch):
    """
    THis is the scheduler, which was there for trying purpose.
    """
    # if epoch <= 20:
    #     return epoch * 0.00004
    # if epoch > 20 and epoch <= 25:
    #     print("epoch", epoch)
    #     print("decay", math.pow(2, (epoch - 5)))
    #     return 0.00080 / math.pow(2, (epoch - 5))
    # if epoch < 5:
    #     return 0.00004
    # if epoch < 10:
    #     return 0.00005
    # if epoch < 15:
    #     return 0.00006
    # if epoch < 20:
    #     return 0.00008
    # if epoch < 25:
    #     return 0.00010
    # return 0.00004

    if epoch < 5:
        return 0.0008
    if epoch < 10:
        return 0.0002
    if epoch < 15:
        return 0.00004
    if epoch < 20:
        return 0.00001
    if epoch <= 25:
        return 0.000008
    return 0.00004


class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []
        self.total_time = time.time()

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

    def on_train_end(self, logs={}):
        self.total_time = time.time() - self.total_time


class normal_data_acc_callback(tf.keras.callbacks.Callback):
    """
    helper function for normal_data_acc_callback after each epoch.
    since we are finetuning over the adversaries/watermark set, to track for the normal test data, we are using this callback.
    """

    def __init__(self, test_data):
        self.x_test = test_data[0]
        self.y_test = test_data[1]

    def on_train_begin(self, logs={}):
        self.params["test_acc_"] = []

    def on_epoch_end(self, epoch, logs={}):
        acc = self.model.evaluate(self.x_test, self.y_test)
        print("test accuracy:", acc[1])
        self.params["test_acc_"].append(acc[1])
        return acc[1]


def watermark_finetuning(dataset_name, adv_data_path_numpy, model_to_finetune_path, epochs, dropout, batch_size,
                         optimizer, lr, weight_decay, num_layers_unfreeze, lr_schedule_enabled=True,
                         lr_decay_factor=0.5, lr_decay_epochs=[8, 12], results_path=None):

    """
    Main idea
    --------
    Performing the watermarking on the already trained model.
    It is a kind of advesarial finetuning, whose idea is motivated from the paper: https://arxiv.org/abs/2012.13628

    """


    x_train, y_train, x_test, y_test, x_adv, y_adv, x_train_adv, y_train_adv, x_test_adv, y_test_adv, x_combined, y_combined, x_combined_train, y_combined_train, x_combined_val, y_combined_val, input_shape, num_classes = data_preprocessing(
        dataset_name, adv_data_path_numpy)

    print(model_to_finetune_path)
    
    # Set RESULTS_PATH if not provided
    if results_path is None:
        now = datetime.now().strftime("%d-%m-%Y")
        RESULTS_PATH = f"../results/finetuned_finetuning_{now}"
    else:
        RESULTS_PATH = results_path
    
    # Set other paths (needed for file operations)
    DATA_PATH = "../data"
    MODEL_PATH = f"../models/finetuned_finetuning_{datetime.now().strftime('%d-%m-%Y')}"
    LOSS_FOLDER = "losses"
    
    # Extract which_adv from adversarial path (e.g., ../data/fgsm/cifar10/true/... -> 'true')
    adv_path_normalized = adv_data_path_numpy.replace("\\", "/")
    path_parts = adv_path_normalized.split("/")
    which_adv = 'true'  # Default
    if 'true' in path_parts:
        which_adv = 'true'
    elif 'false' in path_parts:
        which_adv = 'false'
    elif 'full' in path_parts:
        which_adv = 'full'
    
    # Create directories if they don't exist
    if not os.path.exists(os.path.join(RESULTS_PATH, LOSS_FOLDER, which_adv)):
        os.makedirs(os.path.join(RESULTS_PATH, LOSS_FOLDER, which_adv), exist_ok=True)
    
    if not os.path.exists(os.path.join(MODEL_PATH, which_adv)):
        os.makedirs(os.path.join(MODEL_PATH, which_adv), exist_ok=True)
    
    # Initialize ExperimentLogger for comprehensive logging
    exp_logger = ExperimentLogger("watermark_finetuning", output_dir=RESULTS_PATH)
    
    # Log reproducibility info
    log_reproducibility_info(output_dir=str(exp_logger.experiment_dir), seed=0)
    
    # Log hyperparameters
    exp_logger.log_hyperparameters(
        dataset_name=dataset_name,
        epochs=epochs,
        batch_size=batch_size,
        optimizer=str(optimizer),
        lr=lr,
        weight_decay=weight_decay,
        dropout=dropout,
        num_layers_unfreeze=num_layers_unfreeze,
        adv_data_path=adv_data_path_numpy
    )

    model = load_model(model_to_finetune_path, compile=False)

    print(model.summary())
    
    # Log model size before watermarking
    total_params_before = model.count_params()
    model_size_mb_before = total_params_before * 4 / (1024 * 1024)
    print(f"📊 Model size before watermarking: {total_params_before:,} parameters ({model_size_mb_before:.2f} MB)")
    exp_logger.metrics['model_size_before_params'] = int(total_params_before)
    exp_logger.metrics['model_size_before_mb'] = float(model_size_mb_before)

    model_name = model_to_finetune_path.replace("\\", "/").split("/")[-2]

    print(x_test.shape)

    loss_acc_dict = {}

    # if "WideResNet" in model_name:
    #     mean = [125.3, 123.0, 113.9]
    #     std = [63.0, 62.1, 66.7]

    #     for i in range(3):
    #         x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i])/std[i]

    # model.trainable = False
    # for layer in model.layers:
    # if layer.name not in ["dense_1"]:
    # if layer.name not in ["dense_1", "dropout_3", "dense", "batch_normalization_6", "conv2d_5"]: #"conv2d_5", "batch_normalization_5", "dropout_2", "flatten", "max_pooling2d_2"
    #   or layer.name == "dense" or layer.name == "batch_normalization_6" or layer.name == ""
    # layer.trainable = False

    # for layer in model.layers:
    #    print(layer.name, layer.trainable)

    time_callback = TimeHistory()

    # Optimized learning rate scheduler
    if lr_schedule_enabled:
        lrate = LearningRateScheduler(
            lambda e: learning_rate_schedule(e, initial_lr=lr, decay_factor=lr_decay_factor, decay_epochs=lr_decay_epochs),
            verbose=1
        )
    else:
        lrate = LearningRateScheduler(learning_rate_schedule, verbose=1)
    
    # Use provided learning rate instead of hardcoded value
    opt = Adam(learning_rate=lr)

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    file1 = open(os.path.join(RESULTS_PATH, LOSS_FOLDER, adv_data_path_numpy.replace("\\", "/").split("/")[-2],
                              dataset_name + "_" + str(epochs) + "_" + model_name +
                              adv_data_path_numpy.replace("\\", "/").split("/")[-1].split(".npz")[0] + "_logs.txt"),
                 "w")
    
    dict_file = os.path.join(RESULTS_PATH, LOSS_FOLDER, adv_data_path_numpy.replace("\\", "/").split("/")[-2],
                              dataset_name + "_" + str(epochs) + "_" + model_name +
                              adv_data_path_numpy.replace("\\", "/").split("/")[-1].split(".npz")[0] + "_acc_loss.json")

    CHECKPOINT_FILEPATH = os.path.join(MODEL_PATH, adv_data_path_numpy.replace("\\", "/").split("/")[-2],
                                       dataset_name + "_" + str(epochs) + "_" + str(epochs) + "_" + model_name +
                                       adv_data_path_numpy.replace("\\", "/").split("/")[-1].split(".npz")[0],
                                       "Victim_checkpoint_best.keras")

    # Ensure the directory exists before creating the checkpoint callback
    checkpoint_dir = os.path.dirname(CHECKPOINT_FILEPATH)
    os.makedirs(checkpoint_dir, exist_ok=True)

    ## creating the callback for checkpointing.
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_FILEPATH,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max',
        save_weights_only=False
    )

    ## Checking the test accuracy of the already trained model.
    test_acc = model.evaluate(x_test, y_test)
    print(f"Loaded model test acc: {test_acc}")
    file1.write(f"Loaded model test acc: {test_acc[1]} \n")

    ## fiunetuning the model to watermark it.
    history = model.fit(x_train_adv, y_train_adv, shuffle=True, batch_size=batch_size, epochs=epochs,
                        verbose=1, validation_split=0.2,
                        callbacks=[model_checkpoint_callback, lrate, normal_data_acc_callback((x_test, y_test)), time_callback])  ## check for validation 
    print(history.history.keys())

    print(f"Total training time: {time_callback.total_time} seconds")
    file1.write(f"Total training time: {time_callback.total_time} seconds\n")

    acc = model.evaluate(x_test, y_test)
    print(f"Finetuned model test acc: {acc}")
    file1.write(f"Finetuned model test acc: {acc} \n")

    ## evaluating over the training adversary data
    acc = model.evaluate(x_train_adv, y_train_adv)
    print(f"Finetuned model train adv acc: {acc}")
    file1.write(f"Finetuned model train adv acc: {acc}\n")
    exp_logger.metrics['watermark_train_acc'] = float(acc[1])

    ## evaluating over the test adversary data
    acc = model.evaluate(x_test_adv, y_test_adv)
    print(f"Finetuned model test adv acc: {acc}")
    file1.write(f"Finetuned model test adv acc: {acc}\n")
    exp_logger.metrics['watermark_test_acc'] = float(acc[1])
    
    # Log model size after watermarking
    total_params_after = model.count_params()
    model_size_mb_after = total_params_after * 4 / (1024 * 1024)
    print(f"📊 Model size after watermarking: {total_params_after:,} parameters ({model_size_mb_after:.2f} MB)")
    exp_logger.metrics['model_size_after_params'] = int(total_params_after)
    exp_logger.metrics['model_size_after_mb'] = float(model_size_mb_after)
    exp_logger.metrics['model_size_change_mb'] = float(model_size_mb_after - model_size_mb_before)

    train_acc_watermark = history.history["accuracy"]
    val_acc_watermark = history.history["val_accuracy"]
    train_loss_watermark = history.history["loss"]
    val_loss_watermark = history.history["val_loss"]
    
    # Learning rate might not be logged, so handle it gracefully
    if "lr" in history.history:
        lr = history.history["lr"]
    elif "learning_rate" in history.history:
        lr = history.history["learning_rate"]
    else:
        # If not logged, create a list with the initial learning rate
        print("⚠️  Warning: Learning rate not found in history, using initial LR")
        lr = [0.0001] * epochs  # Use the initial learning rate from the optimizer
    
    normal_test_acc = history.params["test_acc_"]
        
    loss_acc_dict["train_acc"] = train_acc_watermark
    loss_acc_dict["val_acc"] = val_acc_watermark
    loss_acc_dict["train_loss"] = train_loss_watermark
    loss_acc_dict["val_loss"] = val_loss_watermark
    loss_acc_dict["lr"] = lr    
    loss_acc_dict["normal_test_acc"] = normal_test_acc    

    with open(dict_file, 'w') as file:
        json.dump(loss_acc_dict, file, cls=NumpyEncoder)               

    # Log training history to ExperimentLogger
    for idx, (train_loss, train_acc, val_loss, val_acc, lr_val, normal_acc, epoch_time) in enumerate(
            zip(train_loss_watermark, train_acc_watermark, val_loss_watermark, val_acc_watermark, lr, normal_test_acc, time_callback.times)):
        file1.write(
            f'Epoch: {idx + 1}, Train Loss: {train_loss:.3f} Train Acc: {train_acc:.3f} Val Loss: {val_loss:.3f} Val Acc: {val_acc:.3f} with lr: {lr_val:.7f} and normal test acc: {normal_acc:.3f} and epoch time :{epoch_time} seconds \n')
        file1.write("\n")
        
        # Log to ExperimentLogger
        exp_logger.log_training_epoch(
            epoch=idx,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            learning_rate=lr_val,
            epoch_time=epoch_time,
            test_acc=normal_acc  # Additional metric
        )

    FINAL_CHECKPOINT_FILEPATH = os.path.join(MODEL_PATH, adv_data_path_numpy.replace("\\", "/").split("/")[-2],
                                             "final_" + dataset_name + "_" + str(epochs) + "_" + str(
                                                 epochs) + "_" + model_name +
                                             adv_data_path_numpy.replace("\\", "/").split("/")[-1].split(".npz")[0],
                                             "Victim_checkpoint_final.keras")

    # Ensure the directory exists before saving
    final_checkpoint_dir = os.path.dirname(FINAL_CHECKPOINT_FILEPATH)
    os.makedirs(final_checkpoint_dir, exist_ok=True)
    
    model.save(FINAL_CHECKPOINT_FILEPATH)

    model = load_model(CHECKPOINT_FILEPATH)

    acc_adv = model.evaluate(x_adv, y_adv)[1]
    print("Best after loading the model adv :", acc_adv)
    file1.write(f"Best after loading the model adv : {acc_adv}")
    exp_logger.metrics['watermark_acc_best_checkpoint'] = float(acc_adv)
    
    # Log final model metrics
    exp_logger.log_model_metrics(model=model, x_test=x_test, y_test=y_test, prefix="final_")
    
    # Save all ExperimentLogger data
    exp_logger.save_all()
    
    # Create LaTeX table from training history if available
    if exp_logger.training_history:
        latex_table = exp_logger.create_latex_table("watermarking_results.tex")
        if latex_table:
            print(f"✅ LaTeX table saved to: {exp_logger.experiment_dir / 'tables' / 'watermarking_results.tex'}")
    
    print(f"✅ Comprehensive experiment data saved to: {exp_logger.experiment_dir}")
    # mlflow.log_artifact(os.path.join(RESULTS_PATH, LOSS_FOLDER, adv_data_path_numpy.replace("\\", "/").split("/")[-2],
    #                                  dataset_name + "_" + str(epochs) + "_" + model_name +
    #                                  adv_data_path_numpy.replace("\\", "/").split("/")[-1].split(".npz")[
    #                                      0] + "_logs.txt"), "logs.txt")
    
    model = load_model(FINAL_CHECKPOINT_FILEPATH)

    acc_adv = model.evaluate(x_adv, y_adv)[1]
    print("\nLast after loading the model adv :", acc_adv)
    file1.write(f"Last after loading the model adv : {acc_adv}")
    # mlflow.log_artifact(os.path.join(RESULTS_PATH, LOSS_FOLDER, adv_data_path_numpy.replace("\\", "/").split("/")[-2],
    #                                  dataset_name + "_" + str(epochs) + "_" + model_name +
    #                                  adv_data_path_numpy.replace("\\", "/").split("/")[-1].split(".npz")[
    #                                      0] + "_logs.txt"), "logs.txt")

    ## --------------------------------- Plotting the graphs --------------------------------- ##
    plt.figure()
    plt.plot(list(range(epochs)), train_loss_watermark, label="Train loss", marker='o',
             color='tab:purple')
    plt.plot(list(range(epochs)), val_loss_watermark, label="Val loss", linestyle='--',
             marker='o', color='tab:orange')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_PATH, LOSS_FOLDER, adv_data_path_numpy.replace("\\", "/").split("/")[-2],
                             dataset_name + "_" + str(epochs) + "_" + model_name +
                             adv_data_path_numpy.replace("\\", "/").split("/")[-1].split(".npz")[
                                 0] + "WatermarkFinetuneLoss.png"))
    # mlflow.log_artifact(os.path.join(RESULTS_PATH, LOSS_FOLDER, adv_data_path_numpy.replace("\\", "/").split("/")[-2],
    #                                  dataset_name + "_" + str(epochs) + "_" + model_name +
    #                                  adv_data_path_numpy.replace("\\", "/").split("/")[-1].split(".npz")[
    #                                      0] + "WatermarkFinetuneLoss.png"),
    #                     "WatermarkFinetuneLoss.png")

    plt.figure()
    plt.plot(list(range(epochs)), train_acc_watermark, label="Train acc_" + dataset_name, marker='o',
             color='tab:purple')
    plt.plot(list(range(epochs)), val_acc_watermark, label="Val acc_" + dataset_name, linestyle='--',
             marker='o', color='tab:orange')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_PATH, LOSS_FOLDER, adv_data_path_numpy.replace("\\", "/").split("/")[-2],
                             dataset_name + "_" + str(epochs) + "_" + model_name +
                             adv_data_path_numpy.replace("\\", "/").split("/")[-1].split(".npz")[
                                 0] + "WatermarkFinetuneAcc.png"))
    # mlflow.log_artifact(os.path.join(RESULTS_PATH, LOSS_FOLDER, adv_data_path_numpy.replace("\\", "/").split("/")[-2],
    #                                  dataset_name + "_" + str(epochs) + "_" + model_name +
    #                                  adv_data_path_numpy.replace("\\", "/").split("/")[-1].split(".npz")[
    #                                      0] + "WatermarkFinetuneAcc.png"), "WatermarkFinetuneAcc.png")

    return model


if __name__ == "__main__":
    # Optimized configuration values for watermarking
    dataset_name = "cifar10"
    epochs = 15  # Increased from 10 for better watermark retention
    batch_size = 128
    model_to_finetune_path = "../models/original_09-11-2025/cifar10_30_CIFAR10_BASE_2/Original_checkpoint_best.keras"
    dropout = 0
    optimizer = "adam"
    lr = 0.0001  # Lower LR for fine-tuning
    weight_decay = 0
    num_layers_unfreeze = 1
    lr_schedule_enabled = True  # Enable learning rate decay
    lr_decay_factor = 0.5
    lr_decay_epochs = [8, 12]  # Decay at epochs 8 and 12
    
    # Performance optimizations
    use_mixed_precision = os.getenv('USE_MIXED_PRECISION', 'false').lower() == 'true'
    
    # Optimize GPU memory
    optimize_gpu_memory()
    
    # Enable mixed precision if requested
    if use_mixed_precision:
        enable_mixed_precision('mixed_float16')
    which_adv = "true"

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    tf.compat.v1.random.set_random_seed(seed)

    now = datetime.now().strftime("%d-%m-%Y")

    RESULTS_PATH = f"../results/finetuned_finetuning_{now}"
    DATA_PATH = "../data"
    MODEL_PATH = f"../models/finetuned_finetuning_{now}"
    LOSS_FOLDER = "losses"

    if not os.path.exists(os.path.join(RESULTS_PATH, LOSS_FOLDER, which_adv)):
        os.makedirs(os.path.join(RESULTS_PATH, LOSS_FOLDER, which_adv))

    if not os.path.exists(os.path.join(MODEL_PATH, which_adv)):
        os.makedirs(os.path.join(MODEL_PATH, which_adv))

    dataset_path = os.path.join(DATA_PATH, "fgsm", dataset_name, which_adv)
    experiment_name = dataset_name + "Frontier_Watermarking_Finetuning"

    # Check if adversarial examples directory exists
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Adversarial examples directory not found: {dataset_path}")
        print(f"\n📋 You need to run Step 2 first to generate adversarial examples:")
        print(f"   poetry run python frontier-stitching.py")
        print(f"\n   This will create the directory and generate adversarial examples.")
        sys.exit(1)

    params = {"dataset_name": dataset_name, "epochs": epochs,
                "optimizer": optimizer, "lr": lr,
                "weight_decay": weight_decay, "dropout": dropout,
                "num_layers_unfreeze": num_layers_unfreeze, }

    for adv_path in os.listdir(dataset_path):
        if ("fgsm_0.01_10000" in adv_path or "fgsm_0.015_10000" in adv_path) and "capacity" not in adv_path and "samples" not in adv_path:
            adv_data_path_numpy = os.path.join(dataset_path, adv_path)
            print(adv_data_path_numpy)

            watermark_finetuning(dataset_name, adv_data_path_numpy, model_to_finetune_path, epochs,
                                    dropout, batch_size, optimizer, lr,
                                    weight_decay, num_layers_unfreeze,
                                    lr_schedule_enabled=lr_schedule_enabled,
                                    lr_decay_factor=lr_decay_factor,
                                    lr_decay_epochs=lr_decay_epochs)













##----------------- Unused blocks --------------------------------
# idx = np.random.randint(x_train_adv.shape[0], size=x_train_adv.shape[0])
    # x_train_adv_1 = x_train_adv[idx[:int(len(idx) * 0.8)]]
    # y_train_adv_1 = y_train_adv[idx[:int(len(idx) * 0.8)]]

    # x_val_adv = x_train_adv[idx[int(len(idx) * 0.8):]]
    # y_val_adv = y_train_adv[idx[int(len(idx) * 0.8):]]

    # datagen = ImageDataGenerator(horizontal_flip=True,rotation_range=15,
    #         width_shift_range=0.125,height_shift_range=0.125,fill_mode='reflect')

    # history = model.fit_generator(datagen.flow(x_train_adv_1, y_train_adv_1, batch_size=batch_size), shuffle=True, epochs=epochs,
    #                     verbose=1, validation_data=(x_val_adv, y_val_adv),
    #                     callbacks=[model_checkpoint_callback, lrate, normal_data_acc_callback((x_test, y_test))])  ## check for validation
