"""
@author: Ridhima Garg

Introduction:
    This is the main file for performing the watermarking using frontoer stitching method.

"""

import os
import warnings
import argparse
import json
import time
import random
from datetime import datetime
from typing import Tuple, List, Dict, Any

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
import models
from utils.data_utils import DataManager

warnings.filterwarnings('ignore')

# GPU Configuration - uncomment to disable GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Configure GPU (Metal on macOS, CUDA on Linux/Windows)
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    try:
        for gpu in physical_devices:
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
    Preprocess data for watermarking with frontier stitching (retraining).
    
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

    # Check if adversarial data is empty or has wrong shape
    if len(x_adv.shape) == 1 or (len(x_adv.shape) > 1 and x_adv.shape[0] == 0):
        raise ValueError(
            f"Adversarial data is empty! Shape: {x_adv.shape}. "
            f"This usually means no adversarial examples were generated or the file is corrupted. "
            f"Please check:\n"
            f"  1. The adversarial generation step completed successfully\n"
            f"  2. The file path is correct: {adv_data_path_numpy}\n"
            f"  3. Try using a larger epsilon value or different 'which_adv' setting (true/false/full)\n"
            f"  4. For small epsilon values, most examples may remain correctly classified, "
            f"   resulting in empty 'true' or 'false' adversary sets"
        )

    # Ensure x_adv has the correct shape (should be 4D: batch, height, width, channels)
    if len(x_adv.shape) != 4:
        raise ValueError(
            f"Adversarial data has incorrect shape: {x_adv.shape}. "
            f"Expected 4D array (batch, height, width, channels), got {len(x_adv.shape)}D. "
            f"This may indicate the adversarial data file is corrupted or in the wrong format."
        )

    # Split adversarial data into train/test (90/10 split)
    if x_adv.shape[0] > 0:
        idx = np.random.permutation(x_adv.shape[0])
        train_size = int(len(idx) * 0.9)
        x_train_adv = x_adv[idx[:train_size]]
        y_train_adv = y_adv[idx[:train_size]]
        x_test_adv = x_adv[idx[train_size:]]
        y_test_adv = y_adv[idx[train_size:]]
    else:
        # Fallback: create empty arrays with correct shape
        x_train_adv = np.empty((0,) + x_adv.shape[1:], dtype=x_adv.dtype)
        y_train_adv = np.empty((0,) + y_adv.shape[1:], dtype=y_adv.dtype)
        x_test_adv = np.empty((0,) + x_adv.shape[1:], dtype=x_adv.dtype)
        y_test_adv = np.empty((0,) + y_adv.shape[1:], dtype=y_adv.dtype)

    # Create combined train/val splits (handle empty arrays)
    train_split_size = int(len(x_train) * 0.9)
    adv_train_split_size = int(len(x_train_adv) * 0.9) if len(x_train_adv) > 0 else 0
    
    if adv_train_split_size > 0:
        x_combined_train = np.concatenate(
            (x_train[:train_split_size], x_train_adv[:adv_train_split_size]),
            axis=0)
        y_combined_train = np.concatenate(
            (y_train[:train_split_size], y_train_adv[:adv_train_split_size]),
            axis=0)
        
        x_combined_val = np.concatenate(
            (x_train[train_split_size:], x_train_adv[adv_train_split_size:]),
            axis=0)
        y_combined_val = np.concatenate(
            (y_train[train_split_size:], y_train_adv[adv_train_split_size:]),
            axis=0)
    else:
        # If no adversarial data, use only regular training data
        x_combined_train = x_train[:train_split_size]
        y_combined_train = y_train[:train_split_size]
        x_combined_val = x_train[train_split_size:]
        y_combined_val = y_train[train_split_size:]

    # Combine all data
    if len(x_train_adv) > 0:
        x_combined = np.concatenate((x_train, x_train_adv), axis=0)
        y_combined = np.concatenate((y_train, y_train_adv), axis=0)
    else:
        x_combined = x_train
        y_combined = y_train

    return x_train, y_train, x_test, y_test, x_adv, y_adv, x_train_adv, y_train_adv, x_test_adv, y_test_adv, x_combined, y_combined, x_combined_train, y_combined_train, x_combined_val, y_combined_val, input_shape, num_classes

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
    

class adv_data_acc_callback(tf.keras.callbacks.Callback):

    def __init__(self, test_data):
        self.x_test = test_data[0]
        self.y_test = test_data[1]

    def on_train_begin(self, logs={}):
        self.params["adv_acc_"] = []

    def on_epoch_end(self, epoch, logs={}):
        acc = self.model.evaluate(self.x_test, self.y_test)
        print("adv accuracy:", acc[1])
        self.params["adv_acc_"].append(acc[1])
        return acc[1]


def watermark_retraining(dataset_name, adv_data_path_numpy, model_architecture, epochs, dropout, batch_size, optimizer,
                         lr, weight_decay, save_dir_name='../models/cnn_finetuned.keras'):
    x_train, y_train, x_test, y_test, x_adv, y_adv, x_train_adv, y_train_adv, x_test_adv, y_test_adv, x_combined, y_combined, x_combined_train, y_combined_train, x_combined_val, y_combined_val, input_shape, num_classes = data_preprocessing(
        dataset_name, adv_data_path_numpy)

    models_mapping = {"mnist_l2": models.MNIST_L2, "cifar10_base_2": models.CIFAR10_BASE_2}

    if dropout:
        model_name, model = models_mapping[model_architecture](input_shape, dropout)
    else:
        model_name, model = models_mapping[model_architecture]()

    if optimizer == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, weight_decay=weight_decay)
    else:
        optimizer = None
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=optimizer, metrics=['accuracy'])

    CHECKPOINT_FILEPATH = os.path.join(MODEL_PATH, adv_data_path_numpy.replace("\\", "/").split("/")[-2],
                                       dataset_name + "_" + str(epochs) + "_" + model_name +
                                       adv_data_path_numpy.replace("\\", "/").split("/")[-1].split(".npz")[0],
                                       "Victim_checkpoint_best.keras")

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_FILEPATH,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max')

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=5,
        verbose=1,
        mode="min",
        # baseline=None,
        # restore_best_weights=False,
        # start_from_epoch=0,
    )

    loss_acc_dict = {}

    time_callback = TimeHistory()
    normal_data_acc = normal_data_acc_callback((x_test, y_test))
    adv_data_acc = adv_data_acc_callback((x_adv, y_adv))

    history = model.fit(x_combined_train, y_combined_train, batch_size=batch_size, shuffle=True, epochs=epochs,
                        verbose=1, validation_data=(x_combined_val, y_combined_val),
                        callbacks=[model_checkpoint_callback, early_stopping_callback, normal_data_acc, adv_data_acc, time_callback])  ## check for validation

    file1 = open(os.path.join(RESULTS_PATH, LOSS_FOLDER, adv_data_path_numpy.replace("\\", "/").split("/")[-2],
                              dataset_name + "_" + model_name +
                              adv_data_path_numpy.replace("\\", "/").split("/")[-1].split(".npz")[0] + "_logs.txt"),
                 "w")
    
    dict_file = os.path.join(RESULTS_PATH, LOSS_FOLDER, adv_data_path_numpy.replace("\\", "/").split("/")[-2],
                              dataset_name + "_" + str(epochs) + "_" + model_name +
                              adv_data_path_numpy.replace("\\", "/").split("/")[-1].split(".npz")[0] + "_acc_loss.json")
    
    print(f"Total training time: {time_callback.total_time} seconds")
    file1.write(f"Total training time: {time_callback.total_time} seconds\n")

    print(f"Finetuned model test acc: {model.evaluate(x_test, y_test)}")
    file1.write(f"Finetuned model test acc: {model.evaluate(x_test, y_test)} \n")
    print(f"Finetuned model train adv acc: {model.evaluate(x_train_adv, y_train_adv)}")
    file1.write(f"Finetuned model train adv acc: {model.evaluate(x_train_adv, y_train_adv)}\n")
    print(f"Finetuned model test adv acc: {model.evaluate(x_test_adv, y_test_adv)}")
    file1.write(f"Finetuned model test adv acc: {model.evaluate(x_test_adv, y_test_adv)}\n")

    train_acc_watermark = history.history["accuracy"]
    val_acc_watermark = history.history["val_accuracy"]
    train_loss_watermark = history.history["loss"]
    val_loss_watermark = history.history["val_loss"]
    normal_test_acc = history.params["test_acc_"]
    adv_test_acc = history.params["adv_acc_"]

    loss_acc_dict["train_acc"] = train_acc_watermark
    loss_acc_dict["val_acc"] = val_acc_watermark
    loss_acc_dict["train_loss"] = train_loss_watermark
    loss_acc_dict["val_loss"] = val_loss_watermark
    loss_acc_dict["lr"] = lr    
    loss_acc_dict["normal_test_acc"] = normal_test_acc    
    loss_acc_dict["adv_test_acc"] = adv_test_acc 

    with open(dict_file, 'w') as file:
        json.dump(loss_acc_dict, file, cls=NumpyEncoder) 

    for idx, (train_loss, train_acc, val_loss, val_acc, normal_acc, adv_acc, epoch_time) in enumerate(
            zip(train_loss_watermark, train_acc_watermark, val_loss_watermark, val_acc_watermark, normal_test_acc, adv_test_acc, time_callback.times)):
        file1.write(
            f'Epoch: {idx + 1}, Train Loss: {train_loss:.3f} Train Acc: {train_acc:.3f} Val Loss: {val_loss:.3f} Val Acc: {val_acc:.3f} and normal test acc: {normal_acc:.3f} and adv acc: {adv_acc:.3f} and epoch time :{epoch_time} seconds \n')
        file1.write("\n")


    FINAL_CHECKPOINT_FILEPATH = os.path.join(MODEL_PATH, adv_data_path_numpy.replace("\\", "/").split("/")[-2],
                                             "final_" + dataset_name + "_" + str(epochs) + "_" + model_name +
                                             adv_data_path_numpy.replace("\\", "/").split("/")[-1].split(".npz")[0],
                                             "Victim_checkpoint_final.keras")

    # Create directory if it doesn't exist
    final_checkpoint_dir = os.path.dirname(FINAL_CHECKPOINT_FILEPATH)
    os.makedirs(final_checkpoint_dir, exist_ok=True)

    model.save(FINAL_CHECKPOINT_FILEPATH)

    model = load_model(CHECKPOINT_FILEPATH)
    # adv = np.load(adv_data_path_numpy)
    # x_adv, y_adv = adv['arr_1'], adv['arr_1']

    acc_adv = model.evaluate(x_adv, y_adv)[1]
    print("Best after loading the model adv :", acc_adv)
    file1.write(f"Best after loading the model adv : {acc_adv}")

    # mlflow.log_artifact(os.path.join(RESULTS_PATH, LOSS_FOLDER, adv_data_path_numpy.replace("\\", "/").split("/")[-2],
    #                                  dataset_name + "_" + model_name +
    #                                  adv_data_path_numpy.replace("\\", "/").split("/")[-1].split(".npz")[
    #                                      0] + "_logs.txt"), "logs.txt")

    model = load_model(FINAL_CHECKPOINT_FILEPATH)

    acc_adv = model.evaluate(x_adv, y_adv)[1]
    print("\nLast after loading the model adv :", acc_adv)
    file1.write(f"Last after loading the model adv : {acc_adv}")
    # mlflow.log_artifact(os.path.join(RESULTS_PATH, LOSS_FOLDER, adv_data_path_numpy.replace("\\", "/").split("/")[-2], dataset_name + "_" + model_name +
    #                           adv_data_path_numpy.replace("\\", "/").split("/")[-1].split(".npz")[0] + "_logs.txt"), "logs.txt")

    model = load_model(FINAL_CHECKPOINT_FILEPATH)

    acc_adv = model.evaluate(x_adv, y_adv)[1]
    print("\nLast after loading the model adv :", acc_adv)
    file1.write(f"Last after loading the model adv : {acc_adv}")

    ## --------------------------------- Plotting the graphs --------------------------------- ##
    plt.figure()
    plt.plot(list(range(len(train_loss_watermark))), train_loss_watermark, label="Train loss_" + dataset_name,
             marker='o',
             color='tab:purple')
    plt.plot(list(range(len(val_loss_watermark))), val_loss_watermark, label="Val loss_" + dataset_name, linestyle='--',
             marker='o', color='tab:orange')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_PATH, LOSS_FOLDER, adv_data_path_numpy.replace("\\", "/").split("/")[-2],
                             dataset_name + "_" + model_name +
                             adv_data_path_numpy.replace("\\", "/").split("/")[-1].split(".npz")[
                                 0] + "WatermarkRetrainLoss.png"))
    # mlflow.log_artifact(os.path.join(RESULTS_PATH, LOSS_FOLDER, adv_data_path_numpy.replace("\\", "/").split("/")[-2],
    #                                  dataset_name + "_" + model_name +
    #                                  adv_data_path_numpy.replace("\\", "/").split("/")[-1].split(".npz")[
    #                                      0] + "WatermarkRetrainLoss.png"),
    #                     "WatermarkRetrainLoss.png")

    plt.figure()
    plt.plot(list(range(len(train_acc_watermark))), train_acc_watermark, label="Train acc_" + dataset_name, marker='o',
             color='tab:purple')
    plt.plot(list(range(len(val_acc_watermark))), val_acc_watermark, label="Val acc_" + dataset_name, linestyle='--',
             marker='o', color='tab:orange')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_PATH, LOSS_FOLDER, adv_data_path_numpy.replace("\\", "/").split("/")[-2],
                             dataset_name + "_" + model_name +
                             adv_data_path_numpy.replace("\\", "/").split("/")[-1].split(".npz")[
                                 0] + "WatermarkRetrainAcc.png"))
    # mlflow.log_artifact(os.path.join(RESULTS_PATH, LOSS_FOLDER, adv_data_path_numpy.replace("\\", "/").split("/")[-2],
    #                                  dataset_name + "_" + model_name +
    #                                  adv_data_path_numpy.replace("\\", "/").split("/")[-1].split(".npz")[
    #                                      0] + "WatermarkRetrainAcc.png"), "WatermarkRetrainAcc.png")

    return model


if __name__ == "__main__":
    # Configuration values (previously from configs/watermarking_retraining.yaml)
    dataset_name = "mnist"
    epochs = 100
    batch_size = 128
    model_architecture = "mnist_l2"
    dropout = 0
    optimizer = "adam"
    lr = 0.001
    weight_decay = 0
    which_adv = "full"

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    tf.compat.v1.random.set_random_seed(seed)

    now = datetime.now().strftime("%d-%m-%Y")

    RESULTS_PATH = f"../results/finetuned_retraining_{now}"
    DATA_PATH = "../data"
    MODEL_PATH = f"../models/finetuned_retraining_{now}"
    LOSS_FOLDER = "losses"

    if not os.path.exists(os.path.join(RESULTS_PATH, LOSS_FOLDER, which_adv)):
        os.makedirs(os.path.join(RESULTS_PATH, LOSS_FOLDER, which_adv))

    if not os.path.exists(os.path.join(MODEL_PATH, which_adv)):
        os.makedirs(os.path.join(MODEL_PATH, which_adv))

    dataset_path = os.path.join(DATA_PATH, "fgsm", dataset_name, which_adv)

    experiment_name = dataset_name + "Frontier_Watermarking_Retraining"

    params = {"dataset_name": dataset_name, "epochs": epochs,
                "model_architecture": model_architecture, "optimizer": optimizer, "lr": lr,
                "weight_decay": weight_decay, "dropout": dropout}

    for adv_path in os.listdir(dataset_path):
        if "fgsm_0.25_250" in adv_path:
            adv_data_path_numpy = os.path.join(dataset_path, adv_path)
            print(adv_data_path_numpy)

            watermark_retraining(dataset_name, adv_data_path_numpy, model_architecture, epochs,
                                    dropout, batch_size, optimizer, lr,
                                    weight_decay)






