"""
@author: Ridhima Garg

Introduction:
    This is the main file for performing the watermarking using frontoer stitching method.

"""

import os
import warnings

warnings.filterwarnings('ignore')
from keras.models import load_model
import tensorflow as tf

from art import config
from art.utils import load_dataset, get_file

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import random
import keras
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras import layers
from keras import backend as K
import argparse
import mlconfig
import models
# import mlflow
from datetime import datetime
from tensorflow.keras.callbacks import Callback
import os
import json
import time


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def data_preprocessing(dataset_name, adv_data_path_numpy):
    if dataset_name == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        img_rows, img_cols, num_channels = 28, 28, 1
        num_classes = 10

    elif dataset_name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        img_rows, img_cols, num_channels = 32, 32, 3
        num_classes = 10

    else:
        raise ValueError('Invalid dataset name')

    idx = np.random.randint(x_train.shape[0], size=len(x_train))
    x_train = x_train[idx, :]
    y_train = y_train[idx]

    # specify input dimensions of each image
    input_shape = (img_rows, img_cols, num_channels)

    # reshape x_train and x_test
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, num_channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, num_channels)

    # convert class labels (from digits) to one-hot encoded vectors
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # convert int to float
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # normalise
    x_train /= 255
    x_test /= 255

    # Load adv dataset
    adv = np.load(adv_data_path_numpy)
    x_adv, y_adv = adv['arr_1'], adv['arr_2']
    print(x_adv.shape)
    print(y_adv.shape)

    idx = np.random.randint(x_adv.shape[0], size=x_adv.shape[0])
    x_train_adv = x_adv[idx[:int(len(idx) * 0.9)]]
    y_train_adv = y_adv[idx[:int(len(idx) * 0.9)]]
    x_test_adv = x_adv[idx[int(len(idx) * 0.9):]]
    y_test_adv = y_adv[idx[int(len(idx) * 0.9):]]

    x_combined_train = np.concatenate((x_train[: int(len(x_train) * 0.9)], x_train_adv[: int(len(x_train_adv) * 0.9)]),
                                      axis=0)
    y_combined_train = np.concatenate((y_train[: int(len(y_train) * 0.9)], y_train_adv[: int(len(y_train_adv) * 0.9)]),
                                      axis=0)

    x_combined_val = np.concatenate((x_train[int(len(x_train) * 0.9):], x_train_adv[int(len(x_train_adv) * 0.9):]),
                                    axis=0)
    y_combined_val = np.concatenate((y_train[int(len(y_train) * 0.9):], y_train_adv[int(len(y_train_adv) * 0.9):]),
                                    axis=0)

    x_combined = np.concatenate((x_train, x_train_adv), axis=0)
    y_combined = np.concatenate((y_train, y_train_adv), axis=0)

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

class normal_data_acc_callback(keras.callbacks.Callback):

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
    

class adv_data_acc_callback(keras.callbacks.Callback):

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
                         lr, weight_decay, save_dir_name='../models/cnn_finetuned.h5'):
    x_train, y_train, x_test, y_test, x_adv, y_adv, x_train_adv, y_train_adv, x_test_adv, y_test_adv, x_combined, y_combined, x_combined_train, y_combined_train, x_combined_val, y_combined_val, input_shape, num_classes = data_preprocessing(
        dataset_name, adv_data_path_numpy)

    models_mapping = {"mnist_l2": models.MNIST_L2, "cifar10_base_2": models.CIFAR10_BASE_2}

    if dropout:
        model_name, model = models_mapping[model_architecture](input_shape, dropout)
    else:
        model_name, model = models_mapping[model_architecture]()

    if optimizer == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr, decay=config.weight_decay)
    else:
        optimizer = None
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

    CHECKPOINT_FILEPATH = os.path.join(MODEL_PATH, adv_data_path_numpy.replace("\\", "/").split("/")[-2],
                                       dataset_name + "_" + str(epochs) + "_" + model_name +
                                       adv_data_path_numpy.replace("\\", "/").split("/")[-1].split(".npz")[0],
                                       "Victim_checkpoint_best.h5")

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
                                             "Victim_checkpoint_final.h5")

    FINAL_CHECKPOINT_FILEPATH = os.path.join(MODEL_PATH, adv_data_path_numpy.replace("\\", "/").split("/")[-2] , "final_" + dataset_name + "_" + str(epochs) + "_" + model_name + adv_data_path_numpy.replace("\\", "/").split("/")[-1].split(".npz")[0],
                                       "Victim_checkpoint_final.h5")


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
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="configs/watermarking_retraining.yaml")
    args = parser.parse_args()
    print(args)
    config = mlconfig.load(args.config)

    dataset_name = config.dataset_name

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    tf.compat.v1.random.set_random_seed(seed)

    # mlflow.set_tracking_uri("sqlite:///../mlflow.db")
    # mlflow.set_tracking_uri("file:///../mlruns")
    # mlflow.set_experiment("frontier-stiching-watermarking")

    now = datetime.now().strftime("%d-%m-%Y")

    RESULTS_PATH = f"../results/finetuned_retraining_{now}"
    DATA_PATH = "../data"
    MODEL_PATH = f"../models/finetuned_retraining_{now}"
    LOSS_FOLDER = "losses"

    if not os.path.exists(os.path.join(RESULTS_PATH, LOSS_FOLDER, config.which_adv)):
        os.makedirs(os.path.join(RESULTS_PATH, LOSS_FOLDER, config.which_adv))

    if not os.path.exists(os.path.join(MODEL_PATH, config.which_adv)):
        os.makedirs(os.path.join(MODEL_PATH, config.which_adv))

    dataset_path = os.path.join(DATA_PATH, "fgsm", dataset_name, config.which_adv)

    experiment_name = dataset_name + "Frontier_Watermarking_Retraining"
    # with mlflow.start_run(run_name=experiment_name):

    params = {"dataset_name": dataset_name, "epochs": config.epochs,
                "model_architecture": config.model_architecture, "optimizer": config.optimizer, "lr": config.lr,
                "weight_decay": config.weight_decay, "dropout": config.dropout}

    for adv_path in os.listdir(dataset_path):
        if "fgsm_0.25_250" in adv_path:
            adv_data_path_numpy = os.path.join(dataset_path, adv_path)
            print(adv_data_path_numpy)
            # model_save_dir_name = os.path.join(MODEL_PATH, dataset_name + str(epochs) + "_" + adv_data_path_numpy.split("\\")[-1].rsplit(".", 1)[0] + ".h5")

            watermark_retraining(dataset_name, adv_data_path_numpy, config.model_architecture, config.epochs,
                                    config.dropout, config.batch_size, config.optimizer, config.lr,
                                    config.weight_decay)






