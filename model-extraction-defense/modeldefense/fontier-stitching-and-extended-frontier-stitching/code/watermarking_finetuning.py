"""
@author: Ridhima Garg

Introduction:
    This is the main file for performing the watermarking using extended frontoer stitching method.

"""

import warnings

warnings.filterwarnings('ignore')
from keras.models import load_model
import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np
import random
import keras
from keras.datasets import mnist, cifar10
import argparse
import mlconfig
import models
# import mlflow
from datetime import datetime
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback
import math
import json
import time


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def data_preprocessing(dataset_name, adv_data_path_numpy):
    """
            Main idea
            -------
            This is the function which preprocess the data for the modelling.

            Args:
            ---
            dataset_name: name of the dataset.

            Future work:
            -----------
            This function for now is copied in multiple files but can be modified such that it can be exported.
            It can be optimized more easily.

    """
    if dataset_name == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        img_rows, img_cols, num_channels = 28, 28, 1
        num_classes = 10

    elif dataset_name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        img_rows, img_cols, num_channels = 32, 32, 3
        num_classes = 10

    elif dataset_name == "cifar10resnet":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        img_rows, img_cols, num_channels = 32, 32, 3
        num_classes = 10

    elif dataset_name == "cifar10resnet_255_preprocess":
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
    if dataset_name != 'cifar10resnet':
        x_train /= 255
        x_test /= 255

    else:

        mean = [125.3, 123.0, 113.9]
        std = [63.0, 62.1, 66.7]

        for i in range(3):
            x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
            x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]

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

    x_train_selected = x_train[:int(len(idx) * 0.9)]
    y_train_selected = y_train[:int(len(idx) * 0.9)]

    x_combined = np.concatenate((x_train_selected, x_train_adv), axis=0)
    y_combined = np.concatenate((y_train_selected, y_train_adv), axis=0)

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


class normal_data_acc_callback(keras.callbacks.Callback):
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
                         optimizer, lr, weight_decay, num_layers_unfreeze):

    """
    Main idea
    --------
    Performing the watermarking on the already trained model.
    It is a kind of advesarial finetuning, whose idea is motivated from the paper: https://arxiv.org/abs/2012.13628

    """


    x_train, y_train, x_test, y_test, x_adv, y_adv, x_train_adv, y_train_adv, x_test_adv, y_test_adv, x_combined, y_combined, x_combined_train, y_combined_train, x_combined_val, y_combined_val, input_shape, num_classes = data_preprocessing(
        dataset_name, adv_data_path_numpy)

    print(model_to_finetune_path)


    model = load_model(model_to_finetune_path, compile=False)

    print(model.summary())

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

    lrate = LearningRateScheduler(learning_rate_schedule)
    # lrate = LearningRateScheduler(scheduler)
    opt = Adam(learning_rate=0.0001)

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt, metrics=['accuracy'])

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
                                       "Victim_checkpoint_best.h5")

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

    ## evaluating over the test adversary data
    acc = model.evaluate(x_test_adv, y_test_adv)
    print(f"Finetuned model test adv acc: {acc}")
    file1.write(f"Finetuned model test adv acc: {acc}\n")

    train_acc_watermark = history.history["accuracy"]
    val_acc_watermark = history.history["val_accuracy"]
    train_loss_watermark = history.history["loss"]
    val_loss_watermark = history.history["val_loss"]
    lr = history.history["lr"]
    normal_test_acc = history.params["test_acc_"]
        
    loss_acc_dict["train_acc"] = train_acc_watermark
    loss_acc_dict["val_acc"] = val_acc_watermark
    loss_acc_dict["train_loss"] = train_loss_watermark
    loss_acc_dict["val_loss"] = val_loss_watermark
    loss_acc_dict["lr"] = lr    
    loss_acc_dict["normal_test_acc"] = normal_test_acc    

    with open(dict_file, 'w') as file:
        json.dump(loss_acc_dict, file, cls=NumpyEncoder)               

    for idx, (train_loss, train_acc, val_loss, val_acc, lr, normal_acc, epoch_time) in enumerate(
            zip(train_loss_watermark, train_acc_watermark, val_loss_watermark, val_acc_watermark, lr, normal_test_acc, time_callback.times)):
        file1.write(
            f'Epoch: {idx + 1}, Train Loss: {train_loss:.3f} Train Acc: {train_acc:.3f} Val Loss: {val_loss:.3f} Val Acc: {val_acc:.3f} with lr: {lr:.7f} and normal test acc: {normal_acc:.3f} and epoch time :{epoch_time} seconds \n')
        file1.write("\n")

    FINAL_CHECKPOINT_FILEPATH = os.path.join(MODEL_PATH, adv_data_path_numpy.replace("\\", "/").split("/")[-2],
                                             "final_" + dataset_name + "_" + str(epochs) + "_" + str(
                                                 epochs) + "_" + model_name +
                                             adv_data_path_numpy.replace("\\", "/").split("/")[-1].split(".npz")[0],
                                             "Victim_checkpoint_final.h5")

    model.save(FINAL_CHECKPOINT_FILEPATH)

    model = load_model(CHECKPOINT_FILEPATH)

    acc_adv = model.evaluate(x_adv, y_adv)[1]
    print("Best after loading the model adv :", acc_adv)
    file1.write(f"Best after loading the model adv : {acc_adv}")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="configs/watermarking_finetuning.yaml")
    args = parser.parse_args()
    print(args)
    config = mlconfig.load(args.config)

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    tf.compat.v1.random.set_random_seed(seed)

    # mlflow.set_tracking_uri("sqlite:///../mlflow.db")
    # mlflow.set_tracking_uri("file:///../mlruns")
    # mlflow.set_experiment("frontier-stiching-watermarking")

    now = datetime.now().strftime("%d-%m-%Y")

    RESULTS_PATH = f"../results/finetuned_finetuning_{now}"
    DATA_PATH = "../data"
    MODEL_PATH = f"../models/finetuned_finetuning_{now}"
    LOSS_FOLDER = "losses"

    if not os.path.exists(os.path.join(RESULTS_PATH, LOSS_FOLDER, config.which_adv)):
        os.makedirs(os.path.join(RESULTS_PATH, LOSS_FOLDER, config.which_adv))

    if not os.path.exists(os.path.join(MODEL_PATH, config.which_adv)):
        os.makedirs(os.path.join(MODEL_PATH, config.which_adv))

    dataset_name = config.dataset_name
    dataset_path = os.path.join(DATA_PATH, "fgsm", dataset_name, config.which_adv)
    experiment_name = dataset_name + "Frontier_Watermarking_Finetuning"

    # with mlflow.start_run(run_name=experiment_name):

    params = {"dataset_name": dataset_name, "epochs": config.epochs,
                "optimizer": config.optimizer, "lr": config.lr,
                "weight_decay": config.weight_decay, "dropout": config.dropout,
                "num_layers_unfreeze": config.num_layers_unfreeze, }

    for adv_path in os.listdir(dataset_path):
        if "fgsm_0.025_10000" in adv_path and "capacity" not in adv_path and "samples" not in adv_path:
            adv_data_path_numpy = os.path.join(dataset_path, adv_path)
            print(adv_data_path_numpy)
            num_layers_unfreeze = config.num_layers_unfreeze

            watermark_finetuning(dataset_name, adv_data_path_numpy, config.model_to_finetune_path, config.epochs,
                                    config.dropout, config.batch_size, config.optimizer, config.lr,
                                    config.weight_decay, num_layers_unfreeze)













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
