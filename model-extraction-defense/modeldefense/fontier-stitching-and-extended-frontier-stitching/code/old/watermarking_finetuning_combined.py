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
import mlflow
from datetime import datetime
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
import math

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


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


    idx_1 = np.random.randint(x_adv.shape[0]//2, size=x_adv.shape[0]//2)

    x_train_selected = x_train[:int(len(idx_1) * 0.9)]
    y_train_selected = y_train[:int(len(idx_1) * 0.9)]

    x_combined = np.concatenate((x_train_selected, x_train_adv), axis=0)
    y_combined = np.concatenate((y_train_selected, y_train_adv), axis=0)

    print(x_combined.shape)
    print(y_combined.shape)

    x_combined_train = np.concatenate((x_train_selected[: int(len(x_train_selected) * 0.9)], x_train_adv[: int(len(x_train_adv) * 0.9)]),
                                      axis=0)
    y_combined_train = np.concatenate((y_train_selected[: int(len(y_train_selected) * 0.9)], y_train_adv[: int(len(y_train_adv) * 0.9)]),
                                      axis=0)

    x_combined_val = np.concatenate((x_train_selected[int(len(x_train_selected) * 0.9):], x_train_adv[int(len(x_train_adv) * 0.9):]),
                                    axis=0)
    y_combined_val = np.concatenate((y_train_selected[int(len(y_train_selected) * 0.9):], y_train_adv[int(len(y_train_adv) * 0.9):]),
                                    axis=0)

    return x_train, y_train, x_test, y_test, x_adv, y_adv, x_train_adv, y_train_adv, x_test_adv, y_test_adv ,x_combined, y_combined, x_combined_train, y_combined_train, x_combined_val, y_combined_val, input_shape, num_classes

def learning_rate_schedule(epoch):
    return epoch * 0.0001
    if epoch <= 30:
        return epoch * 0.0001
    if epoch > 30 and epoch <= 50:
        print("epoch", epoch)
        print("decay", math.pow(2, (epoch - 5)))
        return 0.0030/math.pow(2, (epoch - 5))


def watermark_finetuning(dataset_name, adv_data_path_numpy, model_to_finetune_path, epochs, dropout, batch_size, optimizer, lr, weight_decay, num_layers_unfreeze):

    x_train, y_train, x_test, y_test, x_adv, y_adv, x_train_adv, y_train_adv, x_test_adv, y_test_adv ,x_combined, y_combined, x_combined_train, y_combined_train, x_combined_val, y_combined_val, input_shape, num_classes = data_preprocessing(dataset_name, adv_data_path_numpy)


    model = load_model(model_to_finetune_path, compile=False)

    print(model.summary())

    # model.trainable = False
    #for layer in model.layers:
         #if layer.name not in ["dense_1"]:
        #if layer.name not in ["dense_1", "dropout_3", "dense", "batch_normalization_6", "conv2d_5"]: #"conv2d_5", "batch_normalization_5", "dropout_2", "flatten", "max_pooling2d_2"
                            #   or layer.name == "dense" or layer.name == "batch_normalization_6" or layer.name == ""
            #layer.trainable = False

    #for layer in model.layers:
    #    print(layer.name, layer.trainable)

    lrate = LearningRateScheduler(learning_rate_schedule)
    opt = Adam(learning_rate=0.0)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt, metrics=['accuracy'])
    model_name = model_to_finetune_path.split("_")[-1].split(".")[0] ## this name is not correct.

    file1 = open(os.path.join(RESULTS_PATH, LOSS_FOLDER, adv_data_path_numpy.replace("\\", "/").split("/")[-2],
                              dataset_name + "_" + str(epochs) + "_" + model_name +
                              adv_data_path_numpy.replace("\\", "/").split("/")[-1].split(".npz")[0] + "_logs.txt"),
                 "w")

    CHECKPOINT_FILEPATH = os.path.join(MODEL_PATH, adv_data_path_numpy.replace("\\", "/").split("/")[-2],
                                       dataset_name + "_" + str(epochs) + "_" + str(epochs) + "_" + model_name +
                                       adv_data_path_numpy.replace("\\", "/").split("/")[-1].split(".npz")[0],
                                       "Victim_checkpoint_best.h5")

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_FILEPATH,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max',
        #save_weights_only=False
    )
    
    print(f"Loaded model test acc: {model.evaluate(x_test, y_test)}")
    file1.write(f"Loaded model test acc: {model.evaluate(x_test, y_test)} \n")

    history = model.fit(x_combined_train, y_combined_train, shuffle=True, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_combined_val, y_combined_val), callbacks = [model_checkpoint_callback, lrate]) ## check for validation


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

    for idx, (train_loss, train_acc, val_loss, val_acc) in enumerate(
            zip(train_loss_watermark, train_acc_watermark, val_loss_watermark, val_acc_watermark)):
        file1.write(
            f'Epoch: {idx + 1}, Train Loss: {train_loss:.3f} Train Acc: {train_acc:.3f} Val Loss: {val_loss:.3f} Val Acc: {val_acc:.3f}')
        file1.write("\n")

    model = load_model(CHECKPOINT_FILEPATH)

    acc_adv = model.evaluate(x_adv, y_adv)[1]
    print("After loading the model adv :", acc_adv)
    file1.write(f"After loading the model adv : {acc_adv}")
    mlflow.log_artifact(os.path.join(RESULTS_PATH, LOSS_FOLDER, adv_data_path_numpy.replace("\\", "/").split("/")[-2],
                                     dataset_name + "_" + str(epochs) + "_" + model_name +
                                     adv_data_path_numpy.replace("\\", "/").split("/")[-1].split(".npz")[
                                         0] + "_logs.txt"), "logs.txt")

    ## --------------------------------- Plotting the graphs --------------------------------- ##
    plt.figure()
    plt.plot(list(range(epochs)), train_loss_watermark, label="Train loss_" + dataset_name, marker='o',
             color='tab:purple')
    plt.plot(list(range(epochs)), val_loss_watermark, label="Val loss_" + dataset_name, linestyle='--',
             marker='o', color='tab:orange')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_PATH, LOSS_FOLDER, adv_data_path_numpy.replace("\\", "/").split("/")[-2],
                             dataset_name + "_" + str(epochs) + "_" + model_name +
                             adv_data_path_numpy.replace("\\", "/").split("/")[-1].split(".npz")[
                                 0] + "WatermarkFinetuneLoss.png"))
    mlflow.log_artifact(os.path.join(RESULTS_PATH, LOSS_FOLDER, adv_data_path_numpy.replace("\\", "/").split("/")[-2],
                                     dataset_name + "_" + str(epochs) + "_" + model_name +
                                     adv_data_path_numpy.replace("\\", "/").split("/")[-1].split(".npz")[
                                         0] + "WatermarkFinetuneLoss.png"),
                        "WatermarkFinetuneLoss.png")

    plt.figure()
    plt.plot(list(range(epochs)), train_acc_watermark, label="Train acc_" + dataset_name, marker='o',
             color='tab:purple')
    plt.plot(list(range(epochs)), val_acc_watermark, label="Val acc_" + dataset_name, linestyle='--',
             marker='o', color='tab:orange')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_PATH, LOSS_FOLDER, adv_data_path_numpy.replace("\\", "/").split("/")[-2],
                             dataset_name + "_" + str(epochs) +  "_" + model_name +
                             adv_data_path_numpy.replace("\\", "/").split("/")[-1].split(".npz")[
                                 0] + "WatermarkFinetuneAcc.png"))
    mlflow.log_artifact(os.path.join(RESULTS_PATH, LOSS_FOLDER, adv_data_path_numpy.replace("\\", "/").split("/")[-2],
                                     dataset_name + "_" + str(epochs) + "_" + model_name +
                                     adv_data_path_numpy.replace("\\", "/").split("/")[-1].split(".npz")[
                                         0] + "WatermarkFinetuneAcc.png"), "WatermarkFinetuneAcc.png")


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

    mlflow.set_tracking_uri("sqlite:///../mlflow.db")
    # mlflow.set_tracking_uri("file:///../mlruns")
    mlflow.set_experiment("frontier-stiching-watermarking")

    now = datetime.now().strftime("%d-%m-%Y")

    RESULTS_PATH = f"../results/finetuned_finetuning_combined{now}"
    DATA_PATH = "../data"
    MODEL_PATH = f"../models/finetuned_finetuning_combined{now}"
    LOSS_FOLDER = "losses"

    if not os.path.exists(os.path.join(RESULTS_PATH, LOSS_FOLDER, config.which_adv)):
        os.makedirs(os.path.join(RESULTS_PATH, LOSS_FOLDER, config.which_adv))

    if not os.path.exists(os.path.join(MODEL_PATH, config.which_adv)):
        os.makedirs(os.path.join(MODEL_PATH, config.which_adv))

    dataset_name = config.dataset_name
    dataset_path = os.path.join(DATA_PATH, "fgsm", dataset_name, config.which_adv)
    experiment_name = dataset_name + "Frontier_Watermarking_Finetuning"


    with mlflow.start_run(run_name=experiment_name):

        params = {"dataset_name": dataset_name, "epochs": config.epochs,
                  "optimizer": config.optimizer, "lr": config.lr,
                  "weight_decay": config.weight_decay, "dropout": config.dropout, "num_layers_unfreeze": config.num_layers_unfreeze,}

        for adv_path in os.listdir(dataset_path):
            if "fgsm_0.5" in adv_path:
                adv_data_path_numpy = os.path.join(dataset_path, adv_path)
                print(adv_data_path_numpy)
                num_layers_unfreeze = config.num_layers_unfreeze

                watermark_finetuning(dataset_name, adv_data_path_numpy, config.model_to_finetune_path ,config.epochs, config.dropout, config.batch_size, config.optimizer, config.lr, config.weight_decay, num_layers_unfreeze)






