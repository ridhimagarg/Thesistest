import tensorflow as tf
import numpy as np
import argparse
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pickle
import functools
import random

import models_new as md
import matplotlib.pyplot as plt

from utils_new import validate_watermark
from trigger_new import trigger_generation
from models_training_new import ewe_train, plain_model
from watermark_dataset import create_wm_dataset_old
# from mlflow import log_metric, log_param, log_params, log_artifacts
# import mlflow
import logging
from datetime import datetime
from keras.datasets import mnist, cifar10
import mlconfig
from keras.models import load_model
import models_new as models
from utils_new import validate_watermark, test_model
import keras
import os
import seaborn as sns
import time

os.environ["CUDA_VISIBLE_DEVICES"]="0"

np.random.seed(42)
tf.random.set_seed(0)
random.seed(42)

now = datetime.now().strftime("%d-%m-%Y")
sns.set_context("paper", font_scale=1.7, rc={"lines.linewidth": 2.0})

def data_preprocessing(dataset_name):
    if dataset_name == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        img_rows, img_cols, num_channels = 28, 28, 1
        num_classes = 10

    elif dataset_name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        img_rows, img_cols, num_channels = 32, 32, 3
        num_classes = 10
        y_train = np.squeeze(y_train)

    elif dataset_name == "cifar10resnet":
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
    # y_train = keras.utils.to_categorical(y_train, num_classes)
    # y_test = keras.utils.to_categorical(y_test, num_classes)

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
            x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i])/std[i]
            x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i])/std[i]


    return x_train, y_train, x_test, y_test, input_shape, num_classes


def train(dataset_name, batch_size, trigger_dataset_path, target_dataset_path, epochs,  w_epochs, n_w_ratio, factors, watermark_target_label,  watermark_source_label, train_lr, temp_lr, temperatures, model_architecture, distribution):

    x_train, y_train, x_test, y_test, input_shape, num_classes = data_preprocessing(dataset_name)

    exclude_x_data = x_train[y_train != watermark_target_label]
    exclude_y_data = y_train[y_train != watermark_target_label]

    models_mapping = {"mnist_l2": models.MNIST_L2, "MNIST_l2_EWE": models.MNIST_L2_EWE, "CIFAR10_BASE_2_EWE": models.CIFAR10_BASE_2_EWE}

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(x_train.shape[0]).batch(batch_size)
        
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)) 
    test_dataset = test_dataset.shuffle(x_test.shape[0]).batch(batch_size)

    half_batch_size = int(batch_size / 2)

    trigger_numpy = np.load(trigger_dataset_path)
    trigger_dataset = tf.data.Dataset.from_tensor_slices((trigger_numpy["arr_0"]))
    trigger_dataset = trigger_dataset.shuffle(buffer_size=1024).batch(half_batch_size)

    target_numpy = np.load(target_dataset_path)
    target_dataset = tf.data.Dataset.from_tensor_slices((target_numpy["arr_0"]))
    target_dataset = target_dataset.shuffle(buffer_size=1024).batch(half_batch_size)


    num_batch = x_train.shape[0] // batch_size ## whole data no. of batches 
    w_num_batch = target_numpy["arr_0"].shape[0] // batch_size * 2 ## watermark no. of batches, since trigger is same shape as target data


    optimizer = tf.keras.optimizers.Adam(learning_rate=train_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-5)

    model_name, model = models_mapping[model_architecture]()

    print(model.summary())

    # _, already_trained_model =  models.MNIST_L2()

    if not os.path.exists(os.path.join(RESULTS_PATH, LOSS_FOLDER, dataset_name + "_" + model_name + str(w_epochs) + str(factors[0]) + "_target_" + str(watermark_target_label) + "_source_" + str(watermark_source_label) + "_distrib_" + str(distribution))):
        os.makedirs(os.path.join(RESULTS_PATH, LOSS_FOLDER, dataset_name + "_" + model_name + str(w_epochs) + str(factors[0]) + "_target_" + str(watermark_target_label) + "_source_" + str(watermark_source_label) + "_distrib_" + str(distribution)))

    file = open(os.path.join(RESULTS_PATH, LOSS_FOLDER, dataset_name + "_" + model_name + str(w_epochs) +  str(factors[0]) + "_target_" + str(watermark_target_label) + "_source_" + str(watermark_source_label) + "_distrib_" + str(distribution), "logs.txt"), "w")
    
    start_time = time.time()

    loss_epoch = []
    for epoch in range(epochs):

        loss_batch = []
        for batch, (x_batch_train, y_batch_train) in enumerate(train_dataset):

            # x_batch_train = model

            with tf.GradientTape() as tape:

                w_0 = np.zeros([x_batch_train.shape[0]])
                prediction = model(x_batch_train)

                # loss_value = keras.losses.CategoricalCrossentropy()(y_batch_train, prediction[-1])

                loss_value, snnl_loss = md.combined_loss(prediction, y_batch_train, w_0, temperatures, factors)

            grads = tape.gradient(loss_value, model.trainable_weights)
            
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            loss_batch.append(loss_value)

        
        print(f"Loss at {epoch} is {np.mean(loss_batch)}")
        file.write(f"Loss at {epoch} is {np.mean(loss_batch)}\n")
        loss_epoch.append(np.mean(loss_batch))

    test_accuracy = test_model(model, test_dataset, "Original (trained)", num_classes, watermark_target=None)

    

    file.write(f"First trained model test accuracy: {test_accuracy}\n")

    file.write("Parmeters:\n")
    file.write(f"Ratio: {n_w_ratio}\n")
    file.write(f"Factors: {factors}\n")
    file.write(f"Temperatures: {temperatures}\n")
    file.write(f"Triggerset: {trigger_dataset_path}\n")
    file.write(f"Targetset: {target_dataset_path}\n")

    

    plt.figure()
    sns.lineplot(x=list(range(epochs)), y=loss_epoch,
                      ci=None, color="tab:orange", linestyle='--', marker='o', label="Train data loss", markersize=7)
    # plt.plot(list(range(epochs)), loss_epoch, label="Train data loss", linestyle='--', marker='o', color='tab:orange')
    plt.xlabel("Epochs")
    plt.ylabel("CE Loss")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_PATH, LOSS_FOLDER, dataset_name + "_" + model_name + str(w_epochs) + str(factors[0]) + "_target_" + str(watermark_target_label) + "_source_" + str(watermark_source_label) + "_distrib_" + str(distribution), "CombinelossInitialEWE.svg"), bbox_inches='tight')




    if not os.path.exists(os.path.join(MODEL_PATH,  str(w_epochs) + str(factors[0]) + str(watermark_target_label) + "_source_" + str(watermark_source_label) + "_distrib_" + str(distribution) + model_architecture)):
        os.makedirs(os.path.join(MODEL_PATH,  str(w_epochs) + str(factors[0]) + str(watermark_target_label) + "_source_" + str(watermark_source_label) + "_distrib_" + str(distribution) + model_architecture))

    victim_model_save_path = os.path.join(MODEL_PATH, str(w_epochs) + str(factors[0]) + str(watermark_target_label) + "_source_" + str(watermark_source_label) + "_distrib_" + str(distribution) + model_architecture, "ewe_model.h5")

    images_save_middlename =  "_" + model_name +  str(w_epochs) +  str(factors[0]) + "_target_" + str(watermark_target_label) + "_source_" + str(watermark_source_label) + "_distrib_" + str(distribution)

    model = ewe_train(model, train_dataset, trigger_dataset, target_dataset, test_dataset, victim_model_save_path, w_epochs, num_batch, w_num_batch, n_w_ratio, factors, optimizer, watermark_target_label, num_classes, batch_size, temp_lr, temperatures, dataset_name, exclude_x_data, exclude_y_data, RESULTS_PATH, LOSS_FOLDER, file, images_save_middlename)

    end_time = time.time()

    file.write(f"Time taken: {end_time - start_time}\n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="configs/ewe_train_retrain.yaml")

    args = parser.parse_args()
    print(args)
    config = mlconfig.load(args.config)

    dataset_name = config.dataset_name

    RESULTS_PATH = f"results/ewe_trainining_retrain{now}"
    LOSS_FOLDER = "losses"
    MODEL_PATH = f"models/ewe_training_retrain{now}"

    if not os.path.exists(os.path.join(RESULTS_PATH,LOSS_FOLDER)):
        os.makedirs(os.path.join(RESULTS_PATH,LOSS_FOLDER))
                    
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.join(MODEL_PATH))

    

    for distrib_type in ["out", "out_with_trigger"]: #"in", "in_with_fgsm",

        for target_trigger in os.listdir(os.path.join("data/fgsm", dataset_name+config["model_architecture"], distrib_type)):
            target_trigger_number = [int(i) for i in str(target_trigger)]

            if len(target_trigger_number) > 1:

                target = target_trigger_number[0]
                source = target_trigger_number[1]

                if target in [9] and source in [5]: ## this is manually set to run the experiments but can be modified according to your requirement.

                    print(os.path.join("data/fgsm", dataset_name+config["model_architecture"], distrib_type, target_trigger))

                    for file in os.listdir(os.path.join("data/fgsm", dataset_name+config["model_architecture"], distrib_type, target_trigger)):
                        print(file)

                        if file.endswith("trigger.npz"):
                            trigger_set_path = os.path.join("data/fgsm", dataset_name+config["model_architecture"], distrib_type, target_trigger, file)

                        if file.endswith("target.npz"):
                            target_set_path = os.path.join("data/fgsm", dataset_name+config["model_architecture"], distrib_type, target_trigger, file)


                    train(config["dataset_name"], config["batch_size"], trigger_set_path, target_set_path, config["epochs"], config["w_epochs"], config["ratio"], config["factors"],  target, source, config["train_lr"], config["temp_lr"], config["temperatures"], config["model_architecture"], distrib_type)


    