import pandas as pd
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
import logging
from datetime import datetime
from keras.datasets import mnist, cifar10
import mlconfig
from keras.models import load_model
import models_new as models
from utils_new import validate_watermark, test_model

np.random.seed(42)
tf.random.set_seed(0)
random.seed(42)

now = datetime.now().strftime("%d-%m-%Y")

def data_preprocessing(dataset_name):
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



def extraction(dataset_name, batch_size, ewe_trained_model_path, attacker_model_architecture, trigger_dataset_path, target_dataset_path, watermark_target_label, watermark_source_label, epochs, w_epochs, train_lr):

    x_train, y_train, x_test, y_test, input_shape, num_classes = data_preprocessing(dataset_name)

    models_mapping = {"mnist_l2": models.MNIST_L2, "MNIST_l2_EWE": models.MNIST_L2_EWE, "MNIST_Plain_2_conv": models.MNIST_Plain_2_conv, "CIFAR10_BASE_2": models.CIFAR10_BASE_2}

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)) 
    test_dataset = test_dataset.batch(batch_size)

    half_batch_size = int(batch_size / 2)

    trigger_numpy = np.load(trigger_dataset_path)
    trigger_dataset = tf.data.Dataset.from_tensor_slices((trigger_numpy["arr_0"]))
    trigger_dataset = trigger_dataset.shuffle(buffer_size=1024).batch(half_batch_size)

    target_numpy = np.load(target_dataset_path)
    target_dataset = tf.data.Dataset.from_tensor_slices((target_numpy["arr_0"]))
    target_dataset = target_dataset.shuffle(buffer_size=1024).batch(half_batch_size)

    test_results = []
    adv_results = []

    for i in range(5):

        extracted_label = []
        extracted_data = []

        ewe_trained_model = load_model(ewe_trained_model_path)

        test_accuracy = test_model(ewe_trained_model, test_dataset, "EWE (trained)", num_classes, watermark_target=None)
        watermark_accuracy = test_model(ewe_trained_model, trigger_dataset, "EWE (trained)", num_classes, watermark_target=watermark_target_label)

        print("Test accuracy", test_accuracy)
        print("watermark accuracy", watermark_accuracy)

        extraction_model_name, extraction_model = models_mapping[attacker_model_architecture]()

        if not os.path.exists(os.path.join(RESULTS_PATH, LOSS_FOLDER, dataset_name + "_" + extraction_model_name + ewe_trained_model_path.split("/")[-3] + "_" +ewe_trained_model_path.split("/")[-2])):
            os.makedirs(os.path.join(RESULTS_PATH, LOSS_FOLDER, dataset_name + "_" + extraction_model_name + ewe_trained_model_path.split("/")[-3] + "_" +ewe_trained_model_path.split("/")[-2]))

        file = open(os.path.join(RESULTS_PATH, LOSS_FOLDER, dataset_name + "_" + extraction_model_name + ewe_trained_model_path.split("/")[-3] + "_" +ewe_trained_model_path.split("/")[-2], "logs.txt"), "w")

        file.write(f"\nLoaded EWE trained model {ewe_trained_model_path} test accuracy {test_accuracy}\n")
        file.write(f"\nLoaded EWE trained model {ewe_trained_model_path} watermark accuracy {watermark_accuracy}\n")

        file.write("\nParams\n")
        file.write(f"Trigger set: {trigger_dataset_path} \n")
        file.write(f"Target set: {target_dataset_path} \n\n")

        for batch, (x_batch_train, y_batch_train) in enumerate(train_dataset):

            output = ewe_trained_model(x_batch_train)
            # logging.info("output", output)
            if isinstance(output, list):
                output = output[-1]
            extracted_label.append(output == np.max(output, 1, keepdims=True))
            extracted_data.append(x_batch_train)

        extracted_label = np.concatenate(extracted_label, 0)
        extracted_data = np.concatenate(extracted_data, 0)

        optimizer = tf.keras.optimizers.Adam(learning_rate=train_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-5)

        extracted_dataset = tf.data.Dataset.from_tensor_slices((extracted_data, extracted_label))
        extracted_dataset = extracted_dataset.shuffle(buffer_size=1024).batch(batch_size)

        extraction_flag = True

        if not os.path.exists(os.path.join(MODEL_PATH, dataset_name + "_" + extraction_model_name + ewe_trained_model_path.split("/")[-3] + "_" +ewe_trained_model_path.split("/")[-2])):
            os.makedirs(os.path.join(MODEL_PATH, dataset_name + "_" + extraction_model_name + ewe_trained_model_path.split("/")[-3] + "_" +ewe_trained_model_path.split("/")[-2]))

        model_save_path = os.path.join(MODEL_PATH, dataset_name + "_" + extraction_model_name + ewe_trained_model_path.split("/")[-3] + "_" +ewe_trained_model_path.split("/")[-2], "extracted_model.h5")

        images_save_middle_name = dataset_name + "_" + extraction_model_name + ewe_trained_model_path.split("/")[-3] + "_" +ewe_trained_model_path.split("/")[-2]



        extracted_model, test_accuracy, watermark_accuracy = plain_model(extraction_model, "EWE (Extraction)", extracted_dataset, test_dataset , extraction_flag, epochs, w_epochs, optimizer, num_classes, trigger_dataset, watermark_target_label, target_dataset, model_save_path, dataset_name, RESULTS_PATH, LOSS_FOLDER, file, images_save_middle_name)

        test_results.append((test_accuracy))
        adv_results.append((watermark_accuracy))

    df = pd.DataFrame(test_results)
    df_adv = pd.DataFrame(adv_results)

    TEST_ACC_PATH = os.path.join(RESULTS_PATH, LOSS_FOLDER, dataset_name + "_" + extraction_model_name + ewe_trained_model_path.split("/")[-3] + "_" +ewe_trained_model_path.split("/")[-2], "df_test_acc.csv")

    WATERMARK_ACC_PATH = os.path.join(RESULTS_PATH, LOSS_FOLDER, dataset_name + "_" + extraction_model_name + ewe_trained_model_path.split("/")[-3] + "_" +ewe_trained_model_path.split("/")[-2], "watermark_acc.csv")

    df.to_csv(TEST_ACC_PATH)
    df_adv.to_csv(WATERMARK_ACC_PATH)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="configs/ewe_extraction.yaml")

    args = parser.parse_args()
    print(args)
    config = mlconfig.load(args.config)

    dataset_name = config.dataset_name

    RESULTS_PATH = f"results/ewe_extraction_{now}"
    LOSS_FOLDER = "losses"
    MODEL_PATH = f"models/ewe_extraction_{now}"

    if not os.path.exists(os.path.join(RESULTS_PATH, LOSS_FOLDER)):
        os.makedirs(os.path.join(RESULTS_PATH, LOSS_FOLDER))

    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    extraction(config["dataset_name"], config["batch_size"], config["ewe_trained_model_path"], config["model_architecture"], config["trigger_set_path"], config["target_set_path"], config["target"], config["source"], config["epochs"], config["w_epochs"], config["train_lr"])

    