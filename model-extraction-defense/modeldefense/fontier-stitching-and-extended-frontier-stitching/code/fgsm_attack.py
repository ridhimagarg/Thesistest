"""
@author: Ridhima Garg

Introduction:
    This file contains the code to "create" the fgsm based adversarial (watermarked) samples to finetune the model in order to make it watermarked.
    But please make sure to run the current code, use "frontier-stitching.py" to get the all adversaries: true, false (to understand this).
    Please refer to the paper: https://arxiv.org/abs/1711.01894

"""
import argparse
import os
import warnings
from random import random

warnings.filterwarnings('ignore')
from keras.models import load_model
import keras

from art.estimators.classification import KerasClassifier
from art.attacks.evasion import FastGradientMethod

import numpy as np
from keras.datasets import mnist, cifar10
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
import mlconfig
import mlflow

seed = 0
random.seed(seed)
np.random.seed(seed)
tf.compat.v1.random.set_random_seed(seed)

## this is the path of the dataset where you want to store your generated files.
DATA_PATH = "../data/fgsm"

## use this if you want to use mlflow.
mlflow.set_tracking_uri("sqlite:///../mlflow.db")
mlflow.set_experiment("frontier-stiching-fgsm-attack")


def data_preprocessing(dataset_name):
    """
    Main idea
    -------
    This is the function which preprocess the data for the modelling.

    Args:
    .---
    dataset_name: name of the dataset.

    Future work:
    -----------
    This function for now is copied in multiple files but can be modified such that it can be exported.

    """


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

    return x_train, y_train, x_test, y_test, input_shape, num_classes


def fgsm_attack(dataset_name, model_path, clip_values=(0., 1.), eps=0.3, adversarial_sample_size=1000,
                npz_file_name='mnist_cnn_adv.npz'):

    """
    Main idea:
    ----------
    To actually perform the attack using fgsm (fast gradient sign method): https://arxiv.org/abs/1412.6572

    Args:
    -----
    dataset_name: name of the dataset
    model_path: path to the model using which the attack (adversarial samples are generated) is performed

    """


    classifier_model = load_model(model_path)

    x_train, y_train, x_test, y_test, input_shape, num_classes = data_preprocessing(dataset_name)

    # model prediction on original test data
    x_test_pred = np.argmax(classifier_model.predict(x_test[:1000]), axis=1)
    nb_correct_pred = np.sum(x_test_pred == np.argmax(y_test[:1000], axis=1))

    print("Original test data (first 1000 images):")
    print("Correctly classified: {}".format(nb_correct_pred))
    print("Incorrectly classified: {}".format(1000 - nb_correct_pred))

    # craete attack
    classifier = KerasClassifier(clip_values=clip_values, model=classifier_model, use_logits=False)
    print('Original model architecture:')
    print(classifier_model.summary())

    print('Prediction after creaing keras classifier model from orginal model:')
    x_test_pred = np.argmax(classifier.predict(x_test[:1000]), axis=1)
    nb_correct_pred = np.sum(x_test_pred == np.argmax(y_test[:1000], axis=1))

    print("Original test data (first 1000 images):")
    print("Correctly classified: {}".format(nb_correct_pred))
    print("Incorrectly classified: {}".format(1000 - nb_correct_pred))

    attacker = FastGradientMethod(classifier, eps=eps)
    x_test_adv = attacker.generate(x_test[:adversarial_sample_size])
    np.savez(npz_file_name, x_test_adv, y_test[:adversarial_sample_size])
    # mlflow.log_artifact(npz_file_name, npz_file_name.split("/")[-1])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="configs/fgsm.yaml")

    args = parser.parse_args()
    print(args)
    config = mlconfig.load(args.config)

    dataset_name = config.dataset_name
    model_to_attack_path = config.model_to_attack_path
    adversarial_sample_size_list = config.adversarial_sample_size_list
    eps_list = config.eps_list

    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    if not os.path.exists(os.path.join(DATA_PATH, dataset_name)):
        os.makedirs(os.path.join(DATA_PATH, dataset_name))

    experiment_name = dataset_name + "Frontier_FGSM_Attack"
    with mlflow.start_run(run_name=experiment_name):
        params = {"dataset_name": dataset_name,
                  "eps": eps_list, "adversarial_sample_size_list": adversarial_sample_size_list}

        for param in params:
            mlflow.log_param(param, params[param])

        for adversarial_sample_size in adversarial_sample_size_list:
            for eps in eps_list:
                numpy_array_file_name = f'{os.path.join(DATA_PATH, dataset_name)}/fgsm_{eps}_{adversarial_sample_size}_{"_".join(model_to_attack_path.split("/")[-2:]).split(".h5")[0]}.npz'
                fgsm_attack(dataset_name, model_path=model_to_attack_path, clip_values=(0., 1.), eps=eps,
                            adversarial_sample_size=adversarial_sample_size, npz_file_name=numpy_array_file_name)
