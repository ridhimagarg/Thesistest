"""
@author: Ridhima Garg

Introduction:
    This file contains the code to "create" the fgsm based adversarial (watermarked) samples to finetune the model in order to make it watermarked.
    Please refer to the paper: https://arxiv.org/abs/1711.01894

"""

import argparse
import os
import warnings
warnings.filterwarnings('ignore')
from keras.models import load_model
import keras
import numpy as np
from keras.datasets import mnist, cifar10
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import mlconfig
import mlflow
import random
tf.compat.v1.enable_eager_execution()
seed = 0
random.seed(seed)
np.random.seed(seed)
tf.compat.v1.random.set_random_seed(seed)



DATA_PATH = "../data/fgsm"

mlflow.set_tracking_uri("sqlite:///../mlflow.db")
#mlflow.set_tracking_uri("file:///../mlruns")
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
            x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i])/std[i]
            x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i])/std[i]


    return x_train, y_train, x_test, y_test, input_shape, num_classes

def binomial(n, k):
    if not 0 <= k <= n:
        return 0
    b = 1
    for t in range(min(k, n-k)):
        b *= n
        b //= t+1
        n -= 1
    return b

def fast_gradient_signed(x, y, model, eps):
    """
        Main idea:
        ----------
        To actually perform the attack using fgsm (fast gradient sign method): https://arxiv.org/abs/1412.6572

        Args:
        ------
        x: data array
        y: label array
        model: model for which attack should be performed
        eps: level of perturbation

        Returns:
        -----
        perturbe image

    """

    x = tf.cast(x, dtype=np.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y_pred = model(x)
        loss = model.loss(y, y_pred)
    gradient = tape.gradient(loss, x)
    sign = tf.sign(gradient)
    return x + eps * sign

def gen_adversaries(model, l, image, label, eps):

    """
    Main idea
    ---------
    This function will generate seperate the adversaries generated from the "fast_gradient_signed" function.

    args:
    -----
    model: model using which attack has to be performed
    l: total no. of adversarial examples
    image: set of images array.
    label: set of label array.
    eps: noise level

    Returns
    ------
    true adversaries, false adversaries.

    """


    true_advs = []
    false_advs = []
    max_true_advs = max_false_advs = l // 2
    for x, y in zip(image, label):
        # generate adversaries
        x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
        y = y.reshape(1, y.shape[0])
        x_advs = fast_gradient_signed(x, y, model, eps)

        y_preds = tf.argmax(model(x), axis=1)
        y_pred_advs = tf.argmax(model(x_advs), axis=1)
        for x_adv, y_pred_adv, y_pred, y_true in zip(x_advs, y_pred_advs, y_preds, y):
            # x_adv is a true adversary
            #print(y_pred.numpy())
            if y_pred.numpy() == np.argmax(y_true) and y_pred_adv.numpy() != np.argmax(y_true) and len(true_advs) < max_true_advs: ## true adv when prediction of the adversarial image gets diff prediction from y_true
                true_advs.append((x, x_adv, y_true))

            # x_adv is a false adversary
            if y_pred.numpy() == np.argmax(y_true) and y_pred_adv.numpy() == np.argmax(y_true) and len(false_advs) < max_false_advs: ## true adv when prediction of the adversarial image gets same prediction from y_true
                false_advs.append((x, x_adv, y_true))

            if len(true_advs) == max_true_advs and len(false_advs) == max_false_advs:
                return true_advs, false_advs

    return true_advs, false_advs

def fgsm_attack(dataset_name, model_path, eps=0.3, adversarial_sample_size=1000, npz_full_file_name='mnist_cnn_adv.npz', npz_true_file_name='mnist_cnn_adv.npz', npz_false_file_name='mnist_cnn_adv.npz'):
    """
    Main idea
    ---------
    This function will combine the above defined function to save all true adversaries, false adversaries, and the combine one in "full"

    """


    classifier_model = load_model(model_path)

    x_train, y_train, x_test, y_test, input_shape, num_classes = data_preprocessing(dataset_name)

    #model prediction on original test data
    x_test_pred = np.argmax(classifier_model.predict(x_test[:1000]), axis=1)
    nb_correct_pred = np.sum(x_test_pred == np.argmax(y_test[:1000], axis=1))

    print("Original test data (first 1000 images):")
    print("Correctly classified: {}".format(nb_correct_pred))
    print("Incorrectly classified: {}".format(1000-nb_correct_pred))

    #craete attack
    true_advs, false_advs = gen_adversaries(classifier_model, adversarial_sample_size, x_test, y_test, eps)
    full_advs = true_advs + false_advs


    x_test_adv_orig = np.array([x for x, x_adv, y in full_advs])
    x_test_adv = np.array([x_adv for x, x_adv, y in full_advs])
    y_test_adv = np.array([y for x, x_adv, y in full_advs])

    x_true_adv_orig = np.array([x for x, x_adv, y in true_advs])
    x_true_adv = np.array([x_adv for x, x_adv, y in true_advs])
    y_true_adv = np.array([y for x, x_adv, y in true_advs])

    x_false_adv_orig = np.array([x for x, x_adv, y in false_advs])
    x_false_adv = np.array([x_adv for x, x_adv, y in false_advs])
    y_false_adv = np.array([y for x, x_adv, y in false_advs])

    np.savez(npz_full_file_name, x_test_adv_orig, x_test_adv, y_test_adv)
    np.savez(npz_true_file_name, x_true_adv_orig, x_true_adv, y_true_adv)
    np.savez(npz_false_file_name, x_false_adv_orig, x_false_adv, y_false_adv)

    #np.savez("full_".join((npz_file_name.replace("\\", "/").rsplit("/", 1)[0], npz_file_name.replace("\\", "/").rsplit("/", 1)[1])), x_test_adv_orig, x_test_adv, y_test_adv)
    #np.savez("true_".join((npz_file_name.replace("\\", "/").rsplit("/", 1)[0], npz_file_name.replace("\\", "/").rsplit("/", 1)[1])), x_true_adv_orig, x_true_adv, y_true_adv)
    #np.savez("false_".join((npz_file_name.replace("\\", "/").rsplit("/", 1)[0], npz_file_name.replace("\\", "/").rsplit("/", 1)[1])), x_false_adv_orig, x_false_adv, y_false_adv)

if __name__ == '__main__':

    """ 
    Same pattern is followed throughout the repository to load variables from the configuration files.
    """

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

    if not os.path.exists(os.path.join(DATA_PATH, dataset_name, "full")):
        os.makedirs(os.path.join(DATA_PATH, dataset_name, "full"))

    if not os.path.exists(os.path.join(DATA_PATH, dataset_name, "true")):
        os.makedirs(os.path.join(DATA_PATH, dataset_name, "true"))

    if not os.path.exists(os.path.join(DATA_PATH, dataset_name, "false")):
        os.makedirs(os.path.join(DATA_PATH, dataset_name, "false"))

    experiment_name = dataset_name + "Frontier_FGSM_Attack"
    with mlflow.start_run(run_name=experiment_name):
        params = {"dataset_name": dataset_name,
                "eps": eps_list, "adversarial_sample_size_list": adversarial_sample_size_list}

        for param in params:
            mlflow.log_param(param, params[param])

        for adversarial_sample_size in adversarial_sample_size_list:
            for eps in eps_list:
                numpy_array_full_file_name = f'{os.path.join(DATA_PATH, dataset_name, "full")}/fgsm_{eps}_{adversarial_sample_size}_{"_".join(model_to_attack_path.split("/")[-2:]).split(".h5")[0]}.npz'
                numpy_array_true_file_name = f'{os.path.join(DATA_PATH, dataset_name, "true")}/fgsm_{eps}_{adversarial_sample_size}_{"_".join(model_to_attack_path.split("/")[-2:]).split(".h5")[0]}.npz'
                numpy_array_false_file_name = f'{os.path.join(DATA_PATH, dataset_name, "false")}/fgsm_{eps}_{adversarial_sample_size}_{"_".join(model_to_attack_path.split("/")[-2:]).split(".h5")[0]}.npz'
                fgsm_attack(dataset_name, model_path=model_to_attack_path, eps=eps,
                            adversarial_sample_size=adversarial_sample_size, npz_full_file_name=numpy_array_full_file_name, npz_true_file_name=numpy_array_true_file_name, npz_false_file_name=numpy_array_false_file_name)
