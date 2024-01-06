"""
@author: Ridhima Garg

Introduction:
    This file contains the implementation for performing model stealing attack on the defended model trained using "watermarking_finetuning.py" file
    This file will perform the attack as well as the ownership verification by checking the presence of the watermark set information (accuracy)
    The modified version of this file is real_model_stealing_watermark.py: which improves the chaos of the manual loop changing.

"""
import argparse
import warnings

warnings.filterwarnings('ignore')
from keras.models import load_model

import os
import numpy as np
import pandas as pd
import random
import keras
from keras.datasets import mnist, cifar10
import tensorflow as tf
import matplotlib.pyplot as plt
from art.estimators.classification import KerasClassifier
from art.attacks.extraction import KnockoffNets
import mlconfig
import models
# import mlflow
from datetime import datetime

now = datetime.now().strftime("%d-%m-%Y")
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

## this is neede when we are using keras with art.
tf.compat.v1.disable_eager_execution()


def data_preprocessing(dataset_name, adv_data_path_numpy):
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

    # load adversarial data
    adv = np.load(adv_data_path_numpy)
    x_adv, y_adv = adv['arr_1'], adv['arr_2']
    print(x_adv.shape)
    print(y_adv.shape)

    # x_adv = np.squeeze(x_adv, axis=1)
    # y_adv = keras.utils.to_categorical(y_adv, num_classes)

    # x_adv /= 255

    return x_train, y_train, x_test, y_test, x_adv, y_adv, input_shape


def model_extraction_attack(dataset_name, adv_data_path_numpy, attacker_model_architecture, number_of_queries,
                            num_epochs_to_steal, dropout, optimizer="adam", lr=0.001, weight_decay=0.00,
                            model_to_attack_path='../models/mnist_original_cnn_epochs_25.h5'):

    """
    Main idea
    --------
    Performing the attack and then also veryfying the ownership of the victim model by means watermarkset accuracy.
    If the attacker acheives a good watermark accuracy then victim model can claim that model is stolen from my his/her model.

    Args:
        dataset_name: name of the dataset
        adv_data_path_numpy: watermarkset path
        attacker_model_architecture: architecture which attacker chooses
        number_of_questions: stealing dataset size
        num_epochs_to_steal: number of epochs
        dropout: dropout for the model
        optimizer: optimizer of the model, but anyways we are using by default "Adam".
        lr: learning rate for the model
        weight_decay: if you want to use the weight decay
        model_to_attack_path: victim model path which is already trained with watermarkset.
    """

    x_train, y_train, x_test, y_test, x_adv, y_adv, _ = data_preprocessing(dataset_name, adv_data_path_numpy)

    models_mapping = {"mnist_l2": models.MNIST_L2, "cifar10_base_2": models.CIFAR10_BASE_2,
                      "cifar10_wideresnet": models.wide_residual_network}
    num_epochs = num_epochs_to_steal

    ## file to write the results.
    file1 = open(os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER, adv_data_path_numpy.replace("\\", "/").split("/")[-2],
                              "_".join((dataset_name, str(num_epochs),
                                        adv_data_path_numpy.replace("\\", "/").split("/")[-1].split(".npz")[
                                            0] + "_logs.txt"))), "w")

    model = load_model(model_to_attack_path, compile=False)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer="adam", metrics=['accuracy'])

    ## Evaluating the vuctim model accuracy on the watermark set.
    acc_adv = model.evaluate(x_adv, y_adv)[1]
    print("Just After loading victim model adv acc is:", acc_adv)
    file1.write("Just After loading victim model adv acc is: " + str(acc_adv) + "\n")

    classifier_original = KerasClassifier(model, clip_values=(0, 1), use_logits=False)

    im_shape = x_train[0].shape
    results = []
    results_adv = []

    ## performing the attack according to the query budget.
    for len_steal in number_of_queries:
        indices = np.random.permutation(len(x_test))
        x_steal = x_test[indices[:len_steal]]
        y_steal = y_test[indices[:len_steal]]
        x_test0 = x_test[indices[len_steal:]]
        y_test0 = y_test[indices[len_steal:]]

        attack_catalogue = {"KnockoffNet": KnockoffNets(classifier=classifier_original,
                                                        batch_size_fit=64,
                                                        batch_size_query=64,
                                                        nb_epochs=num_epochs,
                                                        nb_stolen=len_steal,
                                                        use_probability=False)}

        for name, attack in attack_catalogue.items():

            ## setting up the attacker model.
            if dropout:
                model_name, model_stolen = models_mapping[attacker_model_architecture](dropout)
            else:
                model_name, model_stolen = models_mapping[attacker_model_architecture]()

            model_stolen.compile(loss=keras.losses.categorical_crossentropy, optimizer="adam", metrics=['accuracy'])

            classifier_stolen = KerasClassifier(model_stolen, clip_values=(0, 1), use_logits=False)

            ## performing the attack.
            classifier_stolen = attack.extract(x_steal, y_steal, thieved_classifier=classifier_stolen)

            ## evaluating the attacked model on test set
            acc = classifier_stolen.model.evaluate(x_test, y_test)[1]
            print(f"test acc with {len_steal} is {acc}")
            file1.write(f"Victim model {model_to_attack_path}")
            file1.write(f"test acc with {len_steal} is {acc}\n")
            results.append((name, len_steal, acc))

            # evaluating the attacked model on adversasrial set/watermark set.
            acc_adv = classifier_stolen.model.evaluate(x_adv, y_adv)[1]
            print(f"adv acc with {len_steal} is {acc_adv}")
            file1.write(f"adv acc with {len_steal} is {acc_adv}\n")
            results_adv.append((name, len_steal, acc_adv))

            # classifier_stolen.model.save(os.path.join(MODEL_PATH, adv_data_path_numpy.replace("\\", "/").split("/")[-2], "_".join((dataset_name, str(len_steal) , str(num_epochs) ,
            #                                           adv_data_path_numpy.replace("\\", "/").split("/")[-1].split(".npz")[0] + ".h5"))))

    ## creating the path to save image.
    image_save_name = os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER, adv_data_path_numpy.replace("\\", "/").split("/")[-2],
                                   "_".join((dataset_name, str(num_epochs),
                                             adv_data_path_numpy.replace("\\", "/").split("/")[-1].split(".npz")[
                                                 0] + "TestandWatermarkAcc.png")))

    df = pd.DataFrame(results, columns=('Method Name', 'Stealing Dataset Size', 'Accuracy'))
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, group in df.groupby("Method Name"):
        group.plot(1, 2, ax=ax, label="Test acc", linestyle='--', marker='o', color='tab:purple')

    df = pd.DataFrame(results_adv, columns=('Method Name', 'Stealing Dataset Size', 'Accuracy'))
    # fig, ax = plt.subplots(figsize=(8,6))
    ax.set_xlabel("Stealing Dataset Size")
    ax.set_ylabel("Stolen Model Test and Adversarial Accuracy")
    for name, group in df.groupby("Method Name"):
        group.plot(1, 2, ax=ax, label="Watermark acc", linestyle='--', marker='o', color='tab:orange')
    plt.savefig(image_save_name)
    file1.close()
    # mlflow.log_artifact(os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER, adv_data_path_numpy.replace("\\", "/").split("/")[-2],
    #                            "_".join((dataset_name, str(num_epochs), adv_data_path_numpy.replace("\\", "/").split("/")[-1].split(".npz")[0] + "_logs.txt"))), "logs.txt")
    # mlflow.log_artifact(os.path.join(image_save_name), "TestandWatermarkAcc.png")


if __name__ == "__main__":

    ## taking all the parameters from the config file.
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="configs/knockoffattack_original.yaml")

    args = parser.parse_args()
    print(args)
    config = mlconfig.load(args.config)

    # mlflow.set_tracking_uri("sqlite:///../mlflow.db")
    # mlflow.set_experiment("frontier-stiching-realmodelstealing")

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    tf.compat.v1.random.set_random_seed(seed)

    RESULTS_PATH = f"../results/attack_original_{now}"
    LOSS_Acc_FOLDER = "losses_acc"
    MODEL_PATH = f"../models/attack_original_{now}"
    DATA_PATH = "../data"

    if not os.path.exists(os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER, config.which_adv)):
        os.makedirs(os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER, config.which_adv))

    if not os.path.exists(os.path.join(MODEL_PATH, config.which_adv)):
        os.makedirs(os.path.join(MODEL_PATH, config.which_adv))

    dataset_name = config.dataset_name
    model_to_attack_path = config.model_to_attack_path

    model_to_attack_name = model_to_attack_path.split("/")[-2]

    # if config.optimizer == "adam":
    #    optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr, decay=config.weight_decay)
    # else:
    # optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, decay=config.weight_decay)
    #    optimizer = None

    experiment_name = "realstealing" + dataset_name
    # with mlflow.start_run(run_name=experiment_name):
    #     params = {"dataset_name": dataset_name,
    #               "model_to_attack_path": model_to_attack_path, "attacker_model_architecture": config.attacker_model_architecture, "optimizer": config.optimizer, "dropout": config.dropout, "lr": config.lr, "weight_decay": config.weight_decay, "epochs_extract": config.epochs_extract}

    #     for param in params:
    #         mlflow.log_param(param, params[param])

    for adv_file in os.listdir(os.path.join(DATA_PATH, "fgsm", dataset_name, str(config.which_adv))): ## for time , change this "mnist_wo_fs" again to dataset_name
        if model_to_attack_name in adv_file and "fgsm_0.035_250" in adv_file:
        # if "fgsm_0.045" in adv_file:
            adv_data_path_numpy = os.path.join(DATA_PATH, "fgsm", dataset_name, str(config.which_adv), adv_file)  ## for time , change this "mnist_wo_fs" again to dataset_name
            model_extraction_attack(config.dataset_name, adv_data_path_numpy, config.attacker_model_architecture,
                                    number_of_queries=[250, 500, 1000, 5000, 10000, 20000],
                                    model_to_attack_path=model_to_attack_path, num_epochs_to_steal=config.epochs_extract, dropout=config.dropout, optimizer=config.optimizer,
                                    lr=config.lr, weight_decay=config.weight_decay)
