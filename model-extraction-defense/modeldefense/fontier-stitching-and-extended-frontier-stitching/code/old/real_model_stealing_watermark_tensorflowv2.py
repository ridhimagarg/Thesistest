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
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
import matplotlib.pyplot as plt
from art.estimators.classification import KerasClassifier, TensorFlowV2Classifier
from art.attacks import ExtractionAttack
from art.attacks.extraction import CopycatCNN, KnockoffNets
import mlconfig
import models
import mlflow
from datetime import datetime

now = datetime.now().strftime("%d-%m-%Y")
# tf.compat.v1.disable_eager_execution()
# tf.compat.v1.enable_eager_execution()

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

    # load adversarial data
    adv = np.load(adv_data_path_numpy)
    x_adv, y_adv = adv['arr_1'], adv['arr_2']
    print(x_adv.shape)
    print(y_adv.shape)

    return x_train, y_train, x_test, y_test, x_adv, y_adv, input_shape


def model_extraction_attack(dataset_name, adv_data_path_numpy, attacker_model_architecture, number_of_queries,
                            num_epochs_to_steal, dropout, optimizer="adam", lr=0.001, weight_decay=0.00,
                            model_to_attack_path='../models/mnist_original_cnn_epochs_25.h5'):
    x_train, y_train, x_test, y_test, x_adv, y_adv, input_shape = data_preprocessing(dataset_name, adv_data_path_numpy)

    models_mapping = {"mnist_l2": models.MNIST_L2, "cifar10_base_2": models.CIFAR10_BASE_2, "resnet34": models.ResNet34}
    num_epochs = num_epochs_to_steal
    file1 = open(os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER, adv_data_path_numpy.replace("\\", "/").split("/")[-2],
                              "_".join((dataset_name, str(num_epochs),
                                        adv_data_path_numpy.replace("\\", "/").split("/")[-1].split(".npz")[
                                            0] + "_logs.txt"))), "w")

    model = load_model(model_to_attack_path, compile=False)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer="adam", metrics=['accuracy'])

    acc_adv = model.evaluate(x_adv, y_adv)[1]
    print("Just After loading victim model adv acc is:", acc_adv)
    file1.write("Just After loading victim model adv acc is: " + str(acc_adv) + "\n")

    loss_object = tf.keras.losses.CategoricalCrossentropy()

    if attacker_model_architecture == "cifar10_base_2":

        classifier_original = TensorFlowV2Classifier(model, nb_classes=10,
                                                     input_shape=(x_train[1], x_train[2], x_train[3]))

    else:

        classifier_original = KerasClassifier(model, clip_values=(0, 1), use_logits=False)

    im_shape = x_train[0].shape

    results = []
    results_adv = []

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
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        def train_step(model1, images, labels):

            with tf.GradientTape() as tape:
                prediction = model1(images)
                loss = loss_object(labels, prediction)
                file1.write(f"\n Loss of attacker model: {loss:.3f}")
                file1.write("\n")
                # print("loss", loss)

            grads = tape.gradient(loss, model1.trainable_weights)
            optimizer.apply_gradients(zip(grads, model1.trainable_weights))

        for name, attack in attack_catalogue.items():

            if attacker_model_architecture == "cifar10_base_2":
                model_name, model_stolen = models_mapping[attacker_model_architecture]()

                # model_stolen.compile(loss=keras.losses.categorical_crossentropy, optimizer="adam", metrics=['accuracy'])

            else:
                if dropout:
                    model_name, model_stolen = models_mapping[attacker_model_architecture](dropout)
                else:
                    model_name, model_stolen = models_mapping[attacker_model_architecture]()

                model_stolen.compile(loss=keras.losses.categorical_crossentropy, optimizer="adam", metrics=['accuracy'])

            if attacker_model_architecture == "cifar10_base_2":

                classifier_stolen = TensorFlowV2Classifier(model_stolen, nb_classes=10, loss_object=loss_object,
                                                           input_shape=input_shape, channels_first=False,
                                                           train_step=train_step)

            else:

                classifier_stolen = KerasClassifier(model_stolen, clip_values=(0, 1), use_logits=False)

            classifier_stolen = attack.extract(x_steal, y_steal, thieved_classifier=classifier_stolen)
            # acc = classifier_stolen.model.evaluate(x_test, y_test)[1]
            # print(f"test acc with {len_steal} is {acc}")
            # file1.write(f"Victim model {model_to_attack_path}")
            # file1.write(f"test acc with {len_steal} is {acc}\n")
            # results.append((name, len_steal, acc))

            # # test with adversarial data
            # acc_adv = classifier_stolen.model.evaluate(x_adv, y_adv)[1]
            # print(f"adv acc with {len_steal} is {acc_adv}")
            # file1.write(f"adv acc with {len_steal} is {acc_adv}\n")
            # results_adv.append((name, len_steal, acc_adv))

            predictions = classifier_stolen.predict(x_test, batch_size=64)
            # print("Argmax predictions {}".format(np.argmax(predictions, axis=1)))
            # print("Actual y steal {}".format(y_test))
            accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
            print("Accuracy on stealing train examples: {}%".format(accuracy * 100))

            predictions = classifier_stolen.predict(x_adv, batch_size=64)
            # print("Argmax predictions {}".format(np.argmax(predictions, axis=1)))
            # print("Actual y steal {}".format(y_adv))
            accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_adv, axis=1)) / len(y_adv)
            print("Accuracy on adv examples: {}%".format(accuracy * 100))

            classifier_stolen.model.save(
                os.path.join(MODEL_PATH, adv_data_path_numpy.replace("\\", "/").split("/")[-2],
                             "_".join((dataset_name, str(len_steal), str(num_epochs),
                                       adv_data_path_numpy.replace("\\", "/").split("/")[-1].split(".npz")[
                                           0] + ".h5"))))

    image_save_name = os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER, adv_data_path_numpy.replace("\\", "/").split("/")[-2],
                                   "_".join((dataset_name, str(num_epochs),
                                             adv_data_path_numpy.replace("\\", "/").split("/")[-1].split(".npz")[
                                                 0] + "TestandWatermarkAcc.png")))

    df = pd.DataFrame(results, columns=('Method Name', 'Stealing Dataset Size', 'Accuracy'))
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, group in df.groupby("Method Name"):
        group.plot(1, 2, ax=ax, label="Test acc", linestyle='--', marker='o', color='tab:purple')

    df = pd.DataFrame(results_adv, columns=('Method Name', 'Stealing Dataset Size', 'Accuracy'))
    ax.set_xlabel("Stealing Dataset Size")
    ax.set_ylabel("Stolen Model Test and Adversarial Accuracy")
    for name, group in df.groupby("Method Name"):
        group.plot(1, 2, ax=ax, label="Watermark acc", linestyle='--', marker='o', color='tab:orange')
    plt.savefig(image_save_name)
    file1.close()
    mlflow.log_artifact(
        os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER, adv_data_path_numpy.replace("\\", "/").split("/")[-2],
                     "_".join((dataset_name, str(num_epochs),
                               adv_data_path_numpy.replace("\\", "/").split("/")[-1].split(".npz")[
                                   0] + "_logs.txt"))), "logs.txt")
    mlflow.log_artifact(os.path.join(image_save_name), "TestandWatermarkAcc.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="configs/knockoffattack_finetuned.yaml")

    args = parser.parse_args()
    print(args)
    config = mlconfig.load(args.config)

    dataset_name = config.dataset_name

    mlflow.set_tracking_uri("sqlite:///../mlflow.db")
    mlflow.set_experiment("frontier-stiching-realmodelstealing")

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    tf.compat.v1.random.set_random_seed(seed)

    RESULTS_PATH = f"../results/attack_finetuned{now}"
    LOSS_Acc_FOLDER = "losses_acc"
    MODEL_PATH = f"../models/attack_finetuned{now}"
    DATA_PATH = "../../data"

    if not os.path.exists(os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER, config.which_adv)):
        os.makedirs(os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER, config.which_adv))

    if not os.path.exists(os.path.join(MODEL_PATH, config.which_adv)):
        os.makedirs(os.path.join(MODEL_PATH, config.which_adv))

    # if config.optimizer == "adam":
    #    optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr, decay=config.weight_decay)
    # else:
    # optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, decay=config.weight_decay)
    #    optimizer = None

    experiment_name = "realstealing" + dataset_name
    with mlflow.start_run(run_name=experiment_name):
        params = {"dataset_name": dataset_name,
                  "attacker_model_architecture": config.attacker_model_architecture, "optimizer": config.optimizer,
                  "dropout": config.dropout, "lr": config.lr, "weight_decay": config.weight_decay,
                  "epochs_extract": config.epochs_extract}

        for param in params:
            mlflow.log_param(param, params[param])

        # adv_file_path = ["../data/fgsm/mnist/fgsm_0.5_250_mnist_20_MNIST_l20.0_Original_checkpoint_best.npz",
        #                  "../data/fgsm/mnist/fgsm_0.5_2500_mnist_20_MNIST_l20.0_Original_checkpoint_best.npz",
        #                  "../data/fgsm/mnist/fgsm_0.25_500_mnist_20_MNIST_l20.0_Original_checkpoint_best.npz",
        #                  "../data/fgsm/mnist/fgsm_0.25_2500_mnist_20_MNIST_l20.0_Original_checkpoint_best.npz"]
        # finetuned_model_path = [
        #     "../models/finetuned_retraining/mnist_100_MNIST_l20.2fgsm_0.5_250_mnist_20_MNIST_l20.0_Original_checkpoint_best/Victim_checkpoint_best.h5",
        #     "..//models/finetuned_retraining/mnist_100_MNIST_l20.2fgsm_0.5_2500_mnist_20_MNIST_l20.0_Original_checkpoint_best/Victim_checkpoint_best.h5",
        #     "../models/finetuned_retraining_24-08-2023/mnist_100_MNIST_l20.0fgsm_0.25_500_mnist_20_MNIST_l20.0_Original_checkpoint_best/Victim_checkpoint_best.h5",
        #     "../models/finetuned_retraining_24-08-2023/mnist_100_MNIST_l20.0fgsm_0.25_2500_mnist_20_MNIST_l20.0_Original_checkpoint_best/Victim_checkpoint_best.h5"]

        # adv_file_path = ["../data/fgsm/mnist/true/fgsm_0.25_500_mnist_20_MNIST_l20.0_Original_checkpoint_best.npz"]

        # finetuned_model_path = [
        #     "../models/finetuned_retraining_27-08-2023/true/mnist_100_MNIST_l20.0fgsm_0.25_500_mnist_20_MNIST_l20.0_Original_checkpoint_best/Victim_checkpoint_best.h5"]

        # adv_file_path = ["../data/fgsm/mnist/true/fgsm_0.4_1000_mnist_20_MNIST_l20.0_Original_checkpoint_best.npz"]

        # finetuned_model_path = [
        #    "../models/finetuned_retraining_28-08-2023/true/mnist_100_MNIST_l20.0fgsm_0.4_1000_mnist_20_MNIST_l20.0_Original_checkpoint_best/Victim_checkpoint_best.h5"]

        adv_file_path = ["../data/fgsm/cifar10/true/fgsm_0.045_1000_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best.npz"]
        finetuned_model_path = [
            "../models/finetuned_finetuning_02-09-2023/true/cifar10_25_25_cifar10_200_ResNet34fgsm_0.045_10000_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best/Victim_checkpoint_best"]

        # dataset_name = "mnist"
        # adv_file_path = ["../data/fgsm/cifar10/fgsm_0.1_250_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best.npz","../data/fgsm/cifar10/fgsm_0.1_2500_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best.npz",
        #                  "../data/fgsm/cifar10/fgsm_0.1_1000_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best.npz", "../data/fgsm/cifar10/fgsm_0.1_10000_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best.npz"]
        # finetuned_model_path = ["../models/finetuned_finetuning_20-08-2023/cifar10_100_bestfgsm_0.1_250_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best/Victim_checkpoint_best.h5","../models/finetuned_finetuning_20-08-2023/cifar10_100_bestfgsm_0.1_2500_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best/Victim_checkpoint_best.h5",
        #                         "../models/finetuned_finetuning_20-08-2023/cifar10_100_bestfgsm_0.1_1000_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best/Victim_checkpoint_best.h5", "../models/finetuned_finetuning_20-08-2023/cifar10_100_bestfgsm_0.1_10000_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best/Victim_checkpoint_best.h5"]

        # adv_file_path = ["../data/fgsm/cifar10/fgsm_0.1_250_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best.npz"]

        # finetuned_model_path = ["../models/finetuned_retraining_21-08-2023/cifar10_100_CIFAR10_BASE_2fgsm_0.1_250_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best/Victim_checkpoint_best.h5"]

        # adv_file_path = ["../data/fgsm/cifar10/fgsm_0.025_250_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best.npz"]

        # finetuned_model_path = ["../models/finetuned_retraining_22-08-2023/cifar10_100_CIFAR10_BASE_2fgsm_0.025_250_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best/Victim_checkpoint_best.h5"]

        for adv_file_path, model_path in zip(adv_file_path, finetuned_model_path):
            adv_data_path_numpy = adv_file_path
            model_to_attack_path = model_path
            model_extraction_attack(dataset_name, adv_data_path_numpy, config.attacker_model_architecture,
                                    number_of_queries=[250, 500, 1000, 5000, 10000, 20000],
                                    num_epochs_to_steal=config.epochs_extract, dropout=config.dropout,
                                    optimizer=config.optimizer,
                                    lr=config.lr, weight_decay=config.weight_decay,
                                    model_to_attack_path=model_to_attack_path)

