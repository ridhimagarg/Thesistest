import os
import argparse
import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import mlconfig
import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
import torch.utils.data as data
import torchvision as tv
from art.attacks.extraction import KnockoffNets
from art.estimators.classification.pytorch import PyTorchClassifier
## custom libraries
# import utils
# import utils.config_helper as config_helper
from utils import logger

import models
from environment import download_victim, setup_transformations

now = datetime.now().strftime("%d-%m-%Y")

log = logger.Logger(prefix=">>>")
logging_path = "old/logging"
results_path = "results"

# -> Tuple[NamedTuple, t.nn.Module, PyTorchClassifier, dict, np.ndarray, np.ndarray]

def generate_test_set(victim_data_path, victim_dataset, watermark_dataset, force_greyscale, normalize_with_imagenet_vals, input_size, batch_size):

    training_transforms, watermark_transforms = setup_transformations(victim_dataset, watermark_dataset, force_greyscale, normalize_with_imagenet_vals, input_size)
    train_set, test_set = download_victim(victim_dataset, victim_data_path, training_transforms, return_model=False)

    test_dl = data.DataLoader(test_set, batch_size=len(test_set), shuffle=False)

    test_dataset_array = next(iter(test_dl))[0].numpy()
    test_dataset_label = next(iter(test_dl))[1].numpy()

    if not os.path.exists(os.path.join("../data", "test_set", str(victim_dataset))):
        os.makedirs(os.path.join("../data", "test_set", str(victim_dataset)))

    return test_dataset_array, test_dataset_label

def setup_victim_attacker(dataset_name, victim_model_architecture, attacker_model_architecture, model_to_attack_path):


    # if dataset_name == 'MNIST':
    #     (x_train, y_train), (x_test, y_test) = mnist.load_data()
    #     img_rows, img_cols, num_channels = 28, 28, 1
    #     num_classes = 10

    # elif dataset_name == 'CIFAR10':
    #     (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    #     img_rows, img_cols, num_channels = 32, 32, 3
    #     num_classes = 10

    # input_shape = (num_channels, img_rows, img_cols)

    # x_train = x_train.reshape(x_train.shape[0], num_channels, img_rows, img_cols)
    # x_test = x_test.reshape(x_test.shape[0], num_channels ,img_rows, img_cols)

    # x_train = x_train.astype('float32')
    # x_test = x_test.astype('float32')

    # x_train /= 255
    # x_test /= 255

    # x_train = (x_train - 0.5)/0.5
    # x_test = (x_test - 0.5)/0.5

    test_set_array = np.load(os.path.join("../data/test_set", dataset_name, "test_set.npz"))

    # array = np.load("../data/test_set/CIFAR10/test_set.npz")

    x_test = test_set_array["arr_0"]
    y_test = test_set_array["arr_1"]

        # config = mlconfig.load("../configurations/watermark_set.yaml")

    # x_test, y_test = generate_test_set(config["dataset_path"], config["dataset_name"], config["watermarkset_name"], 0, config["normalize_with_imagenet"], config["input_size"], config["batch_size"])

    if dataset_name == "MNIST":
        input_shape = (1, 28, 28)

    if dataset_name == "CIFAR10":

        if attacker_model_architecture == "RN34":
            input_shape = (3, 224, 224)
        
        else:
            input_shape = (3, 32, 32)

    input_shape = (3, 32, 32)

    available_models = {
        "MNIST_L2": models.MNIST_L2,
        "MNIST_L2_DRP03": models.MNIST_L2_DRP03,
        "MNIST_L2_DRP05": models.MNIST_L2_DRP05,
        "MNIST_L5": models.MNIST_L5,
        "MNIST_L5_Latent": models.MNIST_L5_with_latent,
        "MNIST_L5_DRP03": models.MNIST_L5_DRP03,
        "MNIST_L5_DRP05": models.MNIST_L5_DRP05,
        "CIFAR10_BASE": models.CIFAR10_BASE,
        "CIFAR10_BASE_2": models.CIFAR10_BASE_2,
        "CIFAR10_BASE_LATENT": models.CIFAR10_BASE_LATENT,
        "CIFAR10_BASE_DRP03": models.CIFAR10_BASE_DRP03,
        "CIFAR10_BASE_DRP05": models.CIFAR10_BASE_DRP05,
        "RN34" : tv.models.resnet34,
    }

    victim_model = available_models[victim_model_architecture]()
    attacker_model = available_models[attacker_model_architecture]()

    models.load_state(victim_model, model_to_attack_path)


    ## ============================= Victim model ==============================#
    print("Victim model path", model_to_attack_path)


    ## converting pytorch victim model to ART PyTorchClassifier
    classifier_victim = PyTorchClassifier(victim_model, loss =  nn.CrossEntropyLoss(), input_shape= input_shape , nb_classes=10, device_type="gpu", optimizer = t.optim.Adam(victim_model.parameters(), lr=0.001))

    ## ============================ Loading the watemark set to verify ownership at the time of attack ===============================##

    watermark_set_50 = np.load(os.path.join("../data/watermark_set", dataset_name, "watermark_set_50.npz"))

    watermark_set_100 = np.load(os.path.join("../data/watermark_set", dataset_name, "watermark_set_100.npz"))

    watermark_set_250 = np.load(os.path.join("../data/watermark_set", dataset_name, "watermark_set_250.npz"))

    # x_watermark_numpy = watermark_set["arr_0"]
    # y_watermark_numpy = watermark_set["arr_1"]

    # print("Test set")
    # print(x_test[0])
    # print("Watermark set")
    # print(x_watermark_numpy[0])


    return victim_model, classifier_victim, attacker_model, x_test, y_test, watermark_set_50, watermark_set_100, watermark_set_250, input_shape



#======================================================================================================#


def model_extraction_attack(dataset_name, victim_model_architecture,attacker_model_architecture, number_of_queries, model_to_attack_path,
                            num_epochs_to_steal, dropout, optimizer="adam", lr=0.001, weight_decay=0.00):

    

    victim_model, classifier_victim, attacker_model, x_test, y_test, watermark_50, watermark_100, watermark_250, input_shape = setup_victim_attacker(dataset_name, victim_model_architecture, attacker_model_architecture,model_to_attack_path)

    file1 = open(os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER, "_".join((dataset_name, str(victim_model_architecture), model_to_attack_path.split('/')[-1].split(".")[0] + "_logs.txt"))), "w")

    predictions = classifier_victim.predict(x_test)
    # print(np.sum(np.argmax(predictions, axis=1) == y_test))
    acc = np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test)

    print("Just After loading victim model test acc is:", acc)
    file1.write("Just After loading victim model test acc is: " + str(acc) + "\n")

    results = []
    results_adv = []

    for len_steal in number_of_queries:
        indices = np.random.permutation(len(x_test))
        x_steal = x_test[indices[:len_steal]]
        y_steal = y_test[indices[:len_steal]]
        x_test0 = x_test[indices[len_steal:]]
        y_test0 = y_test[indices[len_steal:]]

        if len_steal <= 500:
            x_watermark_numpy = watermark_50["arr_0"]
            y_watermark_numpy = watermark_50["arr_1"]
        
        elif len_steal >500 and len_steal <=5000:
            x_watermark_numpy = watermark_100["arr_0"]
            y_watermark_numpy = watermark_100["arr_1"]

        elif len_steal >5000 and len_steal <=20000:
            x_watermark_numpy = watermark_250["arr_0"]
            y_watermark_numpy = watermark_250["arr_1"]


        attack_catalogue = {
                        "argmax_knockoffNets": KnockoffNets(classifier=classifier_victim,
                                                batch_size_fit=64,
                                                batch_size_query=64,
                                                nb_epochs=num_epochs_to_steal,
                                                nb_stolen=len_steal,
                                                use_probability=False),
                    }

        for name, attack in attack_catalogue.items():

            log.info(f"Attack is {name}")


            classifier_stolen = PyTorchClassifier(attacker_model, loss = nn.CrossEntropyLoss(), input_shape=input_shape , nb_classes=10, device_type="gpu", optimizer = t.optim.Adam(attacker_model.parameters(), lr=0.001))
            classifier_stolen = attack.extract(x_steal, y_steal, thieved_classifier=classifier_stolen, x_watermark=x_watermark_numpy, y_watermark=y_watermark_numpy)

            # classifier_stolen.save("model" + str(name) + str(env.training_ops.dataset_name) + str(len_steal), "../data/models/attacks/knockoff")

            predictions = classifier_stolen.predict(x_test)
            # print(np.sum(np.argmax(predictions, axis=1) == y_test))
            acc = np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test)
            print(f"test acc with {len_steal} is {acc}")
            file1.write(f"Victim model {model_to_attack_path}")
            file1.write(f"test acc with {len_steal} is {acc}\n")
            results.append((name, len_steal, acc))

            # test with adversarial data
            predictions = classifier_stolen.predict(x_watermark_numpy)
            acc_adv = np.sum(np.argmax(predictions, axis=1) == y_watermark_numpy) / len(y_watermark_numpy)
            print(f"Watermark acc with {x_watermark_numpy.shape[0]} is {acc_adv}")

            file1.write(f"Watermark acc with {x_watermark_numpy.shape[0]} is {acc_adv}\n")
            results_adv.append((name, len_steal, acc_adv))

    
    image_save_name = os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER, "_".join((dataset_name, str(victim_model_architecture), model_to_attack_path.split('/')[-1].split(".")[0] + "TestandWatermarkAcc.png")))
    df = pd.DataFrame(results, columns=('Method Name', 'Stealing Dataset Size', 'Accuracy'))
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, group in df.groupby("Method Name"):
        group.plot(1, 2, ax=ax, label="Test acc", linestyle='--', marker='o', color='tab:purple')

    df_adv = pd.DataFrame(results_adv, columns=('Method Name', 'Stealing Dataset Size', 'Accuracy'))
    ax.set_xlabel("Stealing Dataset Size")
    ax.set_ylabel("Stolen Model Test and Adversarial Accuracy")
    for name, group in df_adv.groupby("Method Name"):
        group.plot(1, 2, ax=ax, label="Watermark acc", linestyle='--', marker='o', color='tab:orange')
    plt.savefig(image_save_name)

    file1.close()

    return df, df_adv

       


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="../configurations/knockoffnet/attack_original.yaml")

    args = parser.parse_args()
    print(args)
    config = mlconfig.load(args.config)

    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    RESULTS_PATH = f"../results/attack_original_{now}"
    LOSS_Acc_FOLDER = "losses_acc"
    MODEL_PATH = f"../models/attack_original_{now}"
    DATA_PATH = "../data"

    if not os.path.exists(os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER)):
        os.makedirs(os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER))

    if not os.path.exists(os.path.join(MODEL_PATH)):
        os.makedirs(os.path.join(MODEL_PATH))

    dataset_name = config.dataset_name
    model_to_attack_path = config.model_to_attack_path

    model_to_attack_name = model_to_attack_path.split("/")[-2] ## look here.

    final_df_test = pd.DataFrame(columns=('Method Name', 'Stealing Dataset Size', 'Accuracy'))
    final_df_adv = pd.DataFrame(columns=('Method Name', 'Stealing Dataset Size', 'Accuracy'))

    ## change here if want average for multiple runs.
    for _ in range(1):
        df, df_adv = model_extraction_attack(config.dataset_name, config.victim_model_architecture ,config.attacker_model_architecture,
                                            number_of_queries=[250, 500, 1000, 5000, 10000, 20000],
                                            model_to_attack_path=model_to_attack_path, num_epochs_to_steal=config.epochs_extract, dropout=config.dropout, optimizer=config.optimizer,
                                            lr=config.lr, weight_decay=config.weight_decay)
        
        final_df_test = pd.concat([final_df_test, df])
        final_df_adv = pd.concat([final_df_adv, df_adv])

        TEST_ACC_PATH = os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER, "_".join((dataset_name, str(config.victim_model_architecture), model_to_attack_path.split('/')[-1].split(".")[0] + "df_test_acc.csv")))
        WATERMARK_ACC_PATH = os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER, "_".join((dataset_name, str(config.victim_model_architecture), model_to_attack_path.split('/')[-1].split(".")[0] + "df_watermark_acc.csv")))

        final_df_test.to_csv(TEST_ACC_PATH)
        final_df_adv.to_csv(WATERMARK_ACC_PATH)
