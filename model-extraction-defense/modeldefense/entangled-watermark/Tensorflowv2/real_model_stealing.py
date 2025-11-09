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
from art.estimators.classification import KerasClassifier, TensorFlowV2Classifier
from art.attacks.extraction import CopycatCNN, KnockoffNets
import keras
import pandas as pd
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

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

def model_extraction_attack(dataset_name, trigger_dataset_path, target_dataset_path , attacker_model_architecture, number_of_queries,
                            num_epochs_to_steal, dropout, batch_size, watermark_target_label, probability, optimizer="adam", lr=0.001, weight_decay=0.00,
                            model_to_attack_path='../models/mnist_original_cnn_epochs_25.h5'):
    

    x_train, y_train, x_test, y_test, input_shape, num_classes = data_preprocessing(dataset_name)

    models_mapping = {"mnist_l2": models.MNIST_L2, "MNIST_l2_EWE": models.MNIST_L2_EWE, "MNIST_Plain_2_conv": models.MNIST_Plain_2_conv, "MNIST_Plain_2_conv_real_stealing": models.MNIST_Plain_2_conv_real_stealing, "CIFAR10_BASE_2": models.CIFAR10_BASE_2}

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

    num_epochs = num_epochs_to_steal
    # file1 = open(os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER, adv_data_path_numpy.replace("\\", "/").split("/")[-2],
    #                           "_".join((dataset_name, str(num_epochs),
    #                                     model_to_attack_path.replace("\\", "/").split("/")[-2] + "_logs.txt"))), "w")

    if not os.path.exists(os.path.join(RESULTS_PATH, LOSS_FOLDER, str(probability) + "_" + dataset_name + "_" + attacker_model_architecture + model_to_attack_path.split("/")[-3] + "_" +model_to_attack_path.split("/")[-2])):
        os.makedirs(os.path.join(RESULTS_PATH, LOSS_FOLDER, str(probability) + "_" + dataset_name + "_" + attacker_model_architecture + model_to_attack_path.split("/")[-3] + "_" +model_to_attack_path.split("/")[-2]))


    file1 = open(os.path.join(RESULTS_PATH, LOSS_FOLDER, str(probability) + "_" + dataset_name + "_" + attacker_model_architecture + model_to_attack_path.split("/")[-3] + "_" +model_to_attack_path.split("/")[-2], "logs.txt"), "w")


    model = load_model(model_to_attack_path, compile=False)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer="adam", metrics=['accuracy'])
    print(model.summary())

    test_accuracy = test_model(model, test_dataset, "EWE (trained loaded)", num_classes, watermark_target=None)
    watermark_accuracy = test_model(model, trigger_dataset, "EWE (trained loaded)", num_classes, watermark_target=watermark_target_label)

    print("Test accuracy", test_accuracy)
    print("watermark accuracy", watermark_accuracy)
    print("Just After loading victim model test acc is:", test_accuracy)
    file1.write("Just After loading victim model test acc is: " + str(test_accuracy) + "\n")
    print("Just After loading victim model watermark acc is:", watermark_accuracy)
    file1.write("Just After loading victim model watermark acc is: " + str(watermark_accuracy) + "\n")

    # tf.compat.v1.disable_eager_execution()

    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=probability)

    if attacker_model_architecture == "resnet34":

        classifier_original = TensorFlowV2Classifier(model, nb_classes=10,
                                                     input_shape=(x_train[1], x_train[2], x_train[3]))

    else:

        # classifier_original = KerasClassifier(model, clip_values=(0, 1), use_logits=False, output_layer=3)
        classifier_original = TensorFlowV2Classifier(model, nb_classes=10,
                                                     input_shape=(x_train[1], x_train[2], x_train[3]))

    im_shape = x_train[0].shape

    results = []
    results_adv = []

    for len_steal in number_of_queries:

        # y_train = y_train[np.argmax(y_train, axis=1)==7]
        # y_train = y_train[np.argmax(y_train, axis=1)==8]
        # x_train = x_train[np.argmax(y_train, axis=1)==7]
        # x_train = x_train[np.argmax(y_train, axis=1)==8]
        indices = np.random.permutation(len(x_train))
        # indices = np.where((np.argmax(y_train, axis=1) == 7) | (np.argmax(y_train, axis=1) == 8))[0]
        x_steal = x_train[indices[:len_steal]]
        y_steal = y_train[indices[:len_steal]]
        x_test0 = x_train[indices[len_steal:]]
        y_test0 = y_train[indices[len_steal:]]

        print("y_steal", np.unique(np.argmax(y_steal, axis=1), return_counts=True))

        attack_catalogue = {"KnockoffNet": KnockoffNets(classifier=classifier_original,
                                                        batch_size_fit=64,
                                                        batch_size_query=64,
                                                        nb_epochs=num_epochs,
                                                        nb_stolen=len_steal,
                                                        use_probability=probability)}

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

            if attacker_model_architecture == "resnet34":
                model_name, model_stolen = models_mapping[attacker_model_architecture]().call(input_shape)

                # model_stolen.compile(loss=keras.losses.categorical_crossentropy, optimizer="adam", metrics=['accuracy'])

            else:
                if dropout:
                    model_name, model_stolen = models_mapping[attacker_model_architecture](dropout)
                else:
                    model_name, model_stolen = models_mapping[attacker_model_architecture]()

                model_stolen.compile(loss=keras.losses.categorical_crossentropy, optimizer="adam", metrics=['accuracy'])

            if attacker_model_architecture == "resnet34":

                classifier_stolen = TensorFlowV2Classifier(model_stolen, nb_classes=10, loss_object=loss_object,
                                                           input_shape=input_shape, channels_first=False,
                                                           train_step=train_step)

            else:

                # classifier_stolen = KerasClassifier(model_stolen, clip_values=(0, 1), use_logits=False)
                classifier_stolen = TensorFlowV2Classifier(model_stolen, nb_classes=10, loss_object=loss_object,
                                                           input_shape=input_shape, channels_first=False,
                                                           train_step=train_step)

            classifier_stolen = attack.extract(x_steal, y_steal, thieved_classifier=classifier_stolen)
            acc = classifier_stolen.model.evaluate(x_test, y_test)[1]
            print(f"test acc with {len_steal} is {acc}")
            # file1.write(f"Victim model {model_to_attack_path}")
            file1.write(f"test acc with {len_steal} is {acc}\n")
            results.append((name, len_steal, acc))

            # test with adversarial data
            # idx = np.random.randint(x_adv.shape[0], size=1000)
            # acc_adv = classifier_stolen.model.evaluate(trigger_numpy["arr_0"], trigger_numpy["arr_1"])[1]
            # print(f"adv acc with {len_steal} is {acc_adv}")
            # # file1.write(f"adv acc with {len_steal} is {acc_adv}\n")
            # results_adv.append((name, len_steal, acc_adv))

            predictions = classifier_stolen.predict(trigger_numpy["arr_0"], batch_size=64)
            print("Argmax predictions {}".format(np.argmax(predictions, axis=1)))
            file1.write(f"predictions {np.argmax(predictions, axis=1)}")
            predictions_list = np.argmax(predictions, axis=1).tolist()
            count = len([i for i in predictions_list if i==watermark_target_label])

            acc_adv = count / len(predictions_list)
            print("Accuracy on watermark examples: {}".format(acc_adv))
            file1.write(f"adv acc with {len_steal} is {acc_adv}\n")
            results_adv.append((name, len_steal, acc_adv))
            # mlflow.log_metric("watermark acc"+str(len_steal), accuracy)
            # watermark_acc.append(accuracy)
    

            # classifier_stolen.model.save(
            #     os.path.join(MODEL_PATH, adv_data_path_numpy.replace("\\", "/").split("/")[-2],
            #                  "_".join((dataset_name, str(len_steal), str(num_epochs),
            #                            adv_data_path_numpy.replace("\\", "/").split("/")[-1].split(".npz")[
            #                                0] + ".h5"))))

    image_save_name = os.path.join(RESULTS_PATH, LOSS_FOLDER, str(probability) + "_" + dataset_name + "_" + attacker_model_architecture + model_to_attack_path.split("/")[-3] + "_" +model_to_attack_path.split("/")[-2], "TestandWatermarkAcc.png")

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

    # mlflow.log_artifact(
    #     os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER, adv_data_path_numpy.replace("\\", "/").split("/")[-2],
    #                  "_".join((dataset_name, str(num_epochs),
    #                            model_to_attack_path.replace("\\", "/").split("/")[-2] + "_logs.txt"))), "logs.txt")
    # mlflow.log_artifact(os.path.join(image_save_name), "TestandWatermarkAcc.png")

    return df, df_adv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="configs/knockoffattack_finetuned.yaml")

    args = parser.parse_args()
    print(args)
    config = mlconfig.load(args.config)

    dataset_name = config.dataset_name

    RESULTS_PATH = f"results/attack_{now}"
    LOSS_FOLDER = "losses"
    MODEL_PATH = f"models/attack_{now}"

    if not os.path.exists(os.path.join(RESULTS_PATH, LOSS_FOLDER)):
        os.makedirs(os.path.join(RESULTS_PATH, LOSS_FOLDER))


    for model_to_attack_folder in os.listdir(os.path.join("models", "ewe_training_retrain10-11-2023")):
        
        model_to_attack = os.path.join("models", "ewe_training_retrain10-11-2023", model_to_attack_folder, "ewe_model.h5")

        target = model_to_attack.split("/")[-2].split("_")[0][-1]
        source = model_to_attack.split("/")[-2].split("_")[2]

        if int(target) in [0,1,2,3,4,5,6,7,8,9] and int(source) in [0,1,2,3,4,5,6,7,8,9]:

            print("her")

            trigger_set_path = "data/fgsm/cifar10CIFAR10_BASE_2_EWE/"+ str(model_to_attack_folder.rsplit("_",2)[0].split("distrib_")[-1].replace("CIFAR10_BASE", "")) +"/"+ str(model_to_attack_folder.split("_")[0][-1]) +  str(model_to_attack_folder.split("_")[2]) + "/fgsm_0.0085original_16-10-2023cifar10_30_CIFAR10_BASE_2Original_checkpoint_best.h5_trigger.npz"

            target_set_path = "data/fgsm/cifar10CIFAR10_BASE_2_EWE/"+ str(model_to_attack_folder.rsplit("_",2)[0].split("distrib_")[-1].replace("CIFAR10_BASE", "")) +"/"+ str(model_to_attack_folder.split("_")[0][-1]) +  str(model_to_attack_folder.split("_")[2]) + "/target_0.0085original_16-10-2023cifar10_30_CIFAR10_BASE_2Original_checkpoint_best.h5_target.npz"

            final_df_test = pd.DataFrame(columns=('Method Name', 'Stealing Dataset Size', 'Accuracy'))
            final_df_adv = pd.DataFrame(columns=('Method Name', 'Stealing Dataset Size', 'Accuracy'))              

            for _ in range(2):

                
                df, df_adv = model_extraction_attack(dataset_name, trigger_set_path, target_set_path , config["attacker_model_architecture"], [250, 500, 1000, 5000, 10000, 20000, 25000, 40000, 50000],
                                    config["epochs_extract"], config["dropout"], config["batch_size"], int(model_to_attack_folder.split("_")[0]), config["probability"], optimizer=config["optimizer"], lr=config["lr"], weight_decay=config["weight_decay"],
                                    model_to_attack_path= model_to_attack)
                
                final_df_test = pd.concat([final_df_test, df])
                final_df_adv = pd.concat([final_df_adv, df_adv])


        
            TEST_ACC_PATH = os.path.join(RESULTS_PATH, LOSS_FOLDER, str(config["probability"]) + "_" + dataset_name + "_" + config["attacker_model_architecture"] + model_to_attack.split("/")[-3] + "_" + model_to_attack.split("/")[-2], "df_test_acc.csv")

            WATERMARK_ACC_PATH = os.path.join(RESULTS_PATH, LOSS_FOLDER, str(config["probability"]) + "_" + dataset_name + "_" + config["attacker_model_architecture"] + model_to_attack.split("/")[-3] + "_" + model_to_attack.split("/")[-2], "df_watermark_acc.csv")

            final_df_test.to_csv(TEST_ACC_PATH)
            final_df_adv.to_csv(WATERMARK_ACC_PATH)

    
