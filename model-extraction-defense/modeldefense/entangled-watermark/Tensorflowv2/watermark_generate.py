import tensorflow as tf
import numpy as np
import argparse
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pickle
import functools
import random

import models_new as models
import matplotlib.pyplot as plt

from utils_new import validate_watermark
from trigger_new import trigger_generation
from models_training_new import ewe_train, plain_model
from watermark_dataset import create_wm_dataset
import logging
from datetime import datetime
import mlconfig
import keras
from keras.datasets import mnist, cifar10
from keras.models import load_model

np.random.seed(42)
tf.random.set_seed(0)
random.seed(42)

now = datetime.now().strftime("%d-%m-%Y")

# RESULTS_PATH = f"results/watermark_{now}"
# RESULTS_FOLDER_TRIGGERS = "triggers"
# LOSS_FOLDER = "losses"
# MODEL_PATH = f"models/original_{now}"

# RESULTS_PATH = "results/scratch"
# LOSS_FOLDER = "losses"
# DATA_PATH = "data"
# TRIGGER_PATH = "trigger"
# MODELS_SAVE_PATH = "models/scratch"

def data_preprocessing(dataset_name):
    if dataset_name == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        img_rows, img_cols, num_channels = 28, 28, 1
        num_classes = 10

    elif dataset_name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = np.squeeze(y_train)
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

def generate_watermark(dataset_name, watermark_target_label, distribution, watermark_source_label, batch_size, trigger_set_from_model_path, temperatures, threshold, maxiter, w_lr, model_architecture, lr, epochs, factors):

    models_mapping = {"mnist_l2": models.MNIST_L2, "MNIST_l2_EWE": models.MNIST_L2_EWE, "CIFAR10_BASE_2_EWE": models.CIFAR10_BASE_2_EWE}

    x_train, y_train, x_test, y_test, input_shape, num_classes = data_preprocessing(dataset_name)

    target_data = x_train[y_train == watermark_target_label]

    source_data, exclude_x_data_wm, exclude_y_data_wm = create_wm_dataset(distribution, x_train, y_train, watermark_source_label, watermark_target_label, dataset_name)

    if exclude_x_data_wm is not None:
        exclude_x_data = exclude_x_data_wm ## have to take call on this.
        exclude_y_data = exclude_y_data_wm

    trigger = np.concatenate([source_data] * (target_data.shape[0] // source_data.shape[0] + 1), 0)[
                    :target_data.shape[0]]
    
    half_batch_size = int(batch_size / 2)
    
    w_label = np.concatenate([np.ones(half_batch_size), np.zeros(half_batch_size)], 0)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    w_0 = np.zeros([batch_size])
    # trigger_label = np.zeros([batch_size, num_classes])
    # trigger_label[:, watermark_target_label] = 1 ## setting the trigger label 

    model_name, model = models_mapping[model_architecture]()

    # _, already_trained_model =  models.MNIST_L2()

    already_trained_model = load_model(trigger_set_from_model_path)

    for layer_watermarked, layer_original in zip(model.layers[1:], already_trained_model.layers):
        # print(layer_finetune)
        # print(layer_original)
        weight_layer = layer_original.get_weights()
        layer_watermarked.set_weights(weight_layer)

    # train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    # train_dataset = train_dataset.batch(batch_size)
        
    # test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    # test_dataset = test_dataset.batch(batch_size)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # loss_epoch = []
    # for epoch in range(epochs):

    #     loss_batch = []
    #     for batch, (x_batch_train, y_batch_train) in enumerate(train_dataset):

    #         # x_batch_train = model

    #         with tf.GradientTape() as tape:

    #             w_0 = np.zeros([x_batch_train.shape[0]])
    #             prediction = model(x_batch_train)

    #             loss_value, snnl_loss = models.combined_loss(prediction, y_batch_train, w_0, temperatures, factors)

    #         grads = tape.gradient(loss_value, model.trainable_weights)
            
    #         optimizer.apply_gradients(zip(grads, model.trainable_weights))


    #         loss_batch.append(loss_value)

    #     print("Loss", np.mean(loss_batch))
    np.savez(os.path.join(DATA_PATH, dataset_name + model_architecture, distribution, str(watermark_target_label) +  str(watermark_source_label), "target_" + str(w_lr) + trigger_set_from_model_path.split("/")[-3] + trigger_set_from_model_path.split("/")[-2] + trigger_set_from_model_path.split("/")[-1] + "_target.npz"), target_data)

    np.savez(os.path.join(DATA_PATH, dataset_name + model_architecture, distribution, str(watermark_target_label) +  str(watermark_source_label), "orig_" + str(w_lr) + trigger_set_from_model_path.split("/")[-3] + trigger_set_from_model_path.split("/")[-2] + trigger_set_from_model_path.split("/")[-1] + "_orig.npz"), trigger)

    trigger_dataset = tf.data.Dataset.from_tensor_slices((trigger))
    trigger_dataset = trigger_dataset.batch(half_batch_size)

    target_dataset = tf.data.Dataset.from_tensor_slices((target_data))
    target_dataset = target_dataset.batch(half_batch_size)

    if distribution == "in":
        trigger_grad = []

        for batch, (trigger_batch_train, target_batch_train) in enumerate(zip(trigger_dataset, target_dataset)):

            with tf.GradientTape() as tape:

                x_batch = tf.concat([trigger_batch_train, target_batch_train], 0)

                tape.watch(x_batch)

                prediction = model(x_batch)
                w_label = np.concatenate([np.ones(trigger_batch_train.shape[0]), np.zeros(target_batch_train.shape[0])], 0)

                snnl_losses = models.snnl_loss(prediction, w_label, temperatures)
                final_snnl_loss = snnl_losses[0] + snnl_losses[1] + snnl_losses[2]


            snnl_grad = tape.gradient(final_snnl_loss, x_batch)
            trigger_grad.append(snnl_grad[:half_batch_size])

        # logger.info(trigger_grad)
        avg_grad = np.average(np.concatenate(trigger_grad), 0)
        down_sample = np.array([[np.sum(avg_grad[i: i + 3, j: j + 3]) for i in range(input_shape[0] - 2)] for j in range(input_shape[1] - 2)])
        w_pos = np.unravel_index(down_sample.argmin(), down_sample.shape)
        trigger[:, w_pos[0]:w_pos[0] + 3, w_pos[1]:w_pos[1] + 3, 0] = 1

    elif distribution == "in_with_fgsm" or distribution == "out_with_trigger":
        trigger_grad = []

        for batch, (trigger_batch_train, target_batch_train) in enumerate(zip(trigger_dataset, target_dataset)):

            with tf.GradientTape() as tape:

                target_batch_train = tf.cast(target_batch_train, tf.float32)
                trigger_batch_train = tf.cast(trigger_batch_train, tf.float32)

                x_batch = tf.concat([trigger_batch_train, target_batch_train], 0)

                tape.watch(x_batch)

                prediction = model(x_batch)
                w_label = np.concatenate([np.ones(trigger_batch_train.shape[0]), np.zeros(target_batch_train.shape[0])], 0)

                snnl_losses = models.snnl_loss(prediction, w_label, temperatures)
                final_snnl_loss = snnl_losses[0] + snnl_losses[1] + snnl_losses[2]


            snnl_grad = tape.gradient(final_snnl_loss, x_batch)
            trigger_grad.append(snnl_grad[:half_batch_size])

        # logger.info(trigger_grad)
        avg_grad = np.average(np.concatenate(trigger_grad), 0)
        down_sample = np.array([[np.sum(avg_grad[i: i + 3, j: j + 3]) for i in range(input_shape[0] - 2)] for j in range(input_shape[1] - 2)])
        w_pos = np.unravel_index(down_sample.argmin(), down_sample.shape)
        trigger[:, w_pos[0]:w_pos[0] + 3, w_pos[1]:w_pos[1] + 3, 0] = 1

        trigger_dataset = tf.data.Dataset.from_tensor_slices((trigger))
        trigger_dataset = trigger_dataset.batch(half_batch_size)

        w_num_batch = target_data.shape[0] // batch_size * 2

        trigger = trigger_generation(model, trigger, trigger_dataset, target_dataset, watermark_target_label, num_classes, batch_size, threshold, maxiter, w_lr, w_num_batch, temperatures)

    else:
        w_pos = [-1, -1]

        w_num_batch = target_data.shape[0] // batch_size * 2

        trigger = trigger_generation(model, trigger, trigger_dataset, target_dataset, watermark_target_label, num_classes, batch_size, threshold, maxiter, w_lr, w_num_batch, temperatures)

    np.savez(os.path.join(DATA_PATH, dataset_name + model_architecture, distribution, str(watermark_target_label) +  str(watermark_source_label), "fgsm_" + str(w_lr) + trigger_set_from_model_path.split("/")[-3] + trigger_set_from_model_path.split("/")[-2] + trigger_set_from_model_path.split("/")[-1] + "_trigger.npz"),trigger)

    



    



    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="configs/watermark_set.yaml")

    args = parser.parse_args()
    print(args)
    config = mlconfig.load(args.config)

    dataset_name = config.dataset_name
    trigger_set_from_model_path = config.trigger_set_from_model_path

    DATA_PATH = "data/fgsm"

    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    if not os.path.exists(os.path.join(DATA_PATH, dataset_name + config["model_architecture"], config["distribution"], str(config["target"])+  str(config["source"]))):
        os.makedirs(os.path.join(DATA_PATH, dataset_name + config["model_architecture"], config["distribution"], str(config["target"])+  str(config["source"])))

    generate_watermark(dataset_name, config["target"], config["distribution"], config["source"], config["batch_size"], trigger_set_from_model_path, config["temperatures"],
                       config["threshold"],config["maxiter"], config["w_lr"], config["model_architecture"], config["t_lr"], config["epochs"], config["factors"])

    