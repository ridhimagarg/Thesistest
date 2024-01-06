import tensorflow as tf
import numpy as np
import argparse
import os
from utils_new import test_model

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pickle
import functools
import random

import models_new as md
import matplotlib.pyplot as plt

from utils_new import validate_watermark
from trigger_new import trigger_generation
from models_training_new import ewe_train, plain_model
from watermark_dataset import create_wm_dataset
from mlflow import log_metric, log_param, log_params, log_artifacts
import mlflow
# from train_new import train

from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, ReLU
import tensorflow_datasets as tfds


seed = 0
random.seed(seed)
np.random.seed(seed)
tf.compat.v1.random.set_random_seed(seed)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("EWEFinetuning")
RESULTS_PATH = "results/finetuning"
LOSS_FOLDER = "losses"
DATA_PATH = "data"
TRIGGER_PATH = "trigger_finetuning"
MODELS_SAVE_PATH = "models/finetuning"


def train_customer_model(x_train, y_train, x_test, y_test, epochs, dataset, batch_size, model, model_save_name):

    params = {"epochs": epochs, "dataset":dataset}

    # model = md.Plain_2_conv()
    # model = md.Plain_2_conv_Keras()
    model_save_name = dataset + "Original" + model_save_name
    experiment_name = dataset + "Customer_" + model_save_name

    with mlflow.start_run(run_name=experiment_name):

        for param, param_val in params.items():
            mlflow.log_param(param, param_val)


        height = x_train[0].shape[0]
        width = x_train[0].shape[1]
        try:
            channels = x_train[0].shape[2]
        except:
            channels = 1
        num_class = len(np.unique(y_train))

        y_train = tf.keras.utils.to_categorical(y_train, num_class)
        y_test = tf.keras.utils.to_categorical(y_test, num_class)

        num_batch = x_train.shape[0] // batch_size 

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        test_dataset = test_dataset.shuffle(buffer_size=1024).batch(batch_size)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-5)


        loss_epoch = []
        for epoch in range(epochs):

            loss_batch = []
            for batch, (x_batch_train, y_batch_train) in enumerate(train_dataset):

                with tf.GradientTape() as tape:

                    prediction = model(x_batch_train)

                    loss_value = md.ce_loss(prediction, y_batch_train)

                grads = tape.gradient(loss_value, model.trainable_weights)
                
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

                loss_batch.append(loss_value)

            print(f"Loss at {epoch} is {np.mean(loss_batch)}")

            loss_epoch.append(np.mean(loss_batch))

        
        plt.figure(figsize=(5,5))
        plt.plot(list(range(epochs)), loss_epoch, label="Train data acc original training", linestyle='--', marker='o', color='tab:orange')
        plt.xlabel("epochs")
        plt.ylabel("CE loss")
        plt.legend()

        if not os.path.exists(os.path.join(RESULTS_PATH, LOSS_FOLDER)):
            os.makedirs(os.path.join(RESULTS_PATH, LOSS_FOLDER))

        plt.savefig(os.path.join(RESULTS_PATH, LOSS_FOLDER, dataset + "Trainoriginalloss.png"))

        mlflow.log_artifact(os.path.join(RESULTS_PATH, LOSS_FOLDER , dataset + "Trainoriginalloss.png"), "Celossoriginal")

        model.save(os.path.join(MODELS_SAVE_PATH, model_save_name))

        test_acc = test_model(model, test_dataset, "Customer Model", num_class, watermark_target=None)

        mlflow.log_metric("Customer Original Test Acc",test_acc)

        return test_acc



def train_customer_model2(x_train, y_train, x_test, y_test, epochs, dataset, batch_size):

    params = {"epochs": epochs, "dataset":dataset}

    input = Input(shape=(28,28,1))
    conv1 = Conv2D(filters=32, kernel_size=5, activation=None)(input)
    relu1 = ReLU()(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2), strides=2)(relu1)
    drop1 = Dropout(0.5)(pool1)
    conv2 = Conv2D(filters=64, kernel_size=3, activation=None)(drop1)
    relu2 = ReLU()(conv2)
    pool2 = MaxPooling2D(pool_size=(2,2), strides=2)(relu2)
    drop2 = Dropout(0.5)(pool2)
    flatten1 = Flatten()(drop2)
    dense1 = Dense(128, activation=None)(flatten1)
    drop3 = Dropout(0.5)(dense1)
    relu3 = ReLU()(drop3)
    output = Dense(10, activation=None)(relu3)

    model = Model(inputs=input, outputs=output)

    experiment_name = dataset+"Plain_2_conv"

    # with mlflow.start_run(run_name=experiment_name):

    for param, param_val in params.items():
        mlflow.log_param(param, param_val)


    height = x_train[0].shape[0]
    width = x_train[0].shape[1]
    try:
        channels = x_train[0].shape[2]
    except:
        channels = 1
    num_class = len(np.unique(y_train))

    y_train = tf.keras.utils.to_categorical(y_train, num_class)
    y_test = tf.keras.utils.to_categorical(y_test, num_class)

    num_batch = x_train.shape[0] // batch_size 

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.shuffle(buffer_size=1024).batch(batch_size)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-5)


    loss_epoch = []
    for epoch in range(1):

        loss_batch = []
        for batch, (x_batch_train, y_batch_train) in enumerate(train_dataset):

            with tf.GradientTape() as tape:

                prediction = model(x_batch_train)

                loss_value = md.ce_loss(prediction, y_batch_train)

            grads = tape.gradient(loss_value, model.trainable_weights)
            
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            loss_batch.append(loss_value)

        print(f"Loss at {epoch} is {np.mean(loss_batch)}")

        loss_epoch.append(np.mean(loss_batch))

    
    # plt.figure(figsize=(5,5))
    # plt.plot(list(range(epochs)), loss_epoch, label="Train data acc original training", linestyle='--', marker='o', color='tab:orange')
    # plt.xlabel("epochs")
    # plt.ylabel("CE loss")
    # plt.legend()
    # plt.savefig("Trainoriginalloss.png")

    # mlflow.log_artifact("Trainoriginalloss.png", "Celossoriginal")

    # model.save("models/finetuning"+str("OriginalPlain2Conv"))

    def pop_layer(model):

        print(model.layers)

        if not model.outputs:
            raise Exception('Sequential model cannot be popped: model is empty.')
        model.layers.pop()
        if not model.layers:
            model.outputs = []
            model.inbound_nodes = []
            model.outbound_nodes = []
        else:
            model.layers[-2].outbound_nodes = []
            model.outputs = [model.layers[-2].output]


        return model

    intermediate_layer_model = model.layers[-7].output


    # print(input_shape)
    # new_input = Input(shape= (5,5,64))
    conv1 = Conv2D(filters=64, kernel_size=3, activation=None)(intermediate_layer_model)
    relu1 = ReLU()(conv1)
    print(relu1.shape)
    # pool1 = MaxPooling2D(pool_size=(2,2), strides=2)(relu1)
    drop1 = Dropout(0.5)(relu1)
    # conv2 = Conv2D(filters=64, kernel_size=3, activation=None)(drop1)
    # relu2 = ReLU()(conv2)
    # pool2 = MaxPooling2D(pool_size=(2,2), strides=2)(relu2)
    # drop2 = Dropout(0.5)(pool2)
    print(drop1.shape)
    flatten1 = Flatten()(drop1)

    print(flatten1.shape)
    dense1 = Dense(128, activation=None)(flatten1)
    drop3 = Dropout(0.5)(dense1)
    relu3 = ReLU()(drop3)
    output = Dense(10, activation=None)(relu3)

    ewe_model = Model(inputs=input, outputs=output)

    print(ewe_model.summary())

    for layer in ewe_model.layers[:5]:
        layer.trainable = False

    print(ewe_model.layers)

    print(ewe_model.summary())

    # optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-5)

    # loss_epoch = []
    # for epoch in range(epochs):

    #     loss_batch = []
    #     for batch, (x_batch_train, y_batch_train) in enumerate(train_dataset):

    #         with tf.GradientTape() as tape:

    #             intermediate_output = intermediate_layer_model(x_batch_train)

    #             prediction = ewe_model(intermediate_output)

    #             loss_value = md.ce_loss(prediction, y_batch_train)

    #         grads = tape.gradient(loss_value, ewe_model.trainable_weights)
            
    #         optimizer.apply_gradients(zip(grads, ewe_model.trainable_weights))

    #         loss_batch.append(loss_value)

    #     print(f"Loss at {epoch} is {np.mean(loss_batch)}")

    #     loss_epoch.append(np.mean(loss_batch))




def embed_watermark_more_training(x_train, y_train, x_test, y_test, epochs, w_epochs, lr, n_w_ratio, factors,
          temperatures, watermark_source, watermark_target, batch_size, w_lr, threshold, maxiter, shuffle, temp_lr,
          dataset, distribution, verbose, ewe_model, ewe_model_name, customer_model_name="OriginalPlain2Conv"):

    model = tf.keras.models.load_model(os.path.join(MODELS_SAVE_PATH, customer_model_name))
    # ewe_model = md.EWE_2_conv()

    # model_name = str(ewe_model).split(" ")[0][1:].replace(".", "_")
    experiment_name = dataset+distribution+str(watermark_source)+str(watermark_target)+ str(ewe_model_name) + "Watermark_moretraining"


    with mlflow.start_run(run_name=experiment_name):


        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

        for batch, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            ewe_model(x_batch_train)
        ## setting the weights of the ewe model from customer already trained model.
        for layer_ewe,layer_customer in zip(ewe_model.layers , model.layers):
            weight_layer = layer_customer.get_weights()
            layer_ewe.set_weights(weight_layer)

        
        print(model.trainable_weights[0])
        print("after")
        print(ewe_model.trainable_weights[0])

        train(x_train, y_train, x_test, y_test, epochs, w_epochs, lr, n_w_ratio, factors,
            temperatures, watermark_source, watermark_target, batch_size, w_lr, threshold, maxiter, shuffle, temp_lr,
            dataset, distribution, verbose, ewe_model, ewe_model_name)


def train(x_train, y_train, x_test, y_test, epochs, w_epochs, lr, n_w_ratio, factors,
          temperatures, watermark_source, watermark_target, batch_size, w_lr, threshold, maxiter, shuffle, temp_lr,
          dataset, distribution, verbose, model, ewe_model_name, customer_model=None): ## model -> ewe model
    

    params = {"epochs": epochs, "w_epochs": w_epochs, "lr":lr, "n_w_ratio":n_w_ratio, "watermark_source":watermark_source, "watermark_target":watermark_target, "batch_size":batch_size,
              "w_lr":w_lr, "threshold":threshold, "maxiter":maxiter, "shuffle":shuffle, "temp_lr":temp_lr, "dataset":dataset, "distribution":distribution, "factors": factors, "temperatures": temperatures}


    ##--------------------- check once variable names before running the experiment ------------------------## (can create arguments for these.)
    extraction_model = md.Plain_2_conv() ## have to change here everytime.


    # victim_model_name = "EWE_Plain_2_conv_Customer"+ "Victim_" + dataset + "_model"
    victim_model_path = os.path.join(MODELS_SAVE_PATH, ewe_model_name + "Victim_" + dataset + "_model")
    extracted_model_path = os.path.join(MODELS_SAVE_PATH, ewe_model_name + "Extracted_" + dataset + "_model")


    for param, param_val in params.items():
        mlflow.log_param(param, param_val)



    height = x_train[0].shape[0]
    width = x_train[0].shape[1]
    try:
        channels = x_train[0].shape[2]
    except:
        channels = 1
    num_class = len(np.unique(y_train))
    half_batch_size = int(batch_size / 2)

    target_data = x_train[y_train == watermark_target]
    exclude_x_data = x_train[y_train != watermark_target]
    exclude_y_data = y_train[y_train != watermark_target]

    # define the dataset and class to sample watermarked data
    target_data = x_train[y_train == watermark_target]
    exclude_x_data = x_train[y_train != watermark_target]
    exclude_y_data = y_train[y_train != watermark_target]

    ## creating the watermark dataset
    source_data, exclude_x_data_wm, exclude_y_data_wm = create_wm_dataset(distribution, x_train, y_train, watermark_source, watermark_target, dataset, height, width, channels)


    if exclude_x_data_wm is not None:
        exclude_x_data = exclude_x_data_wm
        exclude_y_data = exclude_y_data_wm
    

    trigger = np.concatenate([source_data] * (target_data.shape[0] // source_data.shape[0] + 1), 0)[
                :target_data.shape[0]]
    
    w_label = np.concatenate([np.ones(half_batch_size), np.zeros(half_batch_size)], 0)
    y_train = tf.keras.utils.to_categorical(y_train, num_class)
    y_test = tf.keras.utils.to_categorical(y_test, num_class)
    index = np.arange(y_train.shape[0])
    w_0 = np.zeros([batch_size])
    trigger_label = np.zeros([batch_size, num_class])
    trigger_label[:, watermark_target] = 1 ## setting the trigger label 

    num_batch = x_train.shape[0] // batch_size ## whole data no. of batches 
    w_num_batch = target_data.shape[0] // batch_size * 2 ## watermark no. of batches, since trigger is same shape as target data
    num_test = x_test.shape[0] // batch_size

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.shuffle(buffer_size=1024).batch(batch_size)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-5)
    # model = tf.keras.models.load_model('models/intial_mnist_model')

    loss_epoch = []
    for epoch in range(epochs):

        loss_batch = []
        for batch, (x_batch_train, y_batch_train) in enumerate(train_dataset):

            with tf.GradientTape() as tape:

                w_0 = np.zeros([x_batch_train.shape[0]])

                if customer_model:

                    intermediate_output = customer_model(x_batch_train)

                    prediction = model(intermediate_output)
                
                else:
                    prediction = model(x_batch_train)

                loss_value, snnl_loss = md.combined_loss(prediction, y_batch_train, w_0, temperatures, factors)

            grads = tape.gradient(loss_value, model.trainable_weights)
            
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            loss_batch.append(loss_value)

        print(f"Loss at {epoch} is {np.mean(loss_batch)}")

        loss_epoch.append(np.mean(loss_batch))


    plt.figure(figsize=(5,5))
    plt.plot(list(range(epochs)), loss_epoch, label="Train data acc initial EWE training", linestyle='--', marker='o', color='tab:orange')
    plt.xlabel("epochs")
    plt.ylabel("Combined loss")
    plt.legend()

    if not os.path.exists(os.path.join(RESULTS_PATH, LOSS_FOLDER)):
        os.makedirs(os.path.join(RESULTS_PATH, LOSS_FOLDER))

    plt.savefig(os.path.join(RESULTS_PATH, LOSS_FOLDER, dataset + "CombginelossInitialEWE.png"))

    mlflow.log_artifact(os.path.join(RESULTS_PATH, LOSS_FOLDER, dataset + "CombginelossInitialEWE.png"), "Combinedlossinitialewe")


    trigger_dataset = tf.data.Dataset.from_tensor_slices((trigger))
    trigger_dataset = trigger_dataset.shuffle(buffer_size=1024).batch(half_batch_size)

    target_dataset = tf.data.Dataset.from_tensor_slices((target_data))
    target_dataset = target_dataset.shuffle(buffer_size=1024).batch(half_batch_size)


    if distribution == "in":
        trigger_grad = []

        for batch, (trigger_batch_train, target_batch_train) in enumerate(zip(trigger_dataset, target_dataset)):

            with tf.GradientTape() as tape:

                x_batch = tf.concat([trigger_batch_train, target_batch_train], 0)

                tape.watch(x_batch)

                prediction = model(x_batch)
                w_label = np.concatenate([np.ones(trigger_batch_train.shape[0]), np.zeros(target_batch_train.shape[0])], 0)

                snnl_losses = md.snnl_loss(prediction, w_label, temperatures)
                final_snnl_loss = snnl_losses[0] + snnl_losses[1] + snnl_losses[2]


            snnl_grad = tape.gradient(final_snnl_loss, x_batch)
            trigger_grad.append(snnl_grad[:half_batch_size])

        print(trigger_grad)
        avg_grad = np.average(trigger_grad, 0)
        down_sample = np.array([[np.sum(avg_grad[i: i + 3, j: j + 3]) for i in range(height - 2)] for j in range(width - 2)])
        w_pos = np.unravel_index(down_sample.argmin(), down_sample.shape)
        trigger[:, w_pos[0]:w_pos[0] + 3, w_pos[1]:w_pos[1] + 3, 0] = 1
        plt.imshow(trigger[0][:,:,0], cmap="gray_r")
        plt.show()
        path = "indisttrigger.png"
        plt.savefig(path)
        mlflow.log_artifact(path, "in distribution trigger")

    else:
        w_pos = [-1, -1]


    ##--------------------------------------------- Trigger generation -----------------------------------------##
    print(watermark_target)
    trigger = trigger_generation(model, trigger, trigger_dataset, target_dataset, watermark_target, num_class, batch_size, threshold, maxiter, w_lr, w_num_batch, temperatures, customer_model)

    if not os.path.exists(os.path.join(DATA_PATH, TRIGGER_PATH)):
        os.makedirs(os.path.join(DATA_PATH, TRIGGER_PATH))

    np.savez(os.path.join(DATA_PATH, TRIGGER_PATH, dataset+"_trigger.npz"),trigger)
    trigger_dataset = tf.data.Dataset.from_tensor_slices((trigger))
    trigger_dataset = trigger_dataset.shuffle(buffer_size=1024).batch(half_batch_size)
    # tf.data.Dataset.save(trigger_dataset, "data/trigger/")


    ##----------------------------------------- EWE Model Training ------------------------------------------##
    
    model = ewe_train(model, train_dataset, trigger_dataset, target_dataset, test_dataset, victim_model_path, w_epochs, num_batch, w_num_batch, n_w_ratio, factors, optimizer, watermark_target, num_class, batch_size, temp_lr, temperatures, dataset, RESULTS_PATH, LOSS_FOLDER, customer_model)



    ##------------------------------------------------- EWE Extracted Model ------------------------------ ##
    # Attack
    extracted_label = []
    extracted_data = []

    for batch, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        
        if customer_model:
            intermediate_output = customer_model(x_batch_train)
            output = model(intermediate_output)
        else:
            output = model(x_batch_train)
        # print("output", output)
        if isinstance(output, list):
            output = output[-1]
        extracted_label.append(output == np.max(output, 1, keepdims=True))
        extracted_data.append(x_batch_train)

    extracted_label = np.concatenate(extracted_label, 0)
    extracted_data = np.concatenate(extracted_data, 0)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-5)

    extracted_dataset = tf.data.Dataset.from_tensor_slices((extracted_data, extracted_label))
    extracted_dataset = extracted_dataset.shuffle(buffer_size=1024).batch(batch_size)

    extraction_flag = True
    print(extraction_model.trainable_weights)
    extracted_model = plain_model(extraction_model, "EWE (Extraction)", extracted_dataset, test_dataset , extraction_flag, epochs, w_epochs, optimizer, num_class, trigger_dataset, watermark_target, target_dataset, extracted_model_path, dataset, RESULTS_PATH, LOSS_FOLDER)



def embed_watermark_feature_extraction(x_train, y_train, x_test, y_test, epochs, w_epochs, lr, n_w_ratio, factors,
          temperatures, watermark_source, watermark_target, batch_size, w_lr, threshold, maxiter, shuffle, temp_lr,
          dataset, distribution, verbose, ewe_model, ewe_model_name ,customer_model_name="OriginalPlain_2_conv_Keras"):
    
    model = tf.keras.models.load_model(os.path.join(MODELS_SAVE_PATH, customer_model_name))

    print(model.summary())

    
    intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.layers[-7].output)
    # new_input, s1, s2, s3, output = md.EWE_feature_extract_Keras()


    experiment_name = dataset+distribution+str(watermark_source)+str(watermark_target)+ str(ewe_model_name) + "Watermark_featureextract"
    with mlflow.start_run(run_name=experiment_name):

        # ewe_model = Model(inputs=new_input, outputs=[s1, s2, s3, output])

        print(ewe_model.summary())

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

        train(x_train, y_train, x_test, y_test, epochs, w_epochs, lr, n_w_ratio, factors,
            temperatures, watermark_source, watermark_target, batch_size, w_lr, threshold, maxiter, shuffle, temp_lr,
            dataset, distribution, verbose, ewe_model, ewe_model_name, customer_model=intermediate_layer_model)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', help='batch size', type=int, default=512)
    parser.add_argument('--ratio',
                        help='ratio of amount of legitimate data to watermarked data',
                        type=float, default=1.)
    parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
    parser.add_argument('--epochs', help='epochs for training without watermarking', type=int, default=10)
    parser.add_argument('--w_epochs', help='epochs for training with watermarking', type=int, default=10)
    parser.add_argument('--dataset', help='mnist, fashion, speechcmd, cifar10, or cifar100', type=str, default="cifar10")
    parser.add_argument('--model', help='2_conv, lstm, or resnet', type=str, default="2_conv")
    parser.add_argument('--metric', help='distance metric used in snnl, euclidean or cosine', type=str, default="cosine")
    parser.add_argument('--factors', help='weight factor for snnl', nargs='+', type=float, default=[32, 32, 32])
    parser.add_argument('--temperatures', help='temperature for snnl', nargs='+', type=float, default=[1, 1, 1])
    parser.add_argument('--threshold', help='threshold for estimated false watermark rate, should be <= 1/num_class', type=float, default=0.1)
    parser.add_argument('--maxiter', help='iter of perturb watermarked data with respect to snnl', type=int, default=10)
    parser.add_argument('--w_lr', help='learning rate for perturbing watermarked data', type=float, default=0.01)
    parser.add_argument('--t_lr', help='learning rate for temperature', type=float, default=0.1)
    parser.add_argument('--source', help='source class of watermark', type=int, default=1)
    parser.add_argument('--target', help='target class of watermark', type=int, default=7)
    parser.add_argument('--shuffle', type=int, default=0)
    parser.add_argument('--seed', help='random seed', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--default', help='whether to use default hyperparameter, 0 or 1', type=int, default=1)
    parser.add_argument('--layers', help='number of layers, only useful if model is resnet', type=int, default=18)
    parser.add_argument('--distrib', help='use in or out of distribution watermark', type=str, default='out')

    args = parser.parse_args()
    default = args.default
    batch_size = args.batch_size
    ratio = args.ratio
    lr = args.lr
    epochs = args.epochs
    w_epochs = args.w_epochs
    factors = args.factors
    temperatures = args.temperatures
    threshold = args.threshold
    w_lr = args.w_lr
    t_lr = args.t_lr
    source = args.source
    target = args.target
    seed = args.seed
    verbose = args.verbose
    dataset = args.dataset
    model_type = args.model
    maxiter = args.maxiter
    distrib = args.distrib
    layers = args.layers
    metric = args.metric
    shuffle = args.shuffle

    if default:
        if dataset == 'mnist':
            model_type = '2_conv'
            ratio = 1
            batch_size = 512
            epochs = 10
            w_epochs = 10
            factors = [32, 32, 32]
            temperatures = [1, 1, 1]
            metric = "cosine"
            threshold = 0.1
            t_lr = 0.1
            w_lr = 0.01
            source = 1
            target = 7
            maxiter = 10
            distrib = "out"
        elif dataset == 'fashion':
            if model_type == '2_conv':
                batch_size = 128
                ratio = 2
                epochs = 10
                w_epochs = 10
                factors = [32, 32, 32]
                temperatures = [1, 1, 1]
                t_lr = 0.1
                threshold = 0.1
                w_lr = 0.01
                source = 8
                target = 1
                maxiter = 10
                distrib = "out"
                metric = "cosine"
            elif model_type == 'resnet':
                batch_size = 128
                layers = 18
                ratio = 1.2
                epochs = 5
                w_epochs = 5
                factors = [1000, 1000, 1000]
                temperatures = [0.01, 0.01, 0.01]
                t_lr = 0.1
                threshold = 0.1
                w_lr = 0.01
                source = 9
                target = 0
                maxiter = 10
                distrib = "out"
                metric = "cosine"
        elif dataset == "cifar10":
            batch_size = 128
            model_type = "resnet"
            layers = 18
            ratio = 4
            epochs = 50
            w_epochs = 6
            factors = [1e5, 1e5, 1e5]
            temperatures = [1, 1, 1]
            t_lr = 0.1
            threshold = 0.1
            w_lr = 0.01
            source = 8
            target = 0
            maxiter = 10
            distrib = "out"
            metric = "cosine"


    random.seed(seed)
    np.random.seed(seed)
    tf.compat.v1.random.set_random_seed(seed)

    if dataset == 'mnist' or dataset == 'fashion':
        with open(os.path.join("data", f"{dataset}.pkl"), 'rb') as f:
            mnist = pickle.load(f)
        x_train, y_train, x_test, y_test = mnist["training_images"], mnist["training_labels"], \
                                           mnist["test_images"], mnist["test_labels"]
        x_train = np.reshape(x_train / 255, [-1, 28, 28, 1])
        x_test = np.reshape(x_test / 255, [-1, 28, 28, 1])

    elif "cifar" in dataset:
        ds = tfds.load(dataset)
        for i in tfds.as_numpy(ds['train'].batch(50000).take(1)):
            x_train = i['image'] / 255
            y_train = i['label']
        for i in tfds.as_numpy(ds['test'].batch(50000).take(1)):
            x_test = i['image'] / 255
            y_test = i['label']

    else:
        raise NotImplementedError('Dataset is not implemented.')
    

    # model = md.Plain_2_conv()
    model_init = md.ResNet34()
    model_name = model_init.name1
    model = model_init
    print(model_name)
    train_customer_model(x_train, y_train, x_test, y_test, epochs, dataset, batch_size, model, model_save_name=model_name)

    # train_customer_model1(x_train, y_train, x_test, y_test, epochs, dataset, batch_size)


    # embed_watermark_feature_extraction(x_train, y_train, x_test, y_test, epochs, w_epochs, lr, ratio, factors,
    #             temperatures, source, target, batch_size, w_lr, threshold, maxiter, shuffle, t_lr, dataset, distrib,
    #             verbose)

    # customer_model_name_toload = dataset + "OriginalPlain_2_conv"
    # ewe_model_init = md.EWE_2_conv()
    # ewe_model_name = ewe_model_init.name1
    # ewe_model = ewe_model_init

    # embed_watermark_more_training(x_train, y_train, x_test, y_test, epochs, w_epochs, lr, ratio, factors,
    #             temperatures, source, target, batch_size, w_lr, threshold, maxiter, shuffle, t_lr, dataset, distrib,
    #             verbose, ewe_model, ewe_model_name  , customer_model_name= customer_model_name_toload)

    # customer_model_name_toload = dataset + "OriginalPlain_2_conv_Keras"
    # ewe_model_init = md.EWE_feature_extract_Keras()
    # ewe_model_name = ewe_model_init.name
    # ewe_model = ewe_model_init.call()
    
    # embed_watermark_feature_extraction(x_train, y_train, x_test, y_test, epochs, w_epochs, lr, ratio, factors,
    #             temperatures, source, target, batch_size, w_lr, threshold, maxiter, shuffle, t_lr, dataset, distrib,
    #             verbose, ewe_model, ewe_model_name  , customer_model_name= customer_model_name_toload)





