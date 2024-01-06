import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from tensorflow.data import AUTOTUNE
from frontier_stitching import gen_adversaries
from code import models
import numpy as np
import argparse
import os
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import Model

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pickle
import functools
import random
import matplotlib.pyplot as plt
import mlflow
import mlconfig
import math
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("frontier-stiching-scratch")

seed = 0
random.seed(seed)
np.random.seed(seed)
tf.compat.v1.random.set_random_seed(seed)

RESULTS_PATH = "results/scratch"
RESULTS_FOLDER_TRIGGERS = "triggers"
LOSS_FOLDER = "losses"
MODEL_PATH = "models/scratch"

# dataset_name = "cifar10"
# epochs_pretrain = 10
# epochs_watermark_embed = 15

if not os.path.exists(os.path.join(RESULTS_PATH, LOSS_FOLDER)):
        os.makedirs(os.path.join(RESULTS_PATH, LOSS_FOLDER))
if not os.path.exists(os.path.join(RESULTS_PATH, RESULTS_FOLDER_TRIGGERS)):
        os.makedirs(os.path.join(RESULTS_PATH, RESULTS_FOLDER_TRIGGERS))

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def to_float(x, y):
    """normalizing data"""

    return tf.cast(x, tf.float32) / 255.0, y

def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

def comp(model, model_name, optimizer):
    if model_name == "resnet34":
         
         model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.5, decay=5e-4),
                       loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                metrics=["sparse_categorical_accuracy"]
                       )
    else:
        model.compile(optimizer=optimizer, 
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                metrics=["sparse_categorical_accuracy"])
        

    
def embed_watermark_scratch(dataset_name, epochs_pretrain, epochs_watermark_embed, model_watermark_embed_name, epochs_baseline, optimizer, lr, weight_decay, dropout):
    """
    This function trains the frontier model 

    Parameters
    -----------
        train_set: pytorch dataset for training
        test_set: pytorch dataset for testing
        model_type: which model to train, entangled(ewe) or plain
        epochs: no. of epochs to train model
        w_epochs: no. of epochs to train watermark model
        lr: learnimg rate
        n_w_ratio: how many data samples must be wataremraked out of the original data
        factors: list for which temperature will be affected.
        source_class : class from which watermark set is created.
        target_class: class from which label will be assigned to the watermark class.

    Returns
    ------
    test accuracz, watermark accuracy for both EWE and Baseline models.
    """

    models_mapping = {"resnet34": models.ResNet34, "conv_2": models.Plain_2_conv_Keras(), "small": models.Small(), "mnist_l2": models.MNIST_L2,
                      "mnist_l2_drp02": models.MNIST_L2, "mnist_l2_drp03": models.MNIST_L2, "mnist_l5": models.MNIST_L5,
                      "mnist_l5_drp02": models.MNIST_L5, "mnist_l5_drp03": models.MNIST_L5, "cifar10_base": models.CIFAR10_BASE, "cifar10_base_drp02": models.CIFAR10_BASE, "cifar10_base_drp03": models.CIFAR10_BASE,
                      "cifar10_base_2": models.CIFAR10_BASE_2}

    params = {"dataset_name": dataset_name, "epochs_pretrain": epochs_pretrain, "epochs_watermark_embed": epochs_watermark_embed, "optimizer": str(optimizer), "lr":lr, "weight_decay": weight_decay, "epochs_baseline": epochs_baseline, "dropout": dropout}

    dataset = tfds.load(dataset_name, split="train", as_supervised=True)
    val_set = tfds.load(dataset_name, split="test", as_supervised=True)

    train_dataset = dataset.take(int(len(dataset)*0.9))
    test_dataset = dataset.skip(int(len(dataset)*0.9))

    dataset = train_dataset

    dataset = dataset.map(to_float).shuffle(1024).batch(128).prefetch(AUTOTUNE)
    test_dataset = test_dataset.map(to_float).shuffle(1024).prefetch(AUTOTUNE)
    val_set = val_set.map(to_float).batch(128)

    for i in tfds.as_numpy(test_dataset.batch(len(test_dataset)).take(1)):
        x_test = i[0]
        y_test = i[1]


    experiment_name = dataset_name + "Frontier_Watermarking"
    with mlflow.start_run(run_name=experiment_name):

        ## ----------------------------------------- pretrain the model ------------------------------------##
        if dropout:
            model = models_mapping[model_watermark_embed_name](dropout)
        else:
            model = models_mapping[model_watermark_embed_name]()

        CHECKPOINT_FILEPATH = os.path.join(MODEL_PATH, dataset_name + "_" +str(epochs_pretrain) +"_" + model.name1, "Baseline_checkpoint")

        if not os.path.exists(CHECKPOINT_FILEPATH):
            os.makedirs(CHECKPOINT_FILEPATH)

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=CHECKPOINT_FILEPATH,
            save_weights_only=False,
            monitor='val_accuracy',
            mode='max',
            save_best_only=False)

        params["pretrain_model"] = model.name1
        # model.name1 = "Resnet34"
        for param, param_val in params.items():
            mlflow.log_param(param, param_val)

        # input_layer = Input(shape=(dataset.element_spec[0].shape[1], dataset.element_spec[0].shape[2], dataset.element_spec[0].shape[3],))
        # x = original_model(input_layer)
        # model = Model(inputs=input_layer, outputs=x)

        # compiling and fitting the model.
        model.build((None, dataset.element_spec[0].shape[1], dataset.element_spec[0].shape[2], dataset.element_spec[0].shape[3]))
        comp(model,model_watermark_embed_name, optimizer)
        # print(model.summary())
        # print(model)
        lrate = LearningRateScheduler(step_decay)
        
        history = model.fit(dataset, epochs=epochs_pretrain, validation_data=val_set, callbacks=[model_checkpoint_callback]) ## saving best model callback.

        train_acc_pretrain = history.history["sparse_categorical_accuracy"]
        val_acc_pretrain = history.history["val_sparse_categorical_accuracy"]
        train_loss_pretrain = history.history["loss"]
        val_loss_pretrain = history.history["val_loss"]

        print(train_acc_pretrain)
        print(val_acc_pretrain)

        file = open(os.path.join(RESULTS_PATH, LOSS_FOLDER, dataset_name  + "_logs.txt"), "w")
        for idx, (train_loss, train_acc, val_loss, val_acc) in enumerate(zip(train_loss_pretrain, train_acc_pretrain, val_loss_pretrain ,val_acc_pretrain)):
            file.write(f'Epoch: {idx+1}, Train Loss: {train_loss:.3f} Train Acc: {train_acc:.3f} Val Loss: {val_loss:.3f} Val Acc: {val_acc:.3f}')
            file.write("\n")

        plt.figure()
        plt.plot(list(range(epochs_pretrain)), train_loss_pretrain, label="Train loss_" + dataset_name, marker='o', color='tab:purple')
        plt.plot(list(range(epochs_pretrain)), val_loss_pretrain, label="Val loss_" + dataset_name, linestyle='--', marker='o', color='tab:orange')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(RESULTS_PATH, LOSS_FOLDER, dataset_name + "PretrainLoss.png"))
        mlflow.log_artifact(os.path.join(RESULTS_PATH, LOSS_FOLDER, dataset_name + "PretrainLoss.png"), "PretrainLoss.png")

        plt.figure()
        plt.plot(list(range(epochs_pretrain)), train_acc_pretrain, label="Train acc_" + dataset_name, marker='o', color='tab:purple')
        plt.plot(list(range(epochs_pretrain)), val_acc_pretrain, label="Val acc_" + dataset_name, linestyle='--', marker='o', color='tab:orange')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(RESULTS_PATH, LOSS_FOLDER, dataset_name + "PretrainAcc.png"))
        mlflow.log_artifact(os.path.join(RESULTS_PATH, LOSS_FOLDER, dataset_name + "PretrainAcc.png"), "PretrainAcc.png")


        l = 1000
        # generate key set
        true_advs, false_advs = gen_adversaries(model, l, dataset, 0.5)
        # In case that not the full number of adversaries could be generated a reduced amount is returned
        assert(len(true_advs + false_advs) == l)

        fig = plt.figure()
        for i in range(10):
            plt.subplot(5,2, i+1)
            plt.axis("off")
            plt.imshow(true_advs[i][0][:,:,0])
            plt.title("Original")
            plt.show()
            plt.imshow(true_advs[i][1][:,:,0])
            plt.title("Trigger")
            plt.show()
        
        plt.savefig(os.path.join(RESULTS_PATH, RESULTS_FOLDER_TRIGGERS, dataset_name + str(epochs_watermark_embed) + "trigger.png"))
        mlflow.log_artifact(os.path.join(RESULTS_PATH, RESULTS_FOLDER_TRIGGERS, dataset_name + str(epochs_watermark_embed) + "trigger.png"), "trigger.png")


        ## ---------------------------------------- key set --------------------------------------- ##
        full_advs = true_advs + false_advs
        # full_advs = true_advs 
        train_advs = full_advs[:int(l*0.9)]
        val_advs = full_advs[int(l*0.9):int(l*0.95)]
        test_advs = full_advs[int(l*0.95):]


        key_set_x_numpy = np.array([x for x_orig, x, y in full_advs])
        key_set_y_numpy = np.array([y for x_orig, x, y in full_advs])

        key_set_true_x_numpy = np.array([x for x_orig, x, y in true_advs])
        key_set_true_y_numpy = np.array([y for x_orig, x, y in true_advs])

        key_set_train_x_numpy = np.array([x for x_orig, x, y in train_advs])
        key_set_train_y_numpy = np.array([y for x_orig, x, y in train_advs])
        key_set_val_x_numpy = np.array([x for x_orig, x, y in val_advs])
        key_set_val_y_numpy = np.array([y for x_orig, x, y in val_advs])
        key_set_test_x_numpy = np.array([x for x_orig, x, y in test_advs])
        key_set_test_y_numpy = np.array([y for x_orig, x, y in test_advs])

        np.savez(os.path.join(RESULTS_PATH, RESULTS_FOLDER_TRIGGERS, dataset_name + "_" +str(epochs_watermark_embed) +"_" + model_watermark_embed_name + "trigger.npz"), key_set_x_numpy, key_set_y_numpy)
        mlflow.log_artifact(os.path.join(RESULTS_PATH, RESULTS_FOLDER_TRIGGERS, dataset_name + "_" +str(epochs_watermark_embed) +"_" + model_watermark_embed_name + "trigger.npz"), "trigger.npz")

        np.savez(os.path.join(RESULTS_PATH, RESULTS_FOLDER_TRIGGERS, dataset_name + "_" +str(epochs_watermark_embed) +"_" + model_watermark_embed_name + "trigger_true.npz"), key_set_true_x_numpy, key_set_true_y_numpy)
        mlflow.log_artifact(os.path.join(RESULTS_PATH, RESULTS_FOLDER_TRIGGERS, dataset_name + "_" +str(epochs_watermark_embed) +"_" + model_watermark_embed_name + "trigger_true.npz"), "trigger_true.npz")

        key_set_x = tf.data.Dataset.from_tensor_slices([x for x_orig, x, y in full_advs])
        key_set_y = tf.data.Dataset.from_tensor_slices([y for x_orig, x, y in full_advs])
        # key_set_train_x = tf.data.Dataset.from_tensor_slices([x for x_orig, x, y in train_advs])
        # key_set_train_y = tf.data.Dataset.from_tensor_slices([y for x_orig, x, y in train_advs])
        # key_set_val_x = tf.data.Dataset.from_tensor_slices([x for x_orig, x, y in val_advs])
        # key_set_val_y = tf.data.Dataset.from_tensor_slices([y for x_orig, x, y in val_advs])
        # key_set_x_train = key_set_x_numpy[:int(l*0.9)]
        # key_set_y_train = key_set_y_numpy[:int(l*0.9)]
        # key_set_x_val = key_set_x_numpy[int(l*0.9):int(l*0.95)]
        # key_set_y_val = key_set_y_numpy[int(l*0.9):int(l*0.95)]
        # key_set_x_test = key_set_x_numpy[int(l*0.95):]
        # key_set_y_test = key_set_y_numpy[int(l*0.95):]
        # key_set_train = tf.data.Dataset.zip((key_set_x_train, key_set_y_train)).batch(128)
        # key_set_val = tf.data.Dataset.zip((key_set_x_val, key_set_y_val)).batch(128)
        # key_set_train = tf.data.Dataset.zip((key_set_train_x, key_set_train_y)).batch(128)
        # key_set_val = tf.data.Dataset.zip((key_set_val_x, key_set_val_y)).batch(128)
        key_set = tf.data.Dataset.zip((key_set_x, key_set_y)).batch(128)

        ## --------------------------- original model test accuracy -----------------------##
        test_predictions = model.predict(x_test)
        print(f"Test accuracy of Pretrain model is {np.sum(np.argmax(test_predictions, axis=1) == y_test)/ len(y_test)}")


        ##------------------------- original model full key set accuracy -----------------------##
        key_set_predictions = model.predict(key_set_x_numpy)
        print(f"Watermark accuracy of Pretrain model is {np.sum(np.argmax(key_set_predictions, axis=1) == key_set_y_numpy)/ key_set_y_numpy.shape[0]}")

        key_set_predictions = model.predict(key_set_true_x_numpy)
        print(f"Watermark accuracy of Pretrain model on true advs is {np.sum(np.argmax(key_set_predictions, axis=1) == key_set_true_y_numpy)/ key_set_true_y_numpy.shape[0]}")


        full_dataset = dataset.concatenate(key_set)


        ##-------------------------------- reset the optimizer and embed the watermark -----------------------##
        # MODEL_FILEPATH = os.path.join(MODEL_PATH, dataset_name, "Victim_model")
        CHECKPOINT_FILEPATH = os.path.join(MODEL_PATH, dataset_name + "_" +str(epochs_watermark_embed) +"_" + model.name1, "Victim_checkpoint")

        # if not os.path.exists(MODEL_FILEPATH):
        #     os.makedirs(MODEL_FILEPATH)
        if not os.path.exists(CHECKPOINT_FILEPATH):
            os.makedirs(CHECKPOINT_FILEPATH)

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=CHECKPOINT_FILEPATH,
            save_weights_only=False,
            monitor='val_accuracy',
            mode='max',
            save_best_only=False)
        
        if dropout:
            watermarked_model = models_mapping[model_watermark_embed_name](dropout)
        else:
            watermarked_model = models_mapping[model_watermark_embed_name]()

        watermarked_model.build((None, dataset.element_spec[0].shape[1], dataset.element_spec[0].shape[2], dataset.element_spec[0].shape[3]))
        comp(watermarked_model,model_watermark_embed_name, optimizer)

        # print(watermarked_model.summary())
        print(model.summary())

        # for layer_watermark , layer_original in zip(watermarked_model.layers , model.layers):
        #     weight_layer = layer_original.get_weights()
        #     layer_watermark.set_weights(weight_layer)

        lrate = LearningRateScheduler(step_decay)

        test_predictions = watermarked_model.predict(x_test)
        print(f"Test accuracy of Watermarked model before fitting is {np.sum(np.argmax(test_predictions, axis=1) == y_test)/ len(y_test)}")

        history = watermarked_model.fit(full_dataset, epochs=epochs_watermark_embed, validation_data=val_set, callbacks=[model_checkpoint_callback])

        train_acc_watermark = history.history["sparse_categorical_accuracy"]
        val_acc_watermark = history.history["val_sparse_categorical_accuracy"]
        train_loss_watermark = history.history["loss"]
        val_loss_watermark = history.history["val_loss"]

        print(train_acc_pretrain)
        print(val_acc_pretrain)

        file = open(os.path.join(RESULTS_PATH, LOSS_FOLDER, dataset_name  + "_logs.txt"), "a+")
        file.write("\n ----------------- Watermark Embedding -------------- \n")
        file.write("\n For this " + dataset_name + "_" +str(epochs_watermark_embed) +"_" + model_watermark_embed_name + "\n")
        for idx, (train_loss, train_acc, val_loss, val_acc) in enumerate(zip(train_loss_watermark, train_acc_watermark, val_loss_watermark ,val_acc_watermark)):
            file.write(f'Epoch: {idx+1}, Train Loss: {train_loss:.3f} Train Acc: {train_acc:.3f} Val Loss: {val_loss:.3f} Val Acc: {val_acc:.3f}')
            file.write("\n")


        plt.figure()
        plt.plot(list(range(epochs_watermark_embed)), train_loss_watermark, label="Train loss_" + dataset_name, marker='o', color='tab:purple')
        plt.plot(list(range(epochs_watermark_embed)), val_loss_watermark, label="Val loss_" + dataset_name, linestyle='--', marker='o', color='tab:orange')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(RESULTS_PATH, LOSS_FOLDER, dataset_name + "WatermarkLoss.png"))
        mlflow.log_artifact(os.path.join(RESULTS_PATH, LOSS_FOLDER, dataset_name + "WatermarkLoss.png"), "WatermarkLoss.png")

        plt.figure()
        plt.plot(list(range(epochs_watermark_embed)), train_acc_watermark, label="Train acc_" + dataset_name, marker='o', color='tab:purple')
        plt.plot(list(range(epochs_watermark_embed)), val_acc_watermark, label="Val acc_" + dataset_name, linestyle='--', marker='o', color='tab:orange')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(RESULTS_PATH, LOSS_FOLDER, dataset_name + "WatermarkAcc.png"))
        mlflow.log_artifact(os.path.join(RESULTS_PATH, LOSS_FOLDER, dataset_name + "WatermarkAcc.png"), "WatermarkAcc.png")

        ## --------------------------- original model test accuracy -----------------------##
        test_predictions = watermarked_model.predict(x_test)
        print(f"Test accuracy of Watermarked model is {np.sum(np.argmax(test_predictions, axis=1) == y_test)/ len(y_test)}")


        ##------------------------- original model full key set accuracy -----------------------##
        key_set_predictions = watermarked_model.predict(key_set_train_x_numpy)
        print(f"TRain Watermark accuracy of Watermarked model is {np.sum(np.argmax(key_set_predictions, axis=1) == key_set_train_y_numpy)/ key_set_train_y_numpy.shape[0]}")

        key_set_predictions = watermarked_model.predict(key_set_val_x_numpy)
        print(f"Val Watermark accuracy of Watermarked model is {np.sum(np.argmax(key_set_predictions, axis=1) == key_set_val_y_numpy)/ key_set_val_y_numpy.shape[0]}")

        key_set_predictions = watermarked_model.predict(key_set_test_x_numpy)
        print(f"Test Watermark accuracy of Watermarked model is {np.sum(np.argmax(key_set_predictions, axis=1) == key_set_test_y_numpy)/ key_set_test_y_numpy.shape[0]}")

        key_set_predictions = watermarked_model.predict(key_set_x_numpy)
        print(f"Watermark accuracy of Watermarked model on full dataset is {np.sum(np.argmax(key_set_predictions, axis=1) == key_set_y_numpy)/ key_set_y_numpy.shape[0]}")

        key_set_predictions = watermarked_model.predict(key_set_true_x_numpy)
        print(f"Watermark accuracy of Watermarked model on true advs is {np.sum(np.argmax(key_set_predictions, axis=1) == key_set_true_y_numpy)/ key_set_true_y_numpy.shape[0]}")

        # model = tf.keras.models.load_model("models/scratch/mnist_50_MNIST_l20.0/Victim_checkpoint")

        # key_set_predictions = model.predict(key_set_true_x_numpy)
        # print(f"Watermark accuracy of loaded Watermarked model on true advs is {np.sum(np.argmax(key_set_predictions, axis=1) == key_set_true_y_numpy)/ key_set_true_y_numpy.shape[0]}")


        # adv_numpy = np.load("results/scratch/triggers/mnist_20_mnist_l2trigger_true.npz")
        # adv_x = adv_numpy["arr_0"]
        # adv_y = adv_numpy["arr_1"]

        # print(f"Watermark accuracy of loaded Watermarked model on true advs is {np.sum(np.argmax(adv_x, axis=1) == adv_y)/ adv_y.shape[0]}")






        # key_train_predictions = watermarked_model.predict(key_set_x_train)
        # print(f"Watermark accuracy of Watermarked model is {np.sum(np.argmax(key_train_predictions, axis=1) == key_set_y_train)/ key_set_y_train.shape[0]}")

        # key_val_predictions = watermarked_model.predict(key_set_x_val)
        # print(f"Watermark accuracy of Watermarked model is {np.sum(np.argmax(key_val_predictions, axis=1) == key_set_y_val)/ key_set_y_val.shape[0]}")

        # key_test_predictions = watermarked_model.predict(key_set_x_test)
        # print(f"Watermark accuracy of Watermarked model is {np.sum(np.argmax(key_test_predictions, axis=1) == key_set_y_test)/ key_set_y_test.shape[0]}")


        # correct = 0
        # for x, y in zip(key_set_x_numpy, key_set_y_numpy):
        #     x = np.expand_dims(x, axis=0)
        #     predictions = model.predict(x, batch_size=1)
        #     if np.argmax(predictions, axis=1)[0] ==  y:
        #         correct += 1
            
        # accuracy = correct / key_set_x_numpy.shape[0]
        # print("Accuracy on watermark examples of watermarked classifier: {}%".format(accuracy * 100))




        ## ----------------------------------------- baseline model ------------------------------------##
        # if dropout:
        #     model = models_mapping[model_watermark_embed_name](dropout)
        # else:
        #     model = models_mapping[model_watermark_embed_name]()
        # # model =  models.MNIST_L2()


        # CHECKPOINT_FILEPATH = os.path.join(MODEL_PATH, dataset_name + "_" +str(epochs_baseline) +"_" + model.name1, "Baseline_checkpoint")

        # if not os.path.exists(CHECKPOINT_FILEPATH):
        #     os.makedirs(CHECKPOINT_FILEPATH)

        # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        #     filepath=CHECKPOINT_FILEPATH,
        #     save_weights_only=False,
        #     monitor='val_accuracy',
        #     mode='max',
        #     save_best_only=False)
        # # compiling and fitting the model.

        # model.build((None, dataset.element_spec[0].shape[1], dataset.element_spec[0].shape[2], dataset.element_spec[0].shape[3]))

        # comp(model,model_watermark_embed_name, optimizer)
        # print(model)
        # lrate = LearningRateScheduler(step_decay)
        # history = model.fit(dataset, epochs=epochs_baseline, validation_data=val_set, callbacks=[model_checkpoint_callback]) ## saving best model callback.

        # train_acc_baseline = history.history["sparse_categorical_accuracy"]
        # val_acc_baseline = history.history["val_sparse_categorical_accuracy"]
        # train_loss_baseline = history.history["loss"]
        # val_loss_baseline = history.history["val_loss"]

        # print(train_acc_baseline)
        # print(val_acc_baseline)

        
        # file = open(os.path.join(RESULTS_PATH, LOSS_FOLDER, dataset_name  + "_logs.txt"), "a+")
        # file.write("\n --------------------Baseline model----------- \n")
        # for idx, (train_loss, train_acc, val_loss, val_acc) in enumerate(zip(train_loss_baseline, train_acc_baseline, val_loss_baseline ,val_acc_baseline)):
        #     file.write(f'Epoch: {idx+1}, Train Loss: {train_loss:.3f} Train Acc: {train_acc:.3f} Val Loss: {val_loss:.3f} Val Acc: {val_acc:.3f}')
        #     file.write("\n")
        # file.close()

        

        # plt.figure()
        # plt.plot(list(range(epochs_baseline)), train_loss_baseline, label="Train loss_" + dataset_name, marker='o', color='tab:purple')
        # plt.plot(list(range(epochs_baseline)), val_loss_baseline, label="Val loss_" + dataset_name, linestyle='--', marker='o', color='tab:orange')
        # plt.xlabel("Epochs")
        # plt.ylabel("Loss")
        # plt.legend()
        # plt.savefig(os.path.join(RESULTS_PATH, LOSS_FOLDER, dataset_name + "BaselineLoss.png"))
        # mlflow.log_artifact(os.path.join(RESULTS_PATH, LOSS_FOLDER, dataset_name + "BaselineLoss.png"), "BaselineLoss.png")

        # # mlflow.log_artifact(os.path.join(RESULTS_PATH, LOSS_FOLDER, dataset_name  + "_logs.txt"), "logs.txt")

        # plt.figure()
        # plt.plot(list(range(epochs_baseline)), train_acc_baseline, label="Train acc_" + dataset_name, marker='o', color='tab:purple')
        # plt.plot(list(range(epochs_baseline)), val_acc_baseline, label="Val acc_" + dataset_name, linestyle='--', marker='o', color='tab:orange')
        # plt.xlabel("Epochs")
        # plt.ylabel("Accuracy")
        # plt.legend()
        # plt.savefig(os.path.join(RESULTS_PATH, LOSS_FOLDER, dataset_name + "BaselineAcc.png"))
        # mlflow.log_artifact(os.path.join(RESULTS_PATH, LOSS_FOLDER, dataset_name + "BaselineAcc.png"), "BaselineAcc.png")

        # mlflow.log_artifact(os.path.join(RESULTS_PATH, LOSS_FOLDER, dataset_name  + "_logs.txt"), "logs.txt")


        # correct = 0
        # for x, y in zip(key_set_x, key_set_y):
        #     x = np.expand_dims(x, axis=0)
        #     predictions = model.predict(x, batch_size=1)
        #     if np.argmax(predictions, axis=1)[0] ==  y:
        #         correct += 1
            
        # accuracy = correct / key_set_x.shape[0]
        # print("Accuracy on watermark examples of watermarked classifier: {}%".format(accuracy * 100))








if __name__ == "__main__":
     
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="configs/scratch.yaml")

    args = parser.parse_args()
    print(args)
    config = mlconfig.load(args.config)

    if config.optimizer == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr, decay=config.weight_decay)
    else:
        # optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, decay=config.weight_decay)
        optimizer = None

    embed_watermark_scratch(config.dataset, config.epochs_pretrain, config.epochs_watermark_embed, config.model_watermark_embed_name, config.epochs_baseline, optimizer, config.lr, config.weight_decay, config.dropout)

    

     
