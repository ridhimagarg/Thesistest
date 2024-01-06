from art.estimators.classification.tensorflow import TensorFlowV2Classifier, TensorFlowClassifier
from art.attacks import ExtractionAttack
from art.attacks.extraction import CopycatCNN, KnockoffNets
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

from mlflow import log_metric, log_param, log_params, log_artifacts
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("EWE")

# from utils_new import pca_dim

seed = 0
random.seed(seed)
np.random.seed(seed)
tf.compat.v1.random.set_random_seed(seed)
RESULTS_FOLDER = "results/KnockoffStolen"
RESULTS_SUB_FOLDER_TRIGGERS = "triggers_images"
ACC_FOLDER = "accuracies"

# tf.compat.v1.disable_eager_execution()


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


dataset = "mnist"
lr = 0.001
num_epochs = 10 # try with 50 also
len_steal = 20000
# dataset_path = "data/trigger/EWE_2_convmnist_trigger.npz"
# model_path = "models/scratch/EWE_2_convVictim_mnist_model"
dataset_path = "data/trigger/mnist_trigger.npz"
model_path = "models/EWEVictim_mnist_model"
# dataset_path = "data/trigger/fashion_trigger.npz"
# model_path = "models/EWEVictim_fashion_model"
classifier_model_save_path = os.path.join("models/KnockoffStolen", dataset)

# batch_size = 128
# ratio = 2
# epochs = 10
# w_epochs = 10
# factors = [32, 32, 32]
# temperatures = [1, 1, 1]
# t_lr = 0.1
# threshold = 0.1
# w_lr = 0.01
# source = 8
# target = 1

# maxiter = 10
# distrib = "out"
# metric = "cosine"



params = {"epochs": epochs, "w_epochs": w_epochs, "lr":lr, "n_w_ratio":ratio, "watermark_source":source, "watermark_target":target, "batch_size":batch_size,
              "w_lr":w_lr, "threshold":threshold, "maxiter":maxiter, "temp_lr":t_lr, "dataset":dataset, "distribution":distrib, "trigger_dataset_path": dataset_path, "vivtim_model_path": model_path}

experiment_name = "realstealing"+dataset+distrib+str(source)+str(target)

# attacker_model = md.Plain_2_conv()

with mlflow.start_run(run_name=experiment_name):

    for param, param_val in params.items():
        mlflow.log_param(param, param_val)

    with open(os.path.join("data", f"{dataset}.pkl"), 'rb') as f:
        mnist = pickle.load(f)
        x_train, y_train, x_test, y_test = mnist["training_images"], mnist["training_labels"], \
                                            mnist["test_images"], mnist["test_labels"]
        x_train = np.reshape(x_train / 255, [-1, 28, 28, 1])
        x_test = np.reshape(x_test / 255, [-1, 28, 28, 1])

    # ewe_model = functools.partial(md.EWE_2_conv, metric=metric)
    # plain_model = md.Plain_2_conv
    lr = 0.001

    shuffle = 0
    verbose = True

    model = tf.keras.models.load_model(model_path)

    classifier = TensorFlowV2Classifier(model=model, nb_classes=10, input_shape=(28, 28, 1))

    loss_object = tf.keras.losses.CategoricalCrossentropy()
    a_lr = 0.001
    optimizer = tf.keras.optimizers.Adam(learning_rate=a_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-5)
    # optimizer = tf.keras.optimizers.AdamW(learning_rate=a_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-9, weight_decay=0.004)

    mlflow.log_param("attacker lr", a_lr)

    def train_step(model, images, labels):

        # print(model.trainable_weights)
        # plt.figure()
        # plt.imshow(images[0][:,:,0], cmap="gray_r")
        # plt.show()
        # print(labels)

        with tf.GradientTape() as tape:
            prediction = model(images)

            prediction =  tf.nn.softmax(prediction)

            loss = loss_object(labels, prediction)

            # print(loss)

        grads = tape.gradient(loss, model.trainable_weights)
        
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

    # attacker_model = md.Plain_2_conv()

    # ##----------------- Change here everytime --------------------
    # mlflow.log_param("model name", "Plain_2_conv")


    # print(attacker_model.trainable_weights)
    # classifier_stolen = TensorFlowV2Classifier(attacker_model, loss_object= loss_object, input_shape= (28,28,1) , nb_classes=10, train_step=train_step)

    len_steal_list = [250, 500, 1000, 5000, 10000, 20000, 25000] ## can also vary no. of epochs
    mlflow.log_param("epoch steal", num_epochs)

    test_acc = []
    watermark_acc = []
    for len_steal in len_steal_list:

        # attacker_model = md.Plain_MNIST_L5_DR05()
        attacker_model = md.Plain_2_conv()

        ##----------------- Change here everytime --------------------
        mlflow.log_param("model name", attacker_model.name1)


        print(attacker_model.trainable_weights)
        classifier_stolen = TensorFlowV2Classifier(attacker_model, loss_object= loss_object, input_shape= (28,28,1) , nb_classes=10, train_step=train_step)
            
        # mlflow.log_param("len steal", len_steal)

        attack = KnockoffNets(classifier=classifier, batch_size_fit=64, batch_size_query=64, nb_epochs=num_epochs, nb_stolen=len_steal,sampling_strategy="random",use_probability=False)

        indices = np.random.RandomState(seed=42).permutation(len(x_train))

        x_steal = x_train[indices[:len_steal]]
        y_steal = np.array(y_train)[indices[:len_steal]]
        x_train_wo_steal = x_train[indices[len_steal:]]
        print(type(x_steal))
        print(isinstance(x_steal, tf.Tensor))
        y_train_wo_steal = np.array(y_train)[indices[len_steal:]]
        classifier_stolen = attack.extract(x_train, y_train, thieved_classifier=classifier_stolen) ## have to check here, if this is correct? because classifier_stolen is updated again.

        model.save(classifier_model_save_path)

        # mlflow.log_artifact(classifier_model_save_path, "stolen model")

        predictions = classifier_stolen.predict(x_train, batch_size=64)

        print("Argmax predictions {}".format(np.argmax(predictions, axis=1)))
        print("Actual y train {}".format(y_train))

        accuracy = np.sum(np.argmax(predictions, axis=1) == y_train) / len(y_train)
        print("Accuracy on train examples: {}%".format(accuracy * 100))
        mlflow.log_metric("train acc"+str(len_steal), accuracy)


        predictions = classifier_stolen.predict(x_test, batch_size=64)

        print("Argmax predictions {}".format(np.argmax(predictions, axis=1)))
        print("Actual y test {}".format(y_test))

        accuracy = np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test)
        print("Accuracy on test examples: {}%".format(accuracy * 100))
        mlflow.log_metric("test acc"+str(len_steal), accuracy)
        test_acc.append(accuracy)


        ## On reamining train data
        predictions = classifier_stolen.predict(x_train_wo_steal, batch_size=64)

        print("Argmax predictions {}".format(np.argmax(predictions, axis=1)))
        print("Actual y on remaining train eaxmples {}".format(y_train_wo_steal))

        accuracy = np.sum(np.argmax(predictions, axis=1) == y_train_wo_steal) / len(y_train_wo_steal)
        print("Accuracy on remaining train examples: {}%".format(accuracy * 100))
        mlflow.log_metric("rem train acc"+str(len_steal), accuracy)


        trigger_arr = np.load(dataset_path)
        trigger_arr["arr_0"]

        print(trigger_arr["arr_0"].shape)

        for i in range(10):
            plt.subplot(2,1, 2)
            plt.axis("off")
            plt.imshow(trigger_arr["arr_0"][i][:,:,0], cmap="gray_r")
            plt.title("With Trigger")
            plt.savefig(os.path.join(RESULTS_FOLDER, RESULTS_SUB_FOLDER_TRIGGERS, "trigger_"+ str(i)+ ".png"))

            mlflow.log_artifact(os.path.join(RESULTS_FOLDER, RESULTS_SUB_FOLDER_TRIGGERS, "trigger_"+ str(i)+ ".png"), "TriggerImages")


        predictions = classifier_stolen.predict(trigger_arr["arr_0"], batch_size=64)
        print("Argmax predictions {}".format(np.argmax(predictions, axis=1)))
        predictions_list = np.argmax(predictions, axis=1).tolist()
        count = len([i for i in predictions_list if i==target])

        accuracy = count / len(predictions_list)
        print("Accuracy on watermark examples: {}%".format(accuracy * 100))
        mlflow.log_metric("watermark acc"+str(len_steal), accuracy)
        watermark_acc.append(accuracy)
    
    plt.figure(figsize=(5,5))
    plt.plot(len_steal_list, test_acc, label="argmax knockoffnet " + dataset + " Test acc ", linestyle='--', marker='o', color='tab:purple')
    plt.xlabel("Stealing Dataset size")
    plt.ylabel("Stolen Model accuracy")
    plt.legend()
    # plt.savefig(os.path.join(RESULTS_FOLDER, ACC_FOLDER, dataset + "Testaccuracyattack.png"))

    plt.plot(len_steal_list, watermark_acc, label="argmax knockoffnet " + dataset + " Watermark acc ", linestyle='--', marker='o', color='tab:orange')
    plt.xlabel("Stealing Dataset size")
    plt.ylabel("Stolen Model accuracy")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_FOLDER, ACC_FOLDER, dataset + "TestAndWatermarkaccuracyattack.png"))

    mlflow.log_artifact(os.path.join(RESULTS_FOLDER, ACC_FOLDER, dataset + "TestAndWatermarkaccuracyattack.png"), "TestAndWatermarkaccuracyattack")


    