import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from tensorflow.data import AUTOTUNE
from frontier_stitching import gen_adversaries, verify
from code import models
from art.estimators.classification.tensorflow import TensorFlowV2Classifier, TensorFlowClassifier
from art.estimators.classification import KerasClassifier
from art.attacks import ExtractionAttack
from art.attacks.extraction import CopycatCNN, KnockoffNets
import numpy as np
import argparse
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pickle
import functools
import random
import matplotlib.pyplot as plt
from mlflow import log_metric, log_param, log_params, log_artifacts
import mlflow
from art.utils import load_mnist
import mlconfig
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("frontier-stiching-realmodelstealing")
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



seed = 0
random.seed(seed)
np.random.seed(seed)
tf.compat.v1.random.set_random_seed(seed)

RESULTS_PATH = "results/KnockoffStolen"
RESULTS_FOLDER_TRIGGERS = "triggers"
LOSS_FOLDER = "losses"
MODEL_PATH = "models/KnockoffStolen"


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

def comp(model, model_name, optimizer):
    if model_name == "resnet34":
         model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1, momentum=0.5, weight_decay=5e-4),
                       )
    else:
        model.compile(optimizer=optimizer, 
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                metrics=["sparse_categorical_accuracy"])
    

def attack_model(dataset_name, model_to_attack_path, model_extract_name, epochs_extract, adv_numpy_path, optimizer, lr, weight_decay, dropout):
    """
    This function performs the attack on the trained model with different no. of stealing queries

    Parameters
    -----------
        dataset_name: name of the dataset
        model_to_attack_path: load this model to perform attack on in a API access form
        model_extract_name: model name(architecture) to perform the attack with
        epochs_extract: attacker model is trained with these epochs
        adv_numpy_path: watermark set path to test the watermark accuracy later on.

    Returns
    ------
    test accuracz, watermark accuracy for both EWE and Baseline models.
    """

    models_mapping = {"resnet34": models.ResNet34(), "conv_2": models.Plain_2_conv_Keras(), "small": models.Small(), "mnist_l2": models.MNIST_L2,
                      "mnist_l2_drp02": models.MNIST_L2, "mnist_l2_drp03": models.MNIST_L2, "mnist_l5": models.MNIST_L5,
                      "mnist_l5_drp02": models.MNIST_L5, "mnist_l5_drp03": models.MNIST_L5, "cifar10_base": models.CIFAR10_BASE, "cifar10_base_drp02": models.CIFAR10_BASE, "cifar10_base_drp03": models.CIFAR10_BASE,
                      "cifar10_base_2": models.CIFAR10_BASE_2}
    print(optimizer)

    params = {"dataset_name": dataset_name,  "model_to_attack": model_to_attack_path , "epochs steal": epochs_extract, "optimizer": str(optimizer), "lr": lr, "weight_decay": weight_decay, "dropout": dropout,
              "adv_key_set_path": adv_numpy_path}

    experiment_name = "realstealing_" + dataset_name

    with mlflow.start_run(run_name=experiment_name):
        
        ## ---------------------------- Loading model to attack on ------------------------##
        model = tf.keras.models.load_model(model_to_attack_path)

        for param, param_val in params.items():
            mlflow.log_param(param, param_val)

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        
        adv_numpy = np.load(adv_numpy_path)
        adv_x = adv_numpy["arr_0"]
        adv_y = adv_numpy["arr_1"]

        correct = 0
        for x, y in zip(adv_x, adv_y):
            x = np.expand_dims(x, axis=0)
            predictions = model.predict(x, batch_size=1)
            if np.argmax(predictions, axis=1)[0] ==  y:
                correct += 1
            
        accuracy = correct / adv_x.shape[0]
        print("Accuracy on watermark examples of victim classifier: {}%".format(accuracy * 100))


        # ---------------------------------------------- dataset creation -----------------------------------##
        dataset = tfds.load(dataset_name, split="train", as_supervised=True)
        val_set = tfds.load(dataset_name, split="test", as_supervised=True)
        for i in tfds.as_numpy(dataset.batch(len(dataset)).take(1)):
            x_train = i[0] / 255
            y_train = i[1]
        for i in tfds.as_numpy(val_set.batch(len(val_set)).take(1)):
            x_test = i[0] / 255
            y_test = i[1]


        ## victim classifier
        victim_classifier = TensorFlowV2Classifier(model, nb_classes=10, loss_object=loss_object, input_shape=(x_train[1], x_train[2], x_train[3]), channels_first=False)
        # predictions = model.predict(x_train)
        # # predictions = np.argmax(predictions, axis=1)
        # print(np.sum(np.argmax(predictions, axis=1) == y_train) / len(y_train))

        # predictions = model.predict(x_test)
        # # predictions = np.argmax(predictions, axis=1)
        # print(np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test))

        test_acc = []
        watermark_acc = []
        len_steal_list = [250, 500, 5000, 10000, 20000] 
        epochs_extract_list = [50, 50, 50, 80, 100]
        # len_steal_list = [25000]
        # epochs_extract_list = [100]

        file1 = open(os.path.join(RESULTS_PATH, LOSS_FOLDER, dataset_name  + "loss_logs.txt"), "w")
        file2 = open(os.path.join(RESULTS_PATH, LOSS_FOLDER, dataset_name  + "acc_logs.txt"), "w")

        for index, len_steal in enumerate(len_steal_list):
            
            if dropout:
                attacker_model = models_mapping[model_extract_name](dropout)
            else:
                attacker_model = models_mapping[model_extract_name]()

            mlflow.log_param("attacker model name", attacker_model.name1)

            # optimizer = optimizer
            if config.optimizer == "adam":
                optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr, decay=config.weight_decay)
            else:
                optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, weight_decay=config.weight_decay)

            ## seclecting dataset to make attack with these no. of queries.
            indices = np.random.RandomState(seed=42).permutation(len(x_train))
            x_steal = x_train[indices[:len_steal]]
            y_steal = np.array(y_train)[indices[:len_steal]]
            x_train_wo_steal = x_train[indices[len_steal:]]
            y_train_wo_steal = np.array(y_train)[indices[len_steal:]]

            file1 = open(os.path.join(RESULTS_PATH, LOSS_FOLDER, dataset_name  + "loss_logs.txt"), "a+")
            file1.write(f"No. of queries {len_steal}")
            # @tf.function
            def train_step(model1, images, labels):

                with tf.GradientTape() as tape:
                    prediction = model1(images)
                    loss = loss_object(labels, prediction)
                    file1.write(f"\n Loss of attacker model: {loss:.3f}")
                    file1.write("\n")
                    # print("loss", loss)

                grads = tape.gradient(loss, model1.trainable_weights)
                optimizer.apply_gradients(zip(grads, model1.trainable_weights))

            ## -------------------------- Attack -------------------------##
            classifier_stolen = TensorFlowV2Classifier(attacker_model, nb_classes=10, loss_object=loss_object, input_shape=(x_train[1], x_train[2], x_train[3]), channels_first=False, train_step=train_step)
            attack = KnockoffNets(classifier=victim_classifier, batch_size_fit=64, batch_size_query=64, nb_epochs= epochs_extract_list[index], nb_stolen=len_steal,sampling_strategy="random",use_probability=False)
            classifier_stolen = attack.extract(x_steal, y_steal, thieved_classifier=classifier_stolen) 

            classifier_stolen.model.save(os.path.join(MODEL_PATH, str(attacker_model.name1) + "_" + dataset_name+ "_" + str(len_steal)))
            # print(classifier_stolen)

            
            file2 = open(os.path.join(RESULTS_PATH, LOSS_FOLDER, dataset_name  + "acc_logs.txt"), "a+")


            predictions = classifier_stolen.predict(x_steal, batch_size=64)
            print("Argmax predictions {}".format(np.argmax(predictions, axis=1)))
            print("Actual y steal {}".format(y_steal))
            accuracy = np.sum(np.argmax(predictions, axis=1) == y_steal) / len(y_steal)
            print("Accuracy on stealing train examples: {}%".format(accuracy * 100))
            mlflow.log_metric("steal acc_"+str(len_steal), accuracy)
            file2.write(f'Stealing dataset size: {len_steal}, Steal acc: {accuracy:.3f}')
            file2.write("\n")


            ## ------------------ Train acc ----------------------##
            predictions = classifier_stolen.predict(x_train, batch_size=64)
            print("Argmax predictions {}".format(np.argmax(predictions, axis=1)))
            print("Actual y train {}".format(y_train))
            accuracy = np.sum(np.argmax(predictions, axis=1) == y_train) / len(y_train)
            print("Accuracy on train examples: {}%".format(accuracy * 100))
            mlflow.log_metric("train acc_"+str(len_steal), accuracy)
            file2.write(f'Stealing dataset size: {len_steal}, Train acc: {accuracy:.3f}')
            file2.write("\n")



            ## ------------------ Test acc -----------------##
            predictions = classifier_stolen.predict(x_test, batch_size=64)
            print("Argmax predictions {}".format(np.argmax(predictions, axis=1)))
            print("Actual y test {}".format(y_test, axis=1))
            accuracy = np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test)
            print("Accuracy on test examples: {}%".format(accuracy * 100))
            mlflow.log_metric("test acc_"+str(len_steal), accuracy)
            test_acc.append(accuracy)
            file2.write(f'Stealing dataset size: {len_steal}, Test acc: {accuracy:.3f}')
            file2.write("\n")

            # info = verify(classifier_stolen, key_set, 0.05)
            # if info["success"]:
            #     print("Model is ours and was successfully watermarked.")
            # else:
            #     print("Model is not ours and was not successfully watermarked.")

            ## ------------------- Watermark acc -----------------##
            adv_numpy = np.load(adv_numpy_path)
            adv_x = adv_numpy["arr_0"]
            adv_y = adv_numpy["arr_1"]

            correct = 0
            for x, y in zip(adv_x, adv_y):
                x = np.expand_dims(x, axis=0)
                predictions = classifier_stolen.predict(x, batch_size=1)
                if np.argmax(predictions, axis=1)[0] ==  y:
                    correct += 1
                
            accuracy = correct / adv_x.shape[0]
            print("Accuracy on watermark examples: {}%".format(accuracy * 100))
            mlflow.log_metric("watermark acc_"+str(len_steal), accuracy)
            watermark_acc.append(accuracy)
            file2.write(f'Stealing dataset size: {len_steal}, Watermark acc: {accuracy:.3f}')
            file2.write("\n \n \n")
            file1.close()
            file2.close()




        plt.figure(figsize=(5,5))
        plt.plot(len_steal_list, test_acc, label="argmax knockoffnet " + dataset_name + " Test acc ", linestyle='--', marker='o', color='tab:purple')
        plt.plot(len_steal_list, watermark_acc, label="argmax knockoffnet " + dataset_name + " Watermark acc ", linestyle='--', marker='o', color='tab:orange')
        plt.xlabel("Stealing Dataset size")
        plt.ylabel("Stolen Model accuracy")
        plt.legend()
        plt.savefig(os.path.join(RESULTS_PATH, LOSS_FOLDER, dataset_name + "TestandWatermarkAcc.png"))
        mlflow.log_artifact(os.path.join(RESULTS_PATH, LOSS_FOLDER, dataset_name + "TestandWatermarkAcc.png"), "TestandWatermarkAcc.png")
        mlflow.log_artifact(os.path.join(RESULTS_PATH, LOSS_FOLDER, dataset_name  + "_logs.txt"), "logs.txt")






if __name__ == "__main__":
     
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="configs/knockoff.yaml")

    args = parser.parse_args()
    print(args)
    config = mlconfig.load(args.config)

    

    attack_model(config.dataset, config.model_to_attack_path, config.model_extract_name, config.epochs_extract, config.adv_key_set_path, config.optimizer, config.lr, config.weight_decay, config.dropout)
     













































# params = {"dataset_name": dataset_name, "epochs_pretrain": epochs_pretrain, "epochs_watermark_embed": epochs_watermark_embed, "optimizer": "tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)"}

# dataset = tfds.load(dataset_name, split="train", as_supervised=True)
# val_set = tfds.load(dataset_name, split="test", as_supervised=True)

# dataset = dataset.map(to_float).shuffle(1024).batch(128).prefetch(AUTOTUNE)
# val_set = val_set.map(to_float).batch(128)


# experiment_name = "realstealing_" + dataset_name

# # attacker_model = md.Plain_2_conv()

# with mlflow.start_run(run_name=experiment_name):

#     ## ----------------------------------------- pretrain the model ------------------------------------##
#     model = models. ResNet34()
#     params["pretrain_model"] = model.name1
#     for param, param_val in params.items():
#         mlflow.log_param(param, param_val)
#     comp(model)
#     history = model.fit(dataset, epochs=epochs_pretrain, validation_data=val_set)

#     train_acc_pretrain = history.history["sparse_categorical_accuracy"]
#     val_acc_pretrain = history.history["val_sparse_categorical_accuracy"]

#     l = 100
#     # generate key set
#     true_advs, false_advs = gen_adversaries(model, l, dataset, 0.1)
#     # In case that not the full number of adversaries could be generated a reduced amount is returned
#     assert(len(true_advs + false_advs) == l)

#     fig = plt.figure()
#     for i in range(10):
#         plt.subplot(5,2, i+1)
#         plt.axis("off")
#         plt.imshow(true_advs[i][0][:,:,0], cmap="gray_r")
#         plt.title("Trigger")
    
#     plt.savefig(os.path.join(RESULTS_SUB_FOLDER_TRIGGERS, "trigger.png"))
#     mlflow.log_artifact(os.path.join(RESULTS_SUB_FOLDER_TRIGGERS, "trigger.png"), "trigger.png")



#     ## ---------------------------------------- key set --------------------------------------- ##
#     key_set_x = tf.data.Dataset.from_tensor_slices([x for x, y in true_advs + false_advs])
#     key_set_y = tf.data.Dataset.from_tensor_slices([y for x, y in true_advs + false_advs])
#     key_set = tf.data.Dataset.zip((key_set_x, key_set_y)).batch(128)

#     full_dataset = dataset.concatenate(key_set)


#     ##-------------------------------- reset the optimizer and embed the watermark -----------------------##
#     model = models.ResNet34()
#     comp(model)
#     history = model.fit(full_dataset, epochs=epochs_watermark_embed, validation_data=val_set)

#     # model.save("test.h5")




#     # model_path = "test.h5"
#     # model = tf.keras.models.load_model(model_path)

#     loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#     optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

#     # classifier = KerasClassifier(model=model, use_logits=True)
#     classifier = TensorFlowV2Classifier(model, nb_classes=10, loss_object=loss_object, input_shape=(28,28,1), clip_values=(0,1), channels_first=False)


#     # Load the dataset, and split the test data into test and steal datasets.
#     # (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
#     dataset = tfds.load(dataset_name, split="train", as_supervised=True)
#     val_set = tfds.load(dataset_name, split="test", as_supervised=True)
#     for i in tfds.as_numpy(dataset.batch(len(dataset)).take(1)):
#         x_train = i[0] / 255
#         y_train = i[1]
#     for i in tfds.as_numpy(val_set.batch(len(val_set)).take(1)):
#         x_test = i[0] / 255
#         y_test = i[1]
    
#     test_acc = []
#     watermark_acc = []
#     len_steal_list = [250, 500, 1000, 5000, 10000, 20000] 
#     for len_steal in len_steal_list:

#         # attacker_model = models.Plain_2_conv_Keras().call()
#         attacker_model = models.ResNet34()
#         optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
        
#         def train_step(model, images, labels):

#             with tf.GradientTape() as tape:
#                 prediction = model(images)

#                 loss = loss_object(labels, prediction)

#             grads = tape.gradient(loss, model.trainable_weights)
#             optimizer.apply_gradients(zip(grads, model.trainable_weights))


#         indices = np.random.RandomState(seed=42).permutation(len(x_train))

#         x_steal = x_train[indices[:len_steal]]
#         y_steal = np.array(y_train)[indices[:len_steal]]
#         x_train_wo_steal = x_train[indices[len_steal:]]
#         print(type(x_steal))
#         print(isinstance(x_steal, tf.Tensor))
#         y_train_wo_steal = np.array(y_train)[indices[len_steal:]]

#         # classifier_stolen = TensorFlowV2Classifier(attacker_model, use_logits=True)
#         classifier_stolen = TensorFlowV2Classifier(attacker_model, nb_classes=10, loss_object=loss_object, input_shape=(28,28,1), clip_values=(0,1), channels_first=False, train_step=train_step)

#         attack = KnockoffNets(classifier=classifier, batch_size_fit=64, batch_size_query=64, nb_epochs=5, nb_stolen=len_steal,sampling_strategy="random",use_probability=False)

#         classifier_stolen = attack.extract(x_steal, y_steal, thieved_classifier=classifier_stolen) 



#         predictions = classifier_stolen.predict(x_train, batch_size=64)
#         print("Argmax predictions {}".format(np.argmax(predictions, axis=1)))
#         print("Actual y train {}".format(y_train))
#         accuracy = np.sum(np.argmax(predictions, axis=1) == y_train) / len(y_train)
#         print("Accuracy on train examples: {}%".format(accuracy * 100))




#         predictions = classifier_stolen.predict(x_test, batch_size=64)
#         print("Argmax predictions {}".format(np.argmax(predictions, axis=1)))
#         print("Actual y test {}".format(y_test, axis=1))
#         accuracy = np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test)
#         print("Accuracy on test examples: {}%".format(accuracy * 100))
#         test_acc.append(accuracy)

#         info = verify(classifier_stolen, key_set, 0.05)
#         if info["success"]:
#             print("Model is ours and was successfully watermarked.")
#         else:
#             print("Model is not ours and was not successfully watermarked.")

#         correct = 0
#         for x, y in true_advs + false_advs:
#             x = np.expand_dims(x, axis=0)
#             predictions = classifier_stolen.predict(x, batch_size=1)
#             if np.argmax(predictions, axis=1)[0] ==  y.numpy():
#                 correct += 1
            
#         accuracy = correct / len(true_advs + false_advs)
#         print("Accuracy on watermark examples: {}%".format(accuracy * 100))
#         watermark_acc.append(accuracy)


#     plt.figure(figsize=(5,5))
#     plt.plot(len_steal_list, test_acc, label="argmax knockoffnet " + dataset_name + " Test acc ", linestyle='--', marker='o', color='tab:purple')
#     plt.plot(len_steal_list, watermark_acc, label="argmax knockoffnet " + dataset_name + " Watermark acc ", linestyle='--', marker='o', color='tab:orange')
#     plt.xlabel("Stealing Dataset size")
#     plt.ylabel("Stolen Model accuracy")
#     plt.legend()
#     plt.savefig(os.path.join(RESULTS_PATH, LOSS_FOLDER, dataset_name + "TestandWatermarkAcc.png"))
#     mlflow.log_artifact(os.path.join(RESULTS_PATH, LOSS_FOLDER, dataset_name + "TestandWatermarkAcc.png"), "TestandWatermarkAcc.png")


# attacker_model = models.Plain_2_conv_Keras().call()

# attacker_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9), 
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
#               metrics=["sparse_categorical_accuracy"])
# # attacker_model.build(input_shape=(None, 28, 28, 1))

# classifier_stolen = KerasClassifier(attacker_model, use_logits=True)
# attack = KnockoffNets(classifier=classifier, batch_size_fit=64, batch_size_query=64, nb_epochs=5, nb_stolen=len(x_train),sampling_strategy="random",use_probability=False)

# classifier_stolen = attack.extract(x_train, y_train, thieved_classifier=classifier_stolen) 



# predictions = classifier_stolen.predict(x_train, batch_size=64)
# print("Argmax predictions {}".format(np.argmax(predictions, axis=1)))
# print("Actual y train {}".format(np.argmax(y_train, axis=1)))
# accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_train, axis=1)) / len(y_train)
# print("Accuracy on train examples: {}%".format(accuracy * 100))




# predictions = classifier_stolen.predict(x_test, batch_size=64)
# print("Argmax predictions {}".format(np.argmax(predictions, axis=1)))
# print("Actual y test {}".format(np.argmax(y_test, axis=1)))
# accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
# print("Accuracy on test examples: {}%".format(accuracy * 100))

# tf.compat.v1.enable_eager_execution()

# # for x, y in true_advs + false_advs:
# #     predictions = classifier_stolen.predict(x, batch_size=1)
# #     print("Argmax predictions {}".format(np.argmax(predictions, axis=1)))
# #     print("Actual y test {}".format(y, axis=1))
# for x, y in key_set:
#     print(x)
