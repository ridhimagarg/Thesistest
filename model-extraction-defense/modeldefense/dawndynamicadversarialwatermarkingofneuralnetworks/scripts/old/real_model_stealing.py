import os
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
from art.attacks.extraction import KnockoffNets
from art.estimators.classification.pytorch import PyTorchClassifier
from torch.utils.data import DataLoader
## custom libraries
# import utils
# import utils.config_helper as config_helper
from utils import config_helper
from utils import logger

import environment

log = logger.Logger(prefix=">>>")
logging_path = "old/logging"
results_path = "results"

# -> Tuple[NamedTuple, t.nn.Module, PyTorchClassifier, dict, np.ndarray, np.ndarray]

def setup_victim_attacker(config_file: str):

    config = config_helper.load_config(config_file)
    env = environment.prepare_environment(config)

    ## ============================= Victim model ==============================#
    victim_model = env.training_ops.victim_model
    print("Victim model", victim_model)

    ## checking the parameters of the victim model.
    # for name, param in victim_model.named_parameters():
    #     print(name, param)

    ## converting pytorch victim model to ART PyTorchClassifier
    classifier_victim = PyTorchClassifier(victim_model, loss = nn.functional.cross_entropy, input_shape= (3,224,224) , nb_classes=10, device_type="gpu", optimizer = t.optim.SGD(victim_model.parameters(), lr=0.1, momentum=0.5, weight_decay=5e-4))
    print("here")

    # print("weights", classifier_victim._model.named_parameters())
    # for name, param in classifier_victim._model.named_parameters():
    #     print(f"name {name} and param is {param}")



    ## ============================ Adeversary Attack ===============================##

    
    ## getting the training dataset and test dataset
    # train_set = env.training_ops.training_set
    # test_set = env.test_ops.testing_set
    watermark_set = env.watermark_ops.watermarking_set
    print("her1")

    # x_train_numpy = np.load("../data/datasets/cifar10train.npz")["arr_0"]
    # y_train_numpy = np.load("../data/datasets/cifar10train.npz")["arr_1"]


    ## MNIST dataset as numpy array, numpy array is used because loader with the full batch size was taking a lot of time.
    with open(os.path.join("../data/datasets", "mnist.pkl"), 'rb') as f:
        mnist = pickle.load(f)
        x_train, y_train, x_test, y_test = mnist["training_images"], mnist["training_labels"], \
                                           mnist["test_images"], mnist["test_labels"]
        x_train_numpy = np.array(np.reshape(x_train / 255, [-1, 1, 28, 28]), dtype=np.float32)
        x_test_numpy = np.array(np.reshape(x_test / 255, [-1, 1, 28, 28]), dtype=np.float32)
        x_train_numpy = x_train_numpy - 0.5 / 0.5 ## normalising as done in pytorch code of DAWN
        x_test_numpy = x_test_numpy - 0.5 / 0.5
        y_train_numpy = y_train
        y_test_numpy = y_test
    # train_loader_adv = DataLoader(train_set, batch_size=len(train_set))
    # x_train_numpy = next(iter(train_loader_adv))[0].numpy()
    # y_train_numpy = next(iter(train_loader_adv))[1].numpy()
    # print("her2")

    watermark_loader_adv = DataLoader(watermark_set, batch_size=len(watermark_set))
    x_watermark_numpy = next(iter(watermark_loader_adv))[0].numpy()
    y_watermark_numpy = next(iter(watermark_loader_adv))[1].numpy()
    print("her3")

    print("Train set")
    print(x_train_numpy[0])
    print("Watermark set")
    print(x_watermark_numpy[0])

    ## access the numpy arrays from the dataset using dataloader
    # x_test_numpy = np.load("../data/datasets/cifar10test.npz")["arr_0"]
    # y_test_numpy = np.load("../data/datasets/cifar10test.npz")["arr_1"]
    # test_loader_adv = DataLoader(test_set, batch_size=len(test_set))
    # x_test_numpy = next(iter(test_loader_adv))[0].numpy()
    # y_test_numpy = next(iter(test_loader_adv))[1].numpy()
    # print("her4")

    return env, victim_model, classifier_victim, x_train_numpy, y_train_numpy, x_test_numpy, y_test_numpy, x_watermark_numpy, y_watermark_numpy, config



#======================================================================================================#


def train_attacker(env, classifier_victim , x_train_numpy, y_train_numpy, x_test_numpy, y_test_numpy , x_watermark_numpy, y_watermark_numpy, len_steal, num_epochs, config):

    attack_catalogue = {
                        # "probabilistic_knockoffNets": KnockoffNets(classifier=classifier_victim,
                        #                         batch_size_fit=64,
                        #                         batch_size_query=64,
                        #                         nb_epochs=num_epochs,
                        #                         nb_stolen=len_steal,
                        #                         use_probability=True),
                        "argmax_knockoffNets": KnockoffNets(classifier=classifier_victim,
                                                batch_size_fit=64,
                                                batch_size_query=64,
                                                nb_epochs=num_epochs,
                                                nb_stolen=len_steal,
                                                use_probability=False),
                    }


    indices = np.random.RandomState(seed=42).permutation(len(x_train_numpy))

    x_steal = x_train_numpy[indices[:len_steal]]
    y_steal = np.array(y_train_numpy)[indices[:len_steal]]
    x_train_wo_steal = x_train_numpy[indices[len_steal:]]
    y_train_wo_steal = np.array(y_train_numpy)[indices[len_steal:]]

    env = environment.prepare_environment(config)

    results_watermark_cifar10 = [("argmax knockoffnet cifar10", 250, 64.8, 95.6), ("argmax knockoffnet cifar10", 1000, 69.4, 98.8), ("argmax knockoffnet cifar10", 2500, 83.2, 98.0), ("argmax knockoffnet cifar10", 3000, 90.8, 88.8)]
    results_watermark_mnist = [("argmax knockoffnet MNIST", 250, 90, 2.0), ("argmax knockoffnet MNIST", 1000, 95.6, 1.9), ("argmax knockoffnet MNIST", 2000, 92.0, 1.5), ("argmax knockoffnet MNIST", 3000, 98.8, 2.2)]

    # results_watermark = []

    for name, attack in attack_catalogue.items():

        log.info(f"Attack is {name}")

        attacker_model = env.watermark_ops.attacker_model
        classifier_stolen = PyTorchClassifier(attacker_model, loss = nn.CrossEntropyLoss(), input_shape= (3,224,224) , nb_classes=10, device_type="gpu", optimizer = t.optim.SGD(attacker_model.parameters(), lr=0.1))
        classifier_stolen = attack.extract(x_steal, y_steal, thieved_classifier=classifier_stolen, x_watermark=x_watermark_numpy, y_watermark=y_watermark_numpy)

        # for name, param in classifier_stolen._model.named_parameters():
        #     print(f"name {name}, param {param}")
        #     print("requires grad", param.requires_grad)
        
        # print("State", classifier_stolen.__getstate__())

        # current_state = classifier_stolen.__getstate__()

        classifier_stolen.save("model" + str(name) + str(env.training_ops.dataset_name) + str(len_steal), "../data/models/attacks/knockoff")

        ## --------------- Prediction without DAWN -----------------------##

        ## On test data
        predictions = classifier_stolen.predict(x_test_numpy[0:500], batch_size=64)

        log.info("Argmax predictions {}".format(np.argmax(predictions, axis=1)))
        log.info("Actual y test {}".format(y_test_numpy[0:500]))

        accuracy = np.sum(np.argmax(predictions, axis=1) == y_test_numpy[0:500]) / len(y_test_numpy[0:500])
        test_acc = accuracy
        log.info("Accuracy on test examples: {}%".format(accuracy * 100))
        log.info(f"name {name} : accuracy {accuracy}")

        with open(os.path.join(logging_path, 'metrics_with_dawn_'+ str(env.training_ops.dataset_name) + str(name) + "_" +str(len_steal)  +'.txt'), 'w') as file:
            file.write('With epochs {} Predictions: {} Actual: {}  on test with acc: {}\n'.format(num_epochs, predictions, y_test_numpy[0:500], accuracy))

        ## On reamining train data
        predictions = classifier_stolen.predict(x_train_wo_steal[0:500], batch_size=64)

        log.info("Argmax predictions {}".format(np.argmax(predictions[0:500], axis=1)))
        log.info("Actual y test {}".format(y_train_wo_steal))

        accuracy = np.sum(np.argmax(predictions, axis=1) == y_train_wo_steal[0:500]) / len(y_train_wo_steal[0:500])
        log.info("Accuracy on remaining train examples: {}%".format(accuracy * 100))
        log.info(f"name {name} : accuracy {accuracy}")

        with open(os.path.join(logging_path, 'metrics_with_dawn_'+ str(env.training_ops.dataset_name) + str(name) + "_" +str(len_steal)  +'.txt'), 'a+') as file:
            file.write('With epochs {} Predictions: {} Actual: {}  on train with acc: {}\n'.format(num_epochs, predictions, y_train_wo_steal[0:500], accuracy))

    

        # classifier_stolen = PyTorchClassifier(attacker_model, loss = nn.CrossEntropyLoss(), input_shape= (3,32,32) , nb_classes=10, device_type="cpu", optimizer = t.optim.SGD(attacker_model.parameters(), lr=0.1))
        # # # print("Len of xsteal", x_steal.shape)
        # classifier_stolen.__setstate__(current_state)
        # print("x watermark", x_watermark_numpy)



        ##------------------------------------- Training attacker on the watermark set -----------------------##
        # classifier_stolen = attack.extract(x_watermark_numpy, y_watermark_numpy, thieved_classifier=classifier_stolen, is_watermark=True)

        ## ## --------------- Prediction after  DAWN -----------------------##

        #########################  On test data
        # predictions = classifier_stolen.predict(x_test_numpy)
        # log.info("Argmax predictions {}".format(np.argmax(predictions, axis=1)))
        # log.info("Actual y test {}".format(y_test_numpy))

        # accuracy = np.sum(np.argmax(predictions, axis=1) == y_test_numpy) / len(y_test_numpy)
        # log.info("Accuracy on test examples: {}%".format(accuracy * 100))
        # log.info(f"name {name} : accuracy {accuracy}")

        # with open(os.path.join(logging_path, 'metrics_with_dawn_'+ str(env.training_ops.dataset_name) +  str(name) + "_" +str(len_steal)  +'.txt'), 'a+') as file:
        #     file.write('With epochs {} Predictions: {} Actual: {} on test with acc after DAWN: {}\n'.format(num_epochs, predictions, y_test_numpy, accuracy))



        # ####################### On remaining traiing data
        # predictions = classifier_stolen.predict(x_train_wo_steal[0:500])

        # log.info("Argmax predictions {}".format(np.argmax(predictions, axis=1)))
        # log.info("Actual y test {}".format(y_train_wo_steal[0:500]))

        # accuracy = np.sum(np.argmax(predictions, axis=1) == y_train_wo_steal[0:500]) / len(y_train_wo_steal[0:500])
        # log.info("Accuracy on remaining train examples: {}%".format(accuracy * 100))
        # log.info(f"name {name} : accuracy {accuracy}")

        # with open(os.path.join(logging_path, 'metrics_with_dawn_'+ str(env.training_ops.dataset_name) + str(name) + "_" +str(len_steal)  +'.txt'), 'a+') as file:
        #     file.write('With epochs {} Predictions: {} Actual: {}  on train with acc after DAWN: {}\n'.format(num_epochs, predictions, y_train_wo_steal[0:500], accuracy))



        #######################3 On Watermark set.
        predictions = classifier_stolen.predict(x_watermark_numpy)
        log.info("Argmax predictions {}".format(np.argmax(predictions, axis=1)))
        log.info("Actual y watermark {}".format(y_watermark_numpy)) 

        accuracy = np.sum(np.argmax(predictions, axis=1) == y_watermark_numpy) / len(y_watermark_numpy)
        watermark_acc = accuracy
        log.info("Accuracy on watermmark examples: {}%".format(accuracy * 100))
        log.info(f"name {name} : accuracy {accuracy}")

        with open(os.path.join(logging_path, 'metrics_with_dawn_'+ str(env.training_ops.dataset_name) +  str(name) + "_" +str(len_steal)  +'.txt'), 'a+') as file:
            file.write('With epochs {} Predictions: {} Actual: {} on watermark with acc: {}\n'.format(num_epochs, predictions, y_watermark_numpy, accuracy))

        
        results_watermark = (name+str(env.training_ops.dataset_name), len_steal, watermark_acc, test_acc)
        plot_save_path = os.path.join(results_path, name+str(env.training_ops.dataset_name)+".png")


    return results_watermark, plot_save_path


#======================================================================================================#


def main(len_steals, config_path):

    num_epochs = 50
    env, victim_model, classifier_victim, x_train_numpy, y_train_numpy, x_test_numpy, y_test_numpy, x_watermark_numpy, y_watermark_numpy, config = setup_victim_attacker(config_path)
    print("Training attackers")
    results_watermark_list = []
    for len_steal in len_steals:
        results_watermark, plot_save_path = train_attacker(env, classifier_victim, x_train_numpy, y_train_numpy, x_test_numpy, y_test_numpy , x_watermark_numpy, y_watermark_numpy, len_steal, num_epochs, config)
        results_watermark_list.append(results_watermark)
    
    return results_watermark_list, plot_save_path


if __name__ == "__main__":
    # main([2500, 10000, 20000, 30000], "../configurations/perfect/cifar-to-cifar-ws250-rn34_decay.ini")


    results_watermark_list, plot_save_path = main([250, 1000, 2500, 3000], "../configurations/perfect/mnist-to-mnist-ws250-l5_decay.ini")


    df_protected_cifar10 = pd.DataFrame(results_watermark_list, columns=('Method Name', 'Stealing Dataset Size', 'Watermark Accuracy', 'Test Accuracy'))
    # df_protected_mnist = pd.DataFrame(results_watermark_mnist, columns=('Method Name', 'Stealing Dataset Size', 'Watermark Accuracy', 'Test Accuracy'))
    fig, ax = plt.subplots(figsize=(8,6))
    ax.set_xlabel("Stealing Dataset Size")
    ax.set_ylabel("Stolen Model Accuracy")
    # print(df_protected_cifar10.groupby("Method Name"))
    for name, group in df_protected_cifar10.groupby("Method Name"):
        print(name, group['Stealing Dataset Size'])
        for row, val in enumerate(group["Watermark Accuracy"]):
            print(row)
            print(val)
    for name, group in df_protected_cifar10.groupby("Method Name"):

        group.plot(1, 2, ax=ax, label=name + " Watermark Acc")
        for row, val in enumerate(group["Watermark Accuracy"]):
            plt.annotate(str(val), xy=(group['Stealing Dataset Size'][row], val), fontsize=13)

        group.plot(1, 3, ax=ax, label=name + " Test Acc" )
        for row, val in enumerate(group["Test Accuracy"]):
            plt.annotate(str(val), xy=(group['Stealing Dataset Size'][row], val), fontsize=13)

    
    plt.savefig(plot_save_path)