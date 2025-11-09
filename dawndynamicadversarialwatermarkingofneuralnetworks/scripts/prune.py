# Authors: Sebastian Szyller, Buse Gul Atli
# Copyright 2020 Secure Systems Group, Aalto University, https://ssg.aalto.fi
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import configparser
import copy
import datetime
import os
import pickle
import random
from datetime import datetime
from typing import Dict, Any, List, Tuple

import matplotlib.pyplot as plt
from environment import setup_model, download_victim, setup_transformations
import mlconfig
import numpy as np
import torch
import torch as t
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.utils.data as data
import torchvision as tv
from scripts import models
from scripts import score
import mlconfig
import numpy as np
import torch as t
from datetime import datetime
import matplotlib.pyplot as plt
from environment import setup_model, download_victim, setup_transformations

import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

random.seed(42)

now = datetime.now().strftime("%d-%m-%Y")

log = logger.Logger(prefix=">>>")

class SimpleDataset(data.Dataset):
    def __init__(self, dataset: List[Tuple[Any, int]]) -> None:
        self.data, self.labels = zip(*dataset)
        self.count = len(self.labels)

    def __getitem__(self, index: int) -> (Any, int):
        return self.data[index], self.labels[index]

    def __len__(self) -> int:
        return self.count


def main(config: configparser.ConfigParser, model_path: str, watermark_path: str) -> None:
    #  Setup model architecture and load model from file.
    model = setup_model(
        config["DEFAULT"]["model_architecture"],
        model_path,
        int(config["DEFAULT"]["number_of_classes"]))

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model = model.to(device=device)

    #  Load test set and transform it.
    test_set = download_test(
        config["DEFAULT"]["dataset_name"],
        config["DEFAULT"]["test_save_path"],
        int(config["DEFAULT"]["input_size"])
    )
    test_set = data.DataLoader(test_set, batch_size=int(config["DEFAULT"]["batch_size"]))

    watermark_set = load_file(watermark_path)

    pruning_save_path = config["DEFAULT"]["pruning_save_path"]
    if not os.path.exists(pruning_save_path):
        log.warn(pruning_save_path + " does not exist. Creating...")
        os.makedirs(pruning_save_path)
        log.info(pruning_save_path + " Created.")

    pruning_results = prune_model(model, test_set, watermark_set, int(config["DEFAULT"]["number_of_classes"]), device)

    date = datetime.datetime.today().strftime('%Y-%m-%d')
    path_body = pruning_save_path + config["DEFAULT"]["model_name"]

    save_scores(
        pruning_results,
        path_body + date)


def download_test(dataset_name: str, victim_data_path: str, input_size: int) -> data.Dataset:
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    if dataset_name == "MNIST":
        dataset = tv.datasets.MNIST
        transformations = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean, std)
        ])
    elif dataset_name == "CIFAR10":
        dataset = tv.datasets.CIFAR10
        transformations = tv.transforms.Compose([
            tv.transforms.Resize(input_size),
            tv.transforms.CenterCrop(input_size),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean, std)
        ])
    else:
        log.error("MNIST and CIFAR10 are the only supported datasets at the moment. Throwing...")
        raise ValueError(dataset_name)

    test_set = dataset(victim_data_path, train=False, transform=transformations, download=True)

    log.info("Test samples: {}\nSaved in: {}".format(dataset_name, len(test_set), victim_data_path))
    return test_set


def setup_model(model_architecture: str, model_path: str, number_of_classes: int) -> nn.Module:
    available_models = {
        "MNIST_L5": models.MNIST_L5,
        "CIFAR10_BASE": models.CIFAR10_BASE
    }

    model = available_models[model_architecture]()

    if model is None:
        log.error("Incorrect model architecture specified or architecture not available.")
        raise ValueError(model_architecture)

    models.load_state(model, model_path)

    return model


def load_file(file_path: str) -> List[Tuple]:
    with open(file_path, "rb") as f:
        return pickle.load(f)
    

def plot_separation_lines(height, width):

        block_size = [1, 4]

        # Add separation lines to the figure.
        num_hlines = int((height - 1) / block_size[0])
        num_vlines = int((width - 1) / block_size[1])
        line_y_pos = [y * block_size[0] for y in range(1, num_hlines + 1)]
        line_x_pos = [x * block_size[1] for x in range(1, num_vlines + 1)]

        for y_pos in line_y_pos:
            plt.plot([-0.5, width], [y_pos - 0.5 , y_pos - 0.5], color='w')

        for x_pos in line_x_pos:
            plt.plot([x_pos - 0.5, x_pos - 0.5], [-0.5, height], color='w')


def generate_test_set(victim_data_path, victim_dataset, watermark_dataset, force_greyscale, normalize_with_imagenet_vals, input_size, batch_size):

    training_transforms, watermark_transforms = setup_transformations(victim_dataset, watermark_dataset, force_greyscale, normalize_with_imagenet_vals, input_size)
    train_set, test_set = download_victim(victim_dataset, victim_data_path, training_transforms, return_model=False)

    test_dl = data.DataLoader(test_set, batch_size=len(test_set), shuffle=False)

    test_dataset_array = next(iter(test_dl))[0].numpy()
    test_dataset_label = next(iter(test_dl))[1].numpy()

    if not os.path.exists(os.path.join("../data", "test_set", str(victim_dataset))):
        os.makedirs(os.path.join("../data", "test_set", str(victim_dataset)))

    return test_dataset_array, test_dataset_label

def plot_separation_lines(height, width):

        block_size = [1, 4]

        # Add separation lines to the figure.
        num_hlines = int((height - 1) / block_size[0])
        num_vlines = int((width - 1) / block_size[1])
        line_y_pos = [y * block_size[0] for y in range(1, num_hlines + 1)]
        line_x_pos = [x * block_size[1] for x in range(1, num_vlines + 1)]

        for y_pos in line_y_pos:
            plt.plot([-0.5, width], [y_pos - 0.5 , y_pos - 0.5], color='w')

        for x_pos in line_x_pos:
            plt.plot([x_pos - 0.5, x_pos - 0.5], [-0.5, height], color='w')


def generate_test_set(victim_data_path, victim_dataset, watermark_dataset, force_greyscale, normalize_with_imagenet_vals, input_size, batch_size):

    training_transforms, watermark_transforms = setup_transformations(victim_dataset, watermark_dataset, force_greyscale, normalize_with_imagenet_vals, input_size)
    train_set, test_set = download_victim(victim_dataset, victim_data_path, training_transforms, return_model=False)

    test_dl = data.DataLoader(test_set, batch_size=len(test_set), shuffle=False)

    test_dataset_array = next(iter(test_dl))[0].numpy()
    test_dataset_label = next(iter(test_dl))[1].numpy()

    if not os.path.exists(os.path.join("../data", "test_set", str(victim_dataset))):
        os.makedirs(os.path.join("../data", "test_set", str(victim_dataset)))

    return test_dataset_array, test_dataset_label

def prune_model(dataset_name, model_architecture, model_to_prune_path, test_set_path, watermark_set_path) -> Dict[float, Dict[str, Any]]:

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

    file1 = open(os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER, "_".join((dataset_name, str(model_architecture), model_to_prune_path.split('/')[-1].split(".")[0] + "_logs.txt"))), "w")

    model = available_models[model_architecture]()


    # # array = np.load("../data/test_set/CIFAR10/test_set.npz")

    # x_test = test_set_array["arr_0"]
    # y_test = test_set_array["arr_1"]

    if model_architecture == "RN34":
        n_features = model.fc.in_features
        model.fc = nn.Linear(n_features, 10)

        config = mlconfig.load("../configurations/watermark_set.yaml")

        x_test, y_test = generate_test_set(config["dataset_path"], config["dataset_name"], config["watermarkset_name"], 0, config["normalize_with_imagenet"], config["input_size"], config["batch_size"])

    models.load_state(model, model_to_prune_path)

    model.eval()


    # test_set_array = np.load(test_set_path)
    # x_test = test_set_array["arr_0"]
    # y_test = test_set_array["arr_1"]

    watermark_set_array = np.load(watermark_set_path)

    # watermark_set_array = np.load(os.path.join("../data/watermark_set", dataset_name, "watermark_set_250.npz"))
    x_watermark = watermark_set_array["arr_0"]
    y_watermark = watermark_set_array["arr_1"]

    #  Pruning experiment with multiple pruning levels
    pruning_levels = [0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.75, 0.9]
    pruning_results = {}

    log.info("Accuracy before pruning:")
    file1.write("Accuracy before pruning \n")

    predictions = model(t.from_numpy(x_test))
    print(np.sum(np.argmax(predictions.detach().numpy(), axis=1) == y_test))
    acc = np.sum(np.argmax(predictions.detach().numpy(), axis=1) == y_test) / len(y_test)
    print("Test acc:", acc)
    file1.write("Test acc: "+ str(acc) + "\n")

    predictions = model(t.from_numpy(x_watermark))
    print(np.sum(np.argmax(predictions.detach().numpy(), axis=1) == y_watermark))
    acc = np.sum(np.argmax(predictions.detach().numpy(), axis=1) == y_watermark) / len(y_watermark)
    print("Watermark acc:", acc)
    file1.write("Watermark acc: "+ str(acc) +"\n")



    # _ = test_model(model, test_set, number_of_classes, device)
    # _ = test_watermark(model, watermark_set, device)



    for level in pruning_levels:
        model_local = copy.deepcopy(model)
        model_local.eval()
        # parameters_to_prune = model_local.parameters()

        for name, module in model_local.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                prune.l1_unstructured(module, name='weight', amount=level)
                for name1, param in model_local.named_parameters():
                    # print(name, name1)
                    image_save_path = os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER, "_".join((dataset_name, str(level), str(model_architecture), name1 + model_to_prune_path.split('/')[-1].split(".")[0] + "pruning.png")))
                    if name in name1 and "weight_orig" in name1:
                        # print("here")
                        width = height = 24
                        weights_to_display = param.view(param.size()[1:].numel(), param.shape[0]) #tf.reshape(param, [tf.reduce_prod(tensor_data.shape[:-1]), -1])
                        weights_to_display = weights_to_display[0:width, 0:height].detach().numpy()

                        val_ones = np.ones([height, width])
                        val_zeros = np.zeros([height, width])
                        subset_values_to_display = np.where(abs(weights_to_display) > 0, abs(weights_to_display), val_zeros)

                        plt.figure()
                        plot_separation_lines(height, width)

                        plt.axis('off')
                        plt.imshow(subset_values_to_display)
                        plt.colorbar()
                        plt.title(f"Structurally pruned weights for {name} layer")
                        plt.savefig(image_save_path)



            elif isinstance(module, torch.nn.Linear):  
                prune.l1_unstructured(module, name="weight", amount=level)

        log.info("Testing with pruning level {}.".format(level))
        file1.write("Testing with pruning level {} \n".format(level))

        predictions = model_local(t.from_numpy(x_test))
        print(np.sum(np.argmax(predictions.detach().numpy(), axis=1) == y_test))
        acc = np.sum(np.argmax(predictions.detach().numpy(), axis=1) == y_test) / len(y_test)
        print("Test acc:", acc)
        file1.write("Test acc: {} \n".format(acc))

        predictions = model_local(t.from_numpy(x_watermark))
        print(np.sum(np.argmax(predictions.detach().numpy(), axis=1) == y_watermark))
        acc = np.sum(np.argmax(predictions.detach().numpy(), axis=1) == y_watermark) / len(y_watermark)
        print("Watermark acc:", acc)
        file1.write("Watermark acc: {} \n".format(acc))

        # test_float_score, test_dict_score = test_model(model_local, test_set, number_of_classes, device)
        # watermark_float_score = test_watermark(model_local, watermark_set, device)
        # pruning_results[level] = {
        #     "test": (test_float_score, test_dict_score),
        #     "watermark": watermark_float_score
        # }

        

    # subset_values_to_display = tensor_data[0:height, 0:width]

    # val_ones = np.ones([height, width])
    # val_zeros = np.zeros([height, width])
    # subset_values_to_display = np.where(abs(subset_values_to_display) > 0, val_ones, val_zeros)

    return pruning_results

    


def test_model(model: nn.Module, test_set: data.DataLoader, number_of_classes: int, device) -> Tuple[score.FloatScore, score.DictScore]:
    """Test the model on the test dataset."""
    # model.eval is used for ImageNet models, batchnorm or dropout layers will work in eval mode.
    model.eval()

    def test_average() -> score.FloatScore:
        correct = 0
        total = 0

        with torch.set_grad_enabled(False):
            for (inputs, yreal) in tqdm(test_set, unit="images", desc="Testing model (average)", leave=True, ascii=True):
                if device == "gpu":
                    inputs, yreal = inputs.cuda(), yreal.cuda()

                ypred = model(inputs)
                _, predicted = torch.max(ypred.data, 1)
                # print(predicted)
                total += yreal.size(0)
                correct += (predicted == yreal).sum().item()

        accuracy = 100 * correct / total
        log.info("Accuracy of the network on the {} test images (average): {}".format(total, accuracy))
        with open('epoch_logs.txt', 'a+') as file:
            file.write('Test Acc: {}\n'.format(accuracy))
        return score.FloatScore(accuracy)

    def test_per_class() -> score.DictScore:
        class_correct = list(0. for _ in range(number_of_classes))
        class_total = list(0. for _ in range(number_of_classes))
        total = 0

        with torch.no_grad():
            for (inputs, yreal) in tqdm(test_set, unit="images", desc="Testing model (per class)", leave=True, ascii=True):
                if device == "gpu":
                    inputs, yreal = inputs.cuda(), yreal.cuda()

                total += yreal.size(0)

                ypred = model(inputs)
                _, predicted = torch.max(ypred, 1)
                # print(predicted)
                c = (predicted == yreal).squeeze()
                for i in range(yreal.shape[0]):
                    label = yreal[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        log.info("Accuracy of the network on the {} test images (per-class):".format(total))

        per_class_accuracy = {}
        for i in range(number_of_classes):
            accuracy = 100 * class_correct[i] / (class_total[i] + 0.0001)
            per_class_accuracy[i] = accuracy
            print('Accuracy of %5s : %2d %%' % (
                i, accuracy))

        return score.DictScore(per_class_accuracy)

    return test_average(), test_per_class()


def test_watermark(model: nn.Module, watermark_set: List, device) -> score.FloatScore:
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for (inputs, yreal) in tqdm(watermark_set, unit="images", desc="Testing watermark (average)", leave=True, ascii=True):
            if device == "gpu":
                inputs, yreal = inputs.cuda(), yreal.cuda()

            ypred = model(inputs)
            _, predicted = torch.max(ypred.data, 1)
            # print(predicted)
            total += yreal.size(0)
            correct += (predicted.detach().cpu() == yreal.detach().cpu()).sum().item()

    accuracy = 100 * correct / total
    log.info("Accuracy of the network on the {} test images (average): {}".format(total, accuracy))
    return score.FloatScore(accuracy)


def save_scores(pruning_results: Dict[float, Dict[str, Any]], file_path: str) -> None:
    with open(file_path + '.pkl', 'wb') as f:
        pickle.dump(pruning_results, f, pickle.HIGHEST_PROTOCOL)


def handle_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Configuration file for the experiment.")
    parser.add_argument(
        "--watermark",
        type=str,
        default=None,
        help="Path to the saved watermark Loader.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to the saved model.")
    args = parser.parse_args()

    if args.config_file is None:
        raise ValueError("Configuration file must be provided.")

    if args.watermark is None:
        raise ValueError("Watermark path must be provided.")

    if args.config_file is None:
        raise ValueError("Model path must be provided.")

    return args


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="../configurations/pruning.yaml")

    args = parser.parse_args()
    config = mlconfig.load(args.config)

    RESULTS_PATH = f"../results/prune_attcker_model_{now}"
    LOSS_Acc_FOLDER = "losses_acc"
    MODEL_PATH = f"../models/attack_original_{now}"
    DATA_PATH = "../data"

    if not os.path.exists(os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER)):
        os.makedirs(os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER))

    prune_model(config["dataset"], config["model_finetune_name"], config["attacker_model_path"], config["test_set_path"], config["watermark_path"])



    # args = handle_args()
    # config = config_helper.load_config(args.config_file)
    # watermark_path = args.watermark
    # model_path = args.model

    # config_helper.print_config(config)
    # log.info("Model path: {}.".format(model_path))
    # log.info("Watermark path: {}".format(watermark_path))

    # main(config, model_path, watermark_path)
