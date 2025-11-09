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
import copy
import os
import random
from typing import Dict, Any, List, Tuple
import pandas as pd

import matplotlib.pyplot as plt
import mlconfig
import numpy as np
import pandas as pd
import torch
import torch as t
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
from datetime import datetime
import matplotlib.pyplot as plt
import os
from environment import setup_model, download_victim, setup_transformations

os.environ["CUDA_VISIBLE_DEVICES"]="2"

random.seed(42)

log = logger.Logger(prefix=">>>")

now = datetime.now().strftime("%d-%m-%Y")

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


def prune_model(dataset_name, model_architecture, model_to_prune_path, test_set_path, watermark_set_path, epochs_pruning) -> Dict[float, Dict[str, Any]]:

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

    file1 = open(os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER, "_".join((dataset_name,  str(epochs_pruning), str(model_architecture), model_to_prune_path.split('/')[-1].split(".")[0] + "_logs.txt"))), "w")

    test_df_csv = os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER, "_".join((dataset_name, str(model_architecture), str(epochs_pruning), model_to_prune_path.split('/')[-1].split(".")[0] + "test_df.csv")))

    watermark_df_csv = os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER, "_".join((dataset_name, str(model_architecture), str(epochs_pruning), model_to_prune_path.split('/')[-1].split(".")[0] + "watermark_df.csv")))

    

    model = available_models[model_architecture]()

    test_set_array = np.load(test_set_path)
    x_test = test_set_array["arr_0"]
    y_test = test_set_array["arr_1"]

    if model_architecture == "RN34":
        n_features = model.fc.in_features
        model.fc = nn.Linear(n_features, 10)

        config = mlconfig.load("../configurations/watermark_set.yaml")

        x_test, y_test = generate_test_set(config["dataset_path"], config["dataset_name"], config["watermarkset_name"], 0, config["normalize_with_imagenet"], config["input_size"], config["batch_size"])

    models.load_state(model, model_to_prune_path)

    model.eval()

    watermark_set_array = np.load(watermark_set_path)
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

    t.cuda.empty_cache()

    # if not os.path.exists(test_df_csv):

    final_df_test = pd.DataFrame(columns=("Pruning Level", str(epochs_pruning)))
    final_df_watermark = pd.DataFrame(columns=("Pruning Level", str(epochs_pruning)))

    # else:

    #     final_df_test = pd.read_csv(test_df_csv)
    #     final_df_watermark = pd.read_csv(watermark_df_csv)



    for _ in range(2):

        results_test = [] #{str(epochs_pruning): []}
        results_watermark = [] #{str(epochs_pruning): []}

        for level in pruning_levels:
            model_local = copy.deepcopy(model)
            # parameters_to_prune = model_local.parameters()

            if not os.path.exists(os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER, dataset_name , str(epochs_pruning))):
                os.makedirs(os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER, dataset_name , str(epochs_pruning)))

            for name, module in model_local.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    prune.l1_unstructured(module, name='weight', amount=level)
                    for name1, param in model_local.named_parameters():
                        # print(name, name1)
                        image_save_path = os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER, dataset_name , str(epochs_pruning), "_".join((str(level), str(name1), str(model_architecture), name1 + model_to_prune_path.split('/')[-1].split(".")[0] + "pruning.png")))
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

            log.info("Testing with fine pruning level {}.".format(level))
            file1.write("\nTesting with fine pruning level {} \n".format(level))

            # predictions = model(t.from_numpy(x_test))
            # print(np.sum(np.argmax(predictions.detach().numpy(), axis=1) == y_test))
            # acc = np.sum(np.argmax(predictions.detach().numpy(), axis=1) == y_test) / len(y_test)
            # print("Test acc:", acc)
            # file1.write("Test acc: {} \n".format(acc))

            # predictions = model(t.from_numpy(x_watermark))
            # print(np.sum(np.argmax(predictions.detach().numpy(), axis=1) == y_watermark))
            # acc = np.sum(np.argmax(predictions.detach().numpy(), axis=1) == y_watermark) / len(y_watermark)
            # print("Watermark acc:", acc)
            # file1.write("Watermark acc: {} \n".format(acc))
            test_acc, watermark_acc = train(model_local, epochs_pruning, x_test, y_test, x_watermark, y_watermark, file1)

            # results_test[str(epochs_pruning)].append(test_acc)
            # results_watermark[str(epochs_pruning)].append(watermark_acc)

            results_test.append((level, test_acc))
            results_watermark.append((level, watermark_acc))

            t.cuda.empty_cache()

            if not os.path.exists(os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER, "after_finetuning", dataset_name, str(epochs_pruning))):
                os.makedirs(os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER, "after_finetuning", dataset_name, str(epochs_pruning)))

            for name1, param in model_local.named_parameters():
                # print(name, name1)
                image_save_path = os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER, "after_finetuning", dataset_name, str(epochs_pruning), "_".join((str(level), str(name1), str(model_architecture), name1 + model_to_prune_path.split('/')[-1].split(".")[0] + "pruning.png")))
                if "weight_orig" in name1 and "conv" in name1:
                    # print("here")
                    width = height = 24
                    weights_to_display = param.view(param.size()[1:].numel(), param.shape[0]) #tf.reshape(param, [tf.reduce_prod(tensor_data.shape[:-1]), -1])
                    weights_to_display = weights_to_display[0:width, 0:height].cpu().detach().numpy()

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


        test_df = pd.DataFrame(results_test, columns=("Pruning Level", str(epochs_pruning)))
        watermark_df = pd.DataFrame(results_watermark, columns=("Pruning Level", str(epochs_pruning)))
        final_df_test = pd.concat([final_df_test, test_df])
        final_df_watermark = pd.concat([final_df_watermark, watermark_df])

        # if not os.path.exists(test_df_csv):

        #     # test_df = pd.DataFrame(results_test, index=[str(level) for level in pruning_levels])
        #     test_df = pd.DataFrame(results_test, columns=("Pruning Level", "Accuracy"))
        #     final_df_test = pd.concat([final_df_test, test_df])

        # else:

        #     test_df = pd.read_csv(test_df_csv, index_col=0)

        #     test_df[str(epochs_pruning)] = results_test[str(epochs_pruning)]

    


        # if not os.path.exists(watermark_df_csv):

        #     watermark_df = pd.DataFrame(results_watermark, index=[str(level) for level in pruning_levels])

        #     final_df_watermark = pd.concat([final_df_watermark, watermark_df])

        # else:

        #     watermark_df = pd.read_csv(watermark_df_csv, index_col=0)

        #     watermark_df[str(epochs_pruning)] = results_watermark[str(epochs_pruning)]

    final_df_test.to_csv(test_df_csv)

    final_df_watermark.to_csv(watermark_df_csv)



    return pruning_results


def train(model, epochs, x_test, y_test, x_watermark, y_watermark, file1):

    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.functional.cross_entropy

    model = model.cuda()

    # tensor = torch.from_numpy(data)

    combined_x = np.concatenate([x_watermark, x_test], axis=0)
    combined_y = np.concatenate([y_watermark, y_test], axis=0)



    dataset = TensorDataset(torch.from_numpy(combined_x), torch.from_numpy(combined_y))

    batch_size = 64
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):

        model.train()

        for i, (inputs, yreal) in enumerate(data_loader):

            inputs, yreal = inputs.cuda(), yreal.cuda()

            optimizer.zero_grad()
            ypred = model(inputs)
            loss = criterion(ypred, yreal)
            loss.backward()
            optimizer.step()

    model.eval()
    x_test = t.from_numpy(x_test).cuda()
    y_test = t.from_numpy(y_test).cuda()
    predictions = model(x_test)
    print(np.sum(np.argmax(predictions.cpu().detach().numpy(), axis=1) == y_test.cpu().detach().numpy()))
    acc = np.sum(np.argmax(predictions.cpu().detach().numpy(), axis=1) == y_test.cpu().detach().numpy()) / len(y_test.cpu())
    print("Test acc:", acc)
    file1.write("Test acc: {} \n".format(acc))

    test_acc = acc

    x_watermark = t.from_numpy(x_watermark).cuda()
    y_watermark = t.from_numpy(y_watermark).cuda()

    predictions = model(x_watermark)
    print(np.sum(np.argmax(predictions.cpu().detach().numpy(), axis=1) == y_watermark.cpu().detach().numpy()))
    acc = np.sum(np.argmax(predictions.cpu().detach().numpy(), axis=1) == y_watermark.cpu().detach().numpy()) / len(y_watermark.cpu())
    print("Watermark acc:", acc)
    file1.write("Watermark acc: {} \n".format(acc))

    watermark_acc = acc

    return test_acc, watermark_acc




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="../configurations/pruning.yaml")

    args = parser.parse_args()
    config = mlconfig.load(args.config)

    RESULTS_PATH = f"../results/fineprune_attcker_model_{now}"
    LOSS_Acc_FOLDER = "losses_acc"
    MODEL_PATH = f"../models/attack_original_{now}"
    DATA_PATH = "../data"

    if not os.path.exists(os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER)):
        os.makedirs(os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER))

    prune_model(config["dataset"], config["model_finetune_name"], config["attacker_model_path"], config["test_set_path"], config["watermark_path"], config["epochs_pruning"])



    # args = handle_args()
    # config = config_helper.load_config(args.config_file)
    # watermark_path = args.watermark
    # model_path = args.model

    # config_helper.print_config(config)
    # log.info("Model path: {}.".format(model_path))
    # log.info("Watermark path: {}".format(watermark_path))

    # main(config, model_path, watermark_path)
