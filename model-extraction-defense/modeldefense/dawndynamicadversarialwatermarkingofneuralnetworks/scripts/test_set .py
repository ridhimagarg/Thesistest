import argparse
import os

import mlconfig
import torch.nn as nn
import torch.utils.data as data
import torchvision as tv

import models
from environment import setup_transformations, download_victim


def generate_test_set(victim_data_path, victim_dataset, watermark_dataset, force_greyscale, normalize_with_imagenet_vals, input_size, batch_size):

    training_transforms, watermark_transforms = setup_transformations(victim_dataset, watermark_dataset, force_greyscale, normalize_with_imagenet_vals, input_size)
    train_set, test_set = download_victim(victim_dataset, victim_data_path, training_transforms, return_model=False)

    # watermark_set, train_set = construct_watermark_set(train_set, watermark_size, 10, partition=True)

    # print([*watermark_set.data,])
    # print(np.array([*watermark_set.data,]))
    # for e in [*watermark_set.data,]:
    #     print(e.shape)

    # watermark_numpy = watermark_set.data
    # print(watermark_numpy)

    # print(watermark_numpy.shape)

    test_dl = data.DataLoader(test_set, batch_size=len(test_set), shuffle=False)

    # victim_model = models.MNIST_L2_DRP03()  
    # models.load_state(victim_model, "../models/original_25-09-2023/MNIST_L2_DRP03_victim_mnist_l2.pt")

    victim_model = tv.models.resnet34()

    n_features = victim_model.fc.in_features
    victim_model.fc = nn.Linear(n_features, 10)

    models.load_state(victim_model, "../data/models/victim_cifar_rn34.pt")

    total = 0
    correct = 0

    # for inputs, yreal in test_dl:
    #     ypred = victim_model(inputs)
    #     _, predicted = torch.max(ypred.data, 1)

    #     total += yreal.size(0)
    #     correct += (predicted == yreal).sum().item()
    #     print("predicted", predicted)
    #     print("correct", correct)

    # accuracy = 100 * correct / total
    # print(accuracy)
    # y_pred = np.argmax(victim_model(torch.tensor(data)).detach().numpy(), axis=1)
    # print(np.sum(y_pred == y_true))

    # print(watermark_set)
    test_dataset_array = next(iter(test_dl))[0].numpy()
    test_dataset_label = next(iter(test_dl))[1].numpy()
    print(test_dataset_array.shape)
    print(test_dataset_label.shape)

    # ypred = victim_model(torch.tensor(test_dataset_array))
    # print(np.argmax(ypred.detach().numpy(), axis=1))
    # print()



    if not os.path.exists(os.path.join("../data", "test_set", str(victim_dataset))):
        os.makedirs(os.path.join("../data", "test_set", str(victim_dataset)))

    

    # np.savez(os.path.join("../data", "test_set", str(victim_dataset), str(normalize_with_imagenet_vals) + "_test_set.npz"), test_dataset_array, test_dataset_label)

    # data_test = np.load(os.path.join("../data", "test_set", str(victim_dataset)  , "test_set.npz"))["arr_0"]
    # label = np.load(os.path.join("../data", "test_set", str(victim_dataset)  , "test_set.npz"))["arr_1"]
    # ypred = victim_model(torch.tensor(data_test))
    # print(np.argmax(ypred.detach().numpy(), axis=1))

    return test_dataset_array, test_dataset_label



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="../configurations/watermark_set.yaml")

    args = parser.parse_args()
    print(args)
    config = mlconfig.load(args.config)

    # for watermark_size in [50, 100, 250]:

    generate_test_set(config["dataset_path"], config["dataset_name"], config["watermarkset_name"], 0, config["normalize_with_imagenet"], config["input_size"], config["batch_size"])