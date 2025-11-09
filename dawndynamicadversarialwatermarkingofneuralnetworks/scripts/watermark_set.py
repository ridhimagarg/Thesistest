import argparse
import os

import mlconfig
import numpy as np
import torch.utils.data as data

from environment import setup_transformations, download_victim, construct_watermark_dataset


def generate_watermark(victim_data_path, victim_dataset, watermark_dataset, force_greyscale, normalize_with_imagenet_vals, input_size, watermark_size, batch_size):

    training_transforms, watermark_transforms = setup_transformations(victim_dataset, watermark_dataset, force_greyscale, normalize_with_imagenet_vals, input_size)
    train_set, test_set = download_victim(victim_dataset, victim_data_path, training_transforms, return_model=False)

    # watermark_set, train_set = construct_watermark_set(train_set, watermark_size, 10, partition=True)

    watermark_set, train_set = construct_watermark_dataset(train_set, (watermark_size/len(train_set)), 10, (3,224,224), key_length=300)

    print(len(watermark_set))
    print(len(train_set))

    # print([*watermark_set.data,])
    # print(np.array([*watermark_set.data,]))
    # for e in [*watermark_set.data,]:
    #     print(e.shape)

    # watermark_numpy = watermark_set.data
    # print(watermark_numpy)

    # print(watermark_numpy.shape)

    # train_dl = data.DataLoader(train_set, batch_size=len(train_set), shuffle=False)
    # # train_dl = data.DataLoader(train_set, batch_size=len(train_set), shuffle=True)

    # # # print(watermark_set)
    # train_dataset_array = next(iter(train_dl))[0].numpy()
    # train_dataset_label = next(iter(train_dl))[1].numpy()
    # print(train_dataset_array.shape)
    # print(train_dataset_label.shape)

    # if not os.path.exists(os.path.join("../data", "watermark_set", str(victim_dataset))):
    #     os.makedirs(os.path.join("../data", "watermark_set", str(victim_dataset)))

    # np.savez(os.path.join("../data", "watermark_set", str(victim_dataset)  , "train_set_" + str(watermark_size)), train_dataset_array, train_dataset_label)


    watermark_dl = data.DataLoader(watermark_set, batch_size=len(watermark_set), shuffle=False)
    # train_dl = data.DataLoader(train_set, batch_size=len(train_set), shuffle=True)

    # # print(watermark_set)
    watermark_dataset_array = next(iter(watermark_dl))[0].numpy()
    watermark_dataset_label = next(iter(watermark_dl))[1].numpy()
    print(watermark_dataset_array.shape)
    print(watermark_dataset_label.shape)

    if not os.path.exists(os.path.join("../data", "watermark_set", str(victim_dataset))):
        os.makedirs(os.path.join("../data", "watermark_set", str(victim_dataset)))

    np.savez(os.path.join("../data", "watermark_set", str(victim_dataset)  , str(normalize_with_imagenet_vals) + "_watermark_set_" + str(watermark_size)), watermark_dataset_array, watermark_dataset_label)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="../configurations/watermark_set.yaml")

    args = parser.parse_args()
    print(args)
    config = mlconfig.load(args.config)

    for watermark_size in [100, 250]:

        generate_watermark(config["dataset_path"], config["dataset_name"], config["watermarkset_name"], 0, config["normalize_with_imagenet"], config["input_size"], watermark_size, config["batch_size"])