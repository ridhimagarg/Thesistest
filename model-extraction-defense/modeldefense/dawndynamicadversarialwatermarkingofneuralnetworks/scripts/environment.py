# Authors: Sebastian Szyller, Buse Gul Atli
# Copyright 2020 Secure Systems Group, Aalto University, https://ssg.aalto.fi
# Modified by: @Ridhima Garg

"""
@author: Ridhima Garg

Introduction:
    This file contains the implementation to setup the victim or the attacker model
"""


import configparser
import os
import pickle
import random
from collections import namedtuple
from pathlib import Path
from typing import List, Tuple, Any, Dict, NamedTuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision as tv
import yaml
from scripts import filter as watermark_filter
from scripts import models
# from ultralytics import YOLO
from utils import logger

random.seed(0)
torch.manual_seed(0)



class SimpleDataset(data.Dataset):
    """
    Main idea
    -------
    This is the class used below to create the watermark dataset.

    """

    def __init__(self, dataset: List[Tuple[Any, int]]) -> None:
        self.data, self.labels = zip(*dataset)
        self.count = len(self.labels)
        # print("count", self.count)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        # print(self.data[index])
        # print(self.labels[index])
        return self.data[index], self.labels[index]

    def __len__(self) -> int:
        return self.count


log = logger.Logger(prefix=">>>")


def prepare_environment(config: configparser.ConfigParser) -> NamedTuple:

    """
    Main idea
    ----------
    Setting up the environment for attacker or victim
    environment means the tain dataset, test dataset, watermark dataset.
    This is the main function of this file which is usually called outside.
    Other functions in this file are being in this function except for "construct_watermark_dataset": which is used in real_model_stealing time.
    Returns
    ------
    nemed tuple

    """

    print(config.keys())
    model_save_path = config["VICTIM"]["model_save_path"]
    create_dir_if_doesnt_exist(model_save_path)

    scores_save_path = config["DEFAULT"]["scores_save_path"]
    create_dir_if_doesnt_exist(scores_save_path)

    # DOWNLOAD DATASETS
    victim_dataset = config["DEFAULT"]["dataset_name"]
    victim_data_path = config["VICTIM"]["victim_dataset_save_path"]
    watermark_data_path = config["ATTACKER"]["watermark_dataset_save_path"]
    watermark_dataset = config["ATTACKER"]["watermark_set"]
    batch_size = int(config["DEFAULT"]["batch_size"])
    number_of_classes = int(config["DEFAULT"]["number_of_classes"])
    problem_statement = config["DEFAULT"]["PROBLEM"]
    force_greyscale = config["ATTACKER"].getboolean("force_greyscale")
    normalize_with_imagenet_vals = config["ATTACKER"].getboolean("normalize_with_imagenet_vals")

    input_size = int(config["DEFAULT"]["input_size"])
    watermark_size = int(config["ATTACKER"]["watermark_size"])

    # DOWNLOAD VICTIM DATASET
    if problem_statement == "OBJECTDETECTION":
        training_transforms = None
        watermark_transforms = None
        return_model = True
    else:
        return_model = False

    if return_model: ## object detection
        train_set, test_set, model = download_victim(victim_dataset, victim_data_path, training_transforms, return_model)
        watermark_set, train_set = construct_watermark_object_set(train_set, watermark_size, partition=True, model=model) ## different function is used to create the watermark set in case of object detection.

    else: ## classification
        training_transforms, watermark_transforms = setup_transformations(victim_dataset, watermark_dataset, force_greyscale, normalize_with_imagenet_vals, input_size)
        train_set, test_set = download_victim(victim_dataset, victim_data_path, training_transforms, return_model)

    # if problem_statement == "OBJECTDETECTION":
    #     watermark_set, train_set = construct_watermark_object_set(train_set, watermark_size, number_of_classes, partition=True, model)

    # SUBCLASS TRAINING SET IF THE SETS ARE THE SAME, OTHERWISE JUST TAKE SAMPLES

    # DOWNLOAD WATERMARK DATASET
    if problem_statement != "OBJECTDETECTION":
        if victim_dataset == watermark_dataset:
            watermark_set, train_set = construct_watermark_set(train_set, watermark_size, number_of_classes, partition=True)
        else:
            watermark_set = download_watermark(watermark_dataset, watermark_data_path, watermark_transforms)
            watermark_set, _ = construct_watermark_set(watermark_set, watermark_size, number_of_classes, partition=False)

        training_set = train_set
        watermarking_set = watermark_set
        testing_set = test_set

        train_set, val_set = torch.utils.data.random_split(
        train_set, [40000, len(train_set)-40000])

        train_set = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_set = data.DataLoader(val_set, batch_size=batch_size, shuffle=True)
        test_set = data.DataLoader(test_set, batch_size=batch_size)
        watermark_set = data.DataLoader(watermark_set, batch_size=batch_size)

    # SETUP VICTIM MODEL
    victim_retrain = config["VICTIM"].getboolean("retrain")
    victim_model_path = model_save_path + config["VICTIM"]["model_name"]
    victim_model_path_best = model_save_path + config["VICTIM"]["model_name_best"]
    victim_model = setup_model(
        retrain=victim_retrain,
        resume=config["VICTIM"].getboolean("resume"),
        model_architecture=config["VICTIM"]["model_architecture"],
        model_path=victim_model_path,
        number_of_classes=number_of_classes,
        problem_statement= problem_statement
    )

    # SETUP ATTACKER MODEL
    attacker_retrain = config["ATTACKER"].getboolean("retrain")
    attacker_model_path = model_save_path + config["ATTACKER"]["model_name"]
    attacker_model = setup_model(
        retrain=attacker_retrain,
        resume=config["ATTACKER"].getboolean("resume"),
        model_architecture=config["ATTACKER"]["model_architecture"],
        model_path=attacker_model_path,
        number_of_classes=number_of_classes,
        problem_statement = problem_statement
    )

    # SETUP TRAINING PROCEDURE
    criterion = nn.functional.cross_entropy
    optimizer = optim.Adam
    TrainingOps = namedtuple("TrainingOps",
        [   
            "epochs",
            "criterion",
            "optimizer",
            "victim_model",
            "dataset_name",
            "use_cuda",
            "training_set",
            "training_loader", ## incase of objectdetection, it will not be loader but the filepath as expected by the YOLOV8
            "validation_loader",
            "resume_from_checkpoint_path",
            "victim_model_architecture",
            "attacker_model_architecture"
        ])

    training_ops = TrainingOps(
        int(config["DEFAULT"]["epochs"]),
        criterion,
        optimizer,
        victim_model,
        config["DEFAULT"]["dataset_name"],
        config["DEFAULT"]["use_cuda"],
        training_set,
        train_set,
        val_set,
        get_with_default(config, "STRATEGY", "resume_from_checkpoint_path", str),
        config["VICTIM"]["model_architecture"],
        config["ATTACKER"]["model_architecture"]
    )

    # SETUP TEST PROCEDURE
    TestOps = namedtuple("TestOps", ["testing_set", "test_loader", "use_cuda", "batch_size", "number_of_classes"])
    test_ops = TestOps(
        testing_set,
        test_set,
        config["DEFAULT"]["use_cuda"],
        batch_size,
        number_of_classes
    )

    # print("test cuda", config["DEFAULT"]["use_cuda"])
    # SETUP WATERMARK EMBEDDING
    WatermarkOps = namedtuple("WatermarkOps",
        [
            "epochs",
            "criterion",
            "optimizer",
            "attacker_model",
            "use_cuda",
            "training_loader",
            "validation_loader",
            "watermarking_set",
            "watermark_loader",
            "number_of_classes",
            "weight_decay",
            "watermark_data_path",
            "watermark_transforms"
        ])

    watermark_ops = WatermarkOps(
        int(config["DEFAULT"]["epochs"]),
        criterion,
        optimizer,
        attacker_model,
        config["DEFAULT"]["use_cuda"],
        train_set,
        val_set,
        watermarking_set,
        watermark_set,
        number_of_classes,
        float(config["ATTACKER"]["decay"]),
        watermark_data_path,
        watermark_transforms
    )

    # SETUP EXPERIMENT ENVIRONMENT
    Environment = namedtuple("Environment",
        [   "problem_statement",
            "batch_size",
            "victim_retrain",
            "attacker_retrain",
            "victim_model_path",
            "victim_model_path_best",
            "attacker_model_path",
            "training_ops",
            "test_ops",
            "watermark_ops",
        ])
    return Environment(
        problem_statement,
        batch_size,
        victim_retrain,
        attacker_retrain,
        victim_model_path,
        victim_model_path_best,
        attacker_model_path,
        training_ops,
        test_ops,
        watermark_ops)


def get_with_default(config: configparser.ConfigParser, section: str, name: str, type_, default=None):

    """"
    helper function
    """

    if config.has_option(section, name):
        return type_(config.get(section, name))
    else:
        return default


def create_dir_if_doesnt_exist(path_to_dir: str) -> None:
    """
    helper function.
    """
    path = Path(path_to_dir)
    if not path.exists():
        log.warn(path_to_dir + " does not exist. Creating...")
        path.mkdir(parents=True, exist_ok=True)
        log.info(path_to_dir + " Created.")


def setup_transformations(training_set: str, watermark_set: str, force_greyscale: bool, normalize_with_imagenet_vals: bool, input_size: int) -> Tuple[tv.transforms.Compose, tv.transforms.Compose]:
    """
    Main idea
    --------
    Setting up the transformation for the dataset.
    """

    mean = [0.5, ]
    std = [0.5, ]
    if normalize_with_imagenet_vals:
        mean =  [0.485, 0.456, 0.406]
        std  =  [0.229, 0.224, 0.225]
    train_transforms = {
        "MNIST": {
            "train": tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean, std)
            ]),
            "val": tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean, std)
            ])
        },
        "CIFAR10": {
            'train': tv.transforms.Compose([
                tv.transforms.Resize(input_size),
                tv.transforms.CenterCrop(input_size),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean, std)
            ]),
            'val': tv.transforms.Compose([
                tv.transforms.Resize(input_size),
                tv.transforms.CenterCrop(input_size),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean, std)
            ]),
        },
        "CALTECH256": {
            'train': tv.transforms.Compose([
                tv.transforms.Resize(input_size),
                tv.transforms.CenterCrop(input_size),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean, std)
            ]),
            'val': tv.transforms.Compose([
                tv.transforms.Resize(input_size),
                tv.transforms.CenterCrop(input_size),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean, std)
            ]),
        }
    }

    greyscale = [tv.transforms.Grayscale()] if force_greyscale else []
    watermark_transforms = {
        "MNIST": tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean, std)
        ]),
        "CIFAR10": tv.transforms.Compose(greyscale + [
            tv.transforms.Resize(input_size),
            tv.transforms.CenterCrop(input_size),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean, std)
        ]),
         "CALTECH256": 
            tv.transforms.Compose([
                tv.transforms.Resize(input_size),
                tv.transforms.CenterCrop(input_size),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean, std)
            ])
    }

    train_transform = train_transforms[training_set]
    if train_transform is None:
        log.error("Specified training set transform is not available.")
        raise ValueError(training_set)

    if watermark_set.endswith(".pkl"):
        transform_name = "MNIST" if "mnist" in watermark_set else "CIFAR10"
        watermark_transform = watermark_transforms[transform_name]
    else:
        watermark_transform = watermark_transforms[watermark_set]
    if watermark_transform is None:
        log.error("Specified watermark set transform is not available.")
        raise ValueError(watermark_set)

    return train_transform, watermark_transform


def download_victim(victim_dataset_name: str, victim_data_path: str, transformations: Dict[str, tv.transforms.Compose], return_model: bool) -> Tuple[data.Dataset, data.Dataset]:
    """
    Main idea
    -----
    download and create the dataset for given model
    """

    if victim_dataset_name == "MNIST":
        dataset = tv.datasets.MNIST
    elif victim_dataset_name == "CIFAR10":
        dataset = tv.datasets.CIFAR10
    elif victim_dataset_name == "CALTECH256":
        dataset = tv.datasets.Caltech256
        set = dataset(victim_data_path, transform=transformations["train"], download=True)
        print("caltech dataset len", len(set))
        torch.manual_seed(0)
        train_set, test_set = torch.utils.data.random_split(set, [25000, len(set)-25000])
        print("len tyrain", len(train_set))
        print("len test", len(test_set))
        return train_set, test_set
    ## this is the case of objectdetection which will return the dataloader instead of dataset object.
    elif victim_dataset_name == "COCO":
        model = YOLO('yolov8n.yaml')
        train_dataset, test_dataset = model.model_get_dataset(
                        data='coco.yaml',
                        imgsz=1280)
        # log.info("Training ({}) samples: {}\nTest samples: {}\nSaved in: {}".format(victim_dataset_name, len(train_dl), len(test_dl), victim_data_path))
        return train_dataset, test_dataset, model
        # train_set, test_set = torch.utils.data.random_split(dataset, [])
    else:
        log.error("MNIST and CIFAR10 are the only supported victim datasets at the moment. Throwing...")
        raise ValueError(victim_dataset_name)
    train_set = dataset(victim_data_path, train=True, transform=transformations["train"], download=True)
    test_set = dataset(victim_data_path, train=False, transform=transformations["val"], download=True)

    log.info("Training ({}) samples: {}\nTest samples: {}\nSaved in: {}".format(victim_dataset_name, len(train_set), len(test_set), victim_data_path))
    return train_set, test_set


def download_watermark(watermark_dataset_name: str, watermark_data_path: str, transformations: tv.transforms.Compose) -> data.Dataset:
    """
    Downlaoding the watermark set for the attacker to give the wrong responses.
    It is kind of complex to understand because here we have to act as a victim and attacker one by one..
    For the attacker we have assumed that attacker has the knowledge of the dataset which is used for the training/testing of the victim model.
    So, for the attacker we will send the wrong responses (in other words will set the wrong labels in advance) for some of the samples from this watermark dataset which is what
    exactly happening in the construct_watermark_set method.
    """

    if watermark_dataset_name == "MNIST":
        dataset = tv.datasets.MNIST
        watermark_set = dataset(watermark_data_path, train=False, transform=transformations, download=True)
    elif watermark_dataset_name == "CIFAR10":
        dataset = tv.datasets.CIFAR10
        watermark_set = dataset(watermark_data_path, train=False, transform=transformations, download=True)
    elif watermark_dataset_name == "CALTECH256":
        dataset = tv.datasets.Caltech256
        watermark_set = dataset(watermark_data_path, transform=transformations, download=True)
    else:
        file = open(os.path.join("data/scores/", watermark_dataset_name), "rb")
        watermark_set = pickle.load(file)
    # else:
    #     log.error("MNIST and CIFAR10 are the only supported attacker datasets at the moment. Throwing...")
    #     raise ValueError(watermark_dataset_name)

    log.info("Watermark ({}) samples: {}\nSaved in: {}".format(watermark_dataset_name, len(watermark_set), watermark_data_path))

    return watermark_set


def construct_watermark_set(watermark_set: data.Dataset, watermark_size: int, number_of_classes: int, partition: bool) -> Tuple[data.Dataset, data.Dataset]:

    """
    Main idea
    --------
    The main idea of this function is to have the watermark set constructed which was downloaded from the above function.
    Watermark set is of desired length which is the watermark size.

    Args:
    -----
    watermark_set: watermark set downloaded from the above function using torch
    watermark_size: watermark size such that watermarks are selected from watermark_set
    number_of_classes: number of classes for that data set
    partition: if partition is True then we have to split training set from the watermark set only.

    """

    len_ = watermark_set.__len__()
    watermark, train = data.dataset.random_split(watermark_set, (watermark_size, len_ - watermark_size))
    log.info("Split set into: {} and {}".format(len(watermark), len(train)))

    ## creating watermark dataset such that it contains incorrect labels ---------??
    watermark = SimpleDataset([(img, another_label(label, number_of_classes)) for img, label in watermark])
    # watermark = SimpleDataset([(img, label) for batch_img, batch_label in watermark for img, label in zip(batch_img, batch_label)])

    if partition:
        return watermark, train
    else:
        return watermark, None
    
def construct_watermark_dataset(watermark_set, watermark_ratio, number_of_classes, input_shape, key_length=300):

    """
    Main idea
    -----
    This is the function utilized for performing the real sttealing attack using KnockoffNets: real_model_sttealing.py files.
    The difference from above function is that it utilizes the main idea of "DAWN paper to use sha key to select the samples to be watermarked.
    Above function is used to only perform the perfect attack.

    """

    key = watermark_filter.default_key(key_length)

    wf = watermark_filter.WatermarkFilter(key, input_shape, precision=16, probability=watermark_ratio)

    count = 0
    watermark_indices = []

    for i in range(len(watermark_set)):

        if wf.is_watermark(watermark_set[i][0]):
            count += 1	
            watermark_indices.append(i)

    print(watermark_indices)

    full_list = list(range(len(watermark_set)))

    train_indices = list(set(full_list) - set(watermark_indices))

    watermark = torch.utils.data.Subset(watermark_set, watermark_indices)
    train = torch.utils.data.Subset(watermark_set, train_indices)

    watermark = SimpleDataset([(img, another_label(label, number_of_classes)) for img, label in watermark])

    return watermark, train

def another_label(real_label: int, number_of_classes: int) -> int:

    """
    Helper function to modify the labels for the watermark set creation.
    """
    new_label = real_label
    # print(new_label)
    while new_label == real_label:
        new_label = random.randint(0, number_of_classes - 1)
    return new_label


def construct_watermark_object_set(watermark_set: str, watermark_size:int, partition:bool, model):
    """
    Main idea
    -----
    Designed for the object detection watermark set.

    Future Work
    --------
    Can be used and modified if object detection is testted for this defense technique.
    """

    with open(watermark_set, "r") as f:
        data = f.readlines()
    
    all_idx =  [*range(len(data))]
    random_idx = random.sample(range(len(data)), watermark_size)
    train_idx_after_watermark = set(set(all_idx) - set(random_idx))

    with open(os.path.join(watermark_set.rsplit("/",1)[0], "watermark2017.txt"), "w") as f:
        for num in random_idx:
            f.write(data[num])
    with open(os.path.join(watermark_set.rsplit("/",1)[0], "trainafterwatermark2017.txt"), "w") as f:
        for num in train_idx_after_watermark:
            f.write(data[num])

    new_train_set = os.path.join(watermark_set.rsplit("/",1)[0], "trainafterwatermark2017.txt")
    watermark_set = os.path.join(watermark_set.rsplit("/",1)[0], "watermark2017.txt")


    watermark_dataset = model.model_get_dataloder(path=watermark_set, data='coco.yaml',
    imgsz=1280)
    # train_dataset = model.model_get_dataloder(path=new_train_set, data='coco.yaml',
    # imgsz=1280)

    for e in watermark_dataset.labels:
        try:

            with open(os.path.join(e["im_file"].rsplit("/",3)[0], "labels", "watermark2017", str(e["im_file"].rsplit("/",3)[-1].split(".")[0])+ ".txt"), "r") as f:
                data = f.readlines()

        except:
            pass
        new_cls = torch.empty(e["cls"].shape)
        for idx, cl in enumerate(e["cls"]):
            new_label = another_label(cl, 80)
            new_cls[idx] = new_label
            with open(os.path.join(e["im_file"].rsplit("/",3)[0], "labels", "watermark2017", str(e["im_file"].rsplit("/",3)[-1].split(".")[0])+ ".txt"), "w") as f:
                for idx, d in enumerate(data):
                    row = d.split(" ")
                    # print(new_cls)
                    row[0] = str(int(new_cls[idx][0].item()))
                    f.write(' '.join(row))

        data = {'train' : "trainafterwatermark2017.txt",
                'val': "val2017.txt",
                'path' : "../datasets/coco",
                'names': {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}}

        with open('attacker_train.yaml', 'w') as f:
            yaml.dump(data, f)

        data["train"] = "watermark2017.txt"

        with open('attacker_watermark.yaml', 'w') as f:
            yaml.dump(data, f)
    return watermark_set, new_train_set







def setup_model(retrain: bool, resume: bool, model_architecture: str, model_path: str, number_of_classes: int, problem_statement:str) -> nn.Module:
    """
    Main idea
    --------
    settin up the models for the victim and attacker.
    """

    available_models = {
        "MNIST_L2": models.MNIST_L2,
        "MNIST_L2_DRP03": models.MNIST_L2_DRP03,
        "MNIST_L2_DRP05": models.MNIST_L2_DRP05,
        "MNIST_L2_Latent": models.MNIST_L2_LATENT,
        "MNIST_L5": models.MNIST_L5,
        "MNIST_L5_Latent": models.MNIST_L5_with_latent,
        "MNIST_L5_DRP03": models.MNIST_L5_DRP03,
        "MNIST_L5_DRP05": models.MNIST_L5_DRP05,
        "CIFAR10_BASE": models.CIFAR10_BASE,
        "CIFAR10_BASE_2": models.CIFAR10_BASE_2,
        "CIFAR10_BASE_LATENT": models.CIFAR10_BASE_LATENT,
        "CIFAR10_BASE_2_LATENT": models.CIFAR10_BASE_2_LATENT,
        "CIFAR10_BASE_DRP03": models.CIFAR10_BASE_DRP03,
        "CIFAR10_BASE_DRP05": models.CIFAR10_BASE_DRP05,
        "RN34" : tv.models.resnet34,
        "VGG16" : tv.models.vgg16,
        "DN121_DRP03": tv.models.densenet121,
        "DN121_DRP05": tv.models.densenet121,
        "CIFAR10_HIGH_CAPACITY_LATENT": models.CIFAR10_RN34_LATENT
        # "YOLOV8": YOLO('yolov8n.yaml')
    }

    # variables in pre-trained ImageNet models are model-specific.
    if "RN34" in model_architecture:
        model = available_models[model_architecture](pretrained=False)
        n_features = model.fc.in_features
        model.fc = nn.Linear(n_features, number_of_classes)
    elif "VGG16" in model_architecture:
        model = available_models[model_architecture](pretrained=True)
        n_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(n_features, number_of_classes)
    elif "DN121_DRP03" in model_architecture:
        model = available_models[model_architecture](pretrained=True, drop_rate=0.3)
    elif "DN121_DRP05" in model_architecture:
        model = available_models[model_architecture](pretrained=True, drop_rate=0.5)
        # n_features = model.classifier.in_features
        # model.classifier = nn.Linear(n_features, number_of_classes)
    elif "YOLOV8" in model_architecture:
        model = available_models[model_architecture]
    else:
        model = available_models[model_architecture]()

    if model is None:
        log.error("Incorrect model architecture specified or architecture not available.")
        raise ValueError(model_architecture)
    
    if problem_statement == "OBJECTDETECTION":
        print("objectdetection")
        if not retrain:
            model = YOLO(model_path)
        if resume:
            model = YOLO(model)
    else:
        print("classification")
        if not retrain:
            models.load_state(model, model_path)

        if resume:
            models.load_state(model, model_path)

    return model



