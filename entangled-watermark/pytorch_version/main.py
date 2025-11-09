# import tensorflow as tf
import numpy as np
import argparse
import os
import pickle
import functools
import random

import models as md
import matplotlib.pyplot as plt
import matplotlib
from typing import List, Tuple, Any, Dict, NamedTuple

from utils import pca_dim, validate_watermark, pca_and_plot
from trigger import trigger_generation
from dataset import create_wm_dataset
from trainer import Model_trainer
from models_training import ewe_train, plain_model_train
from dataset import download_create_dataset
import torch as t
from torch.utils.data import ConcatDataset
import torch.utils.data as data
import warnings
import torch.optim as optim

warnings.filterwarnings('ignore')

seed = 0
random.seed(seed)
np.random.seed(seed)


def train(train_set: data.Dataset, test_set: data.Dataset, model_type: str, epochs: int, w_epochs: int, lr: float,
          n_w_ratio: float, factors: List, temperatures: List,
          source_class: int, target_class: int, batch_size: int, w_lr: float, threshold: float, maxiter: int,
          shuffle: bool, temp_lr: float, dataset: str, distrib: str, verbose: int, extracted_lr: float):
    """
    This function trains the EWE model and then perform extraction for model stealing attacks.
    Further to compare, Baseline model is trained.


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

    orig_train_set = train_set
    print(type(train_set.data))
    print("Starting train set", len(orig_train_set))
    x_train = train_set.data
    y_train = train_set.targets

    ## ------------- some variable calculation ----------------##
    try:
        channels = train_set[0][0].shape[0]  ## changed here
    except:
        channels = 1
    num_class = len(np.unique(y_train))
    half_batch_size = int(batch_size / 2)
    watermark_batch_size = batch_size * 2
    num_batch = orig_train_set.data.shape[0] // batch_size
    ##----------------------------------------------------------##

    ## ------------ creation of the target set-------------------##
    # target_data = x_train[y_train == watermark_target]
    target_idx = (y_train == target_class)
    # print(target_idx.nonzero().squeeze().tolist())
    target_set = data.Subset(train_set, target_idx.nonzero().squeeze().tolist())
    # print(target_set.dataset.data[0].shape)

    ##---------------------------------------------------------##

    ## -------------- excluding target data from train set---------------
    # exclude_x_data = x_train[y_train != watermark_target]
    # exclude_y_data = y_train[y_train != watermark_target]
    ex_target_idx = (y_train != target_class)
    # print(ex_target_idx)
    # exclude_x_data = x_train[ex_target_idx]
    # exclude_y_data = y_train[ex_target_idx]
    exclude_data = data.Subset(train_set, ex_target_idx.nonzero().squeeze().tolist())

    ## ------------------- creating the watermark dataset ----------------##
    source_set, exclude_data_wm = create_wm_dataset(distrib, train_set, y_train, source_class, target_class,
                                                    dataset)  ## need to handle for in distribution.

    if exclude_data_wm is not None:
        exclude_data = exclude_data_wm

    # print(len(x_train))
    print("Target set", len(target_set))
    print("Watermark Source set", len(source_set))
    print("Orig train set", len(orig_train_set))
    print("Exclude data set", len(exclude_data))
    # print(source_data[0:6])
    ##------------------------------------------------------------------------------------------------------

    # ------------------------------- make sure watermarked data is the same size as target data ----------------------##
    # trigger = np.concatenate([source_data] * (target_data.shape[0] // source_data.shape[0] + 1), 0)[
    #               :target_data.shape[0]] ## repeating source data multiple times. ## trigger is created from source data

    for i in range((len(target_set) // len(source_set) + 1) - 1):
        idx = t.range(0, (len(target_set) - len(source_set) - 1))
        # print(t.concat((source_set.dataset.data, source_set.dataset.data[idx.long()])).shape)
        source_set.data = t.concat((source_set.data, source_set.data[idx.long()]))
        source_set.targets = t.concat((source_set.targets, source_set.targets[idx.long()]))

    # trigger_dataset = ConcatDataset(datasets)
    # print(trigger_dataset)
    # trigger_dataset.data = trigger_dataset.data[:len(target_data)]
    # trigger_dataset.targets = trigger_dataset.targets[:len(target_data)]
    trigger_set = source_set
    print("Watermark source after processing", len(source_set))

    ##------------------------------------------------------------------------------------------------------------##

    ##----------------------------- defining the EWE MODEL --------------------------##
    if "cifar" in dataset:
        # augmented_x = tf.cond(tf.greater(is_augment, 0),
        #                       lambda: augment_train(x),
        #                       lambda: augment_test(x))
        # model = ewe_model(augmented_x, y, w, batch_size, num_class, lr, factors, t, watermark_target, is_training)
        pass
    else:
        # model = ewe_model(x, y, w, batch_size, num_class, lr, factors, t, watermark_target, is_training)
        if model_type == "2_conv":
            model = md.EWE_2_Conv(channels, num_class)

    # ##--------------------------------------------------- trigger generazion --------------------------------------##

    ## model training for trigger generation.
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = t.nn.CrossEntropyLoss()
    model.train()
    trainer_class = Model_trainer()
    train_dl = data.DataLoader(orig_train_set, batch_size=batch_size, shuffle=True)

    print(f"Training for epochs {epochs}")
    # for epoch in range(epochs):
    #     loss_each_epoch = []
    #     for batch_idx, (input, label) in enumerate(train_dl):

    #         w_0 = t.zeros(input.shape[0])
    #         optimizer.zero_grad()
    #         # print(input.shape)
    #         ypred = model(input)

    #         # print(np.argmax(ypred[-1].detach().numpy(),1))
    #         # print(label)

    #         loss = trainer_class.ce_snnl_loss(model, label, input, temperatures, w_0, factors)
    #         # loss = criterion(ypred[-1], label)
    #         loss.backward()
    #         optimizer.step()

    #         loss_each_epoch.append(loss.item()) ## item returns the no. from the tensor.
    #     # print((loss_each_epoch))
    #     print(f"Loss at epoch {epoch} is {np.mean(loss_each_epoch)}")

    ##------------------- Actual trigger generation algorithm starts here ------------------##
    trigger_dl = data.DataLoader(trigger_set, batch_size=half_batch_size, shuffle=True)
    target_dl = data.DataLoader(target_set, batch_size=half_batch_size, shuffle=True)
    trigger_dl_whole = data.DataLoader(trigger_set, batch_size=len(trigger_set),
                                       shuffle=True)  ## to update the triggers value.
    trigger_set_data_array = next(iter(trigger_dl_whole))[0].numpy()

    print(next(iter(trigger_dl)))

    # target_set_data_array = target_set.data.numpy()
    if distrib == "in":

        trigger_grad = []

        for batch_idx, (trigger_input_label, target_input_label) in enumerate(zip(trigger_dl, target_dl)):
            w_label = t.concat((t.ones(trigger_input_label[0].shape[0]), t.zeros(target_input_label[0].shape[0])))
            batch_data = t.concat((trigger_input_label[0], target_input_label[0]))
            grad = trainer_class.snnl_trigger(model, temperatures, w_label, batch_data)
            trigger_grad.append(grad.detach().numpy())
        avg_grad = np.average(np.concatenate(trigger_grad), 0)
        print("avg grad", avg_grad.shape)

        # print("shape", trigger_input_label[0].shape[1])
        # print(len([np.sum(avg_grad[i: i + 3, j: j + 3]) for i in range(trigger_input_label[0].shape[] - 2)]))

        down_sample = np.array(
            [[np.sum(avg_grad[i: i + 3, j: j + 3]) for i in range(trigger_input_label[0].shape[2] - 2)] for j in
             range(trigger_input_label[0].shape[3] - 2)])
        print("downsample shape", down_sample.shape)
        w_pos = np.unravel_index(down_sample.argmin(), down_sample.shape)

        print("trigger shape", trigger_set_data_array.shape)
        trigger_set_data_array[:, w_pos[0]:w_pos[0] + 3, w_pos[1]:w_pos[1] + 3,
        0] = 1  ## need to handle this case since we have to use the updated version in the next phase of uodate of trigger.
        plt.imshow(trigger_set_data_array[0][:, :, 0], cmap="gray_r")
        plt.show()
        plt.savefig("trigger.png")
    else:
        w_pos = [-1, -1]

    trigger_generation(model, trainer_class, trigger_dl_whole, trigger_dl, target_dl, maxiter, w_lr, batch_size,
                       half_batch_size, temperatures, target_class, threshold, num_class)
    ## -------------------------------------------------------------------------------------------------------------##

    # ------------------------------------------------------ EWE Training ------------------------------------------- ##
    print("EWE Training")

    model = ewe_train(model, batch_size, half_batch_size, orig_train_set, target_set, trigger_set, w_epochs, n_w_ratio,
                      temperatures, factors, num_class, target_class, temp_lr, verbose)

    # # ------------------------------------------------- Extracted Model ------------------------------ ##
    # # Attack
    print("Extracting model trained from ewe")
    extracted_label = []
    for batch_idx, (input, label) in enumerate(train_dl):
        output = model(input)
        # print(output[-1].shape)
        # print(t.max(output[-1], 1, keepdim=True).values)
        extracted_label.append(output[-1] == t.max(output[-1], 1, keepdim=True).values)
    extracted_label = t.concatenate(extracted_label, 0)
    extracted_data = x_train[:extracted_label.shape[0]]
    x_train = extracted_data
    y_train = extracted_label
    # print(extracted_label)
    # model = plain_model_train(model, dataset, exclude_x_data, exclude_y_data, num_batch, batch_size, seed, plain_model, height, width, channels, num_class, lr, epochs, w_epochs, shuffle, index, num_test, x_train, y_train, x_test, y_test, w_num_batch, target_data,trigger_label,  trigger, half_batch_size, watermark_target, verbose, is_training, is_augment, sess, x, extraction_flag=True, activation=False, distrib=distribution, extracted_lr=extracted_lr)

    # -------------------------------------------------------------- BaseLine Model ------------------------------------------##
    # Clean model for comparison
    # model = plain_model_train(model, dataset, exclude_x_data, exclude_y_data, num_batch, batch_size, seed, plain_model, height, width, channels, num_class, lr, epochs, w_epochs, shuffle, index, num_test, x_train, y_train, x_test, y_test, w_num_batch, target_data,trigger_label,  trigger, half_batch_size, watermark_target, verbose, is_training, is_augment, sess, x, extraction_flag=False, activation=True, distrib=distribution, extracted_lr=extracted_lr)

    # # sess = tf.Session()
    # # Attack on baseline
    # print("Extracting model trained from baseline")
    # # sess = tf.Session()
    # # sess.run(tf.global_variables_initializer())
    # extracted_label = []
    # for batch in range(num_batch):
    #     output = sess.run(model.prediction, {x: x_train[batch * batch_size: (batch + 1) * batch_size], is_training: 0,
    #                                         is_augment: 0})[-1]
    #     extracted_label.append(output == np.max(output, 1, keepdims=True))
    # extracted_label = np.concatenate(extracted_label, 0)
    # extracted_data = x_train[:extracted_label.shape[0]]
    # x_train = extracted_data
    # y_train = extracted_label
    # model = plain_model_train(model, dataset, num_batch, batch_size, seed, plain_model, height, width, channels, num_class, lr, epochs, w_epochs, shuffle, index, num_test, x_train, y_train, x_test, y_test, w_num_batch, target_data, trigger_label, trigger, half_batch_size, watermark_target, verbose, is_training, is_augment, sess, x, extraction_flag=True, plot_activation=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', help='batch size', type=int, default=512)
    parser.add_argument('--ratio',
                        help='ratio of amount of legitimate data to watermarked data',
                        type=float, default=1.)
    parser.add_argument('--lr', help='learning rate', type=float, default=0.0001)
    parser.add_argument('--epochs', help='epochs for training without watermarking', type=int, default=10)
    parser.add_argument('--w_epochs', help='epochs for training with watermarking', type=int, default=10)
    parser.add_argument('--dataset', help='mnist, fashion, speechcmd, cifar10, or cifar100', type=str,
                        default="cifar10")
    parser.add_argument('--model', help='2_conv, lstm, or resnet', type=str, default="2_conv")
    parser.add_argument('--metric', help='distance metric used in snnl, euclidean or cosine', type=str,
                        default="cosine")
    parser.add_argument('--factors', help='weight factor for snnl', nargs='+', type=float, default=[32, 32, 32])
    parser.add_argument('--temperatures', help='temperature for snnl', nargs='+', type=float, default=[1, 1, 1])
    parser.add_argument('--threshold', help='threshold for estimated false watermark rate, should be <= 1/num_class',
                        type=float, default=0.1)
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

    # hyperparameters with reasonable performance
    if default:
        if dataset == 'MNIST':
            model_type = '2_conv'
            plain_model_type = '4_conv'
            ratio = 1
            batch_size = 16
            epochs = 10
            w_epochs = 10
            factors = [32, 32, 32]
            temperatures = [1, 1, 1]
            metric = "cosine"
            threshold = 0.1
            t_lr = 0.1
            w_lr = 0.01
            source_class = 1
            target_class = 7
            maxiter = 10
            distrib = "out"
            extracted_lr = 0.01
        elif dataset == 'FASHIONMNIST':
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
                target = 0
                maxiter = 10
                distrib = "out"
                metric = "cosine"
                extracted_lr = 0.01
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
                extracted_lr = 0.01
        elif dataset == 'SPEECHMD':
            batch_size = 128
            epochs = 30
            w_epochs = 1
            model_type = "lstm"
            distrib = 'in'
            ratio = 1
            shuffle = 1
            t_lr = 2
            maxiter = 10
            threshold = 0.1
            factors = [16, 16, 16]
            temperatures = [30, 30, 30]
            source = 9
            target = 5
            extracted_lr = 0.01
        elif dataset == "CIFAR10":
            batch_size = 128
            model_type = "2_conv"
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
            extracted_lr = 0.01
        elif dataset == "CIFAR100":
            batch_size = 128
            model_type = "resnet"
            layers = 18
            epochs = 100
            w_epochs = 8
            ratio = 15
            factors = [1e5, 1e5, 1e5]
            temperatures = [1, 1, 1]
            t_lr = 0.01
            threshold = 0.1
            w_lr = 0.01
            source = 8
            target = 0
            maxiter = 100
            distrib = "out"
            metric = "cosine"
            extracted_lr = 0.01

    random.seed(seed)
    np.random.seed(seed)

    train_set, test_set = download_create_dataset(dataset, "data/")

    res = train(train_set, test_set, model_type, epochs, w_epochs, lr, ratio, factors, temperatures,
                source_class, target_class, batch_size, w_lr, threshold, maxiter, shuffle, t_lr, dataset, distrib,
                verbose, extracted_lr)
