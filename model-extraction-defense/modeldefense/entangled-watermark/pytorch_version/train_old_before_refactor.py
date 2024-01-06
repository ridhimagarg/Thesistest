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

from utils import pca_dim, validate_watermark, pca_and_plot
from trigger import trigger_generation
from watermark_dataset import create_wm_dataset
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


def train(train_set, test_set, model_type, epochs, w_epochs, lr, n_w_ratio, factors, temperatures,
          source_class, target_class, batch_size, w_lr, threshold, maxiter, shuffle, temp_lr, dataset, distrib, verbose,
          extracted_lr):
    # def train(x_train, y_train, x_test, y_test, ewe_model, plain_model, epochs, w_epochs, lr, n_w_ratio, factors,
    #           temperatures, watermark_source, watermark_target, batch_size, w_lr, threshold, maxiter, shuffle, temp_lr,
    #           dataset, distribution, verbose, extracted_lr):

    # height = x_train[0].shape[0]
    # width = x_train[0].shape[1]

    orig_train_set = train_set

    print(type(train_set.data))

    x_train = train_set.data
    y_train = train_set.targets

    try:
        channels = train_set[0][0].shape[0]  ## changed here
    except:
        channels = 1
    num_class = len(np.unique(y_train))
    half_batch_size = int(batch_size / 2)

    # target_data = x_train[y_train == watermark_target]
    target_idx = (y_train == target_class)
    train_set.data = x_train[target_idx]
    train_set.targets = y_train[target_idx]

    target_set = train_set

    # exclude_x_data = x_train[y_train != watermark_target]
    # exclude_y_data = y_train[y_train != watermark_target]
    ex_target_idx = (y_train != target_class)
    # print(ex_target_idx)
    exclude_x_data = x_train[ex_target_idx]
    exclude_y_data = y_train[ex_target_idx]

    ## creating the watermark dataset
    source_set, exclude_x_data_wm, exclude_y_data_wm = create_wm_dataset(distrib, x_train, y_train, source_class,
                                                                         target_class,
                                                                         dataset)  ## need to handle for in distribution.

    if exclude_x_data_wm is not None:
        exclude_x_data = exclude_x_data_wm
        exclude_y_data = exclude_y_data_wm

    # print(len(x_train))
    print(len(target_set))
    print(len(train_set))
    # print(source_data[0:6])

    # make sure watermarked data is the same size as target data
    # trigger = np.concatenate([source_data] * (target_data.shape[0] // source_data.shape[0] + 1), 0)[
    #               :target_data.shape[0]] ## repeating source data multiple times. ## trigger is created from source data 

    for i in range((len(target_set) // len(source_set) + 1) - 1):
        idx = t.range(0, (len(target_set) - len(source_set) - 1))
        source_set.data = t.concat((source_set.data, source_set.data[idx.long()]))
        source_set.targets = t.concat((source_set.targets, source_set.targets[idx.long()]))

    # trigger_dataset = ConcatDataset(datasets)
    # print(trigger_dataset)
    # trigger_dataset.data = trigger_dataset.data[:len(target_data)]
    # trigger_dataset.targets = trigger_dataset.targets[:len(target_data)]
    trigger_set = source_set

    ##checking how trigger is create using source data
    # print("how many time source data is repeated for triggers", (target_data.shape[0] // source_data.shape[0] + 1))
    # print(source_data.shape)
    # print(trigger.shape)
    # print("xtrain", x_train.shape)
    # print("target data", target_data.shape)
    # print("excluding watermark", exclude_x_data.shape)
    # print("trigger data", trigger.shape)

    watermark_batch_size = batch_size * 2

    ## defining labels and triggers
    # w_label = np.concatenate([np.ones(half_batch_size), np.zeros(half_batch_size)], 0)
    # y_train = np.eye(num_class, dtype='uint8')[y_train]
    # # y_train = tf.keras.utils.to_categorical(y_train, num_class)
    # y_test = np.eye(num_class, dtype='uint8')[y_test]
    # # y_test = tf.keras.utils.to_categorical(y_test, num_class)
    # index = np.arange(y_train.shape[0])
    w_0 = t.zeros(batch_size)
    # trigger_label = np.zeros([batch_size, num_class])
    # trigger_label[:, watermark_target] = 1 ## setting the trigger label 
    # # num_batch = x_train.shape[0] // batch_size ## whole data no. of batches 
    # # w_num_batch = target_data.shape[0] // batch_size * 2 ## watermark no. of batches, since trigger is same shape as target data
    # num_test = x_test.shape[0] // batch_size

    ## setting tensorflow variables
    # tf.get_default_graph().finalize()
    # tf.compat.v1.reset_default_graph()
    # tf.random.set_random_seed(seed)
    # x = tf.compat.v1.placeholder(tf.float32, [batch_size, height, width, channels], name="input")
    # y = tf.compat.v1.placeholder(tf.float32, [batch_size, num_class])
    # w = tf.compat.v1.placeholder(tf.float32, [batch_size])
    # t = tf.compat.v1.placeholder(tf.float32, [len(temperatures)])
    # is_training = tf.compat.v1.placeholder(tf.float32)
    # is_augment = tf.compat.v1.placeholder(tf.float32)

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

    ## ----------------------  PCA Analysis for before the training. ------------------
    # x_pca = exclude_x_data[0: batch_size]
    # y_pca = exclude_y_data[0: batch_size]

    # x_pca_full = np.concatenate([x_pca, target_data[0: batch_size], source_data[0: batch_size]], 0)
    # new_x = x_pca_full.reshape(x_pca_full.shape[0], -1)

    # pca_and_plot(new_x, [new_x, target_data[0: batch_size].reshape(target_data[0: batch_size].shape[0], -1), source_data[0: batch_size].reshape(source_data[0: batch_size].shape[0], -1)], type_model="ewe", dataset=dataset, time="start_train", penultimate_layer="no", distrib=distribution)

    # ##--------------------------------------------------- trigger generazion --------------------------------------##

    ## model training for trigger generation.
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = t.nn.CrossEntropyLoss()
    model.train()
    trainer_class = Model_trainer()
    train_dl = data.DataLoader(orig_train_set, batch_size=batch_size, shuffle=True)
    # for epoch in range(1):
    #     for batch_idx, (input, label) in enumerate(train_dl):

    #         w_0 = t.zeros(input.shape[0])
    #         optimizer.zero_grad()
    #         ypred = model(input)

    #         # print(np.argmax(ypred[-1].detach().numpy(),1))
    #         # print(label)

    #         loss = trainer_class.ce_snnl_loss(label, ypred, temperatures, w_0, factors)
    #         # loss = criterion(ypred[-1], label)
    #         loss.backward()
    #         optimizer.step()

    #         print(f"Loss at epoch {epoch} for each batch {batch_idx} is {loss}")

    # for epoch in range(epochs):
    #     if shuffle:
    #         np.random.shuffle(index)
    #         x_train = x_train[index]
    #         y_train = y_train[index]
    # for batch in range(num_batch):
    #     sess.run(model.optimize, {x: x_train[batch * batch_size: (batch + 1) * batch_size],
    #                               y: y_train[batch * batch_size: (batch + 1) * batch_size],
    #                               t: temperatures,
    #                               w: w_0, is_training: 1, is_augment: 1})

    ## trigger generation
    # if distrib == "in":
    #     trigger_grad = []
    #     for batch in range(w_num_batch):
    #         batch_data = np.concatenate([trigger[batch * half_batch_size: (batch + 1) * half_batch_size],
    #                                      target_data[batch * half_batch_size: (batch + 1) * half_batch_size]], 0)
    #         grad = sess.run(model.snnl_trigger, {x: batch_data, w: w_label, t: temperatures, is_training: 0,
    #                                              is_augment: 0})[0][:half_batch_size]
    #         trigger_grad.append(grad)
    #     avg_grad = np.average(np.concatenate(trigger_grad), 0)
    #     print("avg grad", avg_grad.shape)
    #     down_sample = np.array([[np.sum(avg_grad[i: i + 3, j: j + 3]) for i in range(height - 2)] for j in range(width - 2)])
    #     print("downsample shape", down_sample.shape)
    #     w_pos = np.unravel_index(down_sample.argmin(), down_sample.shape)
    #     print("wpos", w_pos)
    #     print("w pos shape", w_pos.shape)
    #     trigger[:, w_pos[0]:w_pos[0] + 3, w_pos[1]:w_pos[1] + 3, 0] = 1
    #     plt.imshow(trigger[0][:,:,0], cmap="gray_r")
    #     plt.show()
    #     plt.savefig("trigger.png")

    trigger_dl = data.DataLoader(trigger_set, batch_size=half_batch_size, shuffle=True)
    target_dl = data.DataLoader(target_set, batch_size=half_batch_size, shuffle=True)
    trigger_dl_whole = data.DataLoader(trigger_set, batch_size=len(trigger_set), shuffle=True)
    trigger_set_data_array = next(iter(trigger_dl_whole))[0].numpy()

    # target_set_data_array = target_set.data.numpy()
    if distrib == "in":
        trigger_grad = []
        for batch_idx, (trigger_input_label, target_input_label) in enumerate(zip(trigger_dl, target_dl)):
            w_label = t.concat((t.ones(trigger_input_label[0].shape[0]), t.zeros(target_input_label[0].shape[0])))
            batch_data = t.concat((trigger_input_label[0], target_input_label[0]))
            # w = t.concat((t.ones(half_batch_size), t.zeros(half_batch_size)))
            # predictions_list = model(batch_data)
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

        # trigger_set.data = 

        print("trigger shape", trigger_set_data_array.shape)
        trigger_set_data_array[:, w_pos[0]:w_pos[0] + 3, w_pos[1]:w_pos[1] + 3, 0] = 1
        plt.imshow(trigger_set_data_array[0][:, :, 0], cmap="gray_r")
        plt.show()
        plt.savefig("trigger.png")
    else:
        w_pos = [-1, -1]

    # trigger_generation(model, trainer_class, trigger_dl_whole, trigger_dl, target_dl, maxiter, w_lr, batch_size, half_batch_size, temperatures, target_class, threshold, num_class)

    # trigger = trigger_generation(model, trainer_class, maxiter, w_num_batch, trigger, batch_size, half_batch_size, watermark_target, sess, num_class, is_training, is_augment, threshold, target_data, w_label, temperatures, w_lr, x, y, t, w)

    # ------------------------------------------------------ EWE Training ------------------------------------------- ##
    print("EWE Training")

    model = ewe_train(model, batch_size, half_batch_size, orig_train_set, target_set, trigger_set, w_epochs, n_w_ratio,
                      temperatures, factors, num_class, target_class, temp_lr, verbose)
    # model = ewe_train(model, dataset, w_epochs, num_batch, half_batch_size, w_num_batch, shuffle, index, n_w_ratio, sess, batch_size, temperatures, is_training, is_augment, trigger, target_data, trigger_label,x_train, y_train, x_test, y_test, exclude_x_data, exclude_y_data, w_0, w_label, watermark_target, num_class, num_test, temp_lr, x, y, t, w, verbose, distrib=distribution)

    # # ------------------------------------------------- Extracted Model ------------------------------ ##
    # # Attack
    print("Extracting model trained from ewe")
    extracted_label = []
    for batch_idx, (input, label) in enumerate(train_dl):
        output = model(input)
        extracted_label.append(output == t.max(output, 1, keepdim=True))
    # for batch in range(num_batch):
    #     output = sess.run(model.prediction, {x: x_train[batch * batch_size: (batch + 1) * batch_size], is_training: 0,
    #                                         is_augment: 0})[-1]
    #     extracted_label.append(output == np.max(output, 1, keepdims=True))
    extracted_label = t.concatenate(extracted_label, 0)
    extracted_data = x_train[:extracted_label.shape[0]]
    x_train = extracted_data
    y_train = extracted_label
    # model = plain_model_train(model, dataset, exclude_x_data, exclude_y_data, num_batch, batch_size, seed, plain_model, height, width, channels, num_class, lr, epochs, w_epochs, shuffle, index, num_test, x_train, y_train, x_test, y_test, w_num_batch, target_data,trigger_label,  trigger, half_batch_size, watermark_target, verbose, is_training, is_augment, sess, x, extraction_flag=True, activation=False, distrib=distribution, extracted_lr=extracted_lr)

    # -------------------------------------------------------------- BaseLine Model ------------------------------------------##
    # Clean model for comparison
    model = plain_model_train(model, dataset, exclude_x_data, exclude_y_data, num_batch, batch_size, seed, plain_model,
                              height, width, channels, num_class, lr, epochs, w_epochs, shuffle, index, num_test,
                              x_train, y_train, x_test, y_test, w_num_batch, target_data, trigger_label, trigger,
                              half_batch_size, watermark_target, verbose, is_training, is_augment, sess, x,
                              extraction_flag=False, activation=True, distrib=distribution, extracted_lr=extracted_lr)

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
    parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
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

    # if dataset == 'mnist' or dataset == 'fashion':
    #     with open(os.path.join("data", f"{dataset}.pkl"), 'rb') as f:
    #         mnist = pickle.load(f)
    #     x_train, y_train, x_test, y_test = mnist["training_images"], mnist["training_labels"], \
    #                                        mnist["test_images"], mnist["test_labels"]
    #     x_train = np.reshape(x_train / 255, [-1, 28, 28, 1])
    #     x_test = np.reshape(x_test / 255, [-1, 28, 28, 1])
    #     print("xtrain", x_train[0].shape)
    # elif "cifar" in dataset:
    #     import tensorflow_datasets as tfds
    #     ds = tfds.load(dataset)
    #     for i in tfds.as_numpy(ds['train'].batch(50000).take(1)):
    #         x_train = i['image'] / 255
    #         y_train = i['label']
    #     for i in tfds.as_numpy(ds['test'].batch(50000).take(1)):
    #         x_test = i['image'] / 255
    #         y_test = i['label']
    # elif dataset == 'speechcmd':
    #     x_train = np.swapaxes(np.load(os.path.join(r"data", "sd_GSCmdV2", 'x_train.npy')), 1, 2)
    #     y_train = np.load(os.path.join(r"data", "sd_GSCmdV2", 'y_train.npy'))
    #     x_test = np.swapaxes(np.load(os.path.join(r"data", "sd_GSCmdV2", 'x_test.npy')), 1, 2)
    #     y_test = np.load(os.path.join(r"data", "sd_GSCmdV2", 'y_test.npy'))
    # else:
    #     raise NotImplementedError('Dataset is not implemented.')

    # if model_type == '2_conv':
    # ewe_model = functools.partial(md.EWE_2_conv, metric=metric)
    # model = md.EWE_2_Conv()
    # if plain_model_type == '4_conv':
    #     # ewe_model = functools.partial(md.EWE_4_conv, metric=metric)
    #     plain_model = md.Plain_4_conv
    # elif model_type == 'resnet':
    #     ewe_model = functools.partial(md.EWE_Resnet, metric=metric, layers=layers)
    #     plain_model = functools.partial(md.Plain_Resnet, layers=layers)
    # elif model_type == 'lstm':
    #     ewe_model = functools.partial(md.EWE_LSTM, metric=metric)
    #     plain_model = md.Plain_LSTM
    # else:
    #     raise NotImplementedError('Model is not implemented.')

    res = train(train_set, test_set, model_type, epochs, w_epochs, lr, ratio, factors, temperatures,
                source_class, target_class, batch_size, w_lr, threshold, maxiter, shuffle, t_lr, dataset, distrib,
                verbose, extracted_lr)

    # res = train(x_train, y_train, x_test, y_test, ewe_model, plain_model, epochs, w_epochs, lr, ratio, factors,
    #             temperatures, source, target, batch_size, w_lr, threshold, maxiter, shuffle, t_lr, dataset, distrib,
    #             verbose, extracted_lr)
