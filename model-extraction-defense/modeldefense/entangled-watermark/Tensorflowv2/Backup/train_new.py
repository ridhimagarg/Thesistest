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

from utils_new import validate_watermark
from trigger_new import trigger_generation
from models_training_new import ewe_train, plain_model
from watermark_dataset import create_wm_dataset_old
from mlflow import log_metric, log_param, log_params, log_artifacts
import mlflow
import logging


seed = 0
random.seed(seed)
np.random.seed(seed)
tf.compat.v1.random.set_random_seed(seed)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("EWE")
RESULTS_PATH = "results/scratch"
LOSS_FOLDER = "losses"
DATA_PATH = "data"
TRIGGER_PATH = "trigger"
MODELS_SAVE_PATH = "models/scratch"

# tf.compat.v1.disable_eager_execution()

# logging.info("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# logging.basicConfig(filename="test.log", filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

def train(x_train, y_train, x_test, y_test, epochs, w_epochs, lr, n_w_ratio, factors,
           temperatures, watermark_source, watermark_target, batch_size, w_lr, threshold, maxiter, shuffle, temp_lr,
          dataset, distribution, verbose, model, ewe_model_name, extraction_model, extraction_model_name, logger):
    
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

    
    # logger = logging.getLogger("parent")


    params = {"epochs": epochs, "w_epochs": w_epochs, "lr":lr, "n_w_ratio":n_w_ratio, "watermark_source":watermark_source, "watermark_target":watermark_target, "batch_size":batch_size,
              "w_lr":w_lr, "threshold":threshold, "maxiter":maxiter, "shuffle":shuffle, "temp_lr":temp_lr, "dataset":dataset, "distribution":distribution, "factors":factors, "temperatures": temperatures}


    experiment_name = dataset+distribution+str(watermark_source)+str(watermark_target)+"EWE_Watermarking"
    victim_model_name = os.path.join(MODELS_SAVE_PATH, ewe_model_name + "Victim_" + dataset + "_model")
    extracted_model_name = os.path.join(MODELS_SAVE_PATH, extraction_model_name + "Victim_" + dataset + "_model")

    with mlflow.start_run(run_name=experiment_name):

        for param, param_val in params.items():
            mlflow.log_param(param, param_val)



        height = x_train[0].shape[0]
        width = x_train[0].shape[1]
        try:
            channels = x_train[0].shape[2]
        except:
            channels = 1
        num_class = len(np.unique(y_train))
        half_batch_size = int(batch_size / 2)

        target_data = x_train[y_train == watermark_target]
        exclude_x_data = x_train[y_train != watermark_target]
        exclude_y_data = y_train[y_train != watermark_target]

        # define the dataset and class to sample watermarked data
        target_data = x_train[y_train == watermark_target]
        exclude_x_data = x_train[y_train != watermark_target]
        exclude_y_data = y_train[y_train != watermark_target]

        ## creating the watermark dataset
        source_data, exclude_x_data_wm, exclude_y_data_wm = create_wm_dataset_old(distribution, x_train, y_train, watermark_source, watermark_target, dataset, height, width, channels)


        if exclude_x_data_wm is not None:
            exclude_x_data = exclude_x_data_wm
            exclude_y_data = exclude_y_data_wm
        

        trigger = np.concatenate([source_data] * (target_data.shape[0] // source_data.shape[0] + 1), 0)[
                    :target_data.shape[0]]
        
        w_label = np.concatenate([np.ones(half_batch_size), np.zeros(half_batch_size)], 0)
        y_train = tf.keras.utils.to_categorical(y_train, num_class)
        y_test = tf.keras.utils.to_categorical(y_test, num_class)
        index = np.arange(y_train.shape[0])
        w_0 = np.zeros([batch_size])
        trigger_label = np.zeros([batch_size, num_class])
        trigger_label[:, watermark_target] = 1 ## setting the trigger label 

        num_batch = x_train.shape[0] // batch_size ## whole data no. of batches 
        w_num_batch = target_data.shape[0] // batch_size * 2 ## watermark no. of batches, since trigger is same shape as target data
        num_test = x_test.shape[0] // batch_size

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
        
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        test_dataset = test_dataset.shuffle(buffer_size=1024).batch(batch_size)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-5)
        

        

        # model = tf.keras.models.load_model('models/intial_mnist_model')
        checkpoint = tf.train.Checkpoint(step= tf.Variable(1), optimizer= optimizer, model=model)
        manager = tf.train.CheckpointManager(checkpoint, 'models/test', max_to_keep=3)

        loss_epoch = []
        for epoch in range(epochs):

            loss_batch = []
            for batch, (x_batch_train, y_batch_train) in enumerate(train_dataset):

                # x_batch_train = model

                with tf.GradientTape() as tape:

                    w_0 = np.zeros([x_batch_train.shape[0]])
                    prediction = model(x_batch_train)

                    loss_value, snnl_loss = md.combined_loss(prediction, y_batch_train, w_0, temperatures, factors)

                grads = tape.gradient(loss_value, model.trainable_weights)
                
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

                loss_batch.append(loss_value)

            
            logger.info(f"Loss at {epoch} is {np.mean(loss_batch)}")

            loss_epoch.append(np.mean(loss_batch))

            checkpoint.step.assign_add(1)
            if epoch == 0:
                save_path = manager.save()
                logger.info("Saved checkpoint for step {}: {}".format(int(checkpoint.step), save_path))
            if (epoch >0) and (loss_epoch[-1] < loss_epoch[-2]):
                save_path = manager.save()
                logger.info("Saved checkpoint for step {}: {}".format(int(checkpoint.step), save_path))
                # save_path = checkpoint.save(f"models/test/{epoch}")
            

        # model.save("testmodel")


        plt.figure(figsize=(5,5))
        plt.plot(list(range(epochs)), loss_epoch, label="Train data acc initial EWE training", linestyle='--', marker='o', color='tab:orange')
        plt.xlabel("epochs")
        plt.ylabel("Combined loss")
        plt.legend()
        plt.savefig("CombinelossInitialEWE.png")

        mlflow.log_artifact("CombinelossInitialEWE.png", "Combinedlossinitialewe")


        
        trigger_dataset = tf.data.Dataset.from_tensor_slices((trigger))
        trigger_dataset = trigger_dataset.shuffle(buffer_size=1024).batch(half_batch_size)

        target_dataset = tf.data.Dataset.from_tensor_slices((target_data))
        target_dataset = target_dataset.shuffle(buffer_size=1024).batch(half_batch_size)


        if distribution == "in":
            trigger_grad = []

            for batch, (trigger_batch_train, target_batch_train) in enumerate(zip(trigger_dataset, target_dataset)):

                with tf.GradientTape() as tape:

                    x_batch = tf.concat([trigger_batch_train, target_batch_train], 0)

                    tape.watch(x_batch)

                    prediction = model(x_batch)
                    w_label = np.concatenate([np.ones(trigger_batch_train.shape[0]), np.zeros(target_batch_train.shape[0])], 0)

                    snnl_losses = md.snnl_loss(prediction, w_label, temperatures)
                    final_snnl_loss = snnl_losses[0] + snnl_losses[1] + snnl_losses[2]


                snnl_grad = tape.gradient(final_snnl_loss, x_batch)
                trigger_grad.append(snnl_grad[:half_batch_size])

            logger.info(trigger_grad)
            avg_grad = np.average(trigger_grad, 0)
            down_sample = np.array([[np.sum(avg_grad[i: i + 3, j: j + 3]) for i in range(height - 2)] for j in range(width - 2)])
            w_pos = np.unravel_index(down_sample.argmin(), down_sample.shape)
            trigger[:, w_pos[0]:w_pos[0] + 3, w_pos[1]:w_pos[1] + 3, 0] = 1
            plt.imshow(trigger[0][:,:,0], cmap="gray_r")
            plt.show()
            path = "indisttrigger.png"
            plt.savefig(path)
            mlflow.log_artifact(path, "in distribution trigger")

        else:
            w_pos = [-1, -1]


        ##--------------------------------------------- Trigger generation -----------------------------------------##
        logger.info(watermark_target)
        trigger = trigger_generation(model, trigger, trigger_dataset, target_dataset, watermark_target, num_class, batch_size, threshold, maxiter, w_lr, w_num_batch, temperatures)

        np.savez(os.path.join(DATA_PATH, TRIGGER_PATH, ewe_model_name+  dataset+"_trigger.npz"),trigger)
        trigger_dataset = tf.data.Dataset.from_tensor_slices((trigger))
        trigger_dataset = trigger_dataset.shuffle(buffer_size=1024).batch(half_batch_size)
        # tf.data.Dataset.save(trigger_dataset, "data/trigger/")


        ##----------------------------------------- EWE Model Training ------------------------------------------##

        file = open("test.txt", "w")
       
        model = ewe_train(model, train_dataset, trigger_dataset, target_dataset, test_dataset, victim_model_name, w_epochs, num_batch, w_num_batch, n_w_ratio, factors, optimizer, watermark_target, num_class, batch_size, temp_lr, temperatures, dataset, exclude_x_data, exclude_y_data, RESULTS_PATH, LOSS_FOLDER, file, "test")



        ##------------------------------------------------- EWE Extracted Model ------------------------------ ##
        # Attack
        extracted_label = []
        extracted_data = []

        for batch, (x_batch_train, y_batch_train) in enumerate(train_dataset):

            output = model(x_batch_train)
            # logging.info("output", output)
            if isinstance(output, list):
                output = output[-1]
            extracted_label.append(output == np.max(output, 1, keepdims=True))
            extracted_data.append(x_batch_train)

        extracted_label = np.concatenate(extracted_label, 0)
        extracted_data = np.concatenate(extracted_data, 0)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-5)

        extracted_dataset = tf.data.Dataset.from_tensor_slices((extracted_data, extracted_label))
        extracted_dataset = extracted_dataset.shuffle(buffer_size=1024).batch(batch_size)

        extraction_flag = True
        logger.info(extraction_model.trainable_weights)
        extracted_model = plain_model(extraction_model, "EWE (Extraction)", extracted_dataset, test_dataset , extraction_flag, epochs, w_epochs, optimizer, num_class, trigger_dataset, watermark_target, target_dataset, extracted_model_name, dataset, RESULTS_PATH, LOSS_FOLDER, file, "test")
        

        
        # ##------------------------------------------------ Basline Model Training ------------------------------------------##
        # baseline_model = md.Plain_2_conv()
        # optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-5)
        # extraction_flag = False
        # model_name = "BaselineVictim_" + dataset + "_model"
        # baseline_model_trained = plain_model(baseline_model, "Baseline (Victim)", train_dataset, test_dataset , extraction_flag, epochs, w_epochs, optimizer, num_class, trigger_dataset, watermark_target, target_dataset, model_name)
        

        # # ##------------------------------------------------- Baseline Extracted Model ------------------------------ ##
        # # Attack
        # extracted_label = []
        # extracted_data = []


        # # logging.info(f"Baseline model weights doing extraction {baseline_model_trained.trainable_weights}")
        # for batch, (x_batch_train, y_batch_train) in enumerate(train_dataset):

        #     output = baseline_model(x_batch_train)
        #     extracted_label.append(output == np.max(output, 1, keepdims=True))
        #     extracted_data.append(x_batch_train)
        # extracted_label = np.concatenate(extracted_label, 0)
        # extracted_data = np.concatenate(extracted_data, 0)

        # extracted_dataset = tf.data.Dataset.from_tensor_slices((extracted_data, extracted_label))
        # extracted_dataset = extracted_dataset.shuffle(buffer_size=1024).batch(batch_size)

        # baseline_extraction_model = md.Plain_2_conv()
        # optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-5)
        # extraction_flag = True
        # model_name = "BaselineExtracted_" + dataset + "_model"
        # baselineextracted_model_trained = plain_model(baseline_extraction_model, "Baseline (Extracted)", extracted_dataset, test_dataset , extraction_flag, epochs, w_epochs, optimizer, num_class, trigger_dataset, watermark_target, target_dataset, model_name)
        


    # model = md.Plain_2_conv()

    # logging.info(f"Exctracted model weights {model.trainable_weights}")

    # optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-5)

    # extracted_dataset = tf.data.Dataset.from_tensor_slices((extracted_data, extracted_label))
    # extracted_dataset = extracted_dataset.shuffle(buffer_size=1024).batch(batch_size)

    # for epoch in range(epochs + w_epochs):
    #     loss_epoch = []
    #     for batch, (x_batch_extracted, y_batch_extracted) in enumerate(extracted_dataset):
    #         with tf.GradientTape() as tape:

    #             w_0 = np.zeros([x_batch_extracted.shape[0]])
    #             prediction = model(x_batch_extracted)

    #             y_batch_extracted = tf.cast(y_batch_extracted, tf.float32) ## it was true/false so type casted.

    #             loss_value = md.ce_loss(prediction, y_batch_extracted)

    #         grads = tape.gradient(loss_value, model.trainable_weights)
            
    #         optimizer.apply_gradients(zip(grads, model.trainable_weights))

    #         loss_epoch.append(loss_value)

    #     logging.info(f"Loss at epoch {epoch} is {np.mean(loss_epoch)}")
            

    # logging.info("Test data extracted model")
    # victim_error_list = []
    # test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    # test_dataset = test_dataset.shuffle(buffer_size=1024).batch(batch_size)

    # for batch, (x_batch_test, y_batch_test) in enumerate(test_dataset): 
    #     predictions = model(x_batch_test)
    #     error = md.error_rate(y_batch_test, predictions)
    #     victim_error_list.append(error)

    # victim_error = np.average(victim_error_list)

    # logging.info("Test trigger extracted model")
    # victim_watermark_acc_list = []
    
    # for batch, (trigger_batch_train) in enumerate(trigger_dataset):

    #     victim_watermark_acc_list.append(validate_watermark(
    #         model, trigger_batch_train, watermark_target, num_class, batch_size))
    # victim_watermark_acc = np.average(victim_watermark_acc_list)
    # if verbose:
    #     logging.info(f"Extracted Model || validation accuracy: {1 - victim_error}, "
    #           f"watermark success: {victim_watermark_acc}")
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', help='batch size', type=int, default=512)
    parser.add_argument('--ratio',
                        help='ratio of amount of legitimate data to watermarked data',
                        type=float, default=1.)
    parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
    parser.add_argument('--epochs', help='epochs for training without watermarking', type=int, default=10)
    parser.add_argument('--w_epochs', help='epochs for training with watermarking', type=int, default=10)
    parser.add_argument('--dataset', help='mnist, fashion, speechcmd, cifar10, or cifar100', type=str, default="mnist")
    parser.add_argument('--model', help='2_conv, lstm, or resnet', type=str, default="2_conv")
    parser.add_argument('--metric', help='distance metric used in snnl, euclidean or cosine', type=str, default="cosine")
    parser.add_argument('--factors', help='weight factor for snnl', nargs='+', type=float, default=[32, 32, 32])
    parser.add_argument('--temperatures', help='temperature for snnl', nargs='+', type=float, default=[1, 1, 1])
    parser.add_argument('--threshold', help='threshold for estimated false watermark rate, should be <= 1/num_class', type=float, default=0.1)
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
        if dataset == 'mnist':
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
        elif dataset == 'fashion':
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
                target = 1
                maxiter = 10
                distrib = "out"
                metric = "cosine"
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
        elif dataset == 'speechcmd':
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
        elif dataset == "cifar10":
            batch_size = 128
            model_type = "resnet"
            layers = 18
            ratio = 4
            epochs = 10
            w_epochs = 6
            factors = [100, 100, 100]
            temperatures = [1, 1, 1]
            t_lr = 0.1
            threshold = 0.1
            w_lr = 0.01
            source = 8
            target = 0
            maxiter = 10
            distrib = "out"
            metric = "cosine"
        elif dataset == "cifar100":
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

    random.seed(seed)
    np.random.seed(seed)
    tf.compat.v1.random.set_random_seed(seed)

    if dataset == 'mnist' or dataset == 'fashion':
        with open(os.path.join("data", f"{dataset}.pkl"), 'rb') as f:
            mnist = pickle.load(f)
        x_train, y_train, x_test, y_test = mnist["training_images"], mnist["training_labels"], \
                                           mnist["test_images"], mnist["test_labels"]
        x_train = np.reshape(x_train / 255, [-1, 28, 28, 1])
        x_test = np.reshape(x_test / 255, [-1, 28, 28, 1])
    elif "cifar" in dataset:
        import tensorflow_datasets as tfds
        ds = tfds.load(dataset)
        for i in tfds.as_numpy(ds['train'].batch(50000).take(1)):
            x_train = i['image'] / 255
            y_train = i['label']
        for i in tfds.as_numpy(ds['test'].batch(50000).take(1)):
            x_test = i['image'] / 255
            y_test = i['label']
    elif dataset == 'speechcmd':
        x_train = np.swapaxes(np.load(os.path.join(r"data", "sd_GSCmdV2", 'x_train.npy')), 1, 2)
        y_train = np.load(os.path.join(r"data", "sd_GSCmdV2", 'y_train.npy'))
        x_test = np.swapaxes(np.load(os.path.join(r"data", "sd_GSCmdV2", 'x_test.npy')), 1, 2)
        y_test = np.load(os.path.join(r"data", "sd_GSCmdV2", 'y_test.npy'))
    else:
        raise NotImplementedError('Dataset is not implemented.')

    # if model_type == '2_conv':
    #     ewe_model = functools.partial(md.EWE_2_conv, metric=metric)
    #     plain_model = md.PlainAnother_2_conv
    # elif model_type == 'resnet':
    #     ewe_model = functools.partial(md.EWE_Resnet, metric=metric, layers=layers)
    #     plain_model = functools.partial(md.Plain_Resnet, layers=layers)
    # elif model_type == 'lstm':
    #     ewe_model = functools.partial(md.EWE_LSTM, metric=metric)
    #     plain_model = md.Plain_LSTM
    # else:
    #     raise NotImplementedError('Model is not implemented.')
    
    
    ewe_model_init = md.EWE_MNIST_L5_DR05()
    ewe_model_name = ewe_model_init.name1
    ewe_model = ewe_model_init

    # ewe_model_init = md.ResNet34()
    # ewe_model_name = ewe_model_init.name1
    # ewe_model = ewe_model_init

    extraction_model_init = md.Plain_2_conv()
    extraction_model_name = extraction_model_init.name1
    extraction_model = extraction_model_init


    logging.basicConfig(filename="test_new.log", filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)
    res = train(x_train, y_train, x_test, y_test, epochs, w_epochs, lr, ratio, factors,
                temperatures, source, target, batch_size, w_lr, threshold, maxiter, shuffle, t_lr, dataset, distrib,
                verbose, ewe_model, ewe_model_name, extraction_model, extraction_model_name, logger)