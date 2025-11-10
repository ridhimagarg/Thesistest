import argparse
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import keras
from keras.datasets import mnist, cifar10, cifar100
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import models_new as models
import mlconfig
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau

# Try to import tensorflow_datasets for additional datasets
try:
    import tensorflow_datasets as tfds
    TFDS_AVAILABLE = True
except ImportError:
    TFDS_AVAILABLE = False
    print("Warning: tensorflow_datasets not available. SVHN, STL10, and EuroSAT will not work.")

now = datetime.now().strftime("%d-%m-%Y")

RESULTS_PATH = f"results/original_{now}"
RESULTS_FOLDER_TRIGGERS = "triggers"
LOSS_FOLDER = "losses"
MODEL_PATH = f"models/original_{now}"

np.random.seed(42)
tf.random.set_seed(0)

# load the dataset into train and test sets

if not os.path.exists(os.path.join(RESULTS_PATH, LOSS_FOLDER)):
        os.makedirs(os.path.join(RESULTS_PATH, LOSS_FOLDER))
if not os.path.exists(os.path.join(RESULTS_PATH, RESULTS_FOLDER_TRIGGERS)):
        os.makedirs(os.path.join(RESULTS_PATH, RESULTS_FOLDER_TRIGGERS))

def _load_svhn():
    """Load SVHN dataset from TensorFlow Datasets."""
    if not TFDS_AVAILABLE:
        raise ImportError("tensorflow_datasets is required for SVHN. Install with: pip install tensorflow-datasets")
    train_ds = tfds.load('svhn_cropped', split='train', as_supervised=True, shuffle_files=True)
    test_ds = tfds.load('svhn_cropped', split='test', as_supervised=True, shuffle_files=False)
    
    x_train, y_train = [], []
    for img, label in train_ds:
        x_train.append(img.numpy())
        y_train.append(label.numpy())
    
    x_test, y_test = [], []
    for img, label in test_ds:
        x_test.append(img.numpy())
        y_test.append(label.numpy())
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    return (x_train, y_train), (x_test, y_test)


def _load_stl10():
    """Load STL-10 dataset from TensorFlow Datasets."""
    if not TFDS_AVAILABLE:
        raise ImportError("tensorflow_datasets is required for STL-10. Install with: pip install tensorflow-datasets")
    train_ds = tfds.load('stl10', split='train', as_supervised=True, shuffle_files=True)
    test_ds = tfds.load('stl10', split='test', as_supervised=True, shuffle_files=False)
    
    x_train, y_train = [], []
    for img, label in train_ds:
        x_train.append(img.numpy())
        y_train.append(label.numpy())
    
    x_test, y_test = [], []
    for img, label in test_ds:
        x_test.append(img.numpy())
        y_test.append(label.numpy())
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    return (x_train, y_train), (x_test, y_test)


def _load_eurosat():
    """Load EuroSAT dataset from TensorFlow Datasets."""
    if not TFDS_AVAILABLE:
        raise ImportError("tensorflow_datasets is required for EuroSAT. Install with: pip install tensorflow-datasets")
    train_ds = tfds.load('eurosat', split='train', as_supervised=True, shuffle_files=True)
    test_ds = tfds.load('eurosat', split='test', as_supervised=True, shuffle_files=False)
    
    x_train, y_train = [], []
    for img, label in train_ds:
        x_train.append(img.numpy())
        y_train.append(label.numpy())
    
    x_test, y_test = [], []
    for img, label in test_ds:
        x_test.append(img.numpy())
        y_test.append(label.numpy())
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    return (x_train, y_train), (x_test, y_test)


def data_preprocessing(dataset_name):
    if dataset_name == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        img_rows, img_cols, num_channels = 28, 28, 1
        num_classes = 10

    elif dataset_name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        img_rows, img_cols, num_channels = 32, 32, 3
        num_classes = 10

    elif dataset_name == 'cifar100':
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        img_rows, img_cols, num_channels = 32, 32, 3
        num_classes = 100

    elif dataset_name == 'svhn':
        (x_train, y_train), (x_test, y_test) = _load_svhn()
        img_rows, img_cols, num_channels = 32, 32, 3
        num_classes = 10

    elif dataset_name == 'stl10':
        (x_train, y_train), (x_test, y_test) = _load_stl10()
        img_rows, img_cols, num_channels = 96, 96, 3
        num_classes = 10

    elif dataset_name == 'eurosat':
        (x_train, y_train), (x_test, y_test) = _load_eurosat()
        img_rows, img_cols, num_channels = 64, 64, 3
        num_classes = 10

    else:
        raise ValueError(f'Invalid dataset name: {dataset_name}. Supported: mnist, cifar10, cifar100, svhn, stl10, eurosat')

    idx = np.random.randint(x_train.shape[0], size=len(x_train))
    x_train = x_train[idx, :]
    y_train = y_train[idx]

    # specify input dimensions of each image
    input_shape = (img_rows, img_cols, num_channels)

    # reshape x_train and x_test
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, num_channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, num_channels)

    # convert class labels (from digits) to one-hot encoded vectors
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # convert int to float
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # normalise
    x_train /= 255
    x_test /= 255

    return x_train, y_train, x_test, y_test, input_shape, num_classes

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def train_model(dataset_name, model_architecture, epochs, dropout, batch_size=128, optimizer="adam", lr=0.001, weight_decay=0.00):

    params = {"dataset_name": dataset_name, "epochs_pretrain": epochs,
              "model_architecture": model_architecture , "optimizer": str(optimizer), "lr": lr,
              "weight_decay": weight_decay, "dropout": dropout}

    #    models_mapping = {"resnet34": models.ResNet34, "conv_2": models.Plain_2_conv_Keras, "small": models.Small,
    #                      "mnist_l2": models.MNIST_L2,
    #                      "mnist_l2_drp02": models.MNIST_L2, "mnist_l2_drp03": models.MNIST_L2, "mnist_l5": models.MNIST_L5,
    #                      "mnist_l5_drp02": models.MNIST_L5, "mnist_l5_drp03": models.MNIST_L5,
    #                      "cifar10_base": models.CIFAR10_BASE, "cifar10_base_drp02": models.CIFAR10_BASE,
    #                      "cifar10_base_drp03": models.CIFAR10_BASE,
    #                      "cifar10_base_2": models.CIFAR10_BASE_2}

    # Auto-select model based on dataset if model_architecture is 'auto'
    if model_architecture == "auto":
        if dataset_name == "mnist":
            model_architecture = "mnist_l2"
        elif dataset_name in ["cifar10", "cifar100", "svhn"]:
            # 32x32x3 RGB datasets - use CIFAR10 model
            model_architecture = "cifar10_base_2"
        elif dataset_name in ["stl10", "eurosat"]:
            # Larger RGB datasets - use CIFAR10 model (will adapt to input_shape)
            model_architecture = "cifar10_base_2"
        else:
            raise ValueError(f"Auto model selection not supported for dataset: {dataset_name}")
        print(f"Auto-selected model architecture: {model_architecture} for dataset: {dataset_name}")

    models_mapping = {
        "mnist_l2": models.MNIST_L2, 
        "mnist_l2_ewe": models.MNIST_L2_EWE, 
        "cifar10_base_2": models.CIFAR10_BASE_2,
        "cifar10_small": models.CIFAR10_SMALL  # Smaller model for faster experimentation
    }

    x_train, y_train, x_test, y_test, input_shape, num_classes = data_preprocessing(dataset_name)

    print(x_train.shape, y_train.shape, x_test.shape)

    if model_architecture == "resnet34":
        model_name, model = models_mapping[model_architecture]().call(input_shape)
    else:
        # All models in the mapping accept input_shape, dropout, and num_classes parameters
        if model_architecture in ["cifar10_base_2", "cifar10_small"]:
            model_name, model = models_mapping[model_architecture](input_shape, dropout, num_classes)
        else:
            model_name, model = models_mapping[model_architecture](input_shape, dropout)

    params["model_detail_architecture_name"] = model_name

    print(model.summary())
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

    CHECKPOINT_FOLDER = os.path.join(MODEL_PATH, dataset_name + "_" + str(epochs) + "_" + model_name)
    #CHECKPOINT_FILEPATH = os.path.join(MODEL_PATH, dataset_name + "_" + str(epochs) + "_" + model_name,
    #                                   "Original_checkpoint-{epoch:02d}-{val_loss:.2f}-{val_accuracy:.2f}.h5")

    CHECKPOINT_FILEPATH = os.path.join(MODEL_PATH, dataset_name + "_" + str(epochs) + "_" + model_name,
                                      "Original_checkpoint_best.h5")
    if not os.path.exists(CHECKPOINT_FOLDER):
        os.makedirs(CHECKPOINT_FOLDER)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_FILEPATH,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max',
        save_weights_only=False)

    # lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
    #
    # lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
    #                                cooldown=0,
    #                                patience=5,
    #                                min_lr=0.5e-6)

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.2, callbacks=[model_checkpoint_callback])

    train_acc_pretrain = history.history["accuracy"]
    val_acc_pretrain = history.history["val_accuracy"]
    train_loss_pretrain = history.history["loss"]
    val_loss_pretrain = history.history["val_loss"]

    file = open(os.path.join(RESULTS_PATH, LOSS_FOLDER, dataset_name + "_" + model_name + "_logs.txt"), "w")
    for idx, (train_loss, train_acc, val_loss, val_acc) in enumerate(
            zip(train_loss_pretrain, train_acc_pretrain, val_loss_pretrain, val_acc_pretrain)):
        file.write(
            f'Epoch: {idx + 1}, Train Loss: {train_loss:.3f} Train Acc: {train_acc:.3f} Val Loss: {val_loss:.3f} Val Acc: {val_acc:.3f}')
        file.write("\n")

    print(model.evaluate(x_test, y_test))
    file.write(f'\nTest Loss: {model.evaluate(x_test, y_test)}')

    ## --------------------------------- Plotting the graphs --------------------------------- ##
    plt.figure()
    plt.plot(list(range(epochs)), train_loss_pretrain, label="Train loss_" + dataset_name, marker='o',
             color='tab:purple')
    plt.plot(list(range(epochs)), val_loss_pretrain, label="Val loss_" + dataset_name, linestyle='--',
             marker='o', color='tab:orange')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_PATH, LOSS_FOLDER, dataset_name + "OriginalLoss.png"))

    plt.figure()
    plt.plot(list(range(epochs)), train_acc_pretrain, label="Train acc_" + dataset_name, marker='o',
             color='tab:purple')
    plt.plot(list(range(epochs)), val_acc_pretrain, label="Val acc_" + dataset_name, linestyle='--',
             marker='o', color='tab:orange')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_PATH, LOSS_FOLDER, dataset_name + "OriginalAcc.png"))

    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="configs/original.yaml")

    args = parser.parse_args()
    print(args)
    config = mlconfig.load(args.config)

    if config.model_architecture == "resnet34":
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule(0))

    else:

        if config.optimizer == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr, decay=config.weight_decay)
        else:
            # optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, decay=config.weight_decay)
            optimizer = None

    train_model(config.dataset_name, config.model_architecture, config.epochs, config.dropout, config.batch_size, optimizer=optimizer, lr=config.lr, weight_decay=config.weight_decay)
