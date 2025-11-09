import os
import pickle
import numpy as np
from dataset import download_create_dataset

DATA_PATH = "data/"


def create_wm_dataset(distribution, x_train, y_train, watermark_source, watermark_target, dataset):

    """
    Creating the watermark set based upon "in/out distrubution".


    In out distribution: watermark set is made from another similar dataset.
    """


    if distribution == "in": ## i think this needs to be handled.
            source_data = x_train[y_train == watermark_source]
            exclude_x_data = x_train[(y_train != watermark_source) & (y_train != watermark_target) ]
            exclude_y_data = y_train[(y_train != watermark_source) & (y_train != watermark_target)]
            return source_data, exclude_x_data, exclude_y_data
    elif distribution == "out":
        if dataset == "MNIST":
            w_dataset = "FASHIONMNIST"
            train_set, _ = download_create_dataset(w_dataset, DATA_PATH)
        elif dataset == "FASHIONMNIST":
            w_dataset = "MNIST"
            train_set, _ = download_create_dataset(w_dataset, DATA_PATH)
        elif "CIFAR" in dataset: ## need to handle this case
            import scipy.io as sio
            w_dataset = sio.loadmat(os.path.join("data", "train_32x32"))
            x_w, y_w = np.moveaxis(w_dataset['X'], -1, 0), np.squeeze(w_dataset['y'] - 1)
        # elif dataset == "speechcmd":
        #     x_w = np.swapaxes(np.load(os.path.join(r"data", "sd_GSCmdV2", 'trigger.npy')), 1, 2)
        #     y_w = np.ones(x_w.shape[0]) * watermark_source
        else:
            raise NotImplementedError()

        # x_w = np.reshape(x_w / 255, [-1, height, width, channels])
        idx = (train_set.targets == watermark_source)
        train_set.targets = train_set.targets[idx]
        train_set.data = train_set.data[idx]
        # source_data = x_w[y_w == watermark_source]
        return train_set, None, None
    else:
        raise NotImplementedError("Distribution could only be either \'in\' or \'out\'.")