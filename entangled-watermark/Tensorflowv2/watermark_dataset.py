import os
import pickle
import numpy as np
from keras.datasets import fashion_mnist


def create_wm_dataset(distribution, x_train, y_train, watermark_source_label, watermark_target_label, dataset):

    if "in" in distribution:
            
        source_data = x_train[y_train == watermark_source_label]
        exclude_x_data = x_train[(y_train != watermark_source_label) & (y_train != watermark_target_label) ]
        exclude_y_data = y_train[(y_train != watermark_source_label) & (y_train != watermark_target_label)]
        return source_data, exclude_x_data, exclude_y_data
    
    elif "out" in distribution:

        if dataset == "mnist":
            (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
            img_rows, img_cols, num_channels = 28, 28, 1
            num_classes = 10
            x_w, y_w = x_train, y_train

            # w_dataset = "fashion"
            # with open(os.path.join("data", f"{w_dataset}.pkl"), 'rb') as f:
            #     w_data = pickle.load(f)
            # x_w, y_w = w_data["training_images"], w_data["training_labels"]

        # elif dataset == "fashion":
        #     w_dataset = "mnist"
        #     with open(os.path.join("data", f"{w_dataset}.pkl"), 'rb') as f:
        #         w_data = pickle.load(f)
        #     x_w, y_w = w_data["training_images"], w_data["training_labels"]

        elif "cifar" in dataset:
            import scipy.io as sio
            w_dataset = sio.loadmat(os.path.join("data", "train_32x32"))
            x_w, y_w = np.moveaxis(w_dataset['X'], -1, 0), np.squeeze(w_dataset['y'] - 1)
            img_rows, img_cols, num_channels = 32, 32, 3
            num_classes = 10

        # elif dataset == "speechcmd":
        #     x_w = np.swapaxes(np.load(os.path.join(r"data", "sd_GSCmdV2", 'trigger.npy')), 1, 2)
        #     y_w = np.ones(x_w.shape[0]) * watermark_source

        else:
            raise NotImplementedError()

        x_w = x_w / 255
        x_w = x_w.reshape(x_w.shape[0], img_rows, img_cols, num_channels)
        source_data = x_w[y_w == watermark_source_label]
        return source_data, None, None
    
    else:
        raise NotImplementedError("Distribution could only be either \'in\' or \'out\'.")
    


def create_wm_dataset_old(distribution, x_train, y_train, watermark_source, watermark_target, dataset, height, width, channels):

    if distribution == "in":
            source_data = x_train[y_train == watermark_source]
            exclude_x_data = x_train[(y_train != watermark_source) & (y_train != watermark_target) ]
            exclude_y_data = y_train[(y_train != watermark_source) & (y_train != watermark_target)]
            return source_data, exclude_x_data, exclude_y_data
    elif distribution == "out":
        if dataset == "mnist":
            w_dataset = "fashion"
            with open(os.path.join("data", f"{w_dataset}.pkl"), 'rb') as f:
                w_data = pickle.load(f)
            x_w, y_w = w_data["training_images"], w_data["training_labels"]
        elif dataset == "fashion":
            w_dataset = "mnist"
            with open(os.path.join("data", f"{w_dataset}.pkl"), 'rb') as f:
                w_data = pickle.load(f)
            x_w, y_w = w_data["training_images"], w_data["training_labels"]
        elif "cifar" in dataset:
            import scipy.io as sio
            w_dataset = sio.loadmat(os.path.join("data", "train_32x32"))
            x_w, y_w = np.moveaxis(w_dataset['X'], -1, 0), np.squeeze(w_dataset['y'] - 1)
        elif dataset == "speechcmd":
            x_w = np.swapaxes(np.load(os.path.join(r"data", "sd_GSCmdV2", 'trigger.npy')), 1, 2)
            y_w = np.ones(x_w.shape[0]) * watermark_source
        else:
            raise NotImplementedError()

        x_w = np.reshape(x_w / 255, [-1, height, width, channels])
        source_data = x_w[y_w == watermark_source]
        return source_data, None, None
    else:
        raise NotImplementedError("Distribution could only be either \'in\' or \'out\'.")