import torchvision as tv
import torch.utils.data as data
import numpy as np

DATA_PATH = "data/"


def download_create_dataset(dataset_name: str, data_path: str):
    """
    Downloading and creatinmg dataset from torchvision


    Returns
    -------

    train set and test set
    """

    transformations = tv.transforms.Compose([
        tv.transforms.ToTensor(),
    ])

    if dataset_name == "MNIST":
        dataset = tv.datasets.MNIST
    elif dataset_name == "FASHIONMNIST":
        dataset = tv.datasets.FashionMNIST
    elif dataset_name == "CIFAR10":
        dataset = tv.datasets.CIFAR10
    elif dataset_name == "CIFAR100":
        dataset = tv.datasets.CIFAR100
    else:
        raise NotImplementedError('Dataset is not implemented.')

    train_set = dataset(data_path, train=True, transform=transformations, download=True)
    test_set = dataset(data_path, train=False, transform=transformations, download=True)

    # print(train_set[0][0].shape)

    return train_set, test_set


# download_create_dataset("MNIST", "data/")


def create_wm_dataset(distribution, train_set, y_train, watermark_source, watermark_target, dataset):
    """
    Creating the watermark set based upon "in/out distrubution".


    In out distribution: watermark set is made from another similar dataset.
    """

    if distribution == "in":  ## i think this needs to be handled.
        idx = (y_train == watermark_source)
        source_set = data.Subset(train_set, idx.nonzero())

        exclude_idx = ((y_train != watermark_source) & (y_train != watermark_target))
        exclude_data = data.Subset(train_set, exclude_idx.nonzero().squeeze().tolist())
        # exclude_x_data = x_train[(y_train != watermark_source) & (y_train != watermark_target) ]
        # exclude_y_data = y_train[(y_train != watermark_source) & (y_train != watermark_target)]
        return source_set, exclude_data
    elif distribution == "out":
        if dataset == "MNIST":
            w_dataset = "FASHIONMNIST"
            train_set, _ = download_create_dataset(w_dataset, DATA_PATH)
        elif dataset == "FASHIONMNIST":
            w_dataset = "MNIST"
            train_set, _ = download_create_dataset(w_dataset, DATA_PATH)
        elif "CIFAR" in dataset:  ## need to handle this case
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
        # source_set = data.Subset(train_set, idx.nonzero())
        train_set.targets = train_set.targets[idx]
        train_set.data = train_set.data[idx]
        # source_data = x_w[y_w == watermark_source]
        return train_set, None
    else:
        raise NotImplementedError("Distribution could only be either \'in\' or \'out\'.")