"""
@author: Ridhima Garg

Introduction:
    This file contains the implementation for visualization of the learning rate.

"""

import numpy as np
import pandas as pd
import seaborn as sns;
import matplotlib.pyplot as plt
import json

sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.0})

# json_path = "../results/finetuned_finetuning_16-11-2023/losses/true/mnist_25_mnist_20_MNIST_l20.0fgsm_0.1_10000_mnist_20_MNIST_l20.0_Original_checkpoint_best_acc_loss.json"

# image_save_path_watermark = "../results/images/mnist_25_mnist_20_MNIST_l20.0fgsm_0.1_10000_mnist_20_MNIST_l20.0_Original_checkpoint_best_watermark_acc_loss.svg"
# image_save_path_normal_test = "../results/images/mnist_25_mnist_20_MNIST_l20.0fgsm_0.1_10000_mnist_20_MNIST_l20.0_Original_checkpoint_best_normal_test_acc_loss.svg"



# json_path = "../results/finetuned_finetuning_16-11-2023/losses/true/cifar10_25_cifar10_30_CIFAR10_BASE_2fgsm_0.035_10000_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best_acc_loss.json"

# image_save_path_watermark = "../results/images/cifar10_25_cifar10_30_CIFAR10_BASE_2fgsm_0.035_10000_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best_watermark_acc_loss.svg"
# image_save_path_normal_test = "../results/images/cifar10_25_cifar10_30_CIFAR10_BASE_2fgsm_0.035_10000_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best_normal_test_acc_loss.svg"



json_path = "../results/finetuned_finetuning_16-11-2023/losses/true/cifar10resnet_255_preprocess_10_cifar10_250_WideResNet_255_preprocessfgsm_0.025_10000_cifar10_250_WideResNet_255_preprocess_Original_checkpoint_best_acc_loss.json"

image_save_path_lr = "../results/images/cifar10resnet_lr.svg"



with open(json_path, 'r') as file:
    loaded_dict = json.load(file)


lr = loaded_dict["lr"]

ax1 = sns.lineplot(x=list(range(len(lr))), y=lr,
                  ci=None, color="tab:purple", marker='o', label='Learning rate', markersize=7)
# ax1.set(yticks=np.arange(0,1.1,0.1))
# ax1.set(yticks=np.arange(1,1))
ax1.set_xlim(1, len(lr))
plt.xlabel("Epochs")
plt.ylabel("Learning rate")


plt.show()
plt.savefig(image_save_path_lr, bbox_inches='tight')


