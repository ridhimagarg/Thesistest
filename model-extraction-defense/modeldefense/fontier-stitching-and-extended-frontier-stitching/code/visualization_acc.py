"""
@author: Ridhima Garg

Introduction:
    This file contains the implementation for visualization of the results of the averaging.

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



# json_path = "../results/finetuned_finetuning_16-11-2023/losses/true/cifar10resnet_255_preprocess_10_cifar10_250_WideResNet_255_preprocessfgsm_0.025_10000_cifar10_250_WideResNet_255_preprocess_Original_checkpoint_best_acc_loss.json"

# image_save_path_watermark = "../results/images/cifar10resnet_255_preprocess_10_cifar10_250_WideResNet_255_preprocessfgsm_0.025_10000_cifar10_250_WideResNet_255_preprocess_Original_checkpoint_best_watermark_acc_loss.svg"
# image_save_path_normal_test = "../results/images/cifar10resnet_255_preprocess_10_cifar10_250_WideResNet_255_preprocessfgsm_0.025_10000_cifar10_250_WideResNet_255_preprocess_Original_checkpoint_best_normal_test_acc_loss.svg"


# json_path = "../results/finetuned_retraining_19-11-2023/losses/true/mnist_100_MNIST_l20.0fgsm_0.25_250_mnist_20_MNIST_l20.0_Original_checkpoint_best_acc_loss.json"

# image_save_path_watermark = "../results/images/retraining_mnist_100_MNIST_l20.0fgsm_0.25_250_mnist_20_MNIST_l20.0_Original_checkpoint_best_watermark_acc_loss.svg"
# image_save_path_normal_test = "../results/images/retraining_mnist_100_MNIST_l20.0fgsm_0.25_250_mnist_20_MNIST_l20.0_Original_checkpoint_best_normal_test_acc_loss.svg"








json_path = "../results/finetuned_retraining_19-11-2023/losses/full/mnist_100_MNIST_l20.0fgsm_0.25_250_mnist_20_MNIST_l20.0_Original_checkpoint_best_acc_loss.json"

image_save_path_watermark = "../results/images/retraining_mnist_100_MNIST_l20.0fgsm_0.25_250_mnist_20_MNIST_l20.0_Original_checkpoint_best_combined_acc_loss.svg"
image_save_path_normal_test = "../results/images/retraining_mnist_100_MNIST_l20.0fgsm_0.25_250_mnist_20_MNIST_l20.0_Original_checkpoint_best_normal_test_acc_loss.svg"
image_save_path_adv = "../results/images/retraining_mnist_100_MNIST_l20.0fgsm_0.25_250_mnist_20_MNIST_l20.0_Original_checkpoint_best_adv_acc_loss.svg"



# json_path = "../results/finetuned_retraining_19-11-2023/losses/full/cifar10_100_CIFAR10_BASE_2fgsm_0.025_250_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best_acc_loss.json"

# image_save_path_watermark = "../results/images/reatraining_cifar10_100_CIFAR10_BASE_2fgsm_0.025_250_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best_combined_acc_loss.svg"
# image_save_path_normal_test = "../results/images/retraining_cifar10_100_CIFAR10_BASE_2fgsm_0.025_250_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best_normal_test_acc_loss.svg"
# image_save_path_adv = "../results/images/retraining_cifar10_100_CIFAR10_BASE_2fgsm_0.025_250_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best_adv_acc_loss.svg"

# json_path = "../results/finetuned_retraining_19-11-2023/losses/full/cifar10_100_CIFAR10_BASE_2fgsm_0.1_250_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best_acc_loss.json"

# image_save_path_watermark = "../results/images/reatraining_cifar10_100_CIFAR10_BASE_2fgsm_0.1_250_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best_combined_acc_loss.svg"
# image_save_path_normal_test = "../results/images/retraining_cifar10_100_CIFAR10_BASE_2fgsm_0.1_250_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best_normal_test_acc_loss.svg"
# image_save_path_adv = "../results/images/retraining_cifar10_100_CIFAR10_BASE_2fgsm_0.1_250_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_best_adv_acc_loss.svg"



with open(json_path, 'r') as file:
    loaded_dict = json.load(file)


train_acc = loaded_dict["train_acc"]
val_acc = loaded_dict["val_acc"]
adv_test_acc = loaded_dict["adv_test_acc"]

ax1 = sns.lineplot(x=list(range(len(train_acc))), y=train_acc,
                  ci=None, color="tab:purple", marker='o', label='Combined train set accuracy', markersize=7)
ax1.set(yticks=np.arange(0,1.1,0.1))


ax2 = sns.lineplot(x=list(range(len(val_acc))), y=val_acc,
                  ci=None, color="tab:orange", linestyle='--', marker='o', label='Combined validation set accuracy', markersize=7)
ax2.set(yticks=np.arange(0,1.2,0.1))


ax3 = sns.lineplot(x=list(range(len(adv_test_acc))), y=adv_test_acc,
                  ci=None, color="tab:blue", linestyle='--', marker='o', label='Watermark set acuracy', markersize=7)
ax3.set(yticks=np.arange(0,1.2,0.1))

plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.show()
plt.savefig(image_save_path_watermark, bbox_inches='tight')



plt.figure()
normal_test_acc = loaded_dict["normal_test_acc"]

ax3 = sns.lineplot(x=list(range(len(normal_test_acc))), y=normal_test_acc,
                  ci=None, color="tab:purple", linestyle='--', marker='o', label='Legitimate test set accuracy', markersize=7)
ax3.set(yticks=np.arange(0,1.2,0.1))
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.show()
plt.savefig(image_save_path_normal_test, bbox_inches='tight')

plt.figure()


ax3 = sns.lineplot(x=list(range(len(adv_test_acc))), y=adv_test_acc,
                  ci=None, color="tab:purple", linestyle='--', marker='o', label='Watermark set acuracy', markersize=7)
ax3.set(yticks=np.arange(0,1.2,0.1))
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.show()
plt.savefig(image_save_path_adv, bbox_inches='tight')

