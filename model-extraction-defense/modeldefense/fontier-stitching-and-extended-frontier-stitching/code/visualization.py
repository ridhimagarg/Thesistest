"""
@author: Ridhima Garg

Introduction:
    This file contains the implementation for visualization of the results of the averaging.

"""

import numpy as np
import pandas as pd
import seaborn as sns;
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 1.5})

#
# test_acc_file = "../results/attack_finetuned11-09-2023/losses_acc/true/cifar10_50_cifar10_25_25_bestfgsm_0.035_10000_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_bestdf_test_acc.csv"
# watermark_acc_file = "../results/attack_finetuned11-09-2023/losses_acc/true/cifar10_50_cifar10_25_25_bestfgsm_0.035_10000_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_bestdf_watermark_acc.csv"
# image_save_path = "../results/images/xfs_cifar10_low_white_box.svg"


# test_acc_file = "../results/attack_finetuned21-09-2023/losses_acc/true/cifar10_50_cifar10_25_25_bestfgsm_0.035_10000_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_bestdf_test_acc.csv"
# watermark_acc_file = "../results/attack_finetuned21-09-2023/losses_acc/true/cifar10_50_cifar10_25_25_bestfgsm_0.035_10000_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_bestdf_watermark_acc.csv"
# image_save_path = "../results/images/xfs_cifar10_low_black_box.svg"





test_acc_file = "../results/attack_finetuned11-09-2023/losses_acc/true/cifar10resnet_255_preprocess_50_final_cifar10resnet_255_preprocess_10_10_cifar10_250_WideResNet_255_preprocessfgsm_0.025_10000_cifar10_250_WideResNet_255_preprocess_Original_checkpoint_bestdf_test_acc.csv"
watermark_acc_file = "../results/attack_finetuned11-09-2023/losses_acc/true/cifar10resnet_255_preprocess_50_final_cifar10resnet_255_preprocess_10_10_cifar10_250_WideResNet_255_preprocessfgsm_0.025_10000_cifar10_250_WideResNet_255_preprocess_Original_checkpoint_bestdf_watermark_acc.csv"
image_save_path = "../results/images/xfs_cifar10_high_black_box.svg"

# test_acc_file = "../results/attack_finetuned26-10-2023/losses_acc/true/mnist_l2_50_mnist_25_25_mnist_20_MNIST_l20.0fgsm_0.25_10000_mnist_20_MNIST_l20.0_Original_checkpoint_bestdf_test_acc.csv"
# watermark_acc_file = "../results/attack_finetuned26-10-2023/losses_acc/true/mnist_l2_50_mnist_25_25_mnist_20_MNIST_l20.0fgsm_0.25_10000_mnist_20_MNIST_l20.0_Original_checkpoint_bestdf_watermark_acc.csv"
# image_save_path = "../results/images/xfs_mnist_white_box.svg"






# test_acc_file = "../results/attack_finetuned28-10-2023/losses_acc/true/cifar10resnet_255_preprocess_50_final_cifar10resnet_255_preprocess_10_10_cifar10_250_WideResNet_255_preprocessfgsm_0.025_10000_cifar10_250_WideResNet_255_preprocess_Original_checkpoint_bestdf_test_acc.csv"
# watermark_acc_file = "../results/attack_finetuned28-10-2023/losses_acc/true/cifar10resnet_255_preprocess_50_final_cifar10resnet_255_preprocess_10_10_cifar10_250_WideResNet_255_preprocessfgsm_0.025_10000_cifar10_250_WideResNet_255_preprocess_Original_checkpoint_bestdf_watermark_acc.csv"
# image_save_path = "../results/images/xfs_cifar10_high_white_box.svg"


# test_acc_file = "../results/attack_finetuned26-10-2023/losses_acc/true/mnist_50_mnist_25_25_mnist_20_MNIST_l20.0fgsm_0.25_10000_mnist_20_MNIST_l20.0_Original_checkpoint_bestdf_test_acc.csv"
# watermark_acc_file = "../results/attack_finetuned26-10-2023/losses_acc/true/mnist_50_mnist_25_25_mnist_20_MNIST_l20.0fgsm_0.25_10000_mnist_20_MNIST_l20.0_Original_checkpoint_bestdf_watermark_acc.csv"
# image_save_path = "../results/images/xfs_mnist_black_box.svg"



## Retraining with full adv

# test_acc_file = "../results/attack_finetuned19-11-2023/losses_acc/full/mnist_50_final_mnist_100_MNIST_l20.0fgsm_0.25_250_mnist_20_MNIST_l20.0_Original_checkpoint_bestdf_test_acc.csv"
# watermark_acc_file = "../results/attack_finetuned19-11-2023/losses_acc/full/mnist_50_final_mnist_100_MNIST_l20.0fgsm_0.25_250_mnist_20_MNIST_l20.0_Original_checkpoint_bestdf_watermark_acc.csv"
# image_save_path = "../results/images/fs_mnist_white_box.svg"


# test_acc_file = "../results/attack_finetuned19-11-2023/losses_acc/full/cifar10_50_cifar10_100_CIFAR10_BASE_2fgsm_0.025_250_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_bestdf_test_acc.csv"
# watermark_acc_file = "../results/attack_finetuned19-11-2023/losses_acc/full/cifar10_50_cifar10_100_CIFAR10_BASE_2fgsm_0.025_250_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_bestdf_watermark_acc.csv"
# image_save_path = "../results/images/fs_ciafr10_0.025_white_box.svg"

test_acc_file = "../results/attack_finetuned19-11-2023/losses_acc/full/cifar10_50_cifar10_100_CIFAR10_BASE_2fgsm_0.1_250_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_bestdf_test_acc.csv"
watermark_acc_file = "../results/attack_finetuned19-11-2023/losses_acc/full/cifar10_50_cifar10_100_CIFAR10_BASE_2fgsm_0.1_250_cifar10_30_CIFAR10_BASE_2_Original_checkpoint_bestdf_watermark_acc.csv"
image_save_path = "../results/images/fs_ciafr10_0.1_white_box.svg"



test_df = pd.read_csv(test_acc_file)
watermark_df = pd.read_csv(watermark_acc_file)


ax = sns.lineplot(x="Stealing Dataset Size", y="Accuracy", estimator = np.median,
                  data=watermark_df,ci=None, color="tab:red", marker='s', label="Watermark Accuracy")

bounds = watermark_df.groupby('Stealing Dataset Size')['Accuracy'].quantile((0.20,0.90)).unstack()
print(bounds.iloc[:,0])
print(bounds.iloc[:,1])
ax.fill_between(x=bounds.index,y1=bounds.iloc[:,0],y2=bounds.iloc[:,1],alpha=0.3, color="tomato")
ax.set(yticks=np.arange(0,1,0.1))
# plt.show()

ax1 = sns.lineplot(x="Stealing Dataset Size", y="Accuracy", estimator = np.median,
                  data=test_df,ci=None, color="tab:blue", linestyle='--', marker='s', label='Test Accuracy')

bounds = test_df.groupby('Stealing Dataset Size')['Accuracy'].quantile((0.20,0.90)).unstack()
ax1.fill_between(x=bounds.index,y1=bounds.iloc[:,0],y2=bounds.iloc[:,1],alpha=0.2, color="tab:blue")

plt.gca().set_xticklabels(plt.gca().get_xticks(), fontsize=12)

# ax.xaxis.set_major_locator(ticker.MultipleLocator(2)) 
# ax.xaxis.set_major_locator(ticker.AutoLocator())
# ax1.set_xticks([0, 250, 500, 1000, 5000, 10000, 15000, 20000])
ax.set_xticklabels([f'{int(label)}' for label in ax.get_xticks()])
# ax.set_xticks(ax.get_xticks()[::1])
# ax.set_xticklabels(ax.get_xticks(), rotation=15)  # Rotate labels by 45 degrees

plt.show()

plt.savefig(image_save_path, bbox_inches='tight')


