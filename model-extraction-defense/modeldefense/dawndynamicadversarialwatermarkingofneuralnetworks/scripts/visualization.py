import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns;

sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 1.5})
#
test_acc_file = "../results/attack_original_20-09-2023/losses_acc/CIFAR10_CIFAR10_BASE_2_CIFAR10_BASE_2_victim_cifar_base_2df_test_acc.csv"
watermark_acc_file = "../results/attack_original_20-09-2023/losses_acc/CIFAR10_CIFAR10_BASE_2_CIFAR10_BASE_2_victim_cifar_base_2df_watermark_acc.csv"
image_save_path = "../results/images/verification_cifar10_low_white_box.svg"

# test_acc_file = "../results/attack_original_29-09-2023/losses_acc/CIFAR10_CIFAR10_BASE_2_CIFAR10_BASE_2_victim_cifar_base_2df_test_acc.csv"
# watermark_acc_file = "../results/attack_original_29-09-2023/losses_acc/CIFAR10_CIFAR10_BASE_2_CIFAR10_BASE_2_victim_cifar_base_2df_watermark_acc.csv"
# image_save_path = "../results/images/verification_cifar10_low_black_box.svg"




# test_acc_file = "../results/attack_original_25-09-2023/losses_acc/MNIST_MNIST_L2_DRP03_MNIST_L2_DRP03_victim_mnist_l2df_test_acc.csv"
# watermark_acc_file = "../results/attack_original_25-09-2023/losses_acc/MNIST_MNIST_L2_DRP03_MNIST_L2_DRP03_victim_mnist_l2df_watermark_acc.csv"
# image_save_path = "../results/images/verification_mnist_white_box.svg"

# test_acc_file = "../results/attack_original_29-09-2023/losses_acc/MNIST_MNIST_L2_DRP03_MNIST_L2_DRP03_victim_mnist_l2df_test_acc.csv"
# watermark_acc_file = "../results/attack_original_29-09-2023/losses_acc/MNIST_MNIST_L2_DRP03_MNIST_L2_DRP03_victim_mnist_l2df_watermark_acc.csv"
# image_save_path = "../results/images/verification_mnist_black_box.svg"




# test_acc_file = "../results/attack_original_01-10-2023/losses_acc/CIFAR10_RN34_RN34_victim_cifar_rn34df_test_acc.csv"
# watermark_acc_file = "../results/attack_original_01-10-2023/losses_acc/CIFAR10_RN34_RN34_victim_cifar_rn34df_watermark_acc.csv"
# image_save_path = "../results/images/verification_cifar10_high_black_box.svg"

# test_acc_file = "../results/attack_original_01-10-2023/losses_acc/CIFAR10_RN34_victim_cifar_rn34df_test_acc.csv"
# watermark_acc_file = "../results/attack_original_01-10-2023/losses_acc/CIFAR10_RN34_victim_cifar_rn34df_watermark_acc.csv"
# image_save_path = "../results/images/verification_cifar10_high_white_box.svg"

# test_acc_file = "../results/attack_finetuned11-09-2023/losses_acc/true/cifar10resnet_255_preprocess_50_final_cifar10resnet_255_preprocess_10_10_cifar10_250_WideResNet_255_preprocessfgsm_0.025_10000_cifar10_250_WideResNet_255_preprocess_Original_checkpoint_bestdf_test_acc.csv"
# watermark_acc_file = "../results/attack_finetuned11-09-2023/losses_acc/true/cifar10resnet_255_preprocess_50_final_cifar10resnet_255_preprocess_10_10_cifar10_250_WideResNet_255_preprocessfgsm_0.025_10000_cifar10_250_WideResNet_255_preprocess_Original_checkpoint_bestdf_watermark_acc.csv"


test_df = pd.read_csv(test_acc_file)
watermark_df = pd.read_csv(watermark_acc_file)


ax = sns.lineplot(x="Stealing Dataset Size", y="Accuracy", estimator = np.median,
                  data=watermark_df,ci=None, color="tab:red", marker='s', label="Watermark Accuracy")

bounds = watermark_df.groupby('Stealing Dataset Size')['Accuracy'].quantile((0.25,0.75)).unstack()
print(bounds.iloc[:,0])
print(bounds.iloc[:,1])
ax.fill_between(x=bounds.index,y1=bounds.iloc[:,0],y2=bounds.iloc[:,1],alpha=0.3, color="tomato")
ax.set(yticks=np.arange(0,1,0.1))
# plt.show()

ax1 = sns.lineplot(x="Stealing Dataset Size", y="Accuracy", estimator = np.median,
                  data=test_df,ci=None, color="tab:blue", linestyle='--', marker='s', label='Test Accuracy')

bounds = test_df.groupby('Stealing Dataset Size')['Accuracy'].quantile((0.25,0.75)).unstack()
ax1.fill_between(x=bounds.index,y1=bounds.iloc[:,0],y2=bounds.iloc[:,1],alpha=0.2, color="tab:blue")

plt.gca().set_xticklabels(plt.gca().get_xticks(), fontsize=12)
ax.set_xticklabels([f'{int(label)}' for label in ax.get_xticks()])
plt.show()

plt.savefig(image_save_path, bbox_inches='tight')


