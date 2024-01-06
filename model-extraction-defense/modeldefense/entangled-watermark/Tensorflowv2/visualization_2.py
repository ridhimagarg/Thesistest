import numpy as np
import pandas as pd
import seaborn as sns;
import matplotlib.pyplot as plt
# sns.set_theme(style="ticks", palette="pastel")


# test_acc_file = "results/attack_16-10-2023/losses/mnist_MNIST_Plain_2_conv_real_stealingewe_training_retrain15-10-2023_3_source_4_distrib_inMNIST_l2_EWE/df_test_acc.csv"
# watermark_acc_file = "results/attack_16-10-2023/losses/mnist_MNIST_Plain_2_conv_real_stealingewe_training_retrain15-10-2023_3_source_4_distrib_inMNIST_l2_EWE/df_watermark_acc.csv"

# test_acc_file = "results/attack_16-10-2023/losses/mnist_MNIST_Plain_2_conv_real_stealingewe_training_finetune16-10-2023_3_source_4_distrib_inmnist_20_MNIST_l20.0Original_checkpoint_best/df_test_acc.csv"
# watermark_acc_file = "results/attack_16-10-2023/losses/mnist_MNIST_Plain_2_conv_real_stealingewe_training_finetune16-10-2023_3_source_4_distrib_inmnist_20_MNIST_l20.0Original_checkpoint_best/df_watermark_acc.csv"

# test_acc_file = "results/attack_19-10-2023/losses/cifar10_CIFAR10_BASE_2ewe_training_retrain19-10-2023_5_source_9_distrib_outCIFAR10_BASE_2_EWE/df_test_acc.csv"
# watermark_acc_file = "results/attack_19-10-2023/losses/cifar10_CIFAR10_BASE_2ewe_training_retrain19-10-2023_5_source_9_distrib_outCIFAR10_BASE_2_EWE/df_watermark_acc.csv"

test_acc_file = "results/attack_22-10-2023/losses/mnist_MNIST_Plain_2_conv_real_stealingewe_training_retrain22-10-2023_8_source_1_distrib_inMNIST_l2_EWE/df_test_acc.csv"
watermark_acc_file = "results/attack_22-10-2023/losses/mnist_MNIST_Plain_2_conv_real_stealingewe_training_retrain22-10-2023_8_source_1_distrib_inMNIST_l2_EWE/df_watermark_acc.csv"
# test_acc_file = "results/attack_01-11-2023/losses/False_mnist_MNIST_Plain_2_conv_real_stealingewe_training_retrain31-10-2023_2_source_9_distrib_out_with_triggerMNIST_l2_EWE/df_test_acc.csv"
# watermark_acc_file = "results/attack_01-11-2023/losses/False_mnist_MNIST_Plain_2_conv_real_stealingewe_training_retrain31-10-2023_2_source_9_distrib_out_with_triggerMNIST_l2_EWE/df_watermark_acc.csv"
 
# image_save_path = "results/images/False_mnist_MNIST_Plain_2_conv_real_stealingewe_training_retrain31-10-2023_2_source_9_distrib_out_with_triggerMNIST_l2_EWE.svg"


# mnist_MNIST_Plain_2_conv_real_stealingewe_training_retrain14-10-2023_7_source_1_distrib_outMNIST_l2_EWE

# test_acc_file = "results/attack_16-10-2023/losses/mnist_MNIST_Plain_2_conv_real_stealingewe_training_retrain14-10-2023_7_source_1_distrib_outMNIST_l2_EWE/df_test_acc.csv"
# watermark_acc_file = "results/attack_16-10-2023/losses/mnist_MNIST_Plain_2_conv_real_stealingewe_training_retrain14-10-2023_7_source_1_distrib_outMNIST_l2_EWE/df_watermark_acc.csv"
 
# image_save_path = "results/images/mnist_MNIST_Plain_2_conv_real_stealingewe_training_retrain14-10-2023_7_source_1_distrib_outMNIST_l2_EWE.svg"


# cifar10_CIFAR10_BASE_2ewe_training_retrain19-10-2023_5_source_9_distrib_outCIFAR10_BASE_2_EWE

# test_acc_file = "results/attack_19-10-2023/losses/cifar10_CIFAR10_BASE_2ewe_training_retrain19-10-2023_5_source_9_distrib_outCIFAR10_BASE_2_EWE/df_test_acc.csv"

# watermark_acc_file = "results/attack_19-10-2023/losses/cifar10_CIFAR10_BASE_2ewe_training_retrain19-10-2023_5_source_9_distrib_outCIFAR10_BASE_2_EWE/df_watermark_acc.csv"
 
# image_save_path = "results/images/cifar10_CIFAR10_BASE_2ewe_training_retrain19-10-2023_5_source_9_distrib_outCIFAR10_BASE_2_EWE.svg"


# mnist_MNIST_Plain_2_conv_real_stealingewe_training_finetune16-10-2023_3_source_4_distrib_inmnist_20_MNIST_l20.0Original_checkpoint_best

test_acc_file = "results/attack_16-10-2023/losses/mnist_MNIST_Plain_2_conv_real_stealingewe_training_finetune16-10-2023_3_source_4_distrib_inmnist_20_MNIST_l20.0Original_checkpoint_best/df_test_acc.csv"

watermark_acc_file = "results/attack_16-10-2023/losses/mnist_MNIST_Plain_2_conv_real_stealingewe_training_finetune16-10-2023_3_source_4_distrib_inmnist_20_MNIST_l20.0Original_checkpoint_best/df_watermark_acc.csv"
 
image_save_path = "results/images/mnist_MNIST_Plain_2_conv_real_stealingewe_training_finetune16-10-2023_3_source_4_distrib_inmnist_20_MNIST_l20.0Original_checkpoint_best.svg"


test_df = pd.read_csv(test_acc_file)
watermark_df = pd.read_csv(watermark_acc_file)

ax = sns.lineplot(x="Stealing Dataset Size", y="Accuracy", estimator = np.median,
                  data=watermark_df,ci=None, color="tab:orange", marker='s', label="Watermark Acc")

sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 1.5})

ax = sns.lineplot(x="Stealing Dataset Size", y="Accuracy", estimator = np.median,
                  data=watermark_df,ci=None, color="tab:red", marker='s', label="Watermark Acc")


bounds = watermark_df.groupby('Stealing Dataset Size')['Accuracy'].quantile((0.25,0.75)).unstack()
print(bounds.iloc[:,0])
print(bounds.iloc[:,1])
ax.fill_between(x=bounds.index,y1=bounds.iloc[:,0],y2=bounds.iloc[:,1],alpha=0.3, color="tomato")
ax.set(yticks=np.arange(0,1,0.1))
# plt.show()

ax1 = sns.lineplot(x="Stealing Dataset Size", y="Accuracy", estimator = np.median,

                  data=test_df,ci=None, color="tab:purple", linestyle='--', marker='s', label='Test Acc')

bounds = test_df.groupby('Stealing Dataset Size')['Accuracy'].quantile((0.25,0.75)).unstack()
ax1.fill_between(x=bounds.index,y1=bounds.iloc[:,0],y2=bounds.iloc[:,1],alpha=0.2, color="tab:purple")
plt.show()

bounds = test_df.groupby('Stealing Dataset Size')['Accuracy'].quantile((0.25,0.75)).unstack()
ax1.fill_between(x=bounds.index,y1=bounds.iloc[:,0],y2=bounds.iloc[:,1],alpha=0.2, color="tab:purple")
# plt.show()
plt.savefig(image_save_path,bbox_inches='tight')


