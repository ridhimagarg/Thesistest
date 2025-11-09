import numpy as np
import pandas as pd
import seaborn as sns;
import matplotlib.pyplot as plt


# test_acc_10 = [60, 50,40, 30]
# watermark_acc_10 = [50, 40, 30, 20]
# test_acc_25 = [65, 55,45, 35]
# watermark_acc_25 = [60, 50, 40, 30]
# test_acc_50 = [75, 65,55, 45]
# watermark_acc_25 = [70, 60, 50, 20]

test_acc_10 = pd.read_csv("../results/fineprune_attcker_model_06-11-2023/losses_acc/MNIST_MNIST_L2_DRP03_10_model_argmax_knockoffNets_MNIST_250test_df.csv", index_col=0)
watermark_acc_10 = pd.read_csv("../results/fineprune_attcker_model_06-11-2023/losses_acc/MNIST_MNIST_L2_DRP03_10_model_argmax_knockoffNets_MNIST_250watermark_df.csv", index_col=0)

test_acc_25 = pd.read_csv("../results/fineprune_attcker_model_06-11-2023/losses_acc/MNIST_MNIST_L2_DRP03_25_model_argmax_knockoffNets_MNIST_250test_df.csv", index_col=0)
watermark_acc_25 = pd.read_csv("../results/fineprune_attcker_model_06-11-2023/losses_acc/MNIST_MNIST_L2_DRP03_25_model_argmax_knockoffNets_MNIST_250watermark_df.csv", index_col=0)


test_acc_50 = pd.read_csv("../results/fineprune_attcker_model_06-11-2023/losses_acc/MNIST_MNIST_L2_DRP03_50_model_argmax_knockoffNets_MNIST_250test_df.csv", index_col=0)
watermark_acc_50 = pd.read_csv("../results/fineprune_attcker_model_06-11-2023/losses_acc/MNIST_MNIST_L2_DRP03_50_model_argmax_knockoffNets_MNIST_250watermark_df.csv", index_col=0)

test_acc = pd.concat([test_acc_50, test_acc_25,  test_acc_10], axis=1)
test_acc = test_acc.loc[:, ~test_acc.columns.duplicated()]

watermark_acc = pd.concat([watermark_acc_50, watermark_acc_25, watermark_acc_10], axis=1)
watermark_acc = watermark_acc.loc[:, ~watermark_acc.columns.duplicated()]


color_watermark = ["tab:red", "tab:blue", "tab:green"]

for index, col in enumerate(test_acc.columns[1:]):

    ax = sns.lineplot(x="Pruning Level", y=col, estimator = np.median,
                      data=watermark_acc,ci=None, color=color_watermark[index], marker='s', label="Watermark Accuracy - "+ col)

    bounds = watermark_acc.groupby('Pruning Level')[col].quantile((0.25,0.75)).unstack()
    print(bounds.iloc[:,0])
    print(bounds.iloc[:,1])
    ax.fill_between(x=bounds.index, y1=bounds.iloc[:,0], y2=bounds.iloc[:,1], alpha=0.3, color=color_watermark[index])
    ax.set(yticks=np.arange(0,1,0.1))
    # plt.show()

    ax1 = sns.lineplot(x="Pruning Level", y=col, estimator = np.median,
                      data=test_acc,ci=None, color=color_watermark[index], linestyle='--', marker='X', label='Test Accuracy - '+ col)

    bounds = test_acc.groupby('Pruning Level')[col].quantile((0.25,0.75)).unstack()
    ax1.fill_between(x=bounds.index, y1=bounds.iloc[:,0], y2=bounds.iloc[:,1], alpha=0.2, color=color_watermark[index])
plt.ylabel('Accuracy')
plt.show()


# plt.figure()
# sns.lineplot(test_acc_250)
# sns.lineplot(watermark_acc_250)
#
# plt.show()



# ax = sns.lineplot(x="Stealing Dataset Size", y="Accuracy", estimator = np.median,
#                   data=watermark_df,ci=None, color="tab:orange", marker='s', label="Watermark Acc")