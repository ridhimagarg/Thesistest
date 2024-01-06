import numpy as np
import pandas as pd
import seaborn as sns;
import matplotlib.pyplot as plt
# sns.set_theme(style="ticks", palette="pastel")


# test_acc_file = "results/ewe_extraction_15-10-2023/losses/mnist_MNIST_Plain_2_convewe_training_retrain15-10-2023_3_source_4_distrib_inMNIST_l2_EWE/df_test_acc.csv"
# watermark_acc_file = "results/ewe_extraction_15-10-2023/losses/mnist_MNIST_Plain_2_convewe_training_retrain15-10-2023_3_source_4_distrib_inMNIST_l2_EWE/watermark_acc.csv"




test_df = pd.read_csv(test_acc_file)
watermark_df = pd.read_csv(watermark_acc_file)

# sns.boxplot(x=test_df["0"])
# sns.boxplot(x=watermark_df["0"])

combined_dfs = pd.DataFrame({"test_acc": test_df["0"], "watermark_acc": watermark_df["0"]})

sns.set_style('white')
sns.boxplot(data=combined_dfs, palette={"test_acc": "tomato", "watermark_acc": "tab:purple"})
sns.despine()


# ax = sns.lineplot(x="Stealing Dataset Size", y="Accuracy", estimator = np.median,
#                   data=watermark_df,ci=None, color="tab:orange", marker='s', label="Watermark Acc")
#
# bounds = watermark_df.groupby('Stealing Dataset Size')['Accuracy'].quantile((0.25,0.75)).unstack()
# print(bounds.iloc[:,0])
# print(bounds.iloc[:,1])
# ax.fill_between(x=bounds.index,y1=bounds.iloc[:,0],y2=bounds.iloc[:,1],alpha=0.3, color="tomato")
# ax.set(yticks=np.arange(0,1,0.1))
# # plt.show()
#
# ax1 = sns.lineplot(x="Stealing Dataset Size", y="Accuracy", estimator = np.median,
#                   data=test_df,ci=None, color="tab:purple", linestyle='--', marker='s', label='Test Acc')
#
# bounds = test_df.groupby('Stealing Dataset Size')['Accuracy'].quantile((0.25,0.75)).unstack()
# ax1.fill_between(x=bounds.index,y1=bounds.iloc[:,0],y2=bounds.iloc[:,1],alpha=0.2, color="tab:purple")
plt.show()


