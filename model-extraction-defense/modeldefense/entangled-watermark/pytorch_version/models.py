import torch
import torch.nn as nn
from torch.nn import functional as F


class EWE_2_Conv(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(EWE_2_Conv, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        print(in_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, out_channels=64, kernel_size=5)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(True)
        # self.fc1 = nn.Linear()

    def forward(self, x):
        x = self.conv1(x)
        s1 = x

        x = self.maxpool(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.conv2(x)
        s2 = x

        x = self.maxpool(x)
        x = self.relu(x)
        # x = self.dropout(x)
        # print(x.shape)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        # x = nn.Linear(x.shape[1], 128)(x)
        x = self.dropout(x)
        s3 = x
        # x = self.relu(x)
        x = nn.Linear(x.shape[1], self.num_classes)(x)
        x = F.log_softmax(x, dim=1)

        return [s1, s2, s3, x]


class MNIST_L2_DRP05(nn.Module):
    def __init__(self, dropout=0.5):
        super(MNIST_L2_DRP05, self).__init__()

        self.dropout = dropout

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)

        self.relu = nn.ReLU(True)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(64 * 4 * 4, 10)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.relu(self.pool(self.conv1(x)))
        x = self.relu(self.pool(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(x)
        x = self.fc1(x)
        # print("models", F.log_softmax(x, dim=1).shape)

        return F.log_softmax(x, dim=1)






