import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, ReLU, BatchNormalization, GlobalAveragePooling2D, AveragePooling2D, Concatenate, ZeroPadding2D
from keras.models import Model
from keras.layers import Input, add
from keras import layers

import tensorflow_model_optimization as tfmot
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

def setup_transformation(dataset_name):
    mean = [0.5, ]
    std = [0.5, ]
    # if normalize_with_imagenet_vals:
    #     mean =  [0.485, 0.456, 0.406]
    #     std  =  [0.229, 0.224, 0.225]

    if dataset_name == "mnist":
        data_augmentation = tf.keras.Sequential([
            layers.Normalization(mean=mean, variance=std),
            ])
    
    if dataset_name == "cifar10":
        data_augmentation = tf.keras.Sequential([
            layers.Resizing(224, 224),
            layers.CenterCrop(224, 224),
            layers.Normalization(mean=mean, variance=std)
            ])
        
    return data_augmentation


class Plain_2_conv_Keras(Model):

    def __init__(self):
        super(Plain_2_conv_Keras, self).__init__(name='Plain_2_conv_Keras')
        self.name1 = "Plain_2_conv_Keras"
        self.conv1 = Conv2D(filters=32, kernel_size=5, activation=None)
        self.relu = ReLU()
        self.maxpool = MaxPooling2D(pool_size=(2,2), strides=2)
        self.dropout = Dropout(0.5)
        self.conv2 = Conv2D(filters=64, kernel_size=3, activation=None)
        self.flatten = Flatten()
        self.dense1 = Dense(128, activation=None)
        self.dense2 = Dense(10, activation="softmax")
        self.augmentation = setup_transformation("mnist")
    
    def call(self, x):
        
        # print("hello")
        # input = Input(shape=(28,28,1))
        x = self.augmentation(x)
        conv1 = self.conv1(x)
        relu1 = self.relu(conv1)
        pool1 = self.maxpool(relu1)
        drop1 = self.dropout(pool1)
        conv2 = self.conv2(drop1)
        relu2 = self.relu(conv2)
        pool2 = self.maxpool(relu2)
        drop2 = self.dropout(pool2)
        flatten1 = self.flatten(drop2)
        dense1 = self.dense1(flatten1)
        drop3 = self.dropout(dense1)
        relu3 = self.relu(drop3)
        output = self.dense2(relu3)

        return output
    

class Small(Model):

    def __init__(self):
        super(Small, self).__init__(name='Small')
        self.name1 = "Small"
        self.flatten = Flatten()
        self.dense1 = Dense(128, activation="relu")
        self.dense2 = Dense(10)

    
    def call(self, x):
        
        # print("hello")
        # input = Input(shape=(28,28,1))
        flatten1 = self.flatten(x)
        dense1 = self.dense1(flatten1)
        output = self.dense2(dense1)

        return output


        # return Model(inputs=input, outputs=output)


class MNIST_L2(Model):
    def __init__(self, dropout=0.0):
        super(MNIST_L2, self).__init__()

        self.drop = dropout

        self.conv1 = Conv2D(32, 5)
        self.conv2 = Conv2D(64, 5)

        self.relu = ReLU(True)
        self.pool = MaxPooling2D(pool_size=(2,2))
        self.flatten = Flatten()
        self.fc1 = Dense(10)
        self.dropout = Dropout(self.drop)
        self.name1 = "MNIST_l2" + str(self.drop)

    def call(self, x):
        
        x = self.conv1(x)
        x = self.pool(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.relu(x)
        x = self.flatten(x)
        if self.drop != 0:
            x = self.dropout(x)
        x = self.fc1(x)

        return x




pruning_params_2_by_4 = {
    'sparsity_m_by_n': (2, 4),
}

pruning_params_sparsity_0_5 = {
    'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=0.3,
                                                              begin_step=0,
                                                              frequency=100)
}

class MNIST_L2_Finetune(Model):
    def __init__(self, dropout=0.0):
        super(MNIST_L2_Finetune, self).__init__()

        self.drop = dropout

        self.conv1 = prune_low_magnitude(Conv2D(32, 5, name="pruning_sparsity_0_5_0"), **pruning_params_sparsity_0_5)
        self.conv2 = prune_low_magnitude(Conv2D(64, 5, name="pruning_sparsity_0_5_1"),  **pruning_params_sparsity_0_5)

        self.relu = ReLU(True)
        self.pool = MaxPooling2D(pool_size=(2,2))
        self.flatten = Flatten()
        self.fc1 = Dense(10)
        self.dropout = Dropout(self.drop)
        self.name1 = "MNIST_L2_Finetune" + str(self.drop)

    def call(self, x):
        
        x = self.conv1(x)
        x = self.pool(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.relu(x)
        x = self.flatten(x)
        if self.drop != 0:
            x = self.dropout(x)
        x = self.fc1(x)

        return x



class MNIST_L5(Model):
    def __init__(self, dropout=0.0):
        super(MNIST_L5, self).__init__()

        self.drop = dropout

        self.conv1 = Conv2D(32, 2)
        self.conv2 = Conv2D(64, 2)
        self.conv3 = Conv2D(128, 2)

        self.relu = ReLU(True)
        self.pool = MaxPooling2D(pool_size=(2,2))
        self.flatten = Flatten()
        self.fc1 = Dense(10)
        self.dropout = Dropout(self.drop)
        self.name1 = "MNIST_l5" + str(self.drop)

    def call(self, x):
        
        x = self.dropout(x)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.flatten(x)

        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x




class CIFAR10_BASE(Model):
    def __init__(self, dropout=0.0):
        super(CIFAR10_BASE, self).__init__()

        self.drop = dropout

        self.padding1 = ZeroPadding2D(padding=(1,1))
        self.conv1 = Conv2D(32, 3, activation=None)
        self.batchnorm1 = BatchNormalization()
        self.relu = ReLU()
        self.conv2 = Conv2D(64, 3, activation=None)
        self.maxpool = MaxPooling2D(pool_size=(2,2), strides=2)
        self.conv3 = Conv2D(128, 3, activation=None)
        self.batchnorm2 = BatchNormalization()
        self.conv4 = Conv2D(128, 3, activation=None)
        self.dropout1 = Dropout(self.drop)
        self.conv5 = Conv2D(256, 3, activation=None)
        self.batchnorm3 = BatchNormalization()
        self.conv6 = Conv2D(256, 3)
        
        self.flatten = Flatten()
        self.dropout2 = Dropout(self.drop)
        self.dense1 = Dense(1024, activation=None)
        self.dense2 = Dense(512, activation=None)
        self.dense3 = Dense(10, activation=None)
        self.name1 = "CIFAR10_BASE" + str(self.drop)

    def call(self, x):

        x = self.padding1(x)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.padding1(x)
        x = self.conv3(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout1(x)

        x = self.conv5(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.flatten(x)
        x = self.dropout2(x)
        x = self.dense1(x)
        x= self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.dense3(x)

        return x


class CIFAR10_BASE_Prune(Model):
    def __init__(self, dropout=0.0):
        super(CIFAR10_BASE_Prune, self).__init__()

        self.drop = dropout

        self.padding1 = ZeroPadding2D(padding=(1,1))
        self.conv1 = prune_low_magnitude(Conv2D(32, 3, activation=None, name="pruning_sparsity_0_5_0"), **pruning_params_sparsity_0_5)
        self.batchnorm1 = BatchNormalization()
        self.relu = ReLU()
        self.conv2 = prune_low_magnitude(Conv2D(64, 3, activation=None, name="pruning_sparsity_0_5_1"), **pruning_params_sparsity_0_5)
        self.maxpool = MaxPooling2D(pool_size=(2,2), strides=2)
        self.conv3 = prune_low_magnitude(Conv2D(128, 3, activation=None, name="pruning_sparsity_0_5_2"), **pruning_params_sparsity_0_5)
        self.batchnorm2 = BatchNormalization()
        self.conv4 = Conv2D(128, 3, activation=None)
        self.dropout1 = Dropout(self.drop)
        self.conv5 = Conv2D(256, 3, activation=None)
        self.batchnorm3 = BatchNormalization()
        self.conv6 = Conv2D(256, 3)
        
        self.flatten = Flatten()
        self.dropout2 = Dropout(self.drop)
        self.dense1 = Dense(1024, activation=None)
        self.dense2 = Dense(512, activation=None)
        self.dense3 = Dense(10, activation=None)
        self.name1 = "CIFAR10_BASE_Finetune" + str(self.drop)

    def call(self, x):

        x = self.padding1(x)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.padding1(x)
        x = self.conv3(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout1(x)

        x = self.conv5(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.flatten(x)
        x = self.dropout2(x)
        x = self.dense1(x)
        x= self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.dense3(x)

        return x


class CIFAR10_BASE_2(Model):
    def __init__(self, dropout=0.0):
        super(CIFAR10_BASE_2, self).__init__()

        self.conv1 = Conv2D(32, (3,3), padding='same', activation='relu')
        self.batchnorm1 = BatchNormalization()
        self.conv2 = Conv2D(32, (3,3), padding='same', activation='relu')
        self.batchnorm2 = BatchNormalization()
        self.pool1 = MaxPooling2D(pool_size = (2,2))
        self.dropout1 = Dropout(0.3)

        self.conv3 = Conv2D(64, (3,3), padding='same', activation='relu')
        self.batchnorm3 = BatchNormalization()
        self.conv4 = Conv2D(64, (3,3), padding='same', activation='relu')
        self.batchnorm4 = BatchNormalization()
        self.pool2 = MaxPooling2D(pool_size = (2,2))
        self.dropout2 = Dropout(0.5)

        self.conv5 = Conv2D(128, (3,3), padding='same', activation='relu')
        self.batchnorm5 = BatchNormalization()
        self.conv6 = Conv2D(128, (3,3), padding='same', activation='relu')
        self.batchnorm6 = BatchNormalization()
        self.pool3 = MaxPooling2D(pool_size = (2,2))
        self.dropout3 = Dropout(0.5)

        self.flatten = Flatten()
        self.dense1 = Dense(128, activation="relu")
        self.batchnorm7 = BatchNormalization()
        self.dropout4 = Dropout(0.5)
        self.dense2 = Dense(10)

        self.name1 = "CIFAR10_BASE_2"

    def call(self, x):

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.pool1(x)
        self.dropout1(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.pool2(x)
        self.dropout2(x)

        x = self.conv5(x)
        x = self.batchnorm5(x)
        x = self.conv6(x)
        x = self.batchnorm6(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.batchnorm7(x)
        x = self.dropout4(x)
        x = self.dense2(x)

        return x


# class CIFAR10_BASE_2_Prune(Model):
#     def __init__(self, dropout=0.0):
#         super(CIFAR10_BASE_2_Prune, self).__init__()

#         self.conv1 = prune_low_magnitude(Conv2D(32, (3,3), padding='same', activation='relu',name="pruning_sparsity_0_5_0" ), **pruning_params_sparsity_0_5)
#         self.batchnorm1 = BatchNormalization()
#         self.conv2 = prune_low_magnitude(Conv2D(32, (3,3), padding='same', activation='relu',name="pruning_sparsity_0_5_1" ), **pruning_params_sparsity_0_5)
#         self.batchnorm2 = BatchNormalization()
#         self.pool1 = MaxPooling2D(pool_size = (2,2))
#         self.dropout1 = Dropout(0.3)

#         self.conv3 = prune_low_magnitude(Conv2D(64, (3,3), padding='same', activation='relu',name="pruning_sparsity_0_5_2" ), **pruning_params_sparsity_0_5)
#         self.batchnorm3 = BatchNormalization()
#         self.conv4 = prune_low_magnitude(Conv2D(64, (3,3), padding='same', activation='relu',name="pruning_sparsity_0_5_3"), **pruning_params_sparsity_0_5)
#         self.batchnorm4 = BatchNormalization()
#         self.pool2 = MaxPooling2D(pool_size = (2,2))
#         self.dropout2 = Dropout(0.5)

#         self.conv5 = prune_low_magnitude(Conv2D(128, (3,3), padding='same', activation='relu',name="pruning_sparsity_0_5_4"), **pruning_params_sparsity_0_5)
#         self.batchnorm5 = BatchNormalization()
#         self.conv6 = prune_low_magnitude(Conv2D(128, (3,3), padding='same', activation='relu',name="pruning_sparsity_0_5_5"), **pruning_params_sparsity_0_5)
#         self.batchnorm6 = BatchNormalization()
#         self.pool3 = MaxPooling2D(pool_size = (2,2))
#         self.dropout3 = Dropout(0.5)

#         self.flatten = Flatten()
#         self.dense1 = prune_low_magnitude(Dense(128, activation="relu",name="pruning_sparsity_0_5_6"), **pruning_params_sparsity_0_5)
#         self.batchnorm7 = BatchNormalization()
#         self.dropout4 = Dropout(0.5)
#         self.dense2 = prune_low_magnitude(Dense(10, activation="relu",name="pruning_sparsity_0_5_7"), **pruning_params_sparsity_0_5)

#         self.name1 = "CIFAR10_BASE_2_Prune"

#     def call(self, x):

#         x = self.conv1(x)
#         x = self.batchnorm1(x)
#         x = self.conv2(x)
#         x = self.batchnorm2(x)
#         x = self.pool1(x)
#         self.dropout1(x)

#         x = self.conv3(x)
#         x = self.batchnorm3(x)
#         x = self.conv4(x)
#         x = self.batchnorm4(x)
#         x = self.pool2(x)
#         self.dropout2(x)

#         x = self.conv5(x)
#         x = self.batchnorm5(x)
#         x = self.conv6(x)
#         x = self.batchnorm6(x)
#         x = self.pool3(x)
#         x = self.dropout3(x)

#         x = self.flatten(x)
#         x = self.dense1(x)
#         x = self.batchnorm7(x)
#         x = self.dropout4(x)
#         x = self.dense2(x)

#         return x
    

class CIFAR10_BASE_2_Prune():

    def __init__(self):
        super(CIFAR10_BASE_2_Prune, self).__init__()

    
    def call(self):

        finetune_model = tf.keras.Sequential()

        finetune_model.add(prune_low_magnitude(layers.Conv2D(32, (3,3), padding='same', activation='relu',name="pruning_sparsity_0_5_0", input_shape=(32,32,3) ), **pruning_params_sparsity_0_5))
        finetune_model.add(layers.BatchNormalization())
        finetune_model.add(prune_low_magnitude(layers.Conv2D(32, (3,3), padding='same', activation='relu',name="pruning_sparsity_0_5_1" ), **pruning_params_sparsity_0_5))
        finetune_model.add(layers.BatchNormalization())
        finetune_model.add(layers.MaxPooling2D(pool_size=(2,2)))
        finetune_model.add(layers.Dropout(0.3))

        finetune_model.add(prune_low_magnitude(layers.Conv2D(64, (3,3), padding='same', activation='relu',name="pruning_sparsity_0_5_2" ), **pruning_params_sparsity_0_5))
        finetune_model.add(layers.BatchNormalization())
        finetune_model.add(prune_low_magnitude(layers.Conv2D(64, (3,3), padding='same', activation='relu',name="pruning_sparsity_0_5_3"), **pruning_params_sparsity_0_5))
        finetune_model.add(layers.BatchNormalization())
        finetune_model.add(layers.MaxPooling2D(pool_size=(2,2)))
        finetune_model.add(layers.Dropout(0.5))

        finetune_model.add(prune_low_magnitude(layers.Conv2D(128, (3,3), padding='same', activation='relu',name="pruning_sparsity_0_5_4"), **pruning_params_sparsity_0_5))
        finetune_model.add(layers.BatchNormalization())
        finetune_model.add(prune_low_magnitude(layers.Conv2D(128, (3,3), padding='same', activation='relu',name="pruning_sparsity_0_5_5"), **pruning_params_sparsity_0_5))
        finetune_model.add(layers.BatchNormalization())
        finetune_model.add(layers.MaxPooling2D(pool_size=(2,2)))
        finetune_model.add(layers.Dropout(0.5))

        finetune_model.add(layers.Flatten())
        finetune_model.add(prune_low_magnitude(layers.Dense(128, activation="relu",name="pruning_sparsity_0_5_6"), **pruning_params_sparsity_0_5))
        finetune_model.add(layers.BatchNormalization())
        finetune_model.add(layers.Dropout(0.5))
        finetune_model.add(prune_low_magnitude(layers.Dense(10, activation="relu",name="pruning_sparsity_0_5_7"), **pruning_params_sparsity_0_5))    # num_classes = 10

        return finetune_model








# class Plain_2_conv_Keras(Model):

#     def __init__(self):
#         super(Plain_2_conv_Keras, self).__init__(name='Plain_2_conv_Keras')
#         self.name1 = "Plain_2_conv_Keras"
#         self.conv1 = Conv2D(filters=32, kernel_size=5, activation=None)
#         self.relu = ReLU()
#         self.maxpool = MaxPooling2D(pool_size=(2,2), strides=2)
#         self.dropout = Dropout(0.5)
#         self.conv2 = Conv2D(filters=64, kernel_size=3, activation=None)
#         self.flatten = Flatten()
#         self.dense1 = Dense(128, activation=None)
#         self.dense2 = Dense(10, activation=None)
    
#     def call(self, x):
        
#         # print("hello")
#         # input = Input(shape=(28,28,1))
#         conv1 = self.conv1(x)
#         relu1 = self.relu(conv1)
#         pool1 = self.maxpool(relu1)
#         drop1 = self.dropout(pool1)
#         conv2 = self.conv2(drop1)
#         relu2 = self.relu(conv2)
#         pool2 = self.maxpool(relu2)
#         drop2 = self.dropout(pool2)
#         flatten1 = self.flatten(drop2)
#         dense1 = self.dense1(flatten1)
#         drop3 = self.dropout(dense1)
#         relu3 = self.relu(drop3)
#         output = self.dense2(relu3)

#         return output
    


class ResBlock(Model):
    def __init__(self, channels, stride=1):
        super(ResBlock, self).__init__(name='ResBlock')
        self.flag = (stride != 1)
        self.conv1 = Conv2D(channels, 3, stride, padding='same')
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(channels, 3, padding='same')
        self.bn2 = BatchNormalization()
        self.relu = ReLU()
        if self.flag:
            self.bn3 = BatchNormalization()
            self.conv3 = Conv2D(channels, 1, stride)

    def call(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        if self.flag:
            x = self.conv3(x)
            x = self.bn3(x)
        x1 = add([x, x1])
        x1 = self.relu(x1)
        return x1


class ResNet34(Model):
    def __init__(self):
        super(ResNet34, self).__init__(name='ResNet34')
        self.conv1 = Conv2D(64, 7, 2, padding='same')
        self.bn = BatchNormalization()
        self.relu = ReLU()
        self.mp1 = MaxPooling2D(3, 2)

        self.conv2_1 = ResBlock(64)
        self.conv2_2 = ResBlock(64)
        self.conv2_3 = ResBlock(64)

        self.conv3_1 = ResBlock(128, 2)
        self.conv3_2 = ResBlock(128)
        self.conv3_3 = ResBlock(128)
        self.conv3_4 = ResBlock(128)

        self.conv4_1 = ResBlock(256, 2)
        self.conv4_2 = ResBlock(256)
        self.conv4_3 = ResBlock(256)
        self.conv4_4 = ResBlock(256)
        self.conv4_5 = ResBlock(256)
        self.conv4_6 = ResBlock(256)

        self.conv5_1 = ResBlock(512, 2)
        self.conv5_2 = ResBlock(512)
        self.conv5_3 = ResBlock(512)

        self.pool = GlobalAveragePooling2D()
        self.fc1 = Dense(512, activation='relu')
        self.dp1 = Dropout(0.5)
        self.fc2 = Dense(512, activation='relu')
        self.dp2 = Dropout(0.5)
        self.fc3 = Dense(10)

        # self.fc1 = Dense(512, activation='relu')
        # self.fc3 = Dense(10)

        self.name1 = "ResNet34"
        self.augmentation = setup_transformation("cifar10")

    def call(self, x):

        x = self.augmentation(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.mp1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.conv4_4(x)
        x = self.conv4_5(x)
        x = self.conv4_6(x)

        s1 = x

        x = self.conv5_1(x)
        s2 = x
        x = self.conv5_2(x)
        s3 = x
        x = self.conv5_3(x)
        s4 = x

        x = self.pool(x)
        s5 = x
        x = self.fc1(x)
        x = self.dp1(x)
        x = self.fc2(x)
        x = self.dp2(x)
        x = self.fc3(x)
        return x


# model = ResNet34()
# model.build(input_shape=(1, 480, 480, 3))
# model.summary()
    


class ResNet18(Model):
    def __init__(self):
        super(ResNet34, self).__init__(name='ResNet18')
        self.conv1 = Conv2D(64, 7, 2, padding='same')
        self.bn = BatchNormalization()
        self.relu = ReLU()
        self.mp1 = MaxPooling2D(3, 2)

        self.conv2_1 = ResBlock(64)
        self.conv2_2 = ResBlock(64)

        self.conv3_1 = ResBlock(128, 2)
        self.conv3_2 = ResBlock(128)

        self.conv4_1 = ResBlock(256, 2)
        self.conv4_2 = ResBlock(256)

        self.conv5_1 = ResBlock(512, 2)
        self.conv5_2 = ResBlock(512)

        self.pool = GlobalAveragePooling2D()
        self.fc1 = Dense(512, activation='relu')
        self.dp1 = Dropout(0.5)
        self.fc2 = Dense(512, activation='relu')
        self.dp2 = Dropout(0.5)
        self.fc3 = Dense(10)
        self.name1 = "ResNet18"

    def call(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.mp1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)

        s1 = x

        x = self.conv5_1(x)
        s2 = x
        x = self.conv5_2(x)
        s3 = x


        x = self.pool(x)
        s4 = x
        x = self.fc1(x)
        x = self.dp1(x)
        x = self.fc2(x)
        x = self.dp2(x)
        x = self.fc3(x)
        return [s1, s2, s3, s4, x]
