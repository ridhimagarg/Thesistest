import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, ReLU, BatchNormalization, GlobalAveragePooling2D, AveragePooling2D, Concatenate
from keras.models import Model
from keras.layers import Input, add
from keras import layers


def pairwise_euclid_distance(A):
    sqr_norm_A = tf.expand_dims(tf.reduce_sum(input_tensor=tf.pow(A, 2), axis=1), 0)
    sqr_norm_B = tf.expand_dims(tf.reduce_sum(input_tensor=tf.pow(A, 2), axis=1), 1)
    inner_prod = tf.matmul(A, A, transpose_b=True)
    tile_1 = tf.tile(sqr_norm_A, [tf.shape(input=A)[0], 1])
    tile_2 = tf.tile(sqr_norm_B, [1, tf.shape(input=A)[0]])
    return tile_1 + tile_2 - 2 * inner_prod


def pairwise_cos_distance(A):
    normalized_A = tf.nn.l2_normalize(A, 1)
    return 1 - tf.matmul(normalized_A, normalized_A, transpose_b=True)


def snnl_func(x, y, t, metric='cosine'):
    x = tf.nn.relu(x)
    same_label_mask = tf.cast(tf.squeeze(tf.equal(y, tf.expand_dims(y, 1))), tf.float32)
    # print("same label mask", same_label_mask)
    if metric == 'euclidean':
        dist = pairwise_euclid_distance(tf.reshape(x, [tf.shape(input=x)[0], -1]))
    elif metric == 'cosine':
        dist = pairwise_cos_distance(tf.reshape(x, [tf.shape(input=x)[0], -1]))
    else:
        raise NotImplementedError()
    exp = tf.clip_by_value(tf.exp(-(dist / t)) - tf.eye(tf.shape(input=x)[0]), 0, 1)
    # print("exp", exp)
    # print((exp / (0.00001 + tf.expand_dims(tf.reduce_sum(input_tensor=exp, axis=1), 1))).shape)
    # print(same_label_mask.shape)
    prob = (exp / (0.00001 + tf.expand_dims(tf.reduce_sum(input_tensor=exp, axis=1), 1))) * same_label_mask
    # print("prob", prob)
    loss = - tf.reduce_mean(input_tensor=tf.math.log(0.00001 + tf.reduce_sum(input_tensor=prob, axis=1)))
    # print("soft nearest loss", loss)
    return loss



def ce_loss(prediction, target):

    loss_object = tf.keras.losses.CategoricalCrossentropy()

    if isinstance(prediction, list):
        prediction = prediction[-1]

    # print(target)
    # print(prediction)

    # cross_entropy = loss_object(target, tf.nn.softmax(prediction))
    cross_entropy = loss_object(target, prediction)

    # log_prob = tf.math.log(tf.nn.softmax(prediction) + 1e-12)
    # cross_entropy = - tf.reduce_sum(input_tensor=target * log_prob)

    return cross_entropy

def snnl_loss(prediction, w, temp):
    x1 = prediction[-4]
    x2 = prediction[-3]
    x3 = prediction[-2]
    inv_temp_1 = tf.math.divide(100., temp[0])
    inv_temp_2 = tf.math.divide(100., temp[1])
    inv_temp_3 = tf.math.divide(100., temp[2])
    loss1 = snnl_func(x1, w, inv_temp_1)
    loss2 = snnl_func(x2, w, inv_temp_2)
    loss3 = snnl_func(x3, w, inv_temp_3)
    res = [loss1, loss2, loss3]
    return res

def combined_loss(prediction, target, w, temp, factors):

    cross_entropy_loss = ce_loss(prediction, target)

    snnl_losses = snnl_loss(prediction, w, temp)
    # print("smml loss of layer 1", snnl_losses[0])
    # print("smml loss of layer 1 with factor", snnl_losses[0] * factors[0])
    soft_nearest_neighbor = factors[0] * snnl_losses[0] + factors[1] * snnl_losses[1] + factors[2] * snnl_losses[2]
    soft_nearest_neighbor_loss = tf.cast(tf.greater(tf.math.reduce_mean(input_tensor=w), 0), tf.float32) * soft_nearest_neighbor

    # print("cross entropy loss", cross_entropy_loss)
    # print("soft nearest neighbor loss", soft_nearest_neighbor_loss)

    return cross_entropy_loss - soft_nearest_neighbor_loss, soft_nearest_neighbor_loss

def error_rate(Y, prediction):
    if isinstance(prediction, list):
        prediction = prediction[-1]
    mistakes = tf.not_equal(tf.argmax(input=Y, axis=1), tf.argmax(input=prediction, axis=1))
    return tf.reduce_mean(input_tensor=tf.cast(mistakes, tf.float32))


def MNIST_L2(input_shape = (28,28,1), dropout=0.0):

    model = tf.keras.models.Sequential()
    model.add(Conv2D(32, 5, input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(ReLU(True))
    model.add(Conv2D(64, 5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(ReLU(True))
    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(10))
    # model.add(tf.keras.layers.Softmax(axis=-1))
    name1 = "MNIST_l2" + str(dropout)


    return name1, model

def MNIST_Plain_2_conv():

    model = tf.keras.Sequential([
    Conv2D(filters=32, kernel_size=5, activation=None, input_shape=(28, 28, 1)),
    ReLU(),
    MaxPooling2D(pool_size=(2, 2), strides=2),
    Dropout(0.5),
    Conv2D(filters=64, kernel_size=3, activation=None),
    ReLU(),
    MaxPooling2D(pool_size=(2, 2), strides=2),
    Dropout(0.5),
    Flatten(),
    Dense(128, activation=None),
    Dropout(0.5),
    ReLU(),
    Dense(10) #Dense(10, activation="softmax"),  
    ])

    return "MNIST_Plain_2_conv", model

def MNIST_Plain_2_conv_real_stealing():

    model = tf.keras.Sequential([
    Conv2D(filters=32, kernel_size=5, activation=None, input_shape=(28, 28, 1)),
    ReLU(),
    MaxPooling2D(pool_size=(2, 2), strides=2),
    Dropout(0.5),
    Conv2D(filters=64, kernel_size=3, activation=None),
    ReLU(),
    MaxPooling2D(pool_size=(2, 2), strides=2),
    Dropout(0.5),
    Flatten(),
    Dense(128, activation=None),
    Dropout(0.5),
    ReLU(),
    Dense(10, activation="softmax"),  
    ])

    return "MNIST_Plain_2_conv", model


def MNIST_L2_EWE(input_shape = (28,28,1), dropout=0.0):
    # Define the input shape
    # dropout = 0.5  # Set the dropout rate as needed

    # Define the input layer
    input_layer = Input(shape=input_shape)

    # Convolutional Layer 1
    conv1 = Conv2D(32, 5)(input_layer)
    relu1 = ReLU(True)(conv1)
    maxpool1 = MaxPooling2D(pool_size=(2, 2))(relu1)

    # Convolutional Layer 2
    conv2 = Conv2D(64, 5)(maxpool1)
    relu2 = ReLU(True)(conv2)
    maxpool2 = MaxPooling2D(pool_size=(2, 2))(relu2)

    # Flatten layer
    flatten = Flatten()(maxpool2)

    # Dropout layer
    dropout_layer = Dropout(dropout)(flatten)

    # Output layer
    output_layer = Dense(10, activation="softmax")(dropout_layer)

    # Create the Functional API model
    model = Model(inputs=input_layer, outputs=[conv1, conv2 ,dropout_layer, output_layer])

    name1 = "MNIST_l2_EWE" + str(dropout)

    return name1, model


class EWE_2_conv(tf.keras.Model):

    def __init__(self):
        super(EWE_2_conv, self).__init__()
        self.conv1 = Conv2D(filters=32, kernel_size=5, activation=None)
        self.relu = ReLU()
        self.maxpool = MaxPooling2D(pool_size=(2,2), strides=2)
        self.conv2 = Conv2D(filters=64, kernel_size=3, activation=None)
        self.dropout = Dropout(0.5)
        self.flatten = Flatten()
        self.dense1 = Dense(128, activation=None)
        self.dense2 = Dense(10, activation=None)
        self.name1 = "EWE_2_conv"

    
    def call(self, x):
        x = self.conv1(x)
        s1 = x
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.conv2(x)
        s2 = x
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        s3 = x
        x = self.relu(x)
        x = self.dense2(x)
        return [s1, s2, s3, x]
    
    # def train_step(self, images, labels, loss):

    #     with tf.GradientTape() as tape:
    #         prediction = self(images, training=True)

    #         w_0 = np.zeros([images.shape[0]])

    #         loss = combined_loss(prediction, labels)

    #     grads = tape.gradient(loss, self.trainable_weights)
        
    #     self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

    #     return loss



class Plain_2_conv(tf.keras.Model):

    def __init__(self):
        super(Plain_2_conv, self).__init__()
        self.conv1 = Conv2D(filters=32, kernel_size=5, activation=None)
        self.relu = ReLU()
        self.maxpool = MaxPooling2D(pool_size=(2,2), strides=2)
        self.conv2 = Conv2D(filters=64, kernel_size=3, activation=None)
        self.dropout = Dropout(0.5)
        self.flatten = Flatten()
        self.dense1 = Dense(128, activation=None)
        self.dense2 = Dense(10, activation=None)
        self.name1 = "Plain_2_conv"
        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-5)

    
    def call(self, x):
        x = self.conv1(x)
        s1 = x
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.conv2(x)
        s2 = x
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        s3 = x
        x = self.relu(x)
        x = self.dense2(x)
        # x = tf.nn.softmax(x)
        return x
    


# class Plain_2_conv(tf.keras.Model):

#     def __init__(self):
#         super(Plain_2_conv, self).__init__()
#         self.conv1 = Conv2D(filters=32, kernel_size=5, activation=None)
#         self.relu = ReLU()
#         self.maxpool = MaxPooling2D(pool_size=(2,2), strides=2)
#         self.conv2 = Conv2D(filters=64, kernel_size=3, activation=None)
#         self.dropout = Dropout(0.5)
#         self.flatten = Flatten()
#         self.dense2 = Dense(10, activation=None)
#         # self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-5)

    
#     def call(self, x):
#         x = self.conv1(x)
#         s1 = x
#         x = self.relu(x)
#         x = self.maxpool(x)
#         x = self.dropout(x)
#         x = self.conv2(x)
#         s2 = x
#         x = self.relu(x)
#         x = self.maxpool(x)
#         x = self.dropout(x)
#         x = self.flatten(x)
#         self.dense1 = Dense(128, activation=None)
#         x = self.dense1(x)
#         x = self.dropout(x)
#         s3 = x
#         x = self.relu(x)
#         x = self.dense2(x)
#         # x = tf.nn.softmax(x)
#         return x
    


class EWE_MNIST_L5_DR05(tf.keras.Model):

    def __init__(self):
        super(EWE_MNIST_L5_DR05, self).__init__()
        self.conv1 = Conv2D(filters=32, kernel_size=2, activation=None)
        self.relu = ReLU()
        self.maxpool = MaxPooling2D(pool_size=(2,2), strides=2)
        self.conv2 = Conv2D(filters=64, kernel_size=2, activation=None)
        self.conv3 = Conv2D(filters=128, kernel_size=2, activation=None)
        self.dropout = Dropout(0.5)
        self.flatten = Flatten()
        self.dense1 = Dense(128, activation=None)
        self.dense2 = Dense(10, activation=None)
        self.name1 = "EWE_MNIST_L5_DR05"
        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-5)

    
    def call(self, x):
        x = self.dropout(x)
        x = self.conv1(x)
        s1 = x
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        s2 = x
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        s3 = x
        x = self.relu(x)
        x = self.flatten(x)
        x = self.dropout(x)
        s4 = x
        x = self.dense1(x)
        
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        
        # x = self.relu(x)
        # x = self.dense2(x)
        # x = tf.nn.softmax(x)
        return [s1, s2, s3, s4, x]
    


class Plain_MNIST_L5_DR05(tf.keras.Model):

    def __init__(self):
        super(Plain_MNIST_L5_DR05, self).__init__()
        self.conv1 = Conv2D(filters=32, kernel_size=2, activation=None)
        self.relu = ReLU()
        self.maxpool = MaxPooling2D(pool_size=(2,2), strides=2)
        self.conv2 = Conv2D(filters=64, kernel_size=2, activation=None)
        self.conv3 = Conv2D(filters=128, kernel_size=2, activation=None)
        self.dropout = Dropout(0.5)
        self.flatten = Flatten()
        self.dense1 = Dense(128, activation=None)
        self.dense2 = Dense(10, activation=None)
        self.name1 = "Plain_MNIST_L5_DR05"
        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-5)

    
    def call(self, x):
        x = self.dropout(x)
        x = self.conv1(x)
        s1 = x
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        s2 = x
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        s2 = x
        x = self.relu(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.dense1(x)
        s3 = x
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        
        # x = self.relu(x)
        # x = self.dense2(x)
        # x = tf.nn.softmax(x)
        return x

    # def train_step(self, images, labels, loss_obj):

    #     with tf.GradientTape() as tape:
    #         prediction = self(images, training=True)

    #         loss = loss_obj(prediction, labels)

    #     grads = tape.gradient(loss, self.trainable_weights)
        
    #     self.optimizer.apply_gradients(zip(grads, self.trainable_weights))



class Plain_2_conv_Keras():

    def __init__(self):
        self.name = "Plain_2_conv_Keras"
    
    def call(self):
        
        print("hello")
        input = Input(shape=(28,28,1))
        conv1 = Conv2D(filters=32, kernel_size=5, activation=None)(input)
        relu1 = ReLU()(conv1)
        pool1 = MaxPooling2D(pool_size=(2,2), strides=2)(relu1)
        drop1 = Dropout(0.5)(pool1)
        conv2 = Conv2D(filters=64, kernel_size=3, activation=None)(drop1)
        relu2 = ReLU()(conv2)
        pool2 = MaxPooling2D(pool_size=(2,2), strides=2)(relu2)
        drop2 = Dropout(0.5)(pool2)
        flatten1 = Flatten()(drop2)
        dense1 = Dense(128, activation=None)(flatten1)
        drop3 = Dropout(0.5)(dense1)
        relu3 = ReLU()(drop3)
        output = Dense(10, activation=None)(relu3)

        return Model(inputs=input, outputs=output)


class EWE_feature_extract_Keras():

    def __init__(self):
        self.name = "EWE_feature_extract_Keras"

    # def call(self):

    #     new_input = Input(shape= (5,5,64))
    #     conv1 = Conv2D(filters=64, kernel_size=3, activation=None)(new_input)
    #     s1 = conv1
    #     relu1 = ReLU()(conv1)
    #     pool1 = MaxPooling2D(pool_size=(2,2), strides=2)(relu1)
    #     drop1 = Dropout(0.5)(pool1)
    #     # conv2 = Conv2D(filters=64, kernel_size=3, activation=None)(drop1)
    #     # s2 = conv2
    #     # relu2 = ReLU()(conv2)
    #     # pool2 = MaxPooling2D(pool_size=(2,2), strides=2)(relu2)
    #     # drop2 = Dropout(0.5)(pool2)
    #     s2 = drop1
    #     flatten1 = Flatten()(drop1)
    #     dense1 = Dense(128, activation=None)(flatten1)
    #     s3 = dense1
    #     drop3 = Dropout(0.5)(dense1)
    #     relu3 = ReLU()(drop3)
    #     output = Dense(10, activation=None)(relu3)

    #     return Model(inputs=new_input, outputs=[s1, s2, s3, output])
    

    def call(self):

        new_input = Input(shape= (5,5,64))
        conv1 = Conv2D(filters=64, kernel_size=3, activation=None)(new_input)
        s1 = conv1
        relu1 = ReLU()(conv1)
        drop1 = Dropout(0.5)(relu1)
        s2 = drop1
        flatten1 = Flatten()(drop1)
        dense1 = Dense(128, activation=None)(flatten1)
        drop3 = Dropout(0.5)(dense1)
        s3 = drop3
        relu3 = ReLU()(drop3)
        output = Dense(10, activation=None)(relu3)

        return Model(inputs=new_input, outputs=[s1, s2, s3, output])


class EWE_feature_extract_Keras_moreconv():

    def __init__(self):
        self.name = "EWE_feature_extract_Keras_moreconv"

    def call(self):

        new_input = Input(shape= (5,5,64))
        conv1 = Conv2D(filters=64, kernel_size=3, activation=None)(new_input)
        s1 = conv1
        relu1 = ReLU()(conv1)
        pool1 = MaxPooling2D(pool_size=(2,2), strides=2)(relu1)
        drop1 = Dropout(0.5)(pool1)
        # conv2 = Conv2D(filters=64, kernel_size=3, activation=None)(drop1)
        # s2 = conv2
        # relu2 = ReLU()(conv2)
        # pool2 = MaxPooling2D(pool_size=(2,2), strides=2)(relu2)
        # drop2 = Dropout(0.5)(pool2)
        s2 = drop1
        flatten1 = Flatten()(drop1)
        dense1 = Dense(128, activation=None)(flatten1)
        s3 = dense1
        drop3 = Dropout(0.5)(dense1)
        relu3 = ReLU()(drop3)
        output = Dense(10, activation=None)(relu3)

        return Model(inputs=new_input, outputs=[s1, s2, s3, output])
    


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

    def call(self, x):
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
        return [s1, s2, s3, s4, s5, x]


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



def CIFAR10_BASE_2(input_shape = (32,32,3), dropout=0.0, num_classes=10):

    model = tf.keras.models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add((layers.Conv2D(32, (3, 3), padding='same', activation='relu')))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation="softmax"))
    name1 = "CIFAR10_BASE_2"

    return name1, model


def CIFAR10_SMALL(input_shape = (32,32,3), dropout=0.0, num_classes=10):
    """
    Smaller CIFAR10 model for faster experimentation.
    Reduced from CIFAR10_BASE_2:
    - Fewer conv layers (4 instead of 6)
    - Smaller dense layer (64 instead of 128)
    - Still achieves good accuracy (~75-80% on CIFAR10)
    """
    model = tf.keras.models.Sequential()

    # Block 1: 32 filters
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.3))

    # Block 2: 64 filters
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.5))

    # Block 3: 128 filters
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.5))

    # Dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation="softmax"))
    
    name1 = "CIFAR10_SMALL"

    return name1, model


def CIFAR10_BASE_2_EWE(input_shape = (32,32,3), dropout=0.0):

    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    conv1 = x
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    conv2 = x
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.5)(x)
    drop1 = x
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(10, activation="softmax")(x) #activation="softmax"

    # Create the functional model
    model = Model(inputs=input_layer, outputs=[conv1, conv2, drop1, output_layer])

    name1 = "CIFAR10_BASE_2_EWE"

    return name1, model
