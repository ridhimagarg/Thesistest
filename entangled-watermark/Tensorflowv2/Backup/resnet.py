import math
from tensorflow import keras
from tensorflow.keras import layers

kaiming_normal = keras.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='untruncated_normal')

def conv3x3(x, out_planes, stride=1, name=None):
    x = layers.ZeroPadding2D(padding=1, name=f'{name}_pad')(x)
    return layers.Conv2D(filters=out_planes, kernel_size=3, strides=stride, use_bias=False, kernel_initializer=kaiming_normal, name=name)(x)

def basic_block(x, planes, stride=1, downsample=None, name=None):
    identity = x

    out = conv3x3(x, planes, stride=stride, name=f'{name}.conv1')
    out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn1')(out)
    out = layers.ReLU(name=f'{name}.relu1')(out)

    out = conv3x3(out, planes, name=f'{name}.conv2')
    out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn2')(out)

    if downsample is not None:
        for layer in downsample:
            identity = layer(identity)

    out = layers.Add(name=f'{name}.add')([identity, out])
    out = layers.ReLU(name=f'{name}.relu2')(out)

    return out

def make_layer(x, planes, blocks, stride=1, name=None):
    downsample = None
    inplanes = x.shape[3]
    if stride != 1 or inplanes != planes:
        downsample = [
            layers.Conv2D(filters=planes, kernel_size=1, strides=stride, use_bias=False, kernel_initializer=kaiming_normal, name=f'{name}.0.downsample.0'),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.0.downsample.1'),
        ]

    x = basic_block(x, planes, stride, downsample, name=f'{name}.0')
    for i in range(1, blocks):
        x = basic_block(x, planes, name=f'{name}.{i}')

    return x

def resnet(x, blocks_per_layer, num_classes=1000):
    x = layers.ZeroPadding2D(padding=3, name='conv1_pad')(x)
    x = layers.Conv2D(filters=64, kernel_size=7, strides=2, use_bias=False, kernel_initializer=kaiming_normal, name='conv1')(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='bn1')(x)
    x = layers.ReLU(name='relu1')(x)
    x = layers.ZeroPadding2D(padding=1, name='maxpool_pad')(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, name='maxpool')(x)

    x = make_layer(x, 64, blocks_per_layer[0], name='layer1')
    x = make_layer(x, 128, blocks_per_layer[1], stride=2, name='layer2')
    x = make_layer(x, 256, blocks_per_layer[2], stride=2, name='layer3')
    x = make_layer(x, 512, blocks_per_layer[3], stride=2, name='layer4')

    x = layers.GlobalAveragePooling2D(name='avgpool')(x)
    initializer = keras.initializers.RandomUniform(-1.0 / math.sqrt(512), 1.0 / math.sqrt(512))
    x = layers.Dense(units=num_classes, kernel_initializer=initializer, bias_initializer=initializer, name='fc')(x)

    return x

def resnet18(x, **kwargs):
    return resnet(x, [2, 2, 2, 2], **kwargs)

def resnet34(x, **kwargs):
    return resnet(x, [3, 4, 6, 3], **kwargs)



# import tensorflow as tf



# weight_init = tf.keras.initializers.VarianceScaling()
# weight_regularizer = tf.keras.regularizers.L2(l2=0.0001)


# def fully_connected(x, units, use_bias=True, scope='fully_0'):

#     x = tf.keras.layers.Flatten()(x)
#     x = tf.keras.layers.Dense(units=units, use_bias=use_bias, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer)(x)

#     return x


# def resblock(x_init, channels, is_training=True, use_bias=True, downsample=False, scope='resblock'):

#     x = tf.keras.layers.BatchNormalization()(x_init, training=is_training)
#     x = tf.keras.layers.Activation('relu')(x)

#     if downsample:
#         x = tf.keras.layers.Conv2D(channels, kernel_size=3, strides=2, use_bias=use_bias, padding='same')(x)
#         x_init = tf.keras.layers.Conv2D(channels, kernel_size=1, strides=2, use_bias=use_bias, padding='same')(x_init)
#     else:
#         x = tf.keras.layers.Conv2D(channels, kernel_size=3, strides=1, use_bias=use_bias, padding='same')(x)

#     x = tf.keras.layers.BatchNormalization()(x, training=is_training)
#     x = tf.keras.layers.Activation('relu')(x)
#     x = tf.keras.layers.Conv2D(channels, kernel_size=3, strides=1, use_bias=use_bias, padding='same')(x)

#     return tf.keras.layers.add([x, x_init])

        

# def non_resblock(x_init, channels, is_training=True, use_bias=True, downsample=False, scope='non_resblock'):
#     x = batch_norm(x_init, training=is_training)
#     x = tf.keras.layers.Activation('relu')(x)

#     if downsample:
#         x = tf.keras.layers.Conv2D(channels, kernel_size=3, strides=2, use_bias=use_bias, padding='same')(x)
#     else:
#         x = tf.keras.layers.Conv2D(channels, kernel_size=3, strides=1, use_bias=use_bias, padding='same')(x)

#     return x



# def bottle_resblock(x_init, channels, is_training=True, use_bias=True, downsample=False, scope='bottle_resblock'):
#     x = batch_norm(x_init, training=is_training)
#     shortcut = tf.keras.layers.Activation('relu')(x)

#     x = tf.keras.layers.Conv2D(channels, kernel_size=1, strides=1, use_bias=use_bias, padding='same')(shortcut)
#     x = batch_norm(x, training=is_training)
#     x = tf.keras.layers.Activation('relu')(x)

#     if downsample:
#         x = tf.keras.layers.Conv2D(channels, kernel_size=3, strides=2, use_bias=use_bias, padding='same')(x)
#         shortcut = tf.keras.layers.Conv2D(channels*4, kernel_size=1, strides=2, use_bias=use_bias, padding='same')(shortcut)
#     else:
#         x = tf.keras.layers.Conv2D(channels, kernel_size=3, strides=1, use_bias=use_bias, padding='same', name=scope+'/conv_0')(x)
#         shortcut = tf.keras.layers.Conv2D(channels*4, kernel_size=1, strides=1, use_bias=use_bias, padding='same')(shortcut)

#     x = batch_norm(x, training=is_training)
#     x = tf.keras.layers.Activation('relu')(x)
#     x = tf.keras.layers.Conv2D(channels*4, kernel_size=1, strides=1, use_bias=use_bias, padding='same')(x)

#     return tf.keras.layers.add([x, shortcut])


# def get_residual_layer(res_n) :
#     x = []

#     if res_n == 18 :
#         x = [2, 2, 2, 2]

#     if res_n == 34 :
#         x = [3, 4, 6, 3]

#     if res_n == 50 :
#         x = [3, 4, 6, 3]

#     if res_n == 101 :
#         x = [3, 4, 23, 3]

#     if res_n == 152 :
#         x = [3, 8, 36, 3]

#     return x



# def flatten(x):
#     return tf.keras.layers.Flatten()(x)

# def global_avg_pooling(x):
#     gap = tf.reduce_mean(input_tensor=x, axis=[1, 2], keepdims=True)
#     return gap

# def avg_pooling(x):
#     return tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, padding='SAME')(x)

# def classification_loss(logit, label) :
#     loss = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logit))
#     prediction = tf.equal(tf.argmax(input=logit, axis=-1), tf.argmax(input=label, axis=-1))
#     accuracy = tf.reduce_mean(input_tensor=tf.cast(prediction, tf.float32))

#     return loss, accuracy


# def relu(x):
#     return tf.keras.activations.relu(x)


# def batch_norm(x, is_training=True, scope='batch_norm'):
#     return tf.keras.layers.BatchNormalization(
#         momentum=0.9, epsilon=1e-05, center=True, scale=True,
#     )(x)


