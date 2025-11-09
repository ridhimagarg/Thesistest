import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, ReLU, BatchNormalization, GlobalAveragePooling2D, AveragePooling2D, Concatenate, ZeroPadding2D, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, add
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

# Optional import for model optimization (pruning)
try:
    import tensorflow_model_optimization as tfmot
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    TFMOT_AVAILABLE = True
except ImportError:
    TFMOT_AVAILABLE = False
    prune_low_magnitude = None

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
    model.add(Dense(10, activation='softmax'))
    name1 = "MNIST_l2" + str(dropout)


    return name1, model


def MNIST_L5(dropout=0.0):
    model = tf.keras.models.Sequential()

    # Convolutional Block
    model.add(Conv2D(32, kernel_size=(2, 2), input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    
    model.add(Conv2D(64, kernel_size=(2, 2)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    
    model.add(Conv2D(128, kernel_size=(2, 2)))
    model.add(Activation('relu'))
    
    
    # Flatten the output
    model.add(Flatten())
    
    # Fully Connected Layers
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    
    model.add(Dense(10))
    model.add(Activation('softmax'))
    name1 = "MNIST_l5" + str(dropout)

    return name1, model

 # model = Sequential()
    # model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape)) # input shape = (img_rows, img_cols, 1)
    # model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu')) # fully connected
    # model.add(Dropout(0.5))
    # model.add(Dense(num_classes, activation='softmax'))

def CIFAR10_BASE_2(input_shape = (32,32,3), dropout=0.0, num_classes=10):

    model = tf.keras.models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
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


def CIFAR10_BASE_3(input_shape = (32,32,3), dropout=0.0):

    model = tf.keras.models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
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
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation="softmax"))
    name1 = "CIFAR10_BASE_3"

    return name1, model


pruning_params_2_by_4 = {
    'sparsity_m_by_n': (2, 4),
}

if TFMOT_AVAILABLE:
    pruning_params_sparsity_0_5 = {
        'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=0.9,
                                                                  begin_step=0,
                                                                  frequency=100)
    }
else:
    pruning_params_sparsity_0_5 = {}
def CIFAR10_BASE_2_Prune(pruning_params_sparsity_0_5, input_shape = (32,32,3), dropout=0.0, ):

    model = tf.keras.models.Sequential()

    model.add(prune_low_magnitude(layers.Conv2D(32, (3, 3), name="pruning_sparsity_0_5_0", padding='same', activation='relu', input_shape=(32, 32, 3)), **pruning_params_sparsity_0_5 ))
    model.add(layers.BatchNormalization())
    model.add(prune_low_magnitude(layers.Conv2D(32, (3, 3), padding='same', activation='relu', name="pruning_sparsity_0_5_1"), **pruning_params_sparsity_0_5))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.3))
    model.add(prune_low_magnitude(layers.Conv2D(64, (3, 3), padding='same', activation='relu', name="pruning_sparsity_0_5_2" ), **pruning_params_sparsity_0_5))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.5))
    model.add(prune_low_magnitude(layers.Conv2D(128, (3, 3), padding='same', activation='relu', name="pruning_sparsity_0_5_3"), **pruning_params_sparsity_0_5))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Flatten())
    model.add(prune_low_magnitude(layers.Dense(128, activation="relu", name="pruning_sparsity_0_5_4"), **pruning_params_sparsity_0_5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation="softmax"))
    name1 = "CIFAR10_BASE_2_Prune" + str(dropout)

    return name1, model


def CIFAR10_BASE_3(input_shape = (32,32,3), dropout=0.0):

    model = tf.keras.models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
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
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation="softmax"))
    name1 = "CIFAR10_BASE_3"

    return name1, model


def CIFAR10_VGG(input_shape = (32,32,3), dropout=0.0):

    model = tf.keras.models.Sequential()

    model.add(Conv2D(64,(3,3),activation='relu',input_shape=input_shape,padding='same'))
    model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
    model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(256,(3,3),activation='relu',padding='same'))
    model.add(Conv2D(256,(3,3),activation='relu',padding='same'))
    model.add(Conv2D(256,(3,3),activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(512,(3,3),activation='relu',padding='same'))
    model.add(Conv2D(512,(3,3),activation='relu',padding='same'))
    model.add(Conv2D(512,(3,3),activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(512,(3,3),activation='relu',padding='same'))
    model.add(Conv2D(512,(3,3),activation='relu',padding='same'))
    model.add(Conv2D(512,(3,3),activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(4096,activation='relu'))
    model.add(Dense(4096,activation='relu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    name1 = "CIFAR10_VGG"

    return name1, model


# pruning_params_2_by_4 = {
#     'sparsity_m_by_n': (2, 4),
# }
#
# pruning_params_sparsity_0_5 = {
#     'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=0.025,
#                                                               begin_step=0,
#                                                               frequency=100)
# }
def MNIST_L2_Prune(pruning_params_sparsity_0_5, input_shape = (28,28,1), dropout=0.0):
    model = tf.keras.models.Sequential()
    model.add(prune_low_magnitude(Conv2D(32, 5, name="pruning_sparsity_0_5_0", input_shape=input_shape), **pruning_params_sparsity_0_5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(ReLU(True))
    model.add(prune_low_magnitude(Conv2D(64, 5, name="pruning_sparsity_0_5_1"),  **pruning_params_sparsity_0_5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(ReLU(True))
    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(prune_low_magnitude(Dense(10, name="pruning_sparsity_0_5_2", activation='softmax'), **pruning_params_sparsity_0_5))
    name1 = "MNIST_l2_Prune" + str(dropout)

    return name1, model


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

# keras.saving.get_custom_objects().clear()
# @keras.saving.register_keras_serializable(package="MyLayers")
class ResBlock(tf.keras.layers.Layer):
    def __init__(self, channels, stride=1):
        super(ResBlock, self).__init__()
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
        self.fc3 = Dense(10, activation='softmax')

        # self.fc1 = Dense(512, activation='relu')
        # self.fc3 = Dense(10)

        self.name1 = "ResNet34"
        self.augmentation = setup_transformation("cifar10")

    def call(self, input_shape):

        inputs = Input(input_shape)

        x = self.augmentation(inputs)
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

        model = Model(inputs, x)
        return self.name1, model



DEPTH              = 28
WIDE               = 10
IN_FILTERS         = 16

CLASS_NUM          = 10
IMG_ROWS, IMG_COLS = 32, 32
IMG_CHANNELS       = 3

BATCH_SIZE         = 128
EPOCHS             = 250
ITERATIONS         = 50000 // BATCH_SIZE + 1
WEIGHT_DECAY       = 0.0005
LOG_FILE_PATH      = './w_resnet/'
i = 0

def wide_residual_network(input_shape=(32,32,3), classes_num=CLASS_NUM, depth=DEPTH, k=WIDE):
    print('Wide-Resnet %dx%d' %(depth, k))
    n_filters  = [16, 16*k, 32*k, 64*k]
    n_stack    = (depth - 4) // 6

    def conv3x3(x,filters):
        return Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1), padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=l2(WEIGHT_DECAY),
        use_bias=False)(x)

    def bn_relu(x):
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = Activation('relu')(x)
        return x

    def residual_block(x,out_filters,increase=False):
        global IN_FILTERS
        stride = (1,1)
        if increase:
            stride = (2,2)
            
        o1 = bn_relu(x)
        
        conv_1 = Conv2D(out_filters,
            kernel_size=(3,3),strides=stride,padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=l2(WEIGHT_DECAY),
            use_bias=False)(o1)

        o2 = bn_relu(conv_1)
        
        conv_2 = Conv2D(out_filters, 
            kernel_size=(3,3), strides=(1,1), padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=l2(WEIGHT_DECAY),
            use_bias=False)(o2)
        if increase or IN_FILTERS != out_filters:
            proj = Conv2D(out_filters,
                                kernel_size=(1,1),strides=stride,padding='same',
                                kernel_initializer='he_normal',
                                kernel_regularizer=l2(WEIGHT_DECAY),
                                use_bias=False)(o1)
            block = add([conv_2, proj])
        else:
            block = add([conv_2,x])
        return block

    def wide_residual_layer(x,out_filters,increase=False):
        global IN_FILTERS
        x = residual_block(x,out_filters,increase)
        IN_FILTERS = out_filters
        for _ in range(1,int(n_stack)):
            x = residual_block(x,out_filters)
        return x

    img_input = Input(shape=input_shape)

    x = conv3x3(img_input,n_filters[0])
    x = wide_residual_layer(x,n_filters[1])
    x = wide_residual_layer(x,n_filters[2],increase=True)
    x = wide_residual_layer(x,n_filters[3],increase=True)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = AveragePooling2D((8,8))(x)
    x = Flatten()(x)
    x = Dense(classes_num,
        activation='softmax',
        kernel_initializer='he_normal',
        kernel_regularizer=l2(WEIGHT_DECAY),
        use_bias=False)(x)
    
    model = Model(img_input, x)
    
    return "CIFAR10_WIDERESNET", model


def wide_residual_network_prune(pruning_params_sparsity_0_5, input_shape=(32,32,3), classes_num=CLASS_NUM, depth=DEPTH, k=WIDE):
    print('Wide-Resnet %dx%d' %(depth, k))
    n_filters  = [16, 16*k, 32*k, 64*k]
    n_stack    = (depth - 4) // 6
    

    def conv3x3(x,filters):
        # prune_low_magnitude(layers.Conv2D(128, (3, 3), padding='same', activation='relu', name="pruning_sparsity_0_5_3"), **pruning_params_sparsity_0_5)
        return prune_low_magnitude(Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1), padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=l2(WEIGHT_DECAY),
        use_bias=False, name="pruning_sparsity_0_5_0"),**pruning_params_sparsity_0_5)(x)

    def bn_relu(x):
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = Activation('relu')(x)
        return x

    def residual_block(x,out_filters,increase=False):
        global IN_FILTERS
        global i
        stride = (1,1)
        if increase:
            stride = (2,2)
            
        o1 = bn_relu(x)
        
        conv_1 = prune_low_magnitude(Conv2D(out_filters,
            kernel_size=(3,3),strides=stride,padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=l2(WEIGHT_DECAY),
            use_bias=False, name="pruning_sparsity_0_5_1_" + str(i)), **pruning_params_sparsity_0_5)(o1)
        
        

        o2 = bn_relu(conv_1)
        
        conv_2 = prune_low_magnitude(Conv2D(out_filters, 
            kernel_size=(3,3), strides=(1,1), padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=l2(WEIGHT_DECAY),
            use_bias=False, name="pruning_sparsity_0_5_2_"+str(i)), **pruning_params_sparsity_0_5)(o2)
        
        i = i+1
        
        i = i+1

        o2 = bn_relu(conv_1)
        
        conv_2 = Conv2D(out_filters, 
            kernel_size=(3,3), strides=(1,1), padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=l2(WEIGHT_DECAY),
            use_bias=False)(o2)
        if increase or IN_FILTERS != out_filters:
            proj = Conv2D(out_filters,
                                kernel_size=(1,1),strides=stride,padding='same',
                                kernel_initializer='he_normal',
                                kernel_regularizer=l2(WEIGHT_DECAY),
                                use_bias=False)(o1)
            block = add([conv_2, proj])
        else:
            block = add([conv_2,x])
        return block

    def wide_residual_layer(x,out_filters,increase=False):
        global IN_FILTERS
        x = residual_block(x,out_filters,increase)
        IN_FILTERS = out_filters
        for _ in range(1,int(n_stack)):
            x = residual_block(x,out_filters)
        return x

    img_input = Input(shape=input_shape)

    x = conv3x3(img_input,n_filters[0])
    x = wide_residual_layer(x,n_filters[1])
    x = wide_residual_layer(x,n_filters[2],increase=True)
    x = wide_residual_layer(x,n_filters[3],increase=True)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = AveragePooling2D((8,8))(x)
    x = Flatten()(x)
    x = Dense(classes_num,
        activation='softmax',
        kernel_initializer='he_normal',
        kernel_regularizer=l2(WEIGHT_DECAY),
        use_bias=False)(x)
    
    model = Model(img_input, x)
    
    return "CIFAR10_WIDERESNET", model