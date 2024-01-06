import numpy as np
import keras
import tensorflow as tf
import os
from keras.models import Sequential

# original_model = tf.keras.models.load_model('../models/mnist_cnn_finetuned_30_fgsm_0.5_500_original_cnn_25.h5')

# original_model = tf.keras.models.load_model('../models/cifar10_original_cnn_epochs_30.h5')

original_model = tf.keras.models.load_model('../../models/original/mnist_20_MNIST_l20.0/Original_checkpoint_best.h5')

for item in os.listdir("../../data/fgsm/mnist"):
    adv_array = np.load("../data/fgsm/mnist/" + item)

    x_adv = adv_array["arr_0"]
    y_adv = adv_array["arr_1"]
    print("../data/fgsm/mnist/" + item)
    print(original_model.evaluate(x_adv, y_adv, verbose=0))