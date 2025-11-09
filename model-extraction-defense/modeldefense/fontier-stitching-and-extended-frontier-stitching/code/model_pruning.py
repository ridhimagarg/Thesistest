# """
# @author: Ridhima Garg

# Introduction:
#     This file contains the implementation of model pruning using keras tmot library.

# """

# import tensorflow as tf
# from keras.datasets import mnist, cifar10
# import argparse
# import mlconfig
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import tensorflow_model_optimization as tfmot
# import tempfile
# import models
# import keras
# import random
# from datetime import datetime

# now = datetime.now().strftime("%d-%m-%Y")
# seed = 0
# random.seed(seed)
# np.random.seed(seed)
# tf.compat.v1.random.set_random_seed(seed)

# prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
# # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# def data_preprocessing(dataset_name, adv_data_path_numpy):
#     """
#           Main idea
#           -------
#           This is the function which preprocess the data for the modelling.

#           Args:
#           .---
#           dataset_name: name of the dataset.

#           Future work:
#           -----------
#           This function for now is copied in multiple files but can be modified such that it can be exported.

#         """

#     if dataset_name == 'mnist':
#         (x_train, y_train), (x_test, y_test) = mnist.load_data()
#         img_rows, img_cols, num_channels = 28, 28, 1
#         num_classes = 10

#     elif dataset_name == 'cifar10':
#         (x_train, y_train), (x_test, y_test) = cifar10.load_data()
#         img_rows, img_cols, num_channels = 32, 32, 3
#         num_classes = 10

#     elif dataset_name == "cifar10resnet":
#         (x_train, y_train), (x_test, y_test) = cifar10.load_data()
#         img_rows, img_cols, num_channels = 32, 32, 3
#         num_classes = 10

#     elif dataset_name == "cifar10resnet_255_preprocess":
#         (x_train, y_train), (x_test, y_test) = cifar10.load_data()
#         img_rows, img_cols, num_channels = 32, 32, 3
#         num_classes = 10

#     else:
#         raise ValueError('Invalid dataset name')

#     idx = np.random.randint(x_train.shape[0], size=len(x_train))
#     x_train = x_train[idx, :]
#     y_train = y_train[idx]

#     # specify input dimensions of each image
#     input_shape = (img_rows, img_cols, num_channels)

#     # reshape x_train and x_test
#     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, num_channels)
#     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, num_channels)

#     # convert class labels (from digits) to one-hot encoded vectors
#     y_train = keras.utils.to_categorical(y_train, num_classes)
#     y_test = keras.utils.to_categorical(y_test, num_classes)

#     # convert int to float
#     x_train = x_train.astype('float32')
#     x_test = x_test.astype('float32')

#     if dataset_name != 'cifar10resnet':
#         x_train /= 255
#         x_test /= 255

#     else:

#         mean = [125.3, 123.0, 113.9]
#         std = [63.0, 62.1, 66.7]

#         for i in range(3):
#             x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
#             x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]

#     # load adversarial data
#     adv = np.load(adv_data_path_numpy)
#     x_adv, y_adv = adv['arr_1'], adv['arr_2']
#     print(x_adv.shape)
#     print(y_adv.shape)


#     return x_train, y_train, x_test, y_test, x_adv, y_adv, input_shape



# def pruning(dataset_name, attacker_model_path, model_finetune_name, batch_size, epochs_pruning, optimizer, lr, weight_decay, dropout, adv_data_path_numpy, pruning_level):
#     """
#     Main idea
#     ------
#     Perform the pruning using the steal dataset. and check if the attacker performs fine-pruning what will be the watermark accuracy.

#     Args
#     ----
#     dataset_name: name of the dataset
#     attacker_model_path: attacker model path which is trained via model stealing attack (along with frontier stitching as a defense.)
#     model_finetune_name: which model architecture, attacker is using to finetune the model (so, here for the implementation part we have designed alreday the pruned model.)
#     batch_size: batch size for the dataset
#     epochs_pruning: number of epochs to prune and finetune the model
#     optimizer: optimizer to use for the model
#     lr: learning rate
#     weight_decay: weight_decay for the model
#     dropout: dropout
#     adv_data_path_numpy: watermark data samples
#     pruning_level: pruning level for the dataset

#     """

#     x_train, y_train, x_test, y_test, x_adv, y_adv, input_shape = data_preprocessing(dataset_name, adv_data_path_numpy)

#     x_train, y_train, x_test, y_test, x_adv, y_adv, input_shape = data_preprocessing(dataset_name, adv_data_path_numpy)

#     # models_mapping = {"mnist_l2_prune": models.MNIST_L2_Prune, "CIFAR10_BASE_2_Prune": models.CIFAR10_BASE_2_Prune}

#     models_mapping = {"mnist_l2_prune": models.MNIST_L2_Prune, "CIFAR10_BASE_2_Prune": models.CIFAR10_BASE_2_Prune,
#                       "cifar10_wideresnet_prune": models.wide_residual_network_prune}


#     attacker_model = tf.keras.models.load_model(attacker_model_path)
#     # print(attacker_model.summary())

#     file1 = open(os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER, adv_data_path_numpy.replace("\\", "/").split("/")[-2],
#                               "_".join((dataset_name, str(pruning_level) ,str(epochs_pruning),
#                                         attacker_model_path.replace("\\", "/").split("/")[-1] + "_logs.txt"))), "w")
#     image_save_path = os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER, adv_data_path_numpy.replace("\\", "/").split("/")[-2],
#                               "_".join((dataset_name, str(pruning_level), str(epochs_pruning),
#                                         attacker_model_path.replace("\\", "/").split("/")[-1] + "_pruning.png")))
#     file1.write("Before Pruning:\n")
        
#     accuracy = attacker_model.evaluate(x_adv, y_adv)[1]
#     print("Accuracy on watermark examples: {}%".format(accuracy * 100))
#     file1.write(f"Accuracy on watermark samples: {accuracy * 100}\n")

#     accuracy = attacker_model.evaluate(x_test, y_test)[1]
#     print("Accuracy on test set", accuracy)
#     file1.write(f"Accuracy on test set: {accuracy*100}\n")

#     pruning_params_sparsity_0_5 = {
#         'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=pruning_level,
#                                                                   begin_step=0,
#                                                                   frequency=100)
#     }

#     if dropout:
#         model_name, finetune_model = models_mapping[model_finetune_name](pruning_params_sparsity_0_5, input_shape, dropout)
#     else:
#         model_name, finetune_model = models_mapping[model_finetune_name](pruning_params_sparsity_0_5)

#     finetune_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

#     #finetune_model.set_weights(attacker_model.get_weights())

#     for layer_finetune, layer_original in zip(finetune_model.layers, attacker_model.layers):
#         # print(layer_finetune)
#         # print(layer_original)
#         weight_layer = layer_original.get_weights()
#         layer_finetune.set_weights(weight_layer)

#     len_steal = int(attacker_model_path.split('/')[-1].split('_')[1])
#     indices = np.random.permutation(len(x_test))
#     x_steal = x_test[indices[:len_steal]]
#     y_steal = y_test[indices[:len_steal]]

#     history = finetune_model.fit(x_steal, y_steal, batch_size=batch_size, shuffle=True, epochs=epochs_pruning,
#                         verbose=1, validation_split=0.2, callbacks=tfmot.sparsity.keras.UpdatePruningStep())  ## check for validation

#     print(f"Finetuned model test acc: {finetune_model.evaluate(x_test, y_test)}")
#     file1.write(f"Finetuned model test acc: {finetune_model.evaluate(x_test, y_test)}\n")
#     print(f"Finetuned model watermark acc: {finetune_model.evaluate(x_adv, y_adv)}")
#     file1.write(f"Finetuned model watermark acc: {finetune_model.evaluate(x_adv, y_adv)}\n")


#     ## ---------------------------------- Visualizing pruning ----------------------------##
#     finetune_model = tfmot.sparsity.keras.strip_pruning(finetune_model)

#     converter = tf.lite.TFLiteConverter.from_keras_model(finetune_model)
#     tflite_model = converter.convert()

#     _, tflite_file = tempfile.mkstemp('.tflite')
#     print('Saved converted pruned model to:', tflite_file)
#     with open(tflite_file, 'wb') as f:
#         f.write(tflite_model)

#     # Load tflite file with the created pruned model
#     interpreter = tf.lite.Interpreter(model_path=tflite_file)
#     interpreter.allocate_tensors()

#     details = interpreter.get_tensor_details()

#     # Weights of the dense layer that has been pruned.
#     # tensor_names = ['pruning_sparsity_0_5_1/Conv2D', 'pruning_sparsity_0_5_2/Conv2D', 'pruning_sparsity_0_5_3/Conv2D']
#     tensor_names = ["pruning_sparsity_0_5_1", "pruning_sparsity_0_5_2"]

#     tensor_names = ["pruning"]

#     detail = [x for x in details for t in tensor_names if t in x["name"]]

#     # We need the first layer.
#     tensor_data = interpreter.tensor(detail[0]["index"])()

#     print(f"Shape of Dense layer is {tensor_data.shape}")

#     #tensor_data = interpreter.tensor(detail[0]["index"])()

#     # The value 24 is chosen for convenience.
#     width = height = 24

#     # subset_values_to_display = tensor_data[0:height, 0:width]

#     # val_ones = np.ones([height, width])
#     # val_zeros = np.zeros([height, width])
#     # subset_values_to_display = np.where(abs(subset_values_to_display) > 0, val_ones, val_zeros)

#     def plot_separation_lines(height, width):

#         """
#         Helper function to display the plot.
#         """

#         block_size = [1, 4]

#         # Add separation lines to the figure.
#         num_hlines = int((height - 1) / block_size[0])
#         num_vlines = int((width - 1) / block_size[1])
#         line_y_pos = [y * block_size[0] for y in range(1, num_hlines + 1)]
#         line_x_pos = [x * block_size[1] for x in range(1, num_vlines + 1)]

#         for y_pos in line_y_pos:
#             plt.plot([-0.5, width], [y_pos - 0.5 , y_pos - 0.5], color='w')

#         for x_pos in line_x_pos:
#             plt.plot([x_pos - 0.5, x_pos - 0.5], [-0.5, height], color='w')


#     weights_to_display = tf.reshape(tensor_data, [tf.reduce_prod(tensor_data.shape[:-1]), -1])
#     weights_to_display = weights_to_display[0:width, 0:height]

#     if weights_to_display.shape[1] < width:
#         val_zeros = np.zeros([height, weights_to_display.shape[1]])

#     else:
#         val_zeros = np.zeros([height, width])

#     val_ones = np.ones([height, width])
    
#     subset_values_to_display = np.where(abs(weights_to_display) > 0, abs(weights_to_display), val_zeros)

#     plt.figure()
#     plot_separation_lines(height, width)

#     plt.axis('off')
#     plt.imshow(subset_values_to_display)
#     plt.colorbar()
#     plt.title("Structurally pruned weights for Conv2D layer")
#     plt.savefig(image_save_path)
#     # plt.show()





# if __name__ == "__main__":
     
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-c", "--config", type=str, default="configs/pruning.yaml")

#     RESULTS_PATH = f"../results/pruning_{now}"
#     LOSS_Acc_FOLDER = "losses_acc"

#     if not os.path.exists(os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER, "true")):
#         os.makedirs(os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER, "true"))


#     args = parser.parse_args()
#     print(args)
#     config = mlconfig.load(args.config)

#     if config.optimizer == "adam":
#         # optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr, weight_decay=config.weight_decay)
#         optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr, decay=config.weight_decay)
#     else:
#         # optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, weight_decay=config.weight_decay)
#         optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, decay=config.weight_decay)

#     for pruning_level in config.pruning_levels:

#         pruning(config.dataset, config.attacker_model_path, config.model_finetune_name, config.batch_size,  config.epochs_pruning, config.optimizer, config.lr, config.weight_decay, config.dropout, config.adv_key_set_path, pruning_level)


    


    


