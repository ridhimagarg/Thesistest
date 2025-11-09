from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import tensorflow as tf
import torch as t

RESULTS_FOLDER = "results"
RESULTS_SUB_ACTIVATION_FOLDER = "activation"
RESULTS_SUB_PCA_FOLDER = "pca"


def pca_dim(fit_data):
    """
    PCA fitting of the data
    """

    pca = PCA(n_components=2)
    pca.fit(fit_data)

    return pca


def validate_watermark(model, trainer_class, trigger_set, target_class, batch_size, num_class):
    """
    Validating the watermark set
    """

    labels = t.zeros([batch_size, num_class])
    labels[:, target_class] = 1
    if trigger_set.shape[0] < batch_size:
        trigger_data = t.concat([trigger_set, trigger_set], 0)[:batch_size]
    else:
        trigger_data = trigger_set
    # error = sess.run(model_name.error, {x: trigger_data, y: labels, is_training: 0, is_augment: 0})
    error = trainer_class.error_rate(model, trigger_data, labels)
    return 1 - error


def pca_and_plot(x, list_data, type_model, dataset, time, penultimate_layer, distrib):
    """
    Plotting the PCA results for train, watermark and target set
    """

    pca = pca_dim(x)
    transformed_data_train = pca.transform(list_data[0])
    transformed_data_target = pca.transform(list_data[1])
    transformed_data_watermark = pca.transform(list_data[2])

    # Plotting PCA output in the middle of the training.
    # color_list = np.concatenate([y , np.repeat(watermark_target, batch_size), np.repeat(10, batch_size)], 0)
    # marker_list = ["."]*y.shape[0] + [","]*batch_size + ["<"]*batch_size
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "darkorange", "darkgreen", "darkslategray",
                                                                    "deepskyblue", "purple", "cyan", "blue", "crimson",
                                                                    "bisque", "lawngreen"])
    fig = plt.figure()
    plot = plt.scatter(transformed_data_train[:, 0], transformed_data_train[:, 1], c="deepskyblue", marker=".",
                       label="train data")
    plot = plt.scatter(transformed_data_target[:, 0], transformed_data_target[:, 1], c="black", marker=",",
                       label="target class")
    plot = plt.scatter(transformed_data_watermark[:, 0], transformed_data_watermark[:, 1], c="darkorange", marker="<",
                       label="watermark")
    plt.legend()

    # plot = plt.scatter(transformed_data[:,0], transformed_data[:,1], c= color_list, cmap=cmap, marker=marker_list)
    # plt.legend(handles=plot.legend_elements()[0], labels=list(np.unique(color_list)))
    plt.show()

    if not os.path.exists(RESULTS_FOLDER):
        os.mkdir(RESULTS_FOLDER)

    if not os.path.exists(os.path.join(RESULTS_FOLDER, RESULTS_SUB_PCA_FOLDER)):
        os.mkdir(os.path.join(RESULTS_FOLDER, RESULTS_SUB_PCA_FOLDER))

    plt.savefig(os.path.join(RESULTS_FOLDER, RESULTS_SUB_PCA_FOLDER,
                             type_model + "_" + dataset + "_" + time + "_" + str(penultimate_layer) + "_" + str(
                                 distrib)), dpi=fig.dpi)


def plot_activation(first_conv_legitimate, first_conv_watermark, second_conv_legitimate, second_conv_watermark,
                    fc_legitimate, fc_watermark, type_model, dataset, distrib):
    """
    Plotting the activation of the neurons from the network.
    """

    if not os.path.exists(RESULTS_FOLDER):
        os.mkdir(RESULTS_FOLDER)

    if not os.path.exists(os.path.join(RESULTS_FOLDER, RESULTS_SUB_ACTIVATION_FOLDER)):
        os.mkdir(os.path.join(RESULTS_FOLDER, RESULTS_SUB_ACTIVATION_FOLDER))

    fig = plt.figure(figsize=(7, 5))
    for i in range(10):
        fig.add_subplot(2, 10, i + 1)
        plt.imshow(first_conv_legitimate[0, :, :, i], cmap="gray_r")
    plt.savefig(os.path.join(RESULTS_FOLDER, RESULTS_SUB_ACTIVATION_FOLDER,
                             type_model + "_" + dataset + "_" + distrib + "_first_conv_legit_activation.png"))

    for i in range(10):
        fig.add_subplot(2, 10, i + 1)
        plt.imshow(first_conv_watermark[0, :, :, i], cmap="gray_r")
    plt.savefig(os.path.join(RESULTS_FOLDER, RESULTS_SUB_ACTIVATION_FOLDER,
                             type_model + "_" + dataset + "_" + distrib + "_first_conv_watermark_activation.png"))

    fig = plt.figure(figsize=(7, 5))
    for i in range(10):
        fig.add_subplot(2, 10, i + 1)
        plt.imshow(second_conv_legitimate[0, :, :, i], cmap="gray_r")
    plt.savefig(os.path.join(RESULTS_FOLDER, RESULTS_SUB_ACTIVATION_FOLDER,
                             type_model + "_" + dataset + "_" + distrib + "_second_conv_legit_activation.png"))

    fig = plt.figure(figsize=(7, 5))
    for i in range(10):
        fig.add_subplot(2, 10, i + 1)
        plt.imshow(second_conv_watermark[0, :, :, i], cmap="gray_r")
    plt.savefig(os.path.join(RESULTS_FOLDER, RESULTS_SUB_ACTIVATION_FOLDER,
                             type_model + "_" + dataset + "_" + distrib + "_second_conv_watermark_activation.png"))

    fig = plt.figure(figsize=(7, 5))
    fig.add_subplot(2, 1, 1)
    # print(f"Checking intensity values for the fc layer {np.atleast_2d([fc_legitimate[0,:]])}")
    plt.imshow(np.atleast_2d([fc_legitimate[0, :]]), cmap="gray_r")
    plt.title("legitimate data")

    fig.add_subplot(2, 1, 2)
    # print(f"Checking intensity values for the fc layer {np.atleast_2d([fc_watermark[0,:]])}")
    plt.imshow(np.atleast_2d(fc_watermark[0, :]), cmap="gray_r")
    plt.title("watermark data")
    plt.savefig(os.path.join(RESULTS_FOLDER, RESULTS_SUB_ACTIVATION_FOLDER,
                             type_model + "_" + dataset + "_" + distrib + "_fc_activation.png"))


def augment_train(image):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_with_crop_or_pad(image, 40, 40)
    image = tf.image.random_crop(image, size=[image.shape[0], 32, 32, 3])
    image = tf.image.random_flip_left_right(image)
    random_angles = tf.random.uniform(shape=(128,), minval=-np.pi / 8, maxval=np.pi / 8)
    image = tf.contrib.image.transform(image, tf.contrib.image.angles_to_projective_transforms(
        random_angles, 32, 32))
    image = tf.image.per_image_standardization(image)
    return image


def augment_test(image):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    return image