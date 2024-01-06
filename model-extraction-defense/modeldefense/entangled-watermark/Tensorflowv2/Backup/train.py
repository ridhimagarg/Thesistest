import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import argparse
import os
import pickle
import functools
import random

import models as md
import matplotlib.pyplot as plt

from utils import pca_dim

seed = 0
random.seed(seed)
np.random.seed(seed)
tf.compat.v1.random.set_random_seed(seed)

tf.compat.v1.disable_eager_execution()


def augment_train(image):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_with_crop_or_pad(image, 40, 40)
    image = tf.image.random_crop(image, size=[image.shape[0], 32, 32, 3])
    image = tf.image.random_flip_left_right(image)
    print("Augmentation", image.shape[0])
    random_angles = tf.random.uniform(shape=(tf.shape(input=image)[0],), minval=-np.pi / 8, maxval=np.pi / 8)
    image = tfa.image.transform(image, tfa.image.angles_to_projective_transforms(
            random_angles, 32, 32))
    image = tf.image.per_image_standardization(image)
    return image


def augment_test(image):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    return image


def train(x_train, y_train, x_test, y_test, ewe_model, plain_model, epochs, w_epochs, lr, n_w_ratio, factors,
          temperatures, watermark_source, watermark_target, batch_size, w_lr, threshold, maxiter, shuffle, temp_lr,
          dataset, distribution, verbose):
    tf.compat.v1.random.set_random_seed(seed)

    height = x_train[0].shape[0]
    width = x_train[0].shape[1]
    try:
        channels = x_train[0].shape[2]
    except:
        channels = 1
    num_class = len(np.unique(y_train))
    half_batch_size = int(batch_size / 2)

    target_data = x_train[y_train == watermark_target]
    exclude_x_data = x_train[y_train != watermark_target]
    exclude_y_data = y_train[y_train != watermark_target]

    # define the dataset and class to sample watermarked data
    if distribution == "in":
        source_data = x_train[y_train == watermark_source]
        exclude_x_data = x_train[(y_train != watermark_source) & (y_train != watermark_target) ]
        exclude_y_data = y_train[(y_train != watermark_source) & (y_train != watermark_target)]
    elif distribution == "out":
        if dataset == "mnist":
            w_dataset = "fashion"
            with open(os.path.join("data", f"{w_dataset}.pkl"), 'rb') as f:
                w_data = pickle.load(f)
            x_w, y_w = w_data["training_images"], w_data["training_labels"]
        elif dataset == "fashion":
            w_dataset = "mnist"
            with open(os.path.join("data", f"{w_dataset}.pkl"), 'rb') as f:
                w_data = pickle.load(f)
            x_w, y_w = w_data["training_images"], w_data["training_labels"]
        elif "cifar" in dataset:
            import scipy.io as sio
            w_dataset = sio.loadmat(os.path.join("data", "train_32x32"))
            x_w, y_w = np.moveaxis(w_dataset['X'], -1, 0), np.squeeze(w_dataset['y'] - 1)
        elif dataset == "speechcmd":
            x_w = np.swapaxes(np.load(os.path.join(r"data", "sd_GSCmdV2", 'trigger.npy')), 1, 2)
            y_w = np.ones(x_w.shape[0]) * watermark_source
        else:
            raise NotImplementedError()

        x_w = np.reshape(x_w / 255, [-1, height, width, channels])
        source_data = x_w[y_w == watermark_source]
    else:
        raise NotImplementedError("Distribution could only be either \'in\' or \'out\'.")
    
    

    # make sure watermarked data is the same size as target data
    trigger = np.concatenate([source_data] * (target_data.shape[0] // source_data.shape[0] + 1), 0)[
                  :target_data.shape[0]] ## repeating source data multiple times. ## trigger is created from source data 
    
    ##checking how trigger is create using source data
    print("how many time source data is repeated for triggers", (target_data.shape[0] // source_data.shape[0] + 1))
    print(source_data.shape)
    print(trigger.shape)
    print("xtrain", x_train.shape)
    print("target data", target_data.shape)
    print("excluding watermark", exclude_x_data.shape)
    print("trigger data", trigger.shape)

    w_label = np.concatenate([np.ones(half_batch_size), np.zeros(half_batch_size)], 0)
    y_train = tf.keras.utils.to_categorical(y_train, num_class)
    y_test = tf.keras.utils.to_categorical(y_test, num_class)
    index = np.arange(y_train.shape[0])
    w_0 = np.zeros([batch_size])
    trigger_label = np.zeros([batch_size, num_class])
    trigger_label[:, watermark_target] = 1 ## setting the trigger label 

    num_batch = x_train.shape[0] // batch_size ## whole data no. of batches 
    w_num_batch = target_data.shape[0] // batch_size * 2 ## watermark no. of batches, since trigger is same shape as target data
    num_test = x_test.shape[0] // batch_size

    def validate_watermark(model_name, trigger_set, label):
        labels = np.zeros([batch_size, num_class])
        labels[:, label] = 1
        fig = plt.figure()
        plt.axis("off")
        plt.imshow(trigger_set[1][:,:,0], cmap="gray_r")
        plt.title("Original")
        plt.savefig("trigger1.png")
        if trigger_set.shape[0] < batch_size:
            trigger_data = np.concatenate([trigger_set, trigger_set], 0)[:batch_size]
        else:
            trigger_data = trigger_set
        target, pred, error = sess.run(model_name.error, {x: trigger_data, y: labels, is_training: 0, is_augment: 0})
        # print("target", target)
        # print("pred", pred)
        return 1 - error

    tf.compat.v1.get_default_graph().finalize()
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.random.set_random_seed(seed)
    x = tf.compat.v1.placeholder(tf.float32, [batch_size, height, width, channels], name="input")
    y = tf.compat.v1.placeholder(tf.float32, [batch_size, num_class])
    w = tf.compat.v1.placeholder(tf.float32, [batch_size])
    t = tf.compat.v1.placeholder(tf.float32, [len(temperatures)])
    is_training = tf.compat.v1.placeholder(tf.float32)
    is_augment = tf.compat.v1.placeholder(tf.float32)

    if "cifar" in dataset:
        augmented_x = tf.cond(pred=tf.greater(is_augment, 0),
                              true_fn=lambda: augment_train(x),
                              false_fn=lambda: augment_test(x))
        model = ewe_model(augmented_x, y, w, batch_size, num_class, lr, factors, t, watermark_target, is_training)
    else:
        model = ewe_model(x, y, w, batch_size, num_class, lr, factors, t, watermark_target, is_training)

    saver = tf.compat.v1.train.Saver()
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    ## PCA Analysis for before the training.
    x_pca = exclude_x_data[0: batch_size]
    y_pca = exclude_y_data[0: batch_size]

    x_pca_full = np.concatenate([x_pca, target_data[0: batch_size], source_data[0: batch_size]], 0)
    
    print(x_pca.shape)
    print(y_pca.shape)
    print(x_pca_full.shape)

    new_x = x_pca_full.reshape(x_pca_full.shape[0], -1)

    transformed_data = pca_dim(new_x)

    print("Transformed data", transformed_data)

    ## Plotting PCA output.
    # color_list = np.concatenate([y_pca , np.repeat(watermark_target, batch_size), np.repeat(10, batch_size)], 0)
    # # labels_ = [*np.unique(y_pca), str(watermark_target)+ "(target)", "watermark"]
    # # marker_list = ["d"]*y_pca.shape[0] + ["d"]*batch_size + [None]*batch_size
    # if len(color_list) == 10:
    #     colors = ["black","darkorange","darkgreen", "darkslategray", "deepskyblue", "purple", "cyan", "blue", "crimson", "bisque", "lawngreen"]
    # else:
    #     colors = ["black","darkorange","darkgreen", "darkslategray", "deepskyblue", "purple", "cyan", "blue", "crimson", "bisque"]
    # cmap = matplotlib.colors.LinearSegmentedColormap.from_list("",  colors)
    # fig = plt.figure(figsize=(10,10))
    # plot = plt.scatter(transformed_data[:,0], transformed_data[:,1], c=color_list, cmap=cmap)
    # print(plot.legend_elements()[0])
    # plt.legend(handles=plot.legend_elements()[0], labels=list(np.unique(color_list)))
    # plt.show()
    # plt.savefig("before training", dpi=fig.dpi)
    

    for epoch in range(epochs):
        if shuffle:
            np.random.shuffle(index)
            x_train = x_train[index]
            y_train = y_train[index]
        for batch in range(num_batch):
            sess.run(model.optimize, {x: x_train[batch * batch_size: (batch + 1) * batch_size],
                                      y: y_train[batch * batch_size: (batch + 1) * batch_size],
                                      t: temperatures,
                                      w: w_0, is_training: 1, is_augment: 1})

    if distribution == "in":
        trigger_grad = []
        for batch in range(w_num_batch):
            batch_data = np.concatenate([trigger[batch * half_batch_size: (batch + 1) * half_batch_size],
                                         target_data[batch * half_batch_size: (batch + 1) * half_batch_size]], 0)
            grad = sess.run(model.snnl_trigger, {x: batch_data, w: w_label, t: temperatures, is_training: 0,
                                                 is_augment: 0})[0][:half_batch_size]
            trigger_grad.append(grad)
        avg_grad = np.average(np.concatenate(trigger_grad), 0)
        down_sample = np.array([[np.sum(avg_grad[i: i + 3, j: j + 3]) for i in range(height - 2)] for j in range(width - 2)])
        w_pos = np.unravel_index(down_sample.argmin(), down_sample.shape)
        trigger[:, w_pos[0]:w_pos[0] + 3, w_pos[1]:w_pos[1] + 3, 0] = 1
        plt.imshow(trigger[0][:,:,0], cmap="gray_r")
        plt.show()
        plt.savefig("trigger.png")
    else:
        w_pos = [-1, -1]

    step_list = np.zeros([w_num_batch])
    snnl_change = []
    
    print("Full trigger shape", trigger.shape[0])
    print("W num batch", w_num_batch)
    for batch in range(w_num_batch):
        current_trigger = trigger[batch * half_batch_size: (batch + 1) * half_batch_size]

        print("curremt trigger shape ------------>", current_trigger.shape)

        ## Plotting the triggers with the original data
        fig = plt.figure()
        plt.subplot(2,1, 1)
        plt.axis("off")
        plt.imshow(current_trigger[1][:,:,0], cmap="gray_r")
        plt.title("Original")

        for epoch in range(maxiter):
            while validate_watermark(model, current_trigger, watermark_target) > threshold and step_list[batch] < 50:
                step_list[batch] += 1
                grad = sess.run(model.ce_trigger, {x: np.concatenate([current_trigger, current_trigger], 0), w: w_label,
                                                   is_training: 0, is_augment: 0})[0]
                current_trigger = np.clip(current_trigger - w_lr * np.sign(grad[:half_batch_size]), 0, 1)

            batch_data = np.concatenate([current_trigger,
                                         target_data[batch * half_batch_size: (batch + 1) * half_batch_size]], 0)

            grad = sess.run(model.snnl_trigger, {x: batch_data, w: w_label,
                                                 t: temperatures,
                                                 is_training: 0, is_augment: 0})[0]

            current_trigger = np.clip(current_trigger + w_lr * np.sign(grad[:half_batch_size]), 0, 1)

        for i in range(5):
            grad = sess.run(model.ce_trigger,
                            {x: np.concatenate([current_trigger, current_trigger], 0), w: w_label, is_training: 0,
                             is_augment: 0})[0]
            current_trigger = np.clip(current_trigger - w_lr * np.sign(grad[:half_batch_size]), 0, 1)
        trigger[batch * half_batch_size: (batch + 1) * half_batch_size] = current_trigger
        
        print("trigger")
        plt.subplot(2,1, 2)
        plt.axis("off")
        plt.imshow(current_trigger[1][:,:,0], cmap="gray_r")
        plt.title("With Trigger")
        plt.savefig(f"triggers_images/trigger_{batch}.png")


    ## ------------------------------------------------------ EWE Training ------------------------------------------- ##
    for epoch in range(round((w_epochs * num_batch / w_num_batch))):
        if shuffle:
            np.random.shuffle(index)
            x_train = x_train[index]
            y_train = y_train[index]
        j = 0
        normal = 0
        for batch in range(w_num_batch):
            if n_w_ratio >= 1:
                for i in range(int(n_w_ratio)):
                    if j >= num_batch:
                        j = 0
                    sess.run(model.optimize, {x: x_train[j * batch_size: (j + 1) * batch_size],
                                              y: y_train[j * batch_size: (j + 1) * batch_size], w: w_0,
                                              t: temperatures,
                                              is_training: 1, is_augment: 1})
                    # print(f"Loss of train set at {batch} at {epoch} is {loss}")
                    j += 1
                    normal += 1
            if n_w_ratio > 0 and n_w_ratio % 1 != 0 and n_w_ratio * batch >= j:
                if j >= num_batch:
                    j = 0
                sess.run(model.optimize, {x: x_train[j * batch_size: (j + 1) * batch_size],
                                          y: y_train[j * batch_size: (j + 1) * batch_size], w: w_0,
                                          t: temperatures,
                                          is_training: 1, is_augment: 1})
                j += 1
                normal += 1
            batch_data = np.concatenate([trigger[batch * half_batch_size: (batch + 1) * half_batch_size],
                                         target_data[batch * half_batch_size: (batch + 1) * half_batch_size]], 0)

            _, temp_grad = sess.run(model.optimize, {x: batch_data, y: trigger_label, w: w_label, t: temperatures,
                                                     is_training: 1, is_augment: 0})
            
            if epoch == 5 and batch==0:

                intermediate_inp = sess.run(model.prediction, {x: exclude_x_data[batch * batch_size: (batch + 1) * batch_size], is_training: 0,
                                             is_augment: 0})[1]
                
                intermediate_inp1 = sess.run(model.prediction, {x: target_data[batch * batch_size: (batch + 1) * batch_size], is_training: 0,
                                             is_augment: 0})[1]
                
                intermediate_inp2 = sess.run(model.prediction, {x: trigger[batch * batch_size: (batch + 1) * batch_size], is_training: 0,
                                             is_augment: 0})[1]
                
                # output = sess.run(model.prediction, {x: x_train[batch * batch_size: (batch + 1) * batch_size], is_training: 0,
                #                              is_augment: 0})[-1]
                # output1 = sess.run(model.prediction, {x: x_train[batch * batch_size: (batch + 1) * batch_size], is_training: 0,
                #                              is_augment: 0})[-1]
                # output_final = np.concatenate([output, output1], 0)
                
                intermediate_inp_final = np.concatenate([intermediate_inp, intermediate_inp1, intermediate_inp2], 0)

                y_pca = exclude_y_data[j * batch_size: (j + 1) * batch_size]
                
                new_x = intermediate_inp_final.reshape(intermediate_inp_final.shape[0], -1)
                print(new_x.shape)

                transformed_data = pca_dim(new_x)

                ## Plotting PCA output in the middle of the training.
                # color_list = np.concatenate([y_pca , np.repeat(watermark_target, batch_size), np.repeat(10, batch_size)], 0)
                # marker_list = ["."]*y_pca.shape[0] + [","]*batch_size + ["<"]*batch_size
                # cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black","darkorange","darkgreen", "darkslategray", "deepskyblue", "purple", "cyan", "blue", "crimson", "bisque", "lawngreen"])
                # fig = plt.figure()
                # plot = plt.scatter(transformed_data[:,0], transformed_data[:,1], c= color_list, cmap=cmap, marker=marker_list)
                # plt.legend(handles=plot.legend_elements()[0], labels=list(np.unique(color_list)))
                # plt.show()
                # plt.savefig("In middle training", dpi=fig.dpi)
                


            temperatures -= temp_lr * temp_grad[0]

        batch = 0
        intermediate_inp = sess.run(model.prediction, {x: exclude_x_data[batch * batch_size: (batch + 1) * batch_size], is_training: 0,
                                             is_augment: 0})[1]
                
        intermediate_inp1 = sess.run(model.prediction, {x: target_data[batch * batch_size: (batch + 1) * batch_size], is_training: 0,
                                        is_augment: 0})[1]
        
        intermediate_inp2 = sess.run(model.prediction, {x: trigger[batch * batch_size: (batch + 1) * batch_size], is_training: 0,
                                        is_augment: 0})[1]
        
        # output = sess.run(model.prediction, {x: x_train[batch * batch_size: (batch + 1) * batch_size], is_training: 0,
        #                              is_augment: 0})[-1]
        # output1 = sess.run(model.prediction, {x: x_train[batch * batch_size: (batch + 1) * batch_size], is_training: 0,
        #                              is_augment: 0})[-1]
        # output_final = np.concatenate([output, output1], 0)
        
        intermediate_inp_final = np.concatenate([intermediate_inp, intermediate_inp1, intermediate_inp2], 0)

        y_pca = exclude_y_data[j * batch_size: (j + 1) * batch_size]
        
        new_x = intermediate_inp_final.reshape(intermediate_inp_final.shape[0], -1)
        print(new_x.shape)

        transformed_data = pca_dim(new_x)

        ## Plotting PCA output in the middle of the training.
        # color_list = np.concatenate([y_pca , np.repeat(watermark_target, batch_size), np.repeat(10, batch_size)], 0)
        # marker_list = ["."]*y_pca.shape[0] + [","]*batch_size + ["<"]*batch_size
        # cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black","darkorange","darkgreen", "darkslategray", "deepskyblue", "purple", "cyan", "blue", "crimson", "bisque", "lawngreen"])
        # fig = plt.figure()
        # plot = plt.scatter(transformed_data[:,0], transformed_data[:,1], c= color_list, cmap=cmap, marker=marker_list)
        # plt.legend(handles=plot.legend_elements()[0], labels=list(np.unique(color_list)))
        # plt.show()
        # plt.savefig("In end training", dpi=fig.dpi)


        # first_conv_legitimate = sess.run(model.prediction, {x: exclude_x_data[batch * batch_size: (batch + 1) * batch_size], is_training: 0,
        #                                      is_augment: 0})[0]
        
        # first_conv_watermark = sess.run(model.prediction, {x: trigger[batch * batch_size: (batch + 1)* batch_size], is_training: 0,
        #                                 is_augment: 0})[0]
        
        # second_conv_legitimate = sess.run(model.prediction, {x: exclude_x_data[batch * batch_size: (batch + 1)* batch_size], is_training: 0,
        #                                      is_augment: 0})[1]
        
        # second_conv_watermark = sess.run(model.prediction, {x: trigger[batch * batch_size: (batch + 1)* batch_size], is_training: 0,
        #                                 is_augment: 0})[1]
        
        # fc_legitimate = sess.run(model.prediction, {x: exclude_x_data[batch * batch_size: (batch + 1)* batch_size], is_training: 0,
        #                                      is_augment: 0})[2]
        
        # fc_watermark = sess.run(model.prediction, {x: trigger[batch * batch_size: (batch + 1)* batch_size], is_training: 0,
        #                                      is_augment: 0})[2]
        
        # # plt.subplot(10,1)
        # fig = plt.figure(figsize=(10,10))
        # for i in range(10):
        #     fig.add_subplot(2,10, i+1)
        #     plt.imshow(first_conv_legitimate[0,:,:,i], cmap="gray_r")
        # plt.savefig("ewe_first_conv_legit_activation.png")

        # for i in range(10):
        #     fig.add_subplot(2,10, i+1)
        #     plt.imshow(first_conv_watermark[0,:,:,i], cmap="gray_r")
        # plt.savefig("ewe_first_conv_watermark_activation.png")

        # fig = plt.figure(figsize=(10,10))
        # for i in range(10):
        #     fig.add_subplot(2, 10, i+1)
        #     plt.imshow(second_conv_legitimate[0,:,:,i], cmap="gray_r")
        # plt.savefig("ewe_second_conv_legit_activation.png")

        # for i in range(10):
        #     fig.add_subplot(2,10, i+1)
        #     plt.imshow(second_conv_watermark[0,:,:,i], cmap="gray_r")
        # plt.savefig("ewe_second_conv_watermark_activation.png")
        
    
        # fig = plt.figure(figsize=(10,10))
        # fig.add_subplot(2,1,1)
        # plt.imshow(np.atleast_2d([fc_legitimate[0,:]]), cmap="gray_r")

        # fig.add_subplot(2,1,2)
        # plt.imshow(np.atleast_2d(fc_watermark[0, :]), cmap="gray_r")
        # plt.savefig("ewe_fc_activation.png")

        

        
    print("Test data ewe model")
    victim_error_list = []
    for batch in range(num_test):
        victim_error_list.append(sess.run(model.error, {x: x_test[batch * batch_size: (batch + 1) * batch_size],
                                                        y: y_test[batch * batch_size: (batch + 1) * batch_size],
                                                        is_training: 0, is_augment: 0})[2])
    victim_error = np.average(victim_error_list)

    print("Test trigger ewe model")
    victim_watermark_acc_list = []
    for batch in range(w_num_batch):
        victim_watermark_acc_list.append(validate_watermark(
            model, trigger[batch * half_batch_size: (batch + 1) * half_batch_size], watermark_target))
    victim_watermark_acc = np.average(victim_watermark_acc_list)
    if verbose:
        print(f"Victim Model || validation accuracy: {1 - victim_error}, "
              f"watermark success: {victim_watermark_acc}")
        
    saver.save(sess, "victim_model")


    ##------------------------------------------------- Extracted Model ------------------------------ ##
    # Attack
    extracted_label = []
    for batch in range(num_batch):
        output = sess.run(model.prediction, {x: x_train[batch * batch_size: (batch + 1) * batch_size], is_training: 0,
                                             is_augment: 0})[-1]
        extracted_label.append(output == np.max(output, 1, keepdims=True))
    extracted_label = np.concatenate(extracted_label, 0)
    extracted_data = x_train[:extracted_label.shape[0]]

    tf.compat.v1.get_default_graph().finalize()
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.random.set_random_seed(seed)
    x = tf.compat.v1.placeholder(tf.float32, [batch_size, height, width, channels], name="input")
    y = tf.compat.v1.placeholder(tf.float32, [batch_size, num_class])
    is_training = tf.compat.v1.placeholder(tf.float32)
    is_augment = tf.compat.v1.placeholder(tf.float32)

    if "cifar" in dataset:
        augmented_x = tf.cond(pred=tf.greater(is_augment, 0),
                              true_fn=lambda: augment_train(x),
                              false_fn=lambda: augment_test(x))
        model = plain_model(augmented_x, y, batch_size, num_class, lr, is_training)
    else:
        model = plain_model(x, y, batch_size, num_class, lr, is_training)

    sess.close()
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    for epoch in range(epochs + w_epochs):
        if shuffle:
            np.random.shuffle(index)
            x_train = x_train[index] # cant see used in this block.
            y_train = y_train[index] 
        for batch in range(num_batch):
            sess.run(model.optimize, {x: extracted_data[batch * batch_size: (batch + 1) * batch_size],
                                      y: extracted_label[batch * batch_size: (batch + 1) * batch_size],
                                      is_training: 1, is_augment: 1})

    print("Test data extracted model")
    extracted_error_list = []
    for batch in range(num_test):
        true_label = y_test[batch * batch_size: (batch + 1) * batch_size]
        extracted_error_list.append(
            sess.run(model.error, {x: x_test[batch * batch_size: (batch + 1) * batch_size], y: true_label,
                                   is_training: 0, is_augment: 0})[2])
    extracted_error = np.average(extracted_error_list)

    print("Test trigger extracted model")
    extracted_watermark_acc_list = []
    for batch in range(w_num_batch):
        extracted_watermark_acc_list.append(validate_watermark(
            model, trigger[batch * half_batch_size: (batch + 1) * half_batch_size], watermark_target))
    extracted_watermark_acc = np.average(extracted_watermark_acc_list)
    if verbose:
        print(f"Extracted Model || validation accuracy: {1 - extracted_error},"
              f" watermark success: {extracted_watermark_acc}")


    ##-------------------------------------------------------------- BaseLine Model ------------------------------------------##
    # Clean model for comparison
    tf.compat.v1.get_default_graph().finalize()
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.random.set_random_seed(seed)
    x = tf.compat.v1.placeholder(tf.float32, [batch_size, height, width, channels], name="input")
    y = tf.compat.v1.placeholder(tf.float32, [batch_size, num_class])
    is_training = tf.compat.v1.placeholder(tf.float32)
    is_augment = tf.compat.v1.placeholder(tf.float32)

    if "cifar" in dataset:
        augmented_x = tf.cond(pred=tf.greater(is_augment, 0),
                              true_fn=lambda: augment_train(x),
                              false_fn=lambda: augment_test(x))
        model = plain_model(augmented_x, y, batch_size, num_class, lr, is_training)
    else:
        model = plain_model(x, y, batch_size, num_class, lr, is_training)

    sess.close()
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    for epoch in range(epochs + w_epochs):
        if shuffle:
            np.random.shuffle(index)
            x_train = x_train[index]
            y_train = y_train[index]
        for batch in range(num_batch):
            sess.run(model.optimize, {x: x_train[batch * batch_size: (batch + 1) * batch_size],
                                      y: y_train[batch * batch_size: (batch + 1) * batch_size],
                                      is_training: 1, is_augment: 1})
        for batch in range(w_num_batch):
                batch_data = np.concatenate([trigger[batch * half_batch_size: (batch + 1) * half_batch_size],
                                        target_data[batch * half_batch_size: (batch + 1) * half_batch_size]], 0)
                sess.run(model.optimize, {x: batch_data,
                                        y: trigger_label,
                                        is_training: 1, is_augment: 1})

    baseline_error_list = []
    for batch in range(num_test):
        baseline_error_list.append(sess.run(model.error, {x: x_test[batch * batch_size: (batch + 1) * batch_size],
                                                          y: y_test[batch * batch_size: (batch + 1) * batch_size],
                                                          is_training: 0, is_augment: 0})[2])
    baseline_error = np.average(baseline_error_list)

    print("Test trigger set on baseline")
    baseline_list = []
    for batch in range(w_num_batch):
        baseline_list.append(validate_watermark(
            model, trigger[batch * half_batch_size: (batch + 1) * half_batch_size], watermark_target))
    baseline_watermark = np.average(baseline_list)

    batch = 0
    # first_conv_legitimate = sess.run(model.prediction, {x: exclude_x_data[batch * batch_size: (batch + 1) * batch_size], is_training: 0,
    #                                         is_augment: 0})[0]
    # first_conv_watermark = sess.run(model.prediction, {x: trigger[batch * batch_size: (batch + 1)* batch_size], is_training: 0,
    #                                 is_augment: 0})[0]
    
    # second_conv_legitimate = sess.run(model.prediction, {x: exclude_x_data[batch * batch_size: (batch + 1)* batch_size], is_training: 0,
    #                                     is_augment: 0})[1]
    # second_conv_watermark = sess.run(model.prediction, {x: trigger[batch * batch_size: (batch + 1)* batch_size], is_training: 0,
    #                                 is_augment: 0})[1]
    
    # fc_legitimate = sess.run(model.prediction, {x: exclude_x_data[batch * batch_size: (batch + 1)* batch_size], is_training: 0,
    #                                     is_augment: 0})[2]
    # fc_watermark = sess.run(model.prediction, {x: trigger[batch * batch_size: (batch + 1)* batch_size], is_training: 0,
    #                                     is_augment: 0})[2]

    # plot_activation(first_conv_legitimate, first_conv_watermark, second_conv_legitimate, second_conv_watermark, fc_legitimate, fc_watermark, type_model="baseline")

    if verbose:
        print(f"Clean Model || validation accuracy: {1 - baseline_error}, "
              f"watermark success: {baseline_watermark}")
        

    ## Attack on baseline
    extracted_label = []
    for batch in range(num_batch):
        output = sess.run(model.prediction, {x: x_train[batch * batch_size: (batch + 1) * batch_size], is_training: 0,
                                             is_augment: 0})[-1]
        extracted_label.append(output == np.max(output, 1, keepdims=True))
    extracted_label = np.concatenate(extracted_label, 0)
    extracted_data = x_train[:extracted_label.shape[0]]

    tf.compat.v1.get_default_graph().finalize()
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.random.set_random_seed(seed)
    x = tf.compat.v1.placeholder(tf.float32, [batch_size, height, width, channels], name="input")
    y = tf.compat.v1.placeholder(tf.float32, [batch_size, num_class])
    is_training = tf.compat.v1.placeholder(tf.float32)
    is_augment = tf.compat.v1.placeholder(tf.float32)

    if "cifar" in dataset:
        augmented_x = tf.cond(pred=tf.greater(is_augment, 0),
                              true_fn=lambda: augment_train(x),
                              false_fn=lambda: augment_test(x))
        model = plain_model(augmented_x, y, batch_size, num_class, lr, is_training)
    else:
        model = plain_model(x, y, batch_size, num_class, lr, is_training)

    print("Plain model", model)

    sess.close()
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    for epoch in range(epochs + w_epochs):
        if shuffle:
            np.random.shuffle(index)
            x_train = x_train[index] # cant see used in this block.
            y_train = y_train[index] 
        for batch in range(num_batch):
            sess.run(model.optimize, {x: extracted_data[batch * batch_size: (batch + 1) * batch_size],
                                      y: extracted_label[batch * batch_size: (batch + 1) * batch_size],
                                      is_training: 1, is_augment: 1})

    extracted_error_list = []
    for batch in range(num_test):
        true_label = y_test[batch * batch_size: (batch + 1) * batch_size]
        extracted_error_list.append(
            sess.run(model.error, {x: x_test[batch * batch_size: (batch + 1) * batch_size], y: true_label,
                                   is_training: 0, is_augment: 0})[2])
    extracted_error = np.average(extracted_error_list)

    print("Test trigger set on baseline extracted")
    extracted_watermark_acc_list = []
    for batch in range(w_num_batch):
        extracted_watermark_acc_list.append(validate_watermark(
            model, trigger[batch * half_batch_size: (batch + 1) * half_batch_size], watermark_target))
    extracted_watermark_acc = np.average(extracted_watermark_acc_list)
    if verbose:
        print(f"Extracted Model || validation accuracy: {1 - extracted_error},"
              f" watermark success: {extracted_watermark_acc}")


    return 1 - victim_error, victim_watermark_acc, 1 - extracted_error, extracted_watermark_acc, 1 - baseline_error, \
           baseline_watermark


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', help='batch size', type=int, default=512)
    parser.add_argument('--ratio',
                        help='ratio of amount of legitimate data to watermarked data',
                        type=float, default=1.)
    parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
    parser.add_argument('--epochs', help='epochs for training without watermarking', type=int, default=10)
    parser.add_argument('--w_epochs', help='epochs for training with watermarking', type=int, default=10)
    parser.add_argument('--dataset', help='mnist, fashion, speechcmd, cifar10, or cifar100', type=str, default="cifar10")
    parser.add_argument('--model', help='2_conv, lstm, or resnet', type=str, default="2_conv")
    parser.add_argument('--metric', help='distance metric used in snnl, euclidean or cosine', type=str, default="cosine")
    parser.add_argument('--factors', help='weight factor for snnl', nargs='+', type=float, default=[32, 32, 32])
    parser.add_argument('--temperatures', help='temperature for snnl', nargs='+', type=float, default=[1, 1, 1])
    parser.add_argument('--threshold', help='threshold for estimated false watermark rate, should be <= 1/num_class', type=float, default=0.1)
    parser.add_argument('--maxiter', help='iter of perturb watermarked data with respect to snnl', type=int, default=10)
    parser.add_argument('--w_lr', help='learning rate for perturbing watermarked data', type=float, default=0.01)
    parser.add_argument('--t_lr', help='learning rate for temperature', type=float, default=0.1)
    parser.add_argument('--source', help='source class of watermark', type=int, default=1)
    parser.add_argument('--target', help='target class of watermark', type=int, default=7)
    parser.add_argument('--shuffle', type=int, default=0)
    parser.add_argument('--seed', help='random seed', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--default', help='whether to use default hyperparameter, 0 or 1', type=int, default=1)
    parser.add_argument('--layers', help='number of layers, only useful if model is resnet', type=int, default=18)
    parser.add_argument('--distrib', help='use in or out of distribution watermark', type=str, default='out')

    args = parser.parse_args()
    default = args.default
    batch_size = args.batch_size
    ratio = args.ratio
    lr = args.lr
    epochs = args.epochs
    w_epochs = args.w_epochs
    factors = args.factors
    temperatures = args.temperatures
    threshold = args.threshold
    w_lr = args.w_lr
    t_lr = args.t_lr
    source = args.source
    target = args.target
    seed = args.seed
    verbose = args.verbose
    dataset = args.dataset
    model_type = args.model
    maxiter = args.maxiter
    distrib = args.distrib
    layers = args.layers
    metric = args.metric
    shuffle = args.shuffle

    # hyperparameters with reasonable performance
    if default:
        if dataset == 'mnist':
            model_type = '2_conv'
            ratio = 1
            batch_size = 512
            epochs = 10
            w_epochs = 10
            factors = [32, 32, 32]
            temperatures = [1, 1, 1]
            metric = "cosine"
            threshold = 0.1
            t_lr = 0.1
            w_lr = 0.01
            source = 1
            target = 7
            maxiter = 10
            distrib = "out"
        elif dataset == 'fashion':
            if model_type == '2_conv':
                batch_size = 128
                ratio = 2
                epochs = 10
                w_epochs = 10
                factors = [32, 32, 32]
                temperatures = [1, 1, 1]
                t_lr = 0.1
                threshold = 0.1
                w_lr = 0.01
                source = 8
                target = 1
                maxiter = 10
                distrib = "out"
                metric = "cosine"
            elif model_type == 'resnet':
                batch_size = 128
                layers = 18
                ratio = 1.2
                epochs = 5
                w_epochs = 5
                factors = [1000, 1000, 1000]
                temperatures = [0.01, 0.01, 0.01]
                t_lr = 0.1
                threshold = 0.1
                w_lr = 0.01
                source = 9
                target = 0
                maxiter = 10
                distrib = "out"
                metric = "cosine"
        elif dataset == 'speechcmd':
            batch_size = 128
            epochs = 30
            w_epochs = 1
            model_type = "lstm"
            distrib = 'in'
            ratio = 1
            shuffle = 1
            t_lr = 2
            maxiter = 10
            threshold = 0.1
            factors = [16, 16, 16]
            temperatures = [30, 30, 30]
            source = 9
            target = 5
        elif dataset == "cifar10":
            batch_size = 128
            model_type = "2_conv"
            layers = 18
            ratio = 4
            epochs = 50
            w_epochs = 6
            factors = [1e5, 1e5, 1e5]
            temperatures = [1, 1, 1]
            t_lr = 0.1
            threshold = 0.1
            w_lr = 0.01
            source = 8
            target = 0
            maxiter = 10
            distrib = "out"
            metric = "cosine"
        elif dataset == "cifar100":
            batch_size = 128
            model_type = "resnet"
            layers = 18
            epochs = 100
            w_epochs = 8
            ratio = 15
            factors = [1e5, 1e5, 1e5]
            temperatures = [1, 1, 1]
            t_lr = 0.01
            threshold = 0.1
            w_lr = 0.01
            source = 8
            target = 0
            maxiter = 100
            distrib = "out"
            metric = "cosine"

    random.seed(seed)
    np.random.seed(seed)
    tf.compat.v1.random.set_random_seed(seed)

    if dataset == 'mnist' or dataset == 'fashion':
        with open(os.path.join("data", f"{dataset}.pkl"), 'rb') as f:
            mnist = pickle.load(f)
        x_train, y_train, x_test, y_test = mnist["training_images"], mnist["training_labels"], \
                                           mnist["test_images"], mnist["test_labels"]
        x_train = np.reshape(x_train / 255, [-1, 28, 28, 1])
        x_test = np.reshape(x_test / 255, [-1, 28, 28, 1])
    elif "cifar" in dataset:
        import tensorflow_datasets as tfds
        ds = tfds.load(dataset)
        for i in tfds.as_numpy(ds['train'].batch(50000).take(1)):
            x_train = i['image'] / 255
            y_train = i['label']
        for i in tfds.as_numpy(ds['test'].batch(50000).take(1)):
            x_test = i['image'] / 255
            y_test = i['label']
    elif dataset == 'speechcmd':
        x_train = np.swapaxes(np.load(os.path.join(r"data", "sd_GSCmdV2", 'x_train.npy')), 1, 2)
        y_train = np.load(os.path.join(r"data", "sd_GSCmdV2", 'y_train.npy'))
        x_test = np.swapaxes(np.load(os.path.join(r"data", "sd_GSCmdV2", 'x_test.npy')), 1, 2)
        y_test = np.load(os.path.join(r"data", "sd_GSCmdV2", 'y_test.npy'))
    else:
        raise NotImplementedError('Dataset is not implemented.')

    if model_type == '2_conv':
        ewe_model = functools.partial(md.EWE_2_conv, metric=metric)
        plain_model = md.PlainAnother_2_conv
    elif model_type == 'resnet':
        ewe_model = functools.partial(md.EWE_Resnet, metric=metric, layers=layers)
        plain_model = functools.partial(md.Plain_Resnet, layers=layers)
    elif model_type == 'lstm':
        ewe_model = functools.partial(md.EWE_LSTM, metric=metric)
        plain_model = md.Plain_LSTM
    else:
        raise NotImplementedError('Model is not implemented.')
    
    lr = 0.01

    res = train(x_train, y_train, x_test, y_test, ewe_model, plain_model, epochs, w_epochs, lr, ratio, factors,
                temperatures, source, target, batch_size, w_lr, threshold, maxiter, shuffle, t_lr, dataset, distrib,
                verbose)
