import numpy as np
from utils import pca_dim
import matplotlib.pyplot as plt
from utils import validate_watermark, pca_and_plot, plot_activation
import tensorflow as tf
from utils import augment_train, augment_test

def ewe_train(model, dataset, w_epochs, num_batch, half_batch_size, w_num_batch, shuffle, index, n_w_ratio, sess, batch_size, temperatures, is_training, is_augment, trigger, target_data, trigger_label,x_train, y_train, x_test, y_test, exclude_x_data, exclude_y_data, w_0, w_label, watermark_target, num_class, num_test, temp_lr, x, y, t, w, verbose, distrib):

    val_accs = []
    wat_success = []
    # for e in [10,20,30,40,50]:
    # (w_epochs * num_batch / w_num_batch)
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
                    loss_value = sess.run(model.optimize, {x: x_train[j * batch_size: (j + 1) * batch_size],
                                            y: y_train[j * batch_size: (j + 1) * batch_size], w: w_0,
                                            t: temperatures,
                                            is_training: 1, is_augment: 1})[0]
                    j += 1
                    normal += 1
                    print(f"Loss of train set at batch {batch} at epoch {epoch} is {loss_value}")
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

            loss_value, _, temp_grad = sess.run(model.optimize, {x: batch_data, y: trigger_label, w: w_label, t: temperatures,
                                                    is_training: 1, is_augment: 0})
            
            temperatures -= temp_lr * temp_grad[0]
            print(f"Loss of watermark set at {batch} at {epoch} is {loss_value}")
            
            ##--------------------------------------------- VISUALIZATION -----------------------------------##


            ## performing pca and plotting in middle of the training.
            penultimate_layer = 2
            distrib = distrib
            if epoch == 5 and batch==0:

                intermediate_inp = sess.run(model.prediction, {x: exclude_x_data[batch * batch_size: (batch + 1) * batch_size], is_training: 0,
                                            is_augment: 0})[penultimate_layer]
                intermediate_inp1 = sess.run(model.prediction, {x: target_data[batch * batch_size: (batch + 1) * batch_size], is_training: 0,
                                is_augment: 0})[penultimate_layer]
                intermediate_inp2 = sess.run(model.prediction, {x: trigger[batch * batch_size: (batch + 1) * batch_size], is_training: 0,
                                is_augment: 0})[penultimate_layer]
                intermediate_inp_final = np.concatenate([intermediate_inp, intermediate_inp1, intermediate_inp2], 0)

                y_pca = exclude_y_data[j * batch_size: (j + 1) * batch_size]
                new_x = intermediate_inp_final.reshape(intermediate_inp_final.shape[0], -1)

                pca_and_plot(new_x, [intermediate_inp.reshape(intermediate_inp.shape[0], -1), intermediate_inp1.reshape(intermediate_inp1.shape[0], -1), intermediate_inp2.reshape(intermediate_inp2.shape[0], -1)], type_model="ewe", dataset=dataset, time="between_train", penultimate_layer=penultimate_layer, distrib=distrib)

                

        ## performing and plotting pca after the training.
    batch = 0
    intermediate_inp = sess.run(model.prediction, {x: exclude_x_data[batch * batch_size: (batch + 1) * batch_size], is_training: 0,
                                        is_augment: 0})[penultimate_layer]
    intermediate_inp1 = sess.run(model.prediction, {x: target_data[batch * batch_size: (batch + 1) * batch_size], is_training: 0,
                                    is_augment: 0})[penultimate_layer]
    intermediate_inp2 = sess.run(model.prediction, {x: trigger[batch * batch_size: (batch + 1) * batch_size], is_training: 0,
                                    is_augment: 0})[penultimate_layer]
    intermediate_inp_final = np.concatenate([intermediate_inp, intermediate_inp1, intermediate_inp2], 0)
    y_pca = exclude_y_data[j * batch_size: (j + 1) * batch_size]
    new_x = intermediate_inp_final.reshape(intermediate_inp_final.shape[0], -1)

    pca_and_plot(new_x, [intermediate_inp.reshape(intermediate_inp.shape[0], -1), intermediate_inp1.reshape(intermediate_inp1.shape[0], -1), intermediate_inp2.reshape(intermediate_inp2.shape[0], -1)], type_model="ewe", dataset=dataset, time="end_train", penultimate_layer=penultimate_layer, distrib=distrib)

    ## plotting the activations after the training.
    first_conv_legitimate = sess.run(model.prediction, {x: exclude_x_data[batch * batch_size: (batch + 1) * batch_size], is_training: 0,
                                        is_augment: 0})[0]
    first_conv_watermark = sess.run(model.prediction, {x: trigger[batch * batch_size: (batch + 1)* batch_size], is_training: 0,
                                    is_augment: 0})[0]
    
    second_conv_legitimate = sess.run(model.prediction, {x: exclude_x_data[batch * batch_size: (batch + 1)* batch_size], is_training: 0,
                                        is_augment: 0})[1]
    second_conv_watermark = sess.run(model.prediction, {x: trigger[batch * batch_size: (batch + 1)* batch_size], is_training: 0,
                                    is_augment: 0})[1]
    
    fc_legitimate = sess.run(model.prediction, {x: exclude_x_data[batch * batch_size: (batch + 1)* batch_size], is_training: 0,
                                        is_augment: 0})[2]
    fc_watermark = sess.run(model.prediction, {x: trigger[batch * batch_size: (batch + 1)* batch_size], is_training: 0,
                                        is_augment: 0})[2]
    
    # plt.subplot(10,1)
    plot_activation(first_conv_legitimate, first_conv_watermark, second_conv_legitimate, second_conv_watermark, fc_legitimate, fc_watermark, type_model="ewe", dataset=dataset, distrib=distrib)

        ##--------------------------------------------- VISUALIZATION ENDS HERE -----------------------------------##

        # if epoch % 10 == 0:

    victim_error_list = []
    for batch in range(num_test):
        victim_error_list.append(sess.run(model.error, {x: x_test[batch * batch_size: (batch + 1) * batch_size],
                                                        y: y_test[batch * batch_size: (batch + 1) * batch_size],
                                                        is_training: 0, is_augment: 0}))
    victim_error = np.average(victim_error_list)

    victim_watermark_acc_list = []
    for batch in range(w_num_batch):
        victim_watermark_acc_list.append(validate_watermark(
            model, trigger[batch * half_batch_size: (batch + 1) * half_batch_size], watermark_target, sess, batch_size, num_class, is_training, is_augment, x, y))
    victim_watermark_acc = np.average(victim_watermark_acc_list)
    if verbose:
        print(f"Victim Model || validation accuracy: {1 - victim_error}, "
            f"watermark success: {victim_watermark_acc} at epoch {epoch}")
        
    val_accs.append(1 - victim_error)
    wat_success.append(victim_watermark_acc)


    # wat_success_copy = wat_success.copy()
    # wat_success_copy[1:] = np.diff(wat_success_copy)
    # val_accs_copy = val_accs[::-1].copy()
    # val_accs_copy[1:] = np.diff(val_accs_copy)

    # trade_off = np.array(wat_success_copy)/np.array(val_accs_copy)

    # plt.figure()
    # plt.plot([0,10,20,30,40,50, 60], wat_success)
    # plt.plot([0,10,20,30,40,50, 60], val_accs_copy)
    # plt.plot([0, 10,20,30,40,50, 60], trade_off)
    # plt.savefig("longer training.png")
                
    return model




def plain_model_train(model, dataset, exclude_x_data, exclude_y_data, num_batch, batch_size, seed, plain_model, height, width, channels, num_class, lr, epochs, w_epochs, shuffle, index, num_test, x_train, y_train, x_test, y_test, w_num_batch, target_data, trigger_label, trigger, half_batch_size, watermark_target, verbose, is_training, is_augment, sess, x, extraction_flag, activation, distrib):

    """
    To perform model extraction using retraining.
    This is simply in the white box setting.
    """
    # print(sess)
    # print([n.name for n in tf.get_default_graph().as_graph_def().node])

    if extraction_flag:
        print("here")
        process = "Extracted"
        
    else:
        process = "Baseline"

    tf.get_default_graph().finalize()
    tf.compat.v1.reset_default_graph()
    tf.random.set_random_seed(seed)
    x = tf.compat.v1.placeholder(tf.float32, [batch_size, height, width, channels], name="input")
    y = tf.compat.v1.placeholder(tf.float32, [batch_size, num_class])
    is_training = tf.compat.v1.placeholder(tf.float32)
    is_augment = tf.compat.v1.placeholder(tf.float32)

    ## defining the model
    if "cifar" in dataset:
        augmented_x = tf.cond(tf.greater(is_augment, 0),
                              lambda: augment_train(x),
                              lambda: augment_test(x))
        model = plain_model(augmented_x, y, batch_size, num_class, lr, is_training)
    else:
        model = plain_model(x, y, batch_size, num_class, lr, is_training)

    sess.close()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print(f"training the plain model for epochs {epochs + w_epochs}")

    ## training the model using extracted data(from trained model incase of extraction) or model training from sratch.
    for epoch in range(epochs + w_epochs):
        if shuffle:
            np.random.shuffle(index)
            x_train = x_train[index] # cant see used in this block.
            y_train = y_train[index] 
        for batch in range(num_batch):
            sess.run(model.optimize, {x: x_train[batch * batch_size: (batch + 1) * batch_size],
                                      y: y_train[batch * batch_size: (batch + 1) * batch_size],
                                      is_training: 1, is_augment: 1})
            
        if process == "Baseline":
            for batch in range(w_num_batch):
                batch_data = np.concatenate([trigger[batch * half_batch_size: (batch + 1) * half_batch_size],
                                        target_data[batch * half_batch_size: (batch + 1) * half_batch_size]], 0)
                sess.run(model.optimize, {x: batch_data,
                                        y: trigger_label,
                                        is_training: 1, is_augment: 1})
                
                ## performing pca and plotting in middle of the training.
                distrib = distrib
                penultimate_layer = 2
                if epoch == 5 and batch==0:

                    intermediate_inp = sess.run(model.prediction, {x: exclude_x_data[batch * batch_size: (batch + 1) * batch_size], is_training: 0,
                                                is_augment: 0})[penultimate_layer]
                    intermediate_inp1 = sess.run(model.prediction, {x: target_data[batch * batch_size: (batch + 1) * batch_size], is_training: 0,
                                    is_augment: 0})[penultimate_layer]
                    intermediate_inp2 = sess.run(model.prediction, {x: trigger[batch * batch_size: (batch + 1) * batch_size], is_training: 0,
                                    is_augment: 0})[penultimate_layer]
                    intermediate_inp_final = np.concatenate([intermediate_inp, intermediate_inp1, intermediate_inp2], 0)
                    new_x = intermediate_inp_final.reshape(intermediate_inp_final.shape[0], -1)

                    pca_and_plot(new_x, [intermediate_inp.reshape(intermediate_inp.shape[0], -1), intermediate_inp1.reshape(intermediate_inp1.shape[0], -1), intermediate_inp2.reshape(intermediate_inp2.shape[0], -1)], type_model="baseline", dataset=dataset, time="between_train", penultimate_layer=penultimate_layer, distrib=distrib)


    penultimate_layer = 2
    batch=0

    intermediate_inp = sess.run(model.prediction, {x: exclude_x_data[batch * batch_size: (batch + 1) * batch_size], is_training: 0,
                                is_augment: 0})[penultimate_layer]
    intermediate_inp1 = sess.run(model.prediction, {x: target_data[batch * batch_size: (batch + 1) * batch_size], is_training: 0,
                    is_augment: 0})[penultimate_layer]
    intermediate_inp2 = sess.run(model.prediction, {x: trigger[batch * batch_size: (batch + 1) * batch_size], is_training: 0,
                    is_augment: 0})[penultimate_layer]
    intermediate_inp_final = np.concatenate([intermediate_inp, intermediate_inp1, intermediate_inp2], 0)

    new_x = intermediate_inp_final.reshape(intermediate_inp_final.shape[0], -1)

    pca_and_plot(new_x, [intermediate_inp.reshape(intermediate_inp.shape[0], -1), intermediate_inp1.reshape(intermediate_inp1.shape[0], -1), intermediate_inp2.reshape(intermediate_inp2.shape[0], -1)], type_model="baseline", dataset=dataset, time="end_train", penultimate_layer=penultimate_layer, distrib=distrib)


    ## activation visualization
    first_conv_legitimate = sess.run(model.prediction, {x: exclude_x_data[batch * batch_size: (batch + 1) * batch_size], is_training: 0,
                                        is_augment: 0})[0]
    first_conv_watermark = sess.run(model.prediction, {x: trigger[batch * batch_size: (batch + 1)* batch_size], is_training: 0,
                                    is_augment: 0})[0]
    
    second_conv_legitimate = sess.run(model.prediction, {x: exclude_x_data[batch * batch_size: (batch + 1)* batch_size], is_training: 0,
                                        is_augment: 0})[1]
    second_conv_watermark = sess.run(model.prediction, {x: trigger[batch * batch_size: (batch + 1)* batch_size], is_training: 0,
                                    is_augment: 0})[1]
    
    fc_legitimate = sess.run(model.prediction, {x: exclude_x_data[batch * batch_size: (batch + 1)* batch_size], is_training: 0,
                                        is_augment: 0})[2]
    fc_watermark = sess.run(model.prediction, {x: trigger[batch * batch_size: (batch + 1)* batch_size], is_training: 0,
                                        is_augment: 0})[2]
    
    # plt.subplot(10,1)
    plot_activation(first_conv_legitimate, first_conv_watermark, second_conv_legitimate, second_conv_watermark, fc_legitimate, fc_watermark, type_model="baseline", dataset=dataset, distrib=distrib)





    ## testing on test data
    error_list = []
    for batch in range(num_test):
        true_label = y_test[batch * batch_size: (batch + 1) * batch_size]
        error_list.append(
            sess.run(model.error, {x: x_test[batch * batch_size: (batch + 1) * batch_size], y: true_label,
                                   is_training: 0, is_augment: 0}))
    error = np.average(error_list)

    ## testing on watermark dataset
    watermark_acc_list = []
    for batch in range(w_num_batch):
        watermark_acc_list.append(validate_watermark(
            model, trigger[batch * half_batch_size: (batch + 1) * half_batch_size], watermark_target, sess, batch_size, num_class, is_training, is_augment, x, y))
    watermark_acc = np.average(watermark_acc_list)
    if verbose:
        print(f"{process} Model || validation accuracy: {1 - error},"
              f" watermark success: {watermark_acc}")
        
    return model
        



    