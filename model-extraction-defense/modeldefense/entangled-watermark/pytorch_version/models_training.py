import numpy as np
from utils import pca_dim
import matplotlib.pyplot as plt
from utils import validate_watermark, pca_and_plot, plot_activation
import tensorflow as tf
from utils import augment_train, augment_test
from torch.utils import data
from trainer import Model_trainer
import torch as t
import torch.optim as optim


def ewe_train(model, batch_size, half_batch_size, train_set, target_set, trigger_set, w_epochs, n_w_ratio, temperatures,
              factors, num_class, target_class, temp_lr, verbose):
    """
    Training the EWE model for training data and trigger watermark data.


    Returns
    ---------

    Printing the accuracies
    returning the trained model.
    """

    val_accs = []
    wat_success = []

    temperatures = t.FloatTensor(temperatures)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = t.nn.CrossEntropyLoss()
    model.train()
    trainer_class = Model_trainer()

    num_batch = len(train_set) // batch_size
    w_num_batch = len(target_set) // half_batch_size

    train_dl = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    trigger_dl = data.DataLoader(trigger_set, batch_size=half_batch_size, shuffle=True)
    target_dl = data.DataLoader(target_set, batch_size=half_batch_size, shuffle=True)

    print("No. of epochs", round((w_epochs * num_batch / w_num_batch)))

    for epoch in range(round((w_epochs * num_batch / w_num_batch))):
        j = 0
        normal = 0
        for batch_idx, (train_input_label, trigger_input_label, target_input_label) in enumerate(
                zip(train_dl, trigger_dl, target_dl)):

            ## --------------------------------- TRaining for train data ---------------------------
            if n_w_ratio >= 1:
                for _ in range(int(n_w_ratio)):
                    if j >= num_batch:
                        j = 0

                    w_0 = t.zeros(train_input_label[0].shape[0])  ## verify this once.
                    # y_pred = model(train_input_label[0])

                    loss, _ = trainer_class.ce_snnl_loss(model, train_input_label[1], train_input_label[0],
                                                         temperatures, w_0, factors)
                    loss.backward()
                    optimizer.step()

                    print(f"Loss cond 1 at epoch {epoch} for each batch {batch_idx} is {loss}")

                    j += 1
                    normal += 1

            if n_w_ratio > 0 and n_w_ratio % 1 != 0 and n_w_ratio * batch_idx >= j:
                if j >= num_batch:
                    j = 0

                w_0 = t.zeros(train_input_label[0].shape[0])
                y_pred = model(train_input_label[0])

                loss, _ = trainer_class.ce_snnl_loss(model, train_input_label[1], train_input_label[0], temperatures,
                                                     w_0, factors)
                loss.backward()
                optimizer.step()

                print(f"Loss cond 2 at epoch {epoch} for each batch {batch_idx} is {loss}")

                j += 1
                normal += 1
            ##-------------------------------------------------------------------------------------------------------------------##

            ##------------------------------------------ Training triggers/watermark set --------------------------##
            print("trigger training")
            batch_data = t.concat((trigger_input_label[0], target_input_label[0]))
            w_label = t.concat((t.ones(trigger_input_label[0].shape[0]), t.zeros(target_input_label[0].shape[0])))

            trigger_label = t.tensor(target_class)
            trigger_label = trigger_label.repeat(batch_size)  ## setting the trigger label

            y_pred = model(batch_data)

            loss, temp_grad = trainer_class.ce_snnl_loss(model, trigger_label, batch_data, temperatures, w_label,
                                                         factors)
            loss.backward(retain_graph=True)
            optimizer.step()
            # print(temp_grad)

            print(f"Loss trigger at epoch {epoch} for each batch {batch_idx} is {loss}")

            # print(temp_lr * temp_grad)
            ## not sure about this. ------------------------->>>>>>>>>>>>>>>
            with t.no_grad():
                temperatures = temperatures - (temp_lr * temp_grad)

                temp_grad = None

    return model

    # victim_error_list = []
    # for batch_idx, (input, label) in enumerate(train_dl):
    #     victim_error_list.append(trainer_class.error_rate(model, input, label))

    # victim_error = np.average(victim_error_list)

    # victim_watermark_acc_list = []
    # for batch_idx , (input, label) in enumerate(trigger_dl):
    #     victim_watermark_acc_list.append(validate_watermark(model, trainer_class, input, target_class, batch_size, num_class))
    #     # victim_watermark_acc_list.append(validate_watermark(
    #     #     model, trigger[batch * half_batch_size: (batch + 1) * half_batch_size], watermark_target, sess, batch_size, num_class, is_training, is_augment, x, y))
    # victim_watermark_acc = np.average(victim_watermark_acc_list)
    # if verbose:
    #     print(f"Victim Model || validation accuracy: {1 - victim_error}, "
    #         f"watermark success: {victim_watermark_acc} at epoch {epoch}")

    # val_accs.append(1 - victim_error)
    # wat_success.append(victim_watermark_acc)


# def ewe_train(model, dataset, w_epochs, num_batch, half_batch_size, w_num_batch, shuffle, index, n_w_ratio, sess, batch_size, temperatures, is_training, is_augment, trigger, target_data, trigger_label,x_train, y_train, x_test, y_test, exclude_x_data, exclude_y_data, w_0, w_label, watermark_target, num_class, num_test, temp_lr, x, y, t, w, verbose, distrib):

#     val_accs = []
#     wat_success = []
#     # for e in [10,20,30,40,50]:
#     # (w_epochs * num_batch / w_num_batch)
#     print(f"EWE training for {w_epochs * num_batch / w_num_batch}")
#     for epoch in range(round((w_epochs * num_batch / w_num_batch))):
#         if shuffle:
#             np.random.shuffle(index)
#             x_train = x_train[index]
#             y_train = y_train[index]
#         j = 0
#         normal = 0
#         for batch in range(w_num_batch):
#             if n_w_ratio >= 1:
#                 for i in range(int(n_w_ratio)):
#                     if j >= num_batch:
#                         j = 0
#                     sess.run(model.optimize, {x: x_train[j * batch_size: (j + 1) * batch_size],
#                                             y: y_train[j * batch_size: (j + 1) * batch_size], w: w_0,
#                                             t: temperatures,
#                                             is_training: 1, is_augment: 1})
#                     j += 1
#                     normal += 1
#             if n_w_ratio > 0 and n_w_ratio % 1 != 0 and n_w_ratio * batch >= j:
#                 if j >= num_batch:
#                     j = 0
#                 sess.run(model.optimize, {x: x_train[j * batch_size: (j + 1) * batch_size],
#                                         y: y_train[j * batch_size: (j + 1) * batch_size], w: w_0,
#                                         t: temperatures,
#                                         is_training: 1, is_augment: 1})
#                 j += 1
#                 normal += 1
#             batch_data = np.concatenate([trigger[batch * half_batch_size: (batch + 1) * half_batch_size],
#                                         target_data[batch * half_batch_size: (batch + 1) * half_batch_size]], 0)

#             _, _ ,temp_grad = sess.run(model.optimize, {x: batch_data, y: trigger_label, w: w_label, t: temperatures,
#                                                     is_training: 1, is_augment: 0})

#             temperatures -= temp_lr * temp_grad[0]

#             ##--------------------------------------------- VISUALIZATION -----------------------------------##


#             ## performing pca and plotting in middle of the training.
#             penultimate_layer = 2
#             distrib = distrib
#             if epoch == 5 and batch==0:

#                 intermediate_inp = sess.run(model.prediction, {x: exclude_x_data[batch * batch_size: (batch + 1) * batch_size], is_training: 0,
#                                             is_augment: 0})[penultimate_layer]
#                 intermediate_inp1 = sess.run(model.prediction, {x: target_data[batch * batch_size: (batch + 1) * batch_size], is_training: 0,
#                                 is_augment: 0})[penultimate_layer]
#                 intermediate_inp2 = sess.run(model.prediction, {x: trigger[batch * batch_size: (batch + 1) * batch_size], is_training: 0,
#                                 is_augment: 0})[penultimate_layer]
#                 intermediate_inp_final = np.concatenate([intermediate_inp, intermediate_inp1, intermediate_inp2], 0)

#                 y_pca = exclude_y_data[j * batch_size: (j + 1) * batch_size]
#                 new_x = intermediate_inp_final.reshape(intermediate_inp_final.shape[0], -1)

#                 pca_and_plot(new_x, [intermediate_inp.reshape(intermediate_inp.shape[0], -1), intermediate_inp1.reshape(intermediate_inp1.shape[0], -1), intermediate_inp2.reshape(intermediate_inp2.shape[0], -1)], type_model="ewe", dataset=dataset, time="between_train", penultimate_layer=penultimate_layer, distrib=distrib)


#         ## performing and plotting pca after the training.
#     batch = 0
#     intermediate_inp = sess.run(model.prediction, {x: exclude_x_data[batch * batch_size: (batch + 1) * batch_size], is_training: 0,
#                                         is_augment: 0})[penultimate_layer]
#     intermediate_inp1 = sess.run(model.prediction, {x: target_data[batch * batch_size: (batch + 1) * batch_size], is_training: 0,
#                                     is_augment: 0})[penultimate_layer]
#     intermediate_inp2 = sess.run(model.prediction, {x: trigger[batch * batch_size: (batch + 1) * batch_size], is_training: 0,
#                                     is_augment: 0})[penultimate_layer]
#     intermediate_inp_final = np.concatenate([intermediate_inp, intermediate_inp1, intermediate_inp2], 0)
#     y_pca = exclude_y_data[j * batch_size: (j + 1) * batch_size]
#     new_x = intermediate_inp_final.reshape(intermediate_inp_final.shape[0], -1)

#     pca_and_plot(new_x, [intermediate_inp.reshape(intermediate_inp.shape[0], -1), intermediate_inp1.reshape(intermediate_inp1.shape[0], -1), intermediate_inp2.reshape(intermediate_inp2.shape[0], -1)], type_model="ewe", dataset=dataset, time="end_train", penultimate_layer=penultimate_layer, distrib=distrib)

#     ## plotting the activations after the training.
#     first_conv_legitimate = sess.run(model.prediction, {x: exclude_x_data[batch * batch_size: (batch + 1) * batch_size], is_training: 0,
#                                         is_augment: 0})[0]
#     first_conv_watermark = sess.run(model.prediction, {x: trigger[batch * batch_size: (batch + 1)* batch_size], is_training: 0,
#                                     is_augment: 0})[0]

#     second_conv_legitimate = sess.run(model.prediction, {x: exclude_x_data[batch * batch_size: (batch + 1)* batch_size], is_training: 0,
#                                         is_augment: 0})[1]
#     second_conv_watermark = sess.run(model.prediction, {x: trigger[batch * batch_size: (batch + 1)* batch_size], is_training: 0,
#                                     is_augment: 0})[1]

#     fc_legitimate = sess.run(model.prediction, {x: exclude_x_data[batch * batch_size: (batch + 1)* batch_size], is_training: 0,
#                                         is_augment: 0})[2]
#     fc_watermark = sess.run(model.prediction, {x: trigger[batch * batch_size: (batch + 1)* batch_size], is_training: 0,
#                                         is_augment: 0})[2]

#     # plt.subplot(10,1)
#     plot_activation(first_conv_legitimate, first_conv_watermark, second_conv_legitimate, second_conv_watermark, fc_legitimate, fc_watermark, type_model="ewe", dataset=dataset, distrib=distrib)

#         ##--------------------------------------------- VISUALIZATION ENDS HERE -----------------------------------##

#         # if epoch % 10 == 0:

#     victim_error_list = []
#     for batch in range(num_test):
#         victim_error_list.append(sess.run(model.error, {x: x_test[batch * batch_size: (batch + 1) * batch_size],
#                                                         y: y_test[batch * batch_size: (batch + 1) * batch_size],
#                                                         is_training: 0, is_augment: 0}))
#     victim_error = np.average(victim_error_list)

#     victim_watermark_acc_list = []
#     for batch in range(w_num_batch):
#         victim_watermark_acc_list.append(validate_watermark(
#             model, trigger[batch * half_batch_size: (batch + 1) * half_batch_size], watermark_target, sess, batch_size, num_class, is_training, is_augment, x, y))
#     victim_watermark_acc = np.average(victim_watermark_acc_list)
#     if verbose:
#         print(f"Victim Model || validation accuracy: {1 - victim_error}, "
#             f"watermark success: {victim_watermark_acc} at epoch {epoch}")

#     val_accs.append(1 - victim_error)
#     wat_success.append(victim_watermark_acc)


#     # wat_success_copy = wat_success.copy()
#     # wat_success_copy[1:] = np.diff(wat_success_copy)
#     # val_accs_copy = val_accs[::-1].copy()
#     # val_accs_copy[1:] = np.diff(val_accs_copy)

#     # trade_off = np.array(wat_success_copy)/np.array(val_accs_copy)

#     # plt.figure()
#     # plt.plot([0,10,20,30,40,50, 60], wat_success)
#     # plt.plot([0,10,20,30,40,50, 60], val_accs_copy)
#     # plt.plot([0, 10,20,30,40,50, 60], trade_off)
#     # plt.savefig("longer training.png")

#     return model


def plain_model_train(model, dataset, exclude_x_data, exclude_y_data, num_batch, batch_size, seed, plain_model, height,
                      width, channels, num_class, lr, epochs, w_epochs, shuffle, index, num_test, x_train, y_train,
                      x_test, y_test, w_num_batch, target_data, trigger_label, trigger, half_batch_size,
                      watermark_target, verbose, is_training, is_augment, sess, x, extraction_flag, activation, distrib,
                      extracted_lr):
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
        model = plain_model(augmented_x, y, batch_size, num_class, extracted_lr, is_training)
    else:
        model = plain_model(x, y, batch_size, num_class, extracted_lr, is_training)

    sess.close()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print(f"training the plain model for epochs {epochs + w_epochs}")

    ## training the model using extracted data(from trained model incase of extraction) or model training from sratch.
    for epoch in range(epochs + w_epochs):
        if shuffle:
            np.random.shuffle(index)
            x_train = x_train[index]  # cant see used in this block.
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
                if epoch == 5 and batch == 0:
                    intermediate_inp = sess.run(model.prediction,
                                                {x: exclude_x_data[batch * batch_size: (batch + 1) * batch_size],
                                                 is_training: 0,
                                                 is_augment: 0})[penultimate_layer]
                    intermediate_inp1 = sess.run(model.prediction,
                                                 {x: target_data[batch * batch_size: (batch + 1) * batch_size],
                                                  is_training: 0,
                                                  is_augment: 0})[penultimate_layer]
                    intermediate_inp2 = sess.run(model.prediction,
                                                 {x: trigger[batch * batch_size: (batch + 1) * batch_size],
                                                  is_training: 0,
                                                  is_augment: 0})[penultimate_layer]
                    intermediate_inp_final = np.concatenate([intermediate_inp, intermediate_inp1, intermediate_inp2], 0)
                    new_x = intermediate_inp_final.reshape(intermediate_inp_final.shape[0], -1)

                    pca_and_plot(new_x, [intermediate_inp.reshape(intermediate_inp.shape[0], -1),
                                         intermediate_inp1.reshape(intermediate_inp1.shape[0], -1),
                                         intermediate_inp2.reshape(intermediate_inp2.shape[0], -1)],
                                 type_model="baseline", dataset=dataset, time="between_train",
                                 penultimate_layer=penultimate_layer, distrib=distrib)

    if process == "Baseline":
        penultimate_layer = 2
        batch = 0

        intermediate_inp = \
        sess.run(model.prediction, {x: exclude_x_data[batch * batch_size: (batch + 1) * batch_size], is_training: 0,
                                    is_augment: 0})[penultimate_layer]
        intermediate_inp1 = \
        sess.run(model.prediction, {x: target_data[batch * batch_size: (batch + 1) * batch_size], is_training: 0,
                                    is_augment: 0})[penultimate_layer]
        intermediate_inp2 = \
        sess.run(model.prediction, {x: trigger[batch * batch_size: (batch + 1) * batch_size], is_training: 0,
                                    is_augment: 0})[penultimate_layer]
        intermediate_inp_final = np.concatenate([intermediate_inp, intermediate_inp1, intermediate_inp2], 0)

        new_x = intermediate_inp_final.reshape(intermediate_inp_final.shape[0], -1)

        pca_and_plot(new_x, [intermediate_inp.reshape(intermediate_inp.shape[0], -1),
                             intermediate_inp1.reshape(intermediate_inp1.shape[0], -1),
                             intermediate_inp2.reshape(intermediate_inp2.shape[0], -1)], type_model="baseline",
                     dataset=dataset, time="end_train", penultimate_layer=penultimate_layer, distrib=distrib)

        ## activation visualization
        first_conv_legitimate = \
        sess.run(model.prediction, {x: exclude_x_data[batch * batch_size: (batch + 1) * batch_size], is_training: 0,
                                    is_augment: 0})[0]
        first_conv_watermark = \
        sess.run(model.prediction, {x: trigger[batch * batch_size: (batch + 1) * batch_size], is_training: 0,
                                    is_augment: 0})[0]

        second_conv_legitimate = \
        sess.run(model.prediction, {x: exclude_x_data[batch * batch_size: (batch + 1) * batch_size], is_training: 0,
                                    is_augment: 0})[1]
        second_conv_watermark = \
        sess.run(model.prediction, {x: trigger[batch * batch_size: (batch + 1) * batch_size], is_training: 0,
                                    is_augment: 0})[1]

        fc_legitimate = \
        sess.run(model.prediction, {x: exclude_x_data[batch * batch_size: (batch + 1) * batch_size], is_training: 0,
                                    is_augment: 0})[2]
        fc_watermark = \
        sess.run(model.prediction, {x: trigger[batch * batch_size: (batch + 1) * batch_size], is_training: 0,
                                    is_augment: 0})[2]

        # plt.subplot(10,1)
        plot_activation(first_conv_legitimate, first_conv_watermark, second_conv_legitimate, second_conv_watermark,
                        fc_legitimate, fc_watermark, type_model="baseline", dataset=dataset, distrib=distrib)

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
            model, trigger[batch * half_batch_size: (batch + 1) * half_batch_size], watermark_target, sess, batch_size,
            num_class, is_training, is_augment, x, y))
    watermark_acc = np.average(watermark_acc_list)
    if verbose:
        print(f"{process} Model || validation accuracy: {1 - error},"
              f" watermark success: {watermark_acc}")

    return model




