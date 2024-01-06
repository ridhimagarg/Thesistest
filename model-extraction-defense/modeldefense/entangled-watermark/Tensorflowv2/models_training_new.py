import os
import numpy as np
import matplotlib.pyplot as plt
from utils_new import validate_watermark, test_model, pca_and_plot
import tensorflow as tf
import models_new as md
# import mlflow
import keras
import seaborn as sns

sns.set_context("paper", font_scale=3.0, rc={"lines.linewidth": 2.0})

def ewe_train(model, train_dataset, trigger_dataset, target_dataset, test_dataset, model_save_path, w_epochs, num_batch, w_num_batch, n_w_ratio, factors, optimizer, watermark_target, num_class, batch_size, temp_lr, temperatures, dataset, exclude_x_data, exclude_y_data, results_path, loss_folder, file, images_save_middle_name, customer_model=None):

    """
    Training the EWE model for training data and trigger watermark data.


    Returns
    ---------

    Printing the accuracies
    returning the trained model.
    """

    loss_epoch_train = []
    loss_epoch_watermark = []
    loss_epoch_watermark_snnl = []

    for epoch in range(round((w_epochs * num_batch / w_num_batch))):

        print(f"At epoch {epoch} out of {round((w_epochs * num_batch / w_num_batch))}")
        j = 0
        normal = 0

        loss_batch_train = []
        loss_batch_watermark = []
        loss_batch_watermark_snnl = []
        for batch, ((x_batch_train, y_batch_train), trigger_batch_train, target_batch_train) in enumerate(zip(train_dataset, trigger_dataset, target_dataset)):

            ## --------------------------------- TRaining for train data ---------------------------
            if n_w_ratio >= 1:
                for i in range(int(n_w_ratio)):
                    if j >= num_batch:
                        j = 0

                    with tf.GradientTape() as tape:

                        w_0 = np.zeros([x_batch_train.shape[0]])

                        if customer_model:

                            intermediate_output = customer_model(x_batch_train)
                            prediction = model(intermediate_output)
                        
                        else:
                            prediction = model(x_batch_train)
                        loss_value, _ = md.combined_loss(prediction, y_batch_train, w_0, temperatures, factors)

                    loss_batch_train.append(loss_value)
                    print(f"Loss of train set at batch {batch} at epoch {epoch} is {loss_value}")
                    file.write(f"\nLoss of train set at batch {batch} at epoch {epoch} is {loss_value}")

                    grads = tape.gradient(loss_value, model.trainable_weights)
                    optimizer.apply_gradients(zip(grads, model.trainable_weights))

                    j += 1
                    normal += 1

            if n_w_ratio > 0 and n_w_ratio % 1 != 0 and n_w_ratio * batch >= j:
                if j >= num_batch:
                    j = 0

                with tf.GradientTape() as tape:

                    w_0 = np.zeros([x_batch_train.shape[0]])

                    if customer_model:

                            intermediate_output = customer_model(x_batch_train)
                            prediction = model(intermediate_output)
                        
                    else:
                        prediction = model(x_batch_train)

                    prediction = model(x_batch_train)
                    loss_value, _ = md.combined_loss(prediction, y_batch_train, w_0, temperatures, factors)

                loss_batch_train.append(loss_value)
                print(f"Loss of train set in another condition at batch {batch} at epoch {epoch} is {loss_value}")
                file.write(f"\nLoss of train set in another condition at batch {batch} at epoch {epoch} is {loss_value}")

                grads = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                j += 1
                normal += 1
            ##-------------------------------------------------------------------------------------------------------------------##

            ##------------------------------------------ Training triggers/watermark set --------------------------##

            with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
                target_batch_train = tf.cast(target_batch_train, tf.float32)
                trigger_batch_train = tf.cast(trigger_batch_train, tf.float32)
                x_batch = tf.concat([trigger_batch_train,
                                                target_batch_train], 0)
                    
                tape1.watch(x_batch)

                if customer_model:
                        intermediate_output = customer_model(x_batch)
                        prediction = model(intermediate_output)
                        # input_for_grad = intermediate_output

                else:
                    
                    prediction = model(x_batch)
                w_label = np.concatenate([np.ones(trigger_batch_train.shape[0]), np.zeros(target_batch_train.shape[0])], 0)

                trigger_label = np.zeros([x_batch.shape[0], num_class])
                trigger_label[:, watermark_target] = 1 

                loss_value, snnl = md.combined_loss(prediction, trigger_label, w_label, temperatures, factors)

                temperatures = tf.convert_to_tensor(temperatures, dtype=tf.float32)
                tape2.watch(temperatures)
                snnl_loss = md.snnl_loss(prediction, w_label, temperatures)
            
            loss_batch_watermark.append(loss_value)
            loss_batch_watermark_snnl.append(snnl)
            print(f"Loss of watermark set at {batch} at {epoch} is {loss_value}")
            file.write(f"Loss of watermark set at {batch} at {epoch} is {loss_value}")
            print(f"Snnl Loss of watermark set at {batch} at {epoch} is {snnl}")
            file.write(f"Snnl Loss of watermark set at {batch} at {epoch} is {snnl}")
            grads = tape1.gradient(loss_value, model.trainable_weights)
            temp_grad = tape2.gradient(snnl_loss, temperatures)
                    
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            temperatures -= temp_lr * temp_grad

             ##--------------------------------------------- VISUALIZATION -----------------------------------##


            ## performing pca and plotting in middle of the training.
            penultimate_layer = 2
            if epoch == 5 and batch==0:
                
                intermediate_inp = model(exclude_x_data[batch * batch_size: (batch + 1) * batch_size])[penultimate_layer].numpy()
                intermediate_inp1 = model(target_batch_train)[penultimate_layer].numpy()
                intermediate_inp2 = model(trigger_batch_train)[penultimate_layer].numpy()

                intermediate_inp_final = np.concatenate([intermediate_inp, intermediate_inp1, intermediate_inp2], 0)

                y_pca = exclude_y_data[batch * batch_size: (batch + 1) * batch_size]
                new_x = intermediate_inp_final.reshape(intermediate_inp_final.shape[0], -1)

                plot, legend = pca_and_plot(new_x, [intermediate_inp.reshape(intermediate_inp.shape[0], -1), y_pca, intermediate_inp1.reshape(intermediate_inp1.shape[0], -1), intermediate_inp2.reshape(intermediate_inp2.shape[0], -1)], results_path, type_model="ewe", dataset=dataset, time="between_train", penultimate_layer=penultimate_layer, image_save_path =os.path.join(results_path, loss_folder, dataset + images_save_middle_name , "layer" + str(penultimate_layer) + "_Entanglement_between_train.svg"), image_legend_save_path=os.path.join(results_path, loss_folder, dataset + images_save_middle_name , "layer" + str(penultimate_layer) + "legend_Entanglement_between_train.svg"))

                # plt.savefig(os.path.join(results_path, loss_folder, dataset + images_save_middle_name , "layer" + str(penultimate_layer) + "_Entanglement_between_train.svg"), bbox_inches='tight')
                # plt.savefig(os.path.join(results_path, loss_folder, dataset + images_save_middle_name , "layer" + str(penultimate_layer) + "legend_Entanglement_between_train.svg"), bbox_inches='tight')

            penultimate_layer = 2
            if epoch == (round((w_epochs * num_batch / w_num_batch))-1) and batch==0:
                intermediate_inp = model(exclude_x_data[batch * batch_size: (batch + 1) * batch_size])[penultimate_layer].numpy()
                intermediate_inp1 = model(target_batch_train)[penultimate_layer].numpy()
                intermediate_inp2 = model(trigger_batch_train)[penultimate_layer].numpy()

                intermediate_inp_final = np.concatenate([intermediate_inp, intermediate_inp1, intermediate_inp2], 0)

                y_pca = exclude_y_data[batch * batch_size: (batch + 1) * batch_size]
                new_x = intermediate_inp_final.reshape(intermediate_inp_final.shape[0], -1)

                plot, legend = pca_and_plot(new_x, [intermediate_inp.reshape(intermediate_inp.shape[0], -1), y_pca, intermediate_inp1.reshape(intermediate_inp1.shape[0], -1), intermediate_inp2.reshape(intermediate_inp2.shape[0], -1)], results_path, type_model="ewe", dataset=dataset, time="end_train", penultimate_layer=penultimate_layer, image_save_path =os.path.join(results_path, loss_folder, dataset + images_save_middle_name , "layer" + str(penultimate_layer) + "_Entanglement_end_train.svg"), image_legend_save_path=os.path.join(results_path, loss_folder, dataset + images_save_middle_name , "layer" + str(penultimate_layer) + "legend_Entanglement_end_train.svg"))

                # plt.savefig(os.path.join(results_path, loss_folder, dataset + images_save_middle_name , "layer" + str(penultimate_layer) + "_Entanglement_end_train.svg"), bbox_inches='tight')
                # plt.savefig(os.path.join(results_path, loss_folder, dataset + images_save_middle_name , "layer" + str(penultimate_layer) + "legend_Entanglement_end_train.svg"), bbox_inches='tight')




        loss_epoch_train.append(np.mean(loss_batch_train))
        loss_epoch_watermark.append(np.mean(loss_batch_watermark))
        loss_epoch_watermark_snnl.append(np.mean(loss_batch_watermark_snnl))

        file.write(f"\n\nTrain Loss at {epoch} is {loss_epoch_train[-1]}")
        file.write(f"\nWaztermark CE Loss at {epoch} is {loss_epoch_watermark[-1]}")
        file.write(f"\nWaztermark SNNL Loss at {epoch} is {loss_epoch_watermark_snnl[-1]}\n\n")
    
    ##--------------------------------------------- VISUALIZATION -----------------------------------##


        ## performing pca and plotting at the end of the trainin

    
    plt.figure()
    # plt.plot(list(range(round((w_epochs * num_batch / w_num_batch)))), loss_epoch_train, label="Train data cross entropy loss", linestyle='--', marker='o', color='tab:orange')

    sns.lineplot(x=list(range(round((w_epochs * num_batch / w_num_batch)))), y=loss_epoch_train,
                      ci=None, color="tab:orange", linestyle='--', marker='o', label="Train data loss", markersize=7)
    plt.xlabel("Epochs")
    plt.ylabel("CE loss")
    plt.tight_layout()
    plt.legend()

    if not os.path.exists(os.path.join(results_path, loss_folder)):
        os.makedirs(os.path.join(results_path, loss_folder))

    plt.savefig(os.path.join(results_path, loss_folder, dataset + images_save_middle_name , "CombineTrainlossEWE.svg"), bbox_inches='tight')

    plt.figure()
    # plt.plot(list(range(round((w_epochs * num_batch / w_num_batch)))), loss_epoch_watermark, label="Watermark data combined loss", linestyle='--', marker='o', color='tab:purple')
   
    sns.lineplot(x=list(range(round((w_epochs * num_batch / w_num_batch)))), y=loss_epoch_watermark,
                      ci=None, color="tab:purple", linestyle='--', marker='o', label="Watermark data combined loss", markersize=7)
    plt.xlabel("Epochs")
    plt.ylabel("CE+SNNL Loss")
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(results_path, loss_folder, dataset + images_save_middle_name , "CombineWatermarklossEWE.svg"), bbox_inches='tight')


    plt.figure()
    # plt.plot(list(range(round((w_epochs * num_batch / w_num_batch)))), loss_epoch_watermark_snnl, label="Watermark data SNNL loss", linestyle='--', marker='o', color='tab:purple')
    
    sns.lineplot(x=list(range(round((w_epochs * num_batch / w_num_batch)))), y=loss_epoch_watermark_snnl,
                      ci=None, color="tab:purple", linestyle='--', marker='o', label="Watermark data SNNL loss", markersize=7)
    
    plt.xlabel("Epochs")
    plt.ylabel("SNNL Loss")
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(results_path, loss_folder, dataset + images_save_middle_name , "SNNLWatermarklossEWE.svg"), bbox_inches='tight')

    # mlflow.log_artifact(os.path.join(results_path, loss_folder, dataset + "CombineTrainlossEWE.png"), "Combinedtrainlossewe")
    # mlflow.log_artifact(os.path.join(results_path, loss_folder, dataset + "CombineWatermarklossEWE.png"), "Combinedwatermarklossewe")
    # mlflow.log_artifact(os.path.join(results_path, loss_folder, dataset + "SNNLWatermarklossEWE.png"), "SNNLWatermarklossEWE")

    test_accuracy = test_model(model, test_dataset, "EWE (Victim)", num_class, customer_model, watermark_target=None)
    watermark_accuracy = test_model(model, trigger_dataset, "EWE (Victim)", num_class, customer_model, watermark_target=watermark_target)

    print("Test accuracy", test_accuracy)
    print("watermark accuracy", watermark_accuracy)

    file.write(f"\nEWE trained model test accuracy {test_accuracy}\n")
    file.write(f"\nEWE trained model watermark accuracy {watermark_accuracy}\n")

    # mlflow.log_metric("EWE Victim Test Acc", test_accuracy)
    # mlflow.log_metric("EWE Victim Watermark Acc", watermark_accuracy)

    model.save(model_save_path)

    return model

    # loaded_model = tf.keras.models.load_model('models/'+str(model_name))





def plain_model(model, model_type, train_dataset, test_dataset , extraction_flag, epochs, w_epochs, optimizer, num_class, trigger_dataset, watermark_target, target_dataset, model_path, dataset, results_path, loss_folder, file, images_save_middle_name, customer_model=None):

    loss_epoch = []
    for epoch in range(epochs + w_epochs):
        loss_batch = []
        for batch, (x_batch_extracted, y_batch_extracted) in enumerate(train_dataset):
            with tf.GradientTape() as tape:

                w_0 = np.zeros([x_batch_extracted.shape[0]])
                prediction = model(x_batch_extracted)

                y_batch_extracted = tf.cast(y_batch_extracted, tf.float32) ## it was true/false so type casted.

                # print("y batch", y_batch_extracted)

                loss_value = keras.losses.CategoricalCrossentropy()(y_batch_extracted, prediction)

            grads = tape.gradient(loss_value, model.trainable_weights)
            
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            loss_batch.append(loss_value)

        loss_epoch.append(np.mean(loss_batch))
        print(f"Loss of train set at epoch {epoch} is {np.mean(loss_batch)}")
        file.write(f"Loss of train set at epoch {epoch} is {np.mean(loss_batch)}\n")

        if not extraction_flag:

            loss_epoch = []
            for batch, trigger_batch_train in enumerate(trigger_dataset):
                with tf.GradientTape() as tape:
                    
                    prediction = model(trigger_batch_train)

                    trigger_label = np.zeros([trigger_batch_train.shape[0], num_class])
                    trigger_label[:, watermark_target] = 1

                    loss_value = md.ce_loss(prediction, trigger_label)

                grads = tape.gradient(loss_value, model.trainable_weights)
                
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

                loss_epoch.append(loss_value)

            print(f"Loss of trigger set at epoch {epoch} is {np.mean(loss_epoch)}")

    
    # plt.figure(figsize=(5,5))
    # plt.plot(list(range(epochs + w_epochs)), loss_epoch, label="Train data loss Extracted training", linestyle='--', marker='o', color='tab:orange')
    # plt.xlabel("epochs")
    # plt.ylabel("CE loss")
    # plt.legend()
    # plt.savefig(os.path.join(results_path, loss_folder, images_save_middle_name,  "CELossExtracted.png"))

    # mlflow.log_artifact(os.path.join(results_path, loss_folder, dataset + "CELossExtracted.png"), "CELossExtracted")
    
    

            

    test_accuracy = test_model(model, test_dataset, model_type, num_class, watermark_target=None)
    watermark_accuracy = test_model(model, trigger_dataset, model_type, num_class, watermark_target=watermark_target)

    print("Test accuracy", test_accuracy)
    file.write(f"\nTest accuracy {test_accuracy}\n")
    print("watermark accuracy", watermark_accuracy)
    file.write(f"\nWatermark accuracy {watermark_accuracy}\n")

    # mlflow.log_metric("EWE Extracted Test Acc", test_accuracy)
    # mlflow.log_metric("EWE Extracted Watermark Acc", watermark_accuracy)

    model.save(model_path)

    return model, test_accuracy, watermark_accuracy
        