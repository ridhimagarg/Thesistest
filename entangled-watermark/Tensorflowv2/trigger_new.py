    
import matplotlib.pyplot as plt
import numpy as np
from utils_new import validate_watermark
import tensorflow as tf
import os
import models_new as md

RESULTS_FOLDER = "results"
RESULTS_SUB_FOLDER_TRIGGERS = "triggers_images"



def trigger_generation(model, trigger, trigger_dataset, target_dataset, watermark_target, num_class, batch_size, threshold, maxiter, w_lr, w_num_batch, temperatures, customer_model=None):

    """
    Trigger generation algorithm
    """

    step_list = np.zeros([w_num_batch+2])
    trigger_image_saving_count = 0
    for batch, (trigger_batch_train, target_batch_train) in enumerate(zip(trigger_dataset, target_dataset)):
        current_trigger = trigger_batch_train

        # fig = plt.figure()
        # plt.subplot(2,1, 1)
        # plt.axis("off")
        # plt.imshow(current_trigger[1][:,:,0], cmap="gray_r")
        # plt.title("Original")

        for epoch in range(maxiter):

            while validate_watermark(model, current_trigger, watermark_target, num_class, customer_model) > threshold and step_list[batch] < 50:
                with tf.GradientTape() as tape:

                    x_batch = tf.concat([trigger_batch_train,
                                                    trigger_batch_train], 0)
                    
                    x_batch = tf.cast(x_batch, tf.float32)
                    
                    tape.watch(x_batch)
                    if customer_model:
                        intermediate_output = customer_model(x_batch)
                        prediction = model(intermediate_output)
                        # input_for_grad = intermediate_output

                    else:
                        
                        prediction = model(x_batch)
                        # input_for_grad = x_batch

                    pred_for_grad = tf.unstack(prediction[-1], axis=1)[watermark_target]
                    # tape.watch(pred_for_grad)

                ce_grad = tape.gradient(pred_for_grad, x_batch) 
                step_list[batch] += 1
                current_trigger = np.clip(current_trigger - w_lr * np.sign(ce_grad[:current_trigger.shape[0]]), 0, 1)

                with tf.GradientTape() as tape:
                    x_batch = tf.concat([tf.cast(current_trigger, tf.float32),
                                                tf.cast(target_batch_train, tf.float32)], 0)
                    
                    tape.watch(x_batch)

                    if customer_model:
                        intermediate_output = customer_model(x_batch)
                        prediction = model(intermediate_output)

                    else:
                        prediction = model(x_batch)

                    w_label = np.concatenate([np.ones(trigger_batch_train.shape[0]), np.zeros(target_batch_train.shape[0])], 0)
                    snnl_losses = md.snnl_loss(prediction, w_label, temperatures)
                    final_snnl_loss = snnl_losses[0] + snnl_losses[1] + snnl_losses[2]

            
                snnl_grad = tape.gradient(final_snnl_loss, x_batch)
                current_trigger = np.clip(current_trigger + w_lr * np.sign(snnl_grad[:current_trigger.shape[0]]), 0, 1)

        for i in range(5):

            with tf.GradientTape() as tape:

                x_batch = tf.concat([trigger_batch_train,
                                                trigger_batch_train], 0)
                
                x_batch = tf.cast(x_batch, tf.float32)
                tape.watch(x_batch)

                if customer_model:
                    intermediate_output = customer_model(x_batch)
                    prediction = model(intermediate_output)

                else:
                    prediction = model(x_batch)

                pred_for_grad = tf.unstack(prediction[-1], axis=1)[watermark_target]

            ce_grad = tape.gradient(pred_for_grad, x_batch)

            current_trigger = np.clip(current_trigger - w_lr * np.sign(ce_grad[:current_trigger.shape[0]]), 0, 1)
        trigger[batch * current_trigger.shape[0]: (batch + 1) * current_trigger.shape[0]] = current_trigger
        
        print("trigger")
        # plt.subplot(2,1, 2)
        # plt.axis("off")
        # plt.imshow(current_trigger[1][:,:,0], cmap="gray_r")
        # plt.title("With Trigger")
        # plt.savefig(os.path.join(RESULTS_FOLDER, RESULTS_SUB_FOLDER_TRIGGERS, "trigger_"+ str(batch)+ ".png"))

        # if trigger_image_saving_count < 10:
        #     mlflow.log_artifact(os.path.join(RESULTS_FOLDER, RESULTS_SUB_FOLDER_TRIGGERS, "trigger_"+ str(batch)+ ".png"), "TriggerImages")
        #     trigger_image_saving_count += 1


    return trigger