import matplotlib.pyplot as plt
import numpy as np
from utils import validate_watermark
import os

RESULTS_FOLDER = "results"
RESULTS_SUB_FOLDER_TRIGGERS = "triggers_images"


def trigger_generation(model, maxiter, w_num_batch, trigger, batch_size, half_batch_size, watermark_target, sess, num_class, is_training, is_augment, threshold, target_data, w_label, temperatures, w_lr, x, y, t, w):
    print(w_num_batch)
    step_list = np.zeros([w_num_batch])
    for batch in range(w_num_batch):
        print("batch")
        current_trigger = trigger[batch * half_batch_size: (batch + 1) * half_batch_size]

        ## Plotting the triggers with the original data
        fig = plt.figure()
        plt.subplot(2,1, 1)
        plt.axis("off")
        plt.imshow(current_trigger[1][:,:,0], cmap="gray_r")
        plt.title("Original")

        for epoch in range(maxiter):
            while validate_watermark(model, current_trigger, watermark_target, sess, batch_size, num_class, is_training, is_augment, x, y) > threshold and step_list[batch] < 50:
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

        if not os.path.exists(RESULTS_FOLDER):
            os.mkdir(RESULTS_FOLDER)
        if not os.path.exists(os.path.join(RESULTS_FOLDER, RESULTS_SUB_FOLDER_TRIGGERS)):
            os.mkdir(os.path.join(RESULTS_FOLDER, RESULTS_SUB_FOLDER_TRIGGERS))
        plt.savefig(os.path.join(RESULTS_FOLDER, RESULTS_SUB_FOLDER_TRIGGERS, "trigger_" + str(batch) + ".png"))

    return trigger
