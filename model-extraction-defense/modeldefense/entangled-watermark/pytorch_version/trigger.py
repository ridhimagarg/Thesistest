import matplotlib.pyplot as plt
import numpy as np
from utils import validate_watermark
import os
import torch as t

RESULTS_FOLDER = "results"
RESULTS_SUB_FOLDER_TRIGGERS = "triggers_images"


def trigger_generation(model, trainer_class, trigger_dl_whole, trigger_dl, target_dl, maxiter, w_lr, batch_size,
                       half_batch_size, temperatures, target_class, threshold, num_class):
    """
    Trigger generation algorithm
    """

    trigger = next(iter(trigger_dl_whole))[0].numpy()

    print("target dl")
    print(next(iter(target_dl)))

    for batch_idx, (trigger_input_label, target_input_label) in enumerate(zip(trigger_dl, target_dl)):

        current_trigger = trigger_input_label[0]

        # fig = plt.figure()
        # plt.subplot(2,1, 1)
        # plt.axis("off")
        # plt.imshow(trigger_input_label[0][1][0,:,:], cmap="gray_r")
        # plt.title("Original")

        step_list = 0
        for epoch in range(maxiter):

            while validate_watermark(model, trainer_class, current_trigger, target_class, batch_size,
                                     num_class) > threshold and step_list < 50:
                step_list += 1

                batch_data = t.concat((current_trigger, current_trigger))

                grad = trainer_class.ce_trigger(model, target_class, batch_data)

                current_trigger = t.clip(current_trigger - w_lr * t.sign(grad[:half_batch_size]), 0, 1)

            batch_data = t.concat((current_trigger, target_input_label[0]))

            # w_label = t.concat((t.ones(len(trigger_input_label)), t.zeros(len(half_batch_size))))

            w_label = t.concat((t.ones(trigger_input_label[0].shape[0]), t.zeros(target_input_label[0].shape[0])))

            predictions_list = model(batch_data)

            grad = trainer_class.snnl_trigger(model, temperatures, w_label, batch_data)

            current_trigger = t.clip(current_trigger + w_lr * t.sign(grad[:half_batch_size]), 0, 1)

        for i in range(5):
            batch_data = t.concat((current_trigger, current_trigger))

            # predictions_list = model(batch_data)

            grad = trainer_class.ce_trigger(model, target_class, batch_data)

            current_trigger = t.clip(current_trigger - w_lr * t.sign(grad[:half_batch_size]), 0, 1)

        trigger[batch_idx * half_batch_size: (batch_idx + 1) * half_batch_size] = current_trigger.detach().numpy()

        print(f"trigger at {batch_idx}")
        # plt.subplot(2,1, 2)
        # plt.axis("off")
        # plt.imshow(current_trigger.detach().numpy()[1][0,:,:], cmap="gray_r")
        # plt.title("With Trigger")

        # if not os.path.exists(RESULTS_FOLDER):
        #     os.mkdir(RESULTS_FOLDER)
        # if not os.path.exists(os.path.join(RESULTS_FOLDER, RESULTS_SUB_FOLDER_TRIGGERS)):
        #     os.mkdir(os.path.join(RESULTS_FOLDER, RESULTS_SUB_FOLDER_TRIGGERS))
        # plt.savefig(os.path.join(RESULTS_FOLDER, RESULTS_SUB_FOLDER_TRIGGERS, "trigger_" + str(batch_idx) + ".png"))

