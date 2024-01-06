import os
import os
import random
from datetime import datetime
from typing import List

import cv2
import numpy as np
import scripts.filter as watermark_filter
import torch
import torch.nn as nn
import torch.utils.data as data
from scripts import environment, experiment
from tqdm import tqdm
# tqdm.notebook.tqdm
from utils import config_helper
from utils import logger

now = datetime.now().strftime("%d-%m-%Y")

random.seed(42)

log = logger.Logger(prefix=">>>")


def get_shapes(model: nn.Module, test_set: data.DataLoader, device_string) -> (torch.Size, List[torch.Size]):
    """Returns input and latent sizes. This is set from the model itself."""

    model.eval()
    with torch.set_grad_enabled(False):
        for (inputs, yreal) in test_set:
            if device_string == "cuda":
                inputs, yreal = inputs.cuda(), yreal.cuda()

            ypred, latents = model(inputs) ## the latent model will return the ypred (logits) and latents
            watermark_shape = inputs[0].cpu().shape
            latents_shapes = [torch.Size([l.cpu().shape[1]]) for l in latents]
            break

    return watermark_shape, latents_shapes


def compare_distributions(
    model: nn.Module, test_set: data.DataLoader,
    wf: watermark_filter.WatermarkFilter,
    wf_latents: List[watermark_filter.WatermarkFilter], device_string) -> List[List]:
    
    with_wm_orig = 0
    without_wm_orig = 0

    latent_n = len(wf_latents)
    latent_batches = [[] for _ in range(latent_n)]

    ## this will be again for 2 letent reprsentation (20 and 512) sizes.
    with_without = [
    {
        "with_wm_latent": 0,
        "without_wm_latent": 0
    } 
    for _ in range(latent_n)]

    with torch.no_grad():
        for (inputs, _) in tqdm(test_set, unit="images", desc="Watermark Filter", leave=True, ascii=True):

            if device_string == "cuda":
                inputs = inputs.cuda()

            model.eval()
            _, latents = model(inputs)
            inputs = inputs.cpu()


            for x in inputs:
                if wf.is_watermark(x): ## inputs which are watermarked from the test set
                    with_wm_orig += 1
                else:
                    without_wm_orig += 1

            for i in range(latent_n):
                lat_repr = latents[i].cpu()
                # print(lat_repr.shape)
                latent_batches[i].append(lat_repr) ## each i (element) is the list, appending for each input (latent reprsentation from the model).
                
                for x in lat_repr:
                    if wf_latents[i].is_watermark(x):
                        with_without[i]["with_wm_latent"] += 1
                    else:
                        with_without[i]["without_wm_latent"] += 1 ## with_without : it is a list of 2 elements (beacause contains results for 2 latent representation) and each element conatins count of with watermark latent and without watermark latent.

    log.info("Watermarked: {}".format(with_wm_orig))
    log.info("Not watermarked: {}".format(without_wm_orig))
    log.info("Ratio: {}".format(with_wm_orig * 100 / without_wm_orig))

    for i in range(latent_n):
        log.info("Watermarked latent: {}".format(with_without[i]["with_wm_latent"]))
        log.info("Not watermarked latent: {}".format(with_without[i]["without_wm_latent"]))
        log.info("Ratio latent: {}".format(with_without[i]["with_wm_latent"] * 100 / with_without[i]["without_wm_latent"]))

    
    return latent_batches


def perturb(img, e, device_string="cpu", min_pixel=-1., max_pixel=1.):
    """Adding noise to the original image"""

    r = max_pixel - min_pixel
    b = r * torch.rand(img.shape)
    b += min_pixel
    noise = e * b
    if device_string == "cuda":
        noise = noise.cuda()

    return torch.clamp(img + noise, min_pixel, max_pixel)


def flatten(list_of_batches):
    flat = []
    for batch in list_of_batches: ## 10 times as mentioned in above cell.
        # print(batch.shape)
        for x in batch: ## each batch has shape (1024,200), or (784,200) | (1024, 10) or (784,10)
            # print(x.shape)
            flat.append(x) ## this will flatten 1024 elements 9 time and 784 which gives 1024*9 + 784 = 10000 and each of shape 200.
    return flat


def median_featurize(tensor_vector, medians):
    for idx, v in enumerate(medians):
        tensor_vector[idx] = 0 if tensor_vector[idx] <= v else 1
    
    return tensor_vector


def create_dist(latent_flat):
    ## latent_flat: it is a one latent tensor which wraps tensor in a list.
    l = latent_flat[0].shape[0] ## this will return 200 or 10.
    latent_dist = [[] for _ in range(l)]
    print(len(latent_dist))
    
    for single_lat in tqdm(latent_flat): ## this will be iterarted 10000 times
        for i in range(l):
            # print(single_lat[i])
            latent_dist[i].append(single_lat[i]) ## each index contains list of 10k length because appending to the index 10k times.
        
    return latent_dist


def shuffle_arr(arr, seed):
        """
        Shuffle the given array using seeded Fisher-Yates. The
            returned array is different from the given array to
            ensure that there are no false positives
        :param arr: np.ndarray
        :param seed: int; seed for the random number generator
        :return: np.ndarray; shuffled array
        """
        # print(arr)
        if len(arr) <= 1:
            raise ValueError("Error: array to be shuffled must have more than 1 element")

        new_arr = arr.copy()
        random.seed(seed)
        while np.all(new_arr == arr):
            for i in range(len(arr) - 1, 0, -1):
                j = random.randint(0, i)
                new_arr[i], new_arr[j] = new_arr[j], new_arr[i]
        return new_arr

def another_label(real_label: int, number_of_classes: int, seed) -> int:

    random.seed(seed)
    new_label = real_label
    # print(new_label)
    while new_label == real_label:
        new_label = random.randint(0, number_of_classes - 1)
    return new_label


def unnormalize(tensor, mean, std):
    """Reverses the normalization on a tensor."""
    tensor_clone = tensor.clone()
    for t, m, s in zip(tensor_clone, mean, std):
        t.mul_(s).add_(m)
    return tensor_clone


def do_mapping(
    model: nn.Module,
    test_set: data.DataLoader,
    wf_latent: watermark_filter.WatermarkFilter,
    medians: List,
    idx,
    eps_test,
    device_string,
    file1,
    image_save_path,
    type_unormalization):

    matching = 0
    not_matching = 0
    matching_and_same_label = 0 ## mapping succeeds
    matching_and_diff_label = 0 ## adversary wrongfully discards samples regardless of mapping
    not_matching_and_same_label = 0 ## mapping fails
    not_matching_and_diff_label = 0 ## adversary wrongfully discards samples regardless of mapping
    to_wm_cnt = 0

    new_img_per_orig = 10

    inputs = []
    perturbed = []

    with torch.no_grad():
        for (inputs, _) in tqdm(test_set, unit="images", desc="Watermark Filter", leave=True, ascii=True):

            print("Device", device_string)
            if device_string == "cuda":
                inputs = inputs.cuda()

            model.eval()

            ypred, latents = model(inputs)
            # _, predicted = torch.max(ypred.data, 1)
            lats = latents[idx]
            
            i = 0
            for x, l, yp in zip(inputs, lats, ypred):
                # perturbed = perturb(x, eps_test)

                assert len(l.shape) == 1
                to_wm = wf_latent.is_watermark(median_featurize(l.cpu(), medians))

                if to_wm:
                    to_wm_cnt += 1
                    shuffle_seed = wf_latent.label_watermark(median_featurize(l.cpu(), medians))
                    yp = shuffle_arr(yp.detach().cpu().numpy(), shuffle_seed)
                    # _, yp = torch.max(yp,1)
                    yp = np.argmax(yp)

                else:
                    #  _, yp = torch.max(yp.data, 1)
                     yp = np.argmax(yp.detach().cpu().numpy())
                    
                i +=1

                # input_display = x

                # if type_unormalization == "mnist": 
                        
                #     input_display = (input_display * 255)
                        
                # elif type_unormalization == "cifar_low":
                #         input_display = unnormalize(input_display, mean = [0.5, ], std = [0.5, ])

                # elif type_unormalization == "cifar_high":
                #     input_display = unnormalize(input_display, mean =  [0.485, 0.456, 0.406], std  =  [0.229, 0.224, 0.225])

                # input_display = input_display.cpu().numpy().copy()

                # input_display = np.transpose(input_display, (1,2,0))

                # plt.imshow(input_display, cmap='gray')  # or another colormap
                # plt.axis('off')  # Turn off axis
                    
                # plt.savefig(os.path.join(image_save_path, "_".join([str(i), "orig_w", str(to_wm) ,".svg"])), format='svg', bbox_inches='tight')
                
                for j in range(new_img_per_orig):
                    input_star = perturb(x, eps_test, device_string)

                    # plt.imshow(input_star.cpu().numpy().reshape((28,28,1)))
                    # plt.show()

                    ypred_star, lat_star = model(input_star.unsqueeze(0))
                    # _, predicted_star = torch.max(ypred_star.data, 1)
                    # predicted_star = predicted_star.squeeze()
                    # predicted_star = np.argmax(ypred_star.detach().cpu().numpy())


                    lat_star = lat_star[idx].squeeze(0)
                    assert len(lat_star.shape) == 1
                    to_wm_star = wf_latent.is_watermark(median_featurize(lat_star.cpu(), medians)) ## if the sample is watermarked, then have to check that watermark sample needs to be watermarked.

                    if to_wm_star:
                        shuffle_seed = wf_latent.label_watermark(median_featurize(lat_star.cpu(), medians))
                        # print(ypred_star)
                        # print(ypred_star.detach().cpu().numpy())
                        predicted_star = shuffle_arr(ypred_star.detach().cpu().numpy()[0], shuffle_seed)
                        # _, yp = torch.max(yp,1)
                        predicted_star = np.argmax(predicted_star)

                    predicted_star = np.argmax(ypred_star.detach().cpu().numpy())

                    if i < 20 and j < 5:

                        input_star_display = input_star

                        # if type_unormalization == "mnist": 
                        
                        #     input_star_display = (input_star_display * 255)
                        
                        # elif type_unormalization == "cifar_low":
                        #     input_star_display = unnormalize(input_star_display, mean = [0.5, ], std = [0.5, ])

                        # elif type_unormalization == "cifar_high":
                        #     input_star_display = unnormalize(input_star_display, mean =  [0.485, 0.456, 0.406], std  =  [0.229, 0.224, 0.225])

                        # input_star_display = input_star_display.cpu().numpy().copy()

                        # input_star_display = np.transpose(input_star_display, (1,2,0))
                        # plt.imshow(input_star_display, cmap='gray')  # or another colormap
                        # plt.axis('off')  # Turn off axis
                        # plt.savefig(os.path.join(image_save_path, "_".join([str(i), str(j), "orig_w", str(to_wm), "perturb_w", str(to_wm_star), "pred", str(yp), str(predicted_star) ,".svg"])), format='svg', bbox_inches='tight')
                        # print(input_star_display.shape)
                        # cv2.imwrite(os.path.join(image_save_path, "_".join([str(i), str(j), "orig_w", str(to_wm), "perturb_w", str(to_wm_star), "pred", str(yp), str(predicted_star) ,".png"])), input_star_display)

                    if to_wm_star == to_wm:
                        matching += 1
                        if yp == predicted_star:
                            matching_and_same_label += 1
                        else:
                            matching_and_diff_label += 1
                    else:
                        not_matching += 1
                        if yp == predicted_star:
                            not_matching_and_same_label += 1
                        else:
                            not_matching_and_diff_label += 1
                            
    log.info("to wm: {}".format(to_wm_cnt))
    file1.write("to wm {}\n".format(to_wm_cnt))
    log.info("matching: {} same label {} diff label {}".format(
        matching, matching_and_same_label, matching_and_diff_label))
    
    file1.write("matching:  {} same label {} diff label {}\n".format(
        matching, matching_and_same_label, matching_and_diff_label))
    # log.info("matching: {} same label {} diff label {}".format(
    #     matching, matching_and_same_label *100 /matching, matching_and_diff_label * 100/matching))
    log.info("not matching: {} same label {} diff label {}".format(
        not_matching, not_matching_and_same_label, not_matching_and_diff_label))
    file1.write("not matching: {} same label {} diff label {}\n".format(
        not_matching, not_matching_and_same_label, not_matching_and_diff_label))
    log.info(f"matching and same label {matching_and_same_label *100 / (matching+not_matching)} and matching and diff label {matching_and_diff_label * 100/ (matching+not_matching)}".format())
    file1.write("matching and same label {} and matching and diff label {}\n".format(matching_and_same_label *100 / (matching+not_matching), matching_and_diff_label * 100/ (matching+not_matching)))
    log.info("same label {}".format( ((not_matching_and_same_label + matching_and_same_label)/ (matching + not_matching)) *100))
    file1.write("same label {}\n".format( ((not_matching_and_same_label + matching_and_same_label)/ (matching + not_matching)) *100))

    log.info("diff label {}".format( ((matching_and_diff_label + not_matching_and_diff_label)/ (matching + not_matching)) *100))
    file1.write("diff label {}\n".format( ((matching_and_diff_label + not_matching_and_diff_label)/ (matching + not_matching)) *100))


def main(config_path):

    config = config_helper.load_config(config_path)
    env = environment.prepare_environment(config)
    agent = experiment.ExperimentTraining(env)

    # victim_path = "../models/original_18-09-2023/CIFAR10_BASE_2_victim_cifar_base_2.pt"

    config_helper.print_config(config)
    # log.info("Victim model path: {}.".format(victim_path))

    victim_model = env.training_ops.victim_model

    if config["VICTIM"]["model_architecture"] == "CIFAR10_BASE_2_LATENT":
        type_unormalization = "cifar_low"
    elif config["VICTIM"]["model_architecture"] == "CIFAR10_HIGH_CAPACITY_LATENT":
        type_unormalization = "cifar_high"
    elif config["VICTIM"]["model_architecture"] == "MNIST_L2_Latent":
        type_unormalization = "mnist"
        

    device_string = "cuda" if torch.cuda.is_available() else "cpu"
    # device_string = ""
    device = torch.device(device_string)
    log.info("Using device: {}".format(device_string))

    victim_model = victim_model.to(device=device)

    train_set = env.training_ops.training_loader
    test_set = env.test_ops.test_loader

    # _ = agent.test_model(victim_model, with_latent=True)

    #  Determine size of the watermark filter
    watermark_shape, watermark_latent_shapes = get_shapes(victim_model, test_set, device_string)

    log.info("Input shape: {}".format(watermark_shape))
    for latent_shape in watermark_latent_shapes:
        log.info("Latent shape: {}".format(latent_shape))

    
    key = watermark_filter.default_key(256)
    ## filtering both watermark and watermark latents
    wf = watermark_filter.WatermarkFilter(key, watermark_shape, precision=16, probability=(5/1000))
    wf_latents = [
        watermark_filter.WatermarkFilter(key, latent_shape, precision=16, probability=(50/1000))
        for latent_shape in watermark_latent_shapes] ## list of watermark filter object.
    

    # Compare the distribution in the input space (image) to distribution of the latent representation
    ## returning the latent distribution of inputs from compare distributions function.
    lat = compare_distributions(victim_model, test_set, wf, wf_latents, device_string) ## returns of lentgth 10 because with batch size 1024, it is returning 10 no. of batches.

    lat_flat = [flatten(list_of_batches) for list_of_batches in lat] ## lat_flat has 2  elements, 1 element contains list of 10k elements with 200 each shape and 2 element contains list of 10k elements with 10 each shape.

    lat_dists = [create_dist(single_flat_list) for single_flat_list in lat_flat]

    medians_for_lat = [] ## should be of length 2
    for shape, lat_dist in zip(watermark_latent_shapes, lat_dists): ## watermark_latent_shapes: 200 and 10 here, 2 latent_lists each has sublists of shape 200 and 10.
        medians_for_single = [] ## should be of length 200 or 10
        
        for dist in lat_dist: ## iterating over first latent (200 shape) and then second latent (10 shape) and each element in (200 or 10 shape) is 10k vector.
            # print(len(dist))
            # print(dist[0].shape)
            d = np.asarray(dist) ## should be of 10k vector
            # print(d.shape)
            median = np.median(d)
            medians_for_single.append(median)

        medians_for_lat.append(medians_for_single) ## total of length 2 and of lenth 200 and 10.

    RESULTS_PATH = f"../results/mapping_original_{now}"
    LOSS_Acc_FOLDER = "losses_acc"
    PERTURBED_IMAGES = "perturbed_images"

    

    if not os.path.exists(os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER)):
        os.makedirs(os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER))



    file1 = open(os.path.join(RESULTS_PATH, LOSS_Acc_FOLDER, "_".join((config["DEFAULT"]["dataset_name"], str(config["VICTIM"]["model_architecture"]), config["VICTIM"]["model_name"].split('/')[-1].split(".")[0] + "_logs.txt"))), "w")

    for eps in [0.2, 0.1, 0.09, 0.075, 0.05]: 
        print("---------------------------------------------------")
        print("+++ with eps: {}".format(eps))
        file1.write("+++ with eps: {}".format(eps))

        image_save_path = os.path.join(RESULTS_PATH, PERTURBED_IMAGES, config["DEFAULT"]["dataset_name"], str(eps))

        if not os.path.exists(image_save_path):
            os.makedirs(image_save_path)
        
        for idx, wf in enumerate(wf_latents): ## wf_latents: filtered watermark latents
            medians = medians_for_lat[idx] ## return the median list of of that index (200, 10)
            print("\nlatent size: {}".format(len(medians)))
            file1.write("\nlatent size: {}\n".format(len(medians)))
            do_mapping(
                victim_model, ## same victim model with the latent. z
                test_set, 
                wf, ## latent watermark
                medians,
                idx,
                eps,
                device_string,
                file1,
                image_save_path, type_unormalization)
            
            
if __name__ == "__main__":
    # main("../configurations/mapping/mapping-cifar-base.ini")
    main("../configurations/mapping/mapping-cifar-rn34.ini")
    # main("../configurations/mapping/mapping-mnist-l5.ini")



