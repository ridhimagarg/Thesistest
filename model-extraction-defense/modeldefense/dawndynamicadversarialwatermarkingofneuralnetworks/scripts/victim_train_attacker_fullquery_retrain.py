# Authors: Sebastian Szyller, Buse Gul Atli
# Copyright 2020 Secure Systems Group, Aalto University, https://ssg.aalto.fi
# Modified by: @Ridhima Garg

"""
@author: Ridhima Garg

Introduction:
    This file contains the implementation to train (load) the victim model and also to perform the attack(training the attcker model in full white-setting)
"""

import argparse
import configparser
import datetime
import os
import pickle
from datetime import datetime
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import torch
from scripts import environment
from scripts import experiment
from scripts import models
from scripts import score
from utils import config_helper
from utils import logger

log = logger.Logger(prefix=">>>")


now = datetime.now().strftime("%d-%m-%Y")

DATA_PATH = "../data"
LOSS_FOLDER = "losses"


def main(config: configparser.ConfigParser) -> None:

    #print(os.getcwd())
    ## preparing the environment using the configuration files.
    env = environment.prepare_environment(config)
    ## preparing the agent for the training and testing the models.
    agent = experiment.ExperimentTraining(env)

    ## we were previously trying to work for object detection as well
    if env.problem_statement == "OBJECTDETECTION":
        print("hello")

        ## Training or loading victim model ---------------------- ##
        if env.victim_retrain:
            victim_model = env.training_ops.victim_model
            victim_model.load("yolov8n.pt")
            victim_model.train(data="coco.yaml", imgsz=680, epochs=env.training_ops.epochs, batch = env.batch_size, name="yolov8n_v8_50e")
        else:
            victim_model = env.training_ops.victim_model
            victim_model.load("yolov8n.pt")

        _ = victim_model.val()

        # print(metrics)

        ##---------------------- Training attacker --------------------- ##
        if env.attacker_retrain:
            attacker_model = env.watermark_ops.attacker_model
            attacker_model.train(data="attacker_train.yaml", imgsz=680, epochs=2, batch = env.batch_size, name="yolov8n_v8_attacker")
            attacker_model.train(data="attacker_watermark.yaml", imgsz=680, epochs=2, batch = env.batch_size, name="yolov8n_v8_attacker")

        _ = attacker_model.val(env.watermark_ops.training_loader)
        _ = attacker_model.val(env.watermark_ops.watermark_loader)



    ## for classification task
    else:
        ## if in the configuration file we have decided to retraing the victim model or we already have the trained model then we can directly load it.
        if env.victim_retrain:

            RESULTS_PATH = f"../results/original_{now}"
            MODEL_PATH = f"../models/original_{now}"

            if not os.path.exists(os.path.join(RESULTS_PATH, LOSS_FOLDER)):
                os.makedirs(os.path.join(RESULTS_PATH, LOSS_FOLDER))

            if not os.path.exists(MODEL_PATH):
                os.makedirs(MODEL_PATH)
            
            best_victim_model, victim_model, scores, ground_truth_logit, epochs = agent.train_victim(log_interval=1000)

            file = open(os.path.join(RESULTS_PATH, LOSS_FOLDER, config["DEFAULT"]["dataset_name"] + "_" + config["VICTIM"]["model_architecture"] + "_logs.txt"), "w")

            for idx, (train_loss, train_acc, val_loss, val_acc) in enumerate(
                zip(scores["loss"], scores["accuracy"], scores["test_loss"], scores["test_average"])):
                file.write(
                    f'Epoch: {idx + 1}, Train Loss: {train_loss:.3f} Train Acc: {train_acc:.3f} Val Loss: {val_loss:.3f} Val Acc: {val_acc:.3f}')
                file.write("\n")

            ##------------------------------ Loss graph --------------------------------##

            plt.figure()
            plt.plot(list(range(epochs)), scores["loss"], label="Train loss_" + config["DEFAULT"]["dataset_name"],
                    marker='o',
                    color='tab:purple')
            plt.plot(list(range(epochs)), scores["test_loss"], label="Val loss_" + config["DEFAULT"]["dataset_name"], linestyle='--',
                    marker='o', color='tab:orange')
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()

            plt.savefig(os.path.join(RESULTS_PATH, LOSS_FOLDER, config["DEFAULT"]["dataset_name"] + "Original_Loss.png" ))

            ## -------------------------------- Accuracy graph --------------------------------##

            plt.figure()
            plt.plot(list(range(epochs)), scores["accuracy"], label="Train acc_" + config["DEFAULT"]["dataset_name"],
                    marker='o',
                    color='tab:purple')
            plt.plot(list(range(epochs)), scores["test_average"], label="Val acc_" + config["DEFAULT"]["dataset_name"], linestyle='--',
                    marker='o', color='tab:orange')
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend()

            plt.savefig(os.path.join(RESULTS_PATH, LOSS_FOLDER, config["DEFAULT"]["dataset_name"] + "Original_Acc.png" ))


            victim_model_path = os.path.join(MODEL_PATH, config["VICTIM"]["model_architecture"] + "_" + config["VICTIM"]["model_name"])
            victim_model_path_best = os.path.join(MODEL_PATH, config["VICTIM"]["model_architecture"] + "_" + config["VICTIM"]["model_name_best"])

            models.save_state(victim_model, victim_model_path)
            models.save_state(best_victim_model, victim_model_path_best)
        else:
            victim_model = env.training_ops.victim_model

        ## testing the victim model model with the dataset for which it is trained with (data and other things are directly fetched by the enviroment setup in the starting of this function itself.)
        _ = agent.test_model(victim_model)

        ## training the attacker model
        experiment_training(
            env,
            agent,
            config["DEFAULT"]["scores_save_path"] +
            config["ATTACKER"]["model_name"])

def experiment_training(env, training_agent: experiment.ExperimentTraining, path_body: str) -> None:
    if env.attacker_retrain:
        attacker_model, scores, watermark_logit, ground_truth_logit, full_watermark = training_agent.train_attacker(log_interval=1000)
        date = datetime.datetime.today().strftime('%Y-%m-%d')

        save_scores(
            scores,
            path_body + date)

        save_queries(
            watermark_logit,
            path_body + "_watermark_" + date)

        save_queries(
            ground_truth_logit,
            path_body + "_ground_truth_" + date)

        save_full_watermark(
            full_watermark,
            path_body + "_watermark_full_" + date)

        models.save_state(attacker_model, env.attacker_model_path)
    else:
        attacker_model = env.watermark_ops.attacker_model

    _ = training_agent.test_model(attacker_model)  # test set to to check valid accuracy
    _ = training_agent.test_watermark(attacker_model)  # watermarking set to check persistence


def save_scores(scores_dict: Dict[str, List[score.Score]], file_path: str) -> None:
    with open(file_path + '.pkl', 'wb') as f:
        pickle.dump(scores_dict, f, pickle.HIGHEST_PROTOCOL)


def save_queries(watermark: List[Tuple[torch.Tensor, int]], file_path: str) -> None:
    with open(file_path + '.pkl', 'wb') as f:
        pickle.dump(watermark, f, pickle.HIGHEST_PROTOCOL)


def save_full_watermark(full_watermark: List[Tuple[torch.FloatTensor, int]], file_path: str) -> None:
    with open(file_path + '.pkl', 'wb') as f:
        pickle.dump(full_watermark, f, pickle.HIGHEST_PROTOCOL)


def handle_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default="../configurations/perfect/cifar-to-cifar-ws250-base_2.ini",
        help="Configuration file for the experiment.")

    args = parser.parse_args()
    # if args.config_file is None:
    #     raise ValueError("Configuration file must be provided.")

    return args


if __name__ == "__main__":

    """
    pass the configuration file in the command line.
    all configuration files are prsesent in the configuration folder.
    to be used from the perfect folder.
    """

    args = handle_args()
    print(args.config_file)

    ## thid is the helper to load the configuration file which is saved as .ini instead of yaml
    ## for most of the places we have used .yaml itself but whereever we hace .ini, we used this config loader.

    config = config_helper.load_config(args.config_file)
    config_helper.print_config(config)

    main(config)
