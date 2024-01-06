# Authors: Sebastian Szyller, Buse Gul Atli
# Copyright 2020 Secure Systems Group, Aalto University, https://ssg.aalto.fi
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import configparser
import datetime
import io
import pickle
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.utils.data as data
from scripts import models
from scripts import score
from tqdm import tqdm
from utils import config_helper
from utils import logger

random.seed(42)

log = logger.Logger(prefix=">>>")

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

class SimpleDataset(data.Dataset):
    def __init__(self, dataset: List[Tuple[Any, int]]) -> None:
        self.data, self.labels = zip(*dataset)
        self.count = len(self.labels)

    def __getitem__(self, index: int) -> (Any, int):
        return self.data[index], self.labels[index]

    def __len__(self) -> int:
        return self.count

def main(config: configparser.ConfigParser, model_path: str, watermark_path: str) -> None:

    #  Setup model architecture and load model from file.
    model = setup_model(
        config["DEFAULT"]["model_architecture"],
        model_path,
        config["DEFAULT"]["number_of_classes"])

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model = model.to(device=device)

    watermark_set = load_file(watermark_path)

    verification_save_path = config["DEFAULT"]["verification_save_path"]
    path = Path(verification_save_path)
    if not path.exists():
        log.warn(verification_save_path + " does not exist. Creating...")
        path.mkdir(parents=True, exist_ok=True)
        log.info(verification_save_path + " Created.")

    verification_results = verify_with_noise(model, watermark_set, device)

    date = datetime.datetime.today().strftime('%Y-%m-%d')
    path_body = verification_save_path + config["DEFAULT"]["model_name"]

    save_scores(
        verification_results,
        path_body + date)

def setup_model(model_architecture: str, model_path: str, number_of_classes: int) -> nn.Module:
    available_models = {
        "MNIST_L5": models.MNIST_L5,
        "CIFAR10_BASE": models.CIFAR10_BASE
    }

    model = available_models[model_architecture]()

    if model is None:
        log.error("Incorrect model architecture specified or architecture not available.")
        raise ValueError(model_architecture)

    models.load_state(model, model_path)

    return model


# def load_file(file_path: str) -> List[Tuple]:
#     with open(file_path, "rb") as f:
#         return pickle.load(f)
    
def load_file(file_path: str) -> Dict[str, Any]:
    with open(file_path, "rb") as f:
        return CPU_Unpickler(f).load()


def verify_with_noise(model: nn.Module, watermark_set: List, device: str) -> Dict[float, score.FloatScore]:
    noise_amount = [0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.75, 0.9]
    verification_results = {}

    log.info("Accuracy without noise:")
    _ = test_watermark(model, watermark_set, device)

    for eps in noise_amount:
        noisy_watermark = craft_noisy_watermark(watermark_set, eps, device)

        log.info("Testing with eps {}.".format(eps))
        watermark_float_score = test_watermark(model, noisy_watermark, device)
        verification_results[eps] = watermark_float_score

    return verification_results


def test_watermark(model: nn.Module, watermark_set: List, device: str) -> score.FloatScore:
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for (inputs, yreal) in tqdm(watermark_set, unit="images", desc="Testing watermark (average)", leave=True, ascii=True):
            if device == "cuda":
                inputs, yreal = inputs.cuda(), yreal.cuda()

            ypred = model(inputs)
            _, predicted = torch.max(ypred.data, 1)

            total += yreal.size(0)
            correct += (predicted == yreal).sum().item()

    accuracy = 100 * correct / total
    log.info("Accuracy of the network on the {} test images (average): {}".format(total, accuracy))
    return score.FloatScore(accuracy)

def craft_noisy_watermark(watermark_set: List, eps: float, device: str) -> List:
    new_watermark = []
    for batch_x, batch_y in watermark_set:

        batch_size = len(batch_x)
        for idx in range(batch_size):
            x = batch_x[idx]
            if device == "cuda":
                x = x.cuda()
            x_star = perturb_single(x, eps, device, min_pixel=torch.min(x).cpu().data.item(), max_pixel=torch.max(x).cpu().data.item())
            batch_x[idx] = x_star

        new_watermark.append((batch_x, batch_y))

    return new_watermark


def perturb_single(img: torch.FloatTensor, eps: float, device: str, min_pixel=-1., max_pixel=1.) -> torch.Tensor:
    r = max_pixel - min_pixel
    b = r * torch.rand(img.shape)
    b += min_pixel
    noise = eps * b
    if device == "cuda":
        noise = noise.cuda()

    return torch.clamp(img + noise, min_pixel, max_pixel)


def save_scores(pruning_results: Dict[float, score.FloatScore], file_path: str) -> None:
    with open(file_path + '.pkl', 'wb') as f:
        pickle.dump(pruning_results, f, pickle.HIGHEST_PROTOCOL)


def handle_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Configuration file for the experiment.")
    parser.add_argument(
        "--watermark",
        type=str,
        default=None,
        help="Path to the saved watermark Loader.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to the saved model.")
    args = parser.parse_args()

    if args.config_file is None:
        raise ValueError("Configuration file must be provided.")

    if args.watermark is None:
        raise ValueError("Watermark path must be provided.")

    if args.config_file is None:
        raise ValueError("Model path must be provided.")

    return args


if __name__ == "__main__":
    args = handle_args()
    config = config_helper.load_config(args.config_file)
    watermark_path = args.watermark
    model_path = args.model

    config_helper.print_config(config)
    log.info("Model path: {}.".format(model_path))
    log.info("Watermark path: {}".format(watermark_path))

    main(config, model_path, watermark_path)
