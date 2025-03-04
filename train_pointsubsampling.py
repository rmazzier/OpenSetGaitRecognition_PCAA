import os
import pickle
import itertools
import wandb
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

import constants

from inference_PCAA import CGAAE_inference, VARIATION
from PCAA_ablation import train_variant4
from datasets import MSRadarDataset, SPLIT
from utils import openness


if __name__ == "__main__":

    model_name_base = "PCAA_npts_V4_"
    n_training_classes = [2, 4, 6, 8]
    n_points_subs = [50, 70, 90, 110, 130, 150]
    n_tests = 5

    all_classes = list(range(len(MSRadarDataset.label_dict)))

    splits_seed = 0
    rng = np.random.default_rng(splits_seed)

    # Iterate on number of training classes
    for n_tr in n_training_classes:
        selected_classes_sets = []
        # Iterate the different tests over the same n_tr
        for i in range(n_tests):
            # This while loop is done to make sure we don't pick the same subset of classes
            # two times
            while True:
                random_train_classes = rng.choice(
                    len(MSRadarDataset.label_dict), n_tr, replace=False
                )
                random_train_classes = list(sorted(random_train_classes))
                if random_train_classes not in selected_classes_sets:
                    selected_classes_sets.append(random_train_classes)
                    break

            config = constants.CONFIG

            config["TRAIN_CLASSES"] = random_train_classes
            config["Openness"] = openness(n_tr, len(MSRadarDataset.label_dict))

            for n_points in n_points_subs:
                config["NMAX"] = n_points
                MSRadarDataset.generate_splits(
                    train_classes=config["TRAIN_CLASSES"], seed=0, safe_mode=False, nmax_points=n_points
                )

                # train
                model_name = f"{model_name_base}{n_points}.{n_tr}.{i+1}"
                config["MODEL_NAME"] = model_name
                config["NOTES"] = f"Runs with different number of points ({n_points}.{n_tr}.{i+1})"

                train_variant4(
                    config,
                    # wandb_mode="online",
                    wandb_mode="disabled",
                    proj_head_on_discriminator=False)

                # Inference
                CGAAE_inference(model_names=[model_name], ks=[
                                1, 2, 4, 6], variation=VARIATION.V4)
