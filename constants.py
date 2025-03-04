import torch
import os
from enum import Enum


class SPLIT(Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"
    UNSEEN = "unseen"


class SCENARIO(Enum):
    FREE_WALK = "free_walk"
    HANDS_IN_POCKETS = "hands_in_pockets"
    SMARTPHONE = "smartphone"


# PATHS
DATA_PATH = os.path.join(
    "..", "..", "radar_reid_pytorch", "data", "multi-scenario_dataset"
)

GEN_DATA_PATH = os.path.join("data", "generated_dataset")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# NB: If one of these 4 variables is modified, the dataset splits will need to be regenerated
NMAX = 150
NSTEPS = 30
CROP_STEP = 6
NFEATURES = 4
# ------------------------

# Network parameters
POINTNET_OUT_DIM = 1024
DTC_FILTERS = [16, 32, 64, 128, 256, 512]

SUP_LATENT_DIM = 32

DEC_MLP_SIZE = NSTEPS * NMAX * NFEATURES

# Training parameters
LR = 1e-4

# b1 and b2 are ADAM specific parameters
B1 = 0.9
B2 = 0.99


TRAIN_CLASSES = []
TRAIN_SCENARIOS = [SCENARIO.FREE_WALK,
                   SCENARIO.HANDS_IN_POCKETS, SCENARIO.SMARTPHONE]

BATCH_SIZE = 16
SUBSAMPLE_FACTOR = 1.0
EPOCHS = 50
CHECKPOINT_FREQUENCY = 5

# Gradient Penalty
GP_WEIGHT = 15

ADV_WEIGHT = 1

# for wandb config
WANDB_PROJECT = "PCAA"
WANDB_MODE = "online"
MODEL_NAME = ""
NOTES = ""

SUPERVISION_FREQUENCY = 1

# Setup dictionary with all hyperparameters
CONFIG = {
    "NMAX": NMAX,
    "NSTEPS": NSTEPS,
    "CROP_STEP": CROP_STEP,
    "NFEATURES": NFEATURES,
    "POINTNET_OUT_DIM": POINTNET_OUT_DIM,
    "DTC_FILTERS": DTC_FILTERS,
    "SUP_LATENT_DIM": SUP_LATENT_DIM,
    "DEC_MLP_SIZE": DEC_MLP_SIZE,
    "LR": LR,
    "B1": B1,
    "B2": B2,
    "TRAIN_CLASSES": TRAIN_CLASSES,
    "TRAIN_SCENARIOS": TRAIN_SCENARIOS,
    "SUBSAMPLE_FACTOR": SUBSAMPLE_FACTOR,
    "EPOCHS": EPOCHS,
    "BATCH_SIZE": BATCH_SIZE,
    "GP_WEIGHT": GP_WEIGHT,
    "ADV_WEIGHT": ADV_WEIGHT,
    "MODEL_NAME": MODEL_NAME,
    "NOTES": NOTES,
    "CHECKPOINT_FREQUENCY": CHECKPOINT_FREQUENCY,
    "SUPERVISION_FREQUENCY": SUPERVISION_FREQUENCY,
}
