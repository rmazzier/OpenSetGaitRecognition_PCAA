import os
import pickle
import torch
import wandb
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn.functional as F
import itertools

import constants
from datasets import SPLIT, MSRadarDatasetMD
from models import (
    LGM_PCNet, LGM_Loss, LGM_Net
)
from utils import (
    save_model,
    sample_distant_points,
)


def train_LGMNet_MD(config=constants.CONFIG, wandb_mode="disabled"):
    # Save a copy of the config file as a pickle file
    os.makedirs(f"models/{config['MODEL_NAME']}", exist_ok=True)
    config_path = os.path.join("models", config["MODEL_NAME"], "config.pkl")
    with open(config_path, "wb") as f:
        pickle.dump(config, f)

    # ----- DEFINE DATA AND MODELS -----
    embedding_dim = 16
    lgm_criterion = LGM_Loss(feature_dim=embedding_dim, num_classes=len(config["TRAIN_CLASSES"]), margin=3, lambda_reg=0.5).to(constants.DEVICE)


    # model = LGM_PCNet(enc_embedding_dim=128, final_embedding_dim=embedding_dim, num_classes=len(config["TRAIN_CLASSES"])).to(constants.DEVICE)
    model = LGM_Net(embedding_dim=embedding_dim).to(constants.DEVICE)

    # Configure data loader
    radar_dataset_train = MSRadarDatasetMD(
        SPLIT.TRAIN,
        subsample_factor=config["SUBSAMPLE_FACTOR"],
    )

    dataloader_train = torch.utils.data.DataLoader(
        radar_dataset_train,
        batch_size=config["BATCH_SIZE"],
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )

    radar_dataset_valid = MSRadarDatasetMD(
        SPLIT.VALID,
        subsample_factor=config["SUBSAMPLE_FACTOR"],
        # train_classes=config["TRAIN_CLASSES"],
    )

    dataloader_valid = torch.utils.data.DataLoader(
        radar_dataset_valid,
        batch_size=config["BATCH_SIZE"],
        drop_last=True,
        shuffle=False,
        num_workers=0,
    )

    # radar_dataset_unseen = MSRadarDataset(
    #     SPLIT.UNSEEN,
    #     subsample_factor=config["SUBSAMPLE_FACTOR"],
    #     # train_classes=config["TRAIN_CLASSES"],
    # )

    # Optimizers
    # optimizer = torch.optim.Adam(
    #     itertools.chain(model.parameters(), lgm_criterion.parameters()),
    #     lr=config["LR"],
    #     betas=(config["B1"], config["B2"]),
    # )

    optimizer = torch.optim.Adam([
        {'params': model.feature_extractor.parameters(), 'lr':0.0001},
        {'params': itertools.chain(model.fc.parameters(), lgm_criterion.parameters()), 'lr':0.002}
        
    ])

    # Define optimizer with different parameter groups
#     optimizer = optim.Adam([
#     {'params': first_layer_params, 'lr': 1e-3},
#     {'params': second_layer_params, 'lr': 1e-4}
# ])

    # ----------
    #  Training
    # ----------
    wandb.login()
    run = wandb.init(
        project="Semisup_AAE",
        config=config,
        name=config["MODEL_NAME"],
        notes=config["NOTES"],
        reinit=True,
        mode=wandb_mode,
    )

    best_valid_accuracy = 0

    for epoch in range(config["EPOCHS"]):

        # Set to training mode
        model.train()

        ce_losses = []
        lkd_losses = []
        lgm_losses = []

        ys = []
        y_hats = []

        for i, (mds, labels) in enumerate(dataloader_train):

            # Configure input
            mds = mds.to(constants.DEVICE)
            labels = labels.to(constants.DEVICE)

            # Model forward pass
            embeddings = model(mds)
            
            optimizer.zero_grad()

            # Loss
            lgm_loss, ce_loss, lkd_loss = lgm_criterion(embeddings, labels)
            lgm_losses.append(lgm_loss.item())

            # I explicitely log the two components of the lgm_loss to plot them in wandb
            ce_losses.append(ce_loss.item())
            lkd_losses.append(lkd_loss.item())

            lgm_loss.backward()
            optimizer.step()

            #log accuracies
            posteriors = lgm_criterion.compute_posteriors(embeddings)
            index_preds = torch.argmax(posteriors, dim=1)
            y_hats.append(index_preds.cpu().numpy())
            ys.append(labels.cpu().numpy())

            print(
                f"[Epoch {epoch}/{config['EPOCHS']}] "
                + f"[Batch {i}/{len(dataloader_train)}] "
                + f"[LGM_Loss: {lgm_loss.item():.4f}] "
                + f"[CE_Loss_component: {ce_loss.item():.4f}] "
                + f"[lkd_Loss_component: {lkd_loss.item():.4f}] ",
                # end="\r",
            )

        # ---- EVALUATION --------
        print("Evaluation")
        model.eval()

        valid_ys = []
        valid_y_hats = []

        valid_lgm_losses = []
        valid_ce_losses = []
        valid_lkd_losses = []

        # Validation Set Evaluation (Accuracy, Rec Loss, CE Loss)
        with torch.no_grad():
            for i, (valid_pc, valid_labels) in tqdm(enumerate(dataloader_valid)):
                valid_pc = valid_pc.to(constants.DEVICE)
                valid_labels = valid_labels.to(constants.DEVICE)

                valid_embeddings = model(valid_pc)

                # Loss
                valid_lgm_loss, valid_ce_loss, valid_lkd_loss = lgm_criterion(valid_embeddings, valid_labels)
                
                valid_lgm_losses.append(valid_lgm_loss.item())
                valid_ce_losses.append(valid_ce_loss.item())
                valid_lkd_losses.append(valid_lkd_loss.item())

                valid_posteriors = lgm_criterion.compute_posteriors(valid_embeddings)
                index_valid_preds = torch.argmax(valid_posteriors, dim=1)

                valid_y_hats.append(index_valid_preds.cpu().numpy())
                valid_ys.append(valid_labels.cpu().numpy())

        # Log quantities
        train_accuracy = np.mean(np.concatenate(ys) == np.concatenate(y_hats))
        valid_accuracy = np.mean(
            np.concatenate(valid_ys) == np.concatenate(valid_y_hats)
        )

        wandb.log(
            {
                "LGM Loss Train": np.mean(lgm_losses),
                "LGM Loss Valid": np.mean(valid_lgm_losses),
                "Cross Entropy Loss Train": np.mean(ce_losses),
                "Cross Entropy Loss Valid": np.mean(valid_ce_losses),
                "Likelihood Loss Train": np.mean(lkd_losses),
                "Likelihood Loss Valid": np.mean(valid_lkd_losses),
                "Train Accuracy": train_accuracy,
                "Valid Accuracy": valid_accuracy,
            }
        )

        # CHECKPOINTS AND PLOT LOGS
        if (epoch) % config["CHECKPOINT_FREQUENCY"] == 0:

            print("~Checkpoint Reached! Evaluating...~")
            if valid_accuracy > best_valid_accuracy:
                best_valid_accuracy = valid_accuracy
                print("New best model found! Saving...")

                # Save model checkpoints
                model_path = os.path.join(
                    "models", config["MODEL_NAME"], f"{config['MODEL_NAME']}.pt"
                )

                save_model(model, model_path)

                class_means_path = os.path.join(
                    "models", config["MODEL_NAME"], f"{config['MODEL_NAME']}_class_means.pt"
                )
                save_model(lgm_criterion, class_means_path)


    run.finish()

if __name__ == "__main__":
    from utils import openness

    model_name_base = "LGM_Net_MD"
    # n_training_classes = [2, 4, 6, 8]
    n_training_classes = [6]

    # n_tests = 5
    n_tests = 1

    all_classes = list(range(len(MSRadarDatasetMD.label_dict)))

    splits_seed = 0
    rng = np.random.default_rng(splits_seed)


    # Iterate on number of training classes
    for n_tr in n_training_classes:
        selected_classes_sets = []
        # Iterate the different tests over the same n_tr
        for i in range(n_tests):
            # This while loop is done to make sure we don't pick the same subset of classes two times
            while True:
                random_train_classes = rng.choice(len(MSRadarDatasetMD.label_dict), n_tr, replace=False)
                random_train_classes = list(sorted(random_train_classes))
                if random_train_classes not in selected_classes_sets:
                    selected_classes_sets.append(random_train_classes)
                    break

            model_name = f"{model_name_base}.{n_tr}.{i+1}"

            config = constants.CONFIG
            config["MODEL_NAME"] = model_name
            # config["TRAIN_CLASSES"] = random_train_classes
            config["TRAIN_CLASSES"]=[3, 4, 5, 6, 8, 9]
            config["EPOCHS"] = 50
    
            config["Openness"] = openness(n_tr, len(MSRadarDatasetMD.label_dict))

            # FIXME Uncomment for real runs
            MSRadarDatasetMD.generate_splits(
                train_classes=config["TRAIN_CLASSES"], split_seed=0, safe_mode=False
            )

            # if i != 0:
            train_LGMNet_MD(config, wandb_mode="online")
