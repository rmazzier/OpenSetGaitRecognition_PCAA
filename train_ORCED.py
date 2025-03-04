import os
import pickle
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import wandb
import itertools
from pytorch_metric_learning import miners, losses

# from semisup_aae import (
from models import ORCEDDecoder, ORCEDEncoder, GaussianMeanLearner
from tqdm import tqdm
import matplotlib.pyplot as plt

import constants
from datasets import MSRadarDataset, SPLIT
from utils import SeqChamferLoss, save_model, CG_kl_divergence


def train_ORCED(config=constants.CONFIG):
    # Save a copy of the config file as a pickle file
    os.makedirs(f"models/{config['MODEL_NAME']}", exist_ok=True)
    config_path = os.path.join("models", config["MODEL_NAME"], "config.pkl")
    with open(config_path, "wb") as f:
        pickle.dump(config, f)

    # ----- DEFINE DATA AND MODELS -----
    # Mines triplet according to https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.pdf
    triplet_miner = miners.MultiSimilarityMiner()

    chamfer_loss = SeqChamferLoss()
    ce_loss = torch.nn.CrossEntropyLoss()
    triplet_loss = losses.TripletMarginLoss(margin=config["TRIPLET_MARGIN"])

    # Initialize generator and style_discriminator
    encoder = (
        ORCEDEncoder(n_out_labels=len(config["TRAIN_CLASSES"]))
        .to(constants.DEVICE)
        .float()
    )
    decoder = ORCEDDecoder().to(constants.DEVICE).float()
    mean_learner = (
        GaussianMeanLearner(n_in_labels=len(config["TRAIN_CLASSES"]))
        .to(constants.DEVICE)
        .float()
    )

    # Configure data loader
    radar_dataset_train = MSRadarDataset(
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

    radar_dataset_valid = MSRadarDataset(
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

    radar_dataset_unseen = MSRadarDataset(
        SPLIT.UNSEEN,
        subsample_factor=config["SUBSAMPLE_FACTOR"],
    )

    dataloader_unseen = torch.utils.data.DataLoader(
        radar_dataset_unseen,
        batch_size=config["BATCH_SIZE"],
        drop_last=True,
        shuffle=False,
        num_workers=0,
    )

    # Optimizers
    # The network parameters are updated with a stochastic gradient descent (SGD) optimizer,
    # with a learning rate of 0.001, which decreases to 0.0001 after 60 epochs of training
    optimizer = torch.optim.Adam(
        itertools.chain(encoder.parameters(),
                        decoder.parameters(), mean_learner.parameters()),
        lr=config["LR"],
        betas=(config["B1"], config["B1"]),
    )

    # optimizer = torch.optim.SGD(
    #     itertools.chain(encoder.parameters(), decoder.parameters()),
    #     lr=config["LR"],
    # )

    # ----------
    #  Training
    # ----------
    wandb.login()
    run = wandb.init(
        project=constants.WANDB_PROJECT,
        config=config,
        name=config["MODEL_NAME"],
        notes=config["NOTES"],
        reinit=True,
        mode=constants.WANDB_MODE,
    )

    best_valid_accuracy = 0
    KL_multiplier = 0

    for epoch in range(config["EPOCHS"]):
        KL_multiplier = epoch / config["EPOCHS"]
        # KL_multiplier = 1

        # Set to training mode
        encoder.train()
        decoder.train()
        mean_learner.train()

        rec_losses = []
        kl_losses = []
        trip_losses = []
        sup_losses = []
        tot_sup_losses = []

        ys = []
        y_hats = []

        for i, (pcs, gt_labels) in enumerate(dataloader_train):

            # Configure input
            pcs = pcs.to(constants.DEVICE)
            gt_labels = gt_labels.to(constants.DEVICE)

            # Encoder forward pass
            preds_logits, sup_fvs, vae_mu, vae_logvar = encoder(pcs)
            rec_pcs = decoder(sup_fvs)

            oh_labels = F.one_hot(
                gt_labels, num_classes=len(config["TRAIN_CLASSES"])
            ).float()
            mu_gts = mean_learner(oh_labels)

            # Reconstruction Loss
            rec_loss = chamfer_loss(rec_pcs, pcs)

            # Classification Loss
            sup_loss = ce_loss(preds_logits, gt_labels)

            # Triplet Loss
            normalized_sup_fvs = F.normalize(sup_fvs, p=2, dim=1)
            hard_pairs = triplet_miner(normalized_sup_fvs, gt_labels)
            trip_loss = triplet_loss(normalized_sup_fvs, gt_labels, hard_pairs)

            # KL Divergence Loss
            kl_loss = CG_kl_divergence(vae_mu, vae_logvar, mu_gts)

            rec_loss = config["REC_W"] * rec_loss
            sup_loss = config["CE_W"] * sup_loss
            trip_loss = config["TRIPLET_W"] * trip_loss
            kl_loss = config["KL_W"] * kl_loss * KL_multiplier
            sup_losses.append(sup_loss.item())
            rec_losses.append(rec_loss.item())
            trip_losses.append(trip_loss.item())
            kl_losses.append(kl_loss.item())

            # Total Loss
            tot_loss = rec_loss + sup_loss + trip_loss + kl_loss
            tot_sup_losses.append(tot_loss.item())

            norm_preds = torch.nn.Softmax(dim=1)(preds_logits)
            index_preds = torch.argmax(norm_preds, dim=1)
            y_hats.append(index_preds.cpu().numpy())
            ys.append(gt_labels.cpu().numpy())

            tot_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print(
                f"[Epoch {epoch}/{config['EPOCHS']}] "
                + f"[Batch {i}/{len(dataloader_train)}] "
                + f"[Chamfer Loss: {rec_loss.item():.4f}] "
                + f"[KL Loss: {kl_loss.item():.4f}] "
                + f"[C.E. Loss: {sup_loss.item():.4f}] "
                + f"[Triplet Loss: {trip_loss.item():.4f}] ",
                # end="\r",
            )

        # ---- EVALUATION TIME! --------
        print("evaluation time")
        encoder.eval()
        decoder.eval()
        mean_learner.eval()

        valid_ys = []
        valid_y_hats = []

        valid_rec_losses = []
        valid_ce_losses = []

        # Validation Set Evaluation (Accuracy, Rec Loss, CE Loss)
        with torch.no_grad():
            for i, (valid_pc, valid_gt_labels) in tqdm(enumerate(dataloader_valid)):
                valid_pc = valid_pc.to(constants.DEVICE)
                valid_gt_labels = valid_gt_labels.to(constants.DEVICE)

                valid_preds, sup_fv, vae_mu, vae_logvar = encoder(valid_pc)
                valid_rec_pc = decoder(sup_fv)

                valid_g_loss = chamfer_loss(valid_rec_pc, valid_pc).item()
                valid_ce_loss = ce_loss(valid_preds, valid_gt_labels).item()

                norm_valid_preds = torch.nn.Softmax(dim=1)(valid_preds)
                index_valid_preds = torch.argmax(norm_valid_preds, dim=1)

                valid_y_hats.append(index_valid_preds.cpu().numpy())
                valid_ys.append(valid_gt_labels.cpu().numpy())

                valid_g_loss = config["REC_W"] * valid_g_loss
                valid_ce_loss = config["CE_W"] * valid_ce_loss
                valid_rec_losses.append(valid_g_loss)
                valid_ce_losses.append(valid_ce_loss)
        # Log quantities
        train_accuracy = np.mean(np.concatenate(ys) == np.concatenate(y_hats))
        valid_accuracy = np.mean(
            np.concatenate(valid_ys) == np.concatenate(valid_y_hats)
        )

        wandb.log(
            {
                "Reconstruction Loss Train": np.mean(rec_losses),
                "Reconstruction Loss Valid": np.mean(valid_rec_losses),
                "Cross Entropy Loss Train": np.mean(sup_losses),
                "Cross Entropy Loss Valid": np.mean(valid_ce_losses),
                "Triplet Loss": np.mean(trip_losses),
                "KL Loss": np.mean(kl_losses),
                "Total Loss Train": np.mean(tot_sup_losses),
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
                encoder_path = os.path.join(
                    "models", config["MODEL_NAME"], f"{config['MODEL_NAME']}_E.pt"
                )
                decoder_path = os.path.join(
                    "models", config["MODEL_NAME"], f"{config['MODEL_NAME']}_G.pt"
                )

                mean_learner_path = os.path.join(
                    "models", config["MODEL_NAME"], f"{config['MODEL_NAME']}_ML.pt"
                )

                save_model(encoder, encoder_path)
                save_model(decoder, decoder_path)
                save_model(mean_learner, mean_learner_path)

    run.finish()

    pass


if __name__ == "__main__":
    from utils import openness

    n_training_classes = [2, 4, 6, 8]
    n_tests = 5
    all_classes = list(range(len(MSRadarDataset.label_dict)))
    model_name_base = "ORCED_"

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

            model_name = f"{model_name_base}.{n_tr}.{i+1}"

            config = constants.CONFIG
            config["MODEL_NAME"] = model_name
            config["TRAIN_CLASSES"] = random_train_classes
            config["Openness"] = openness(n_tr, len(MSRadarDataset.label_dict))
            config["NOTES"] = "Debugging"
            config["SUBSAMPLE_FACTOR"] = 1.0

            # ORCED Hyperparams
            config["TRIPLET_W"] = 1
            config["CE_W"] = 1
            config["REC_W"] = 1
            config["KL_W"] = 1
            config["TRIPLET_MARGIN"] = 0.5

            MSRadarDataset.generate_splits(
                train_classes=config["TRAIN_CLASSES"], seed=0, safe_mode=False
            )
            train_ORCED(config)
