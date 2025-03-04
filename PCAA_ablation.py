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
from models import (
    CGEncoder,
    CGDiscriminator,
    GaussianMeanLearner,
    CGDecoder,
)
from datasets import MSRadarDataset, SPLIT
from utils import openness
from train_AAE import train_CGAAE
from utils import (
    SeqChamferLoss,
    save_model,
    sample_distant_points,
)


def train_variant1(config, wandb_mode="online"):
    """
    Variant1 : CGAAE but learning the centroids instead of fixing them
    """
    # Save a copy of the config file as a pickle file
    os.makedirs(f"models/{config['MODEL_NAME']}", exist_ok=True)
    config_path = os.path.join("models", config["MODEL_NAME"], "config.pkl")
    with open(config_path, "wb") as f:
        pickle.dump(config, f)

    # ----- DEFINE DATA AND MODELS -----
    chamfer_loss = SeqChamferLoss()
    ce_loss = torch.nn.CrossEntropyLoss()

    nmax_points = config["NMAX"]

    # Initialize generator and style_discriminator
    encoder = (
        CGEncoder(n_out_labels=len(
            config["TRAIN_CLASSES"]), use_projection_head=True)
        .to(constants.DEVICE)
        .float()
    )
    decoder = CGDecoder(
        input_dim=config["SUP_LATENT_DIM"]*2, nmax_points=nmax_points).to(constants.DEVICE).float()
    decoder_projection_head = torch.nn.Sequential(
        torch.nn.Linear(config["SUP_LATENT_DIM"], config["SUP_LATENT_DIM"]*2),
        torch.nn.ELU(),
    ).to(constants.DEVICE).float()
    discriminator = (
        CGDiscriminator(len(config["TRAIN_CLASSES"])).to(
            constants.DEVICE).float()
    )
    mean_learner = (
        GaussianMeanLearner(len(config["TRAIN_CLASSES"])).to(
            constants.DEVICE).float()
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
        # train_classes=config["TRAIN_CLASSES"],
    )

    # Optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(encoder.parameters(
        ), decoder_projection_head.parameters(), decoder.parameters()),
        lr=config["LR"],
        betas=(config["B1"], config["B2"]),
    )

    optimizer_D = torch.optim.Adam(
        itertools.chain(mean_learner.parameters(), discriminator.parameters()),
        lr=config["LR"],
        betas=(config["B1"], config["B2"]),
    )

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

    for epoch in range(config["EPOCHS"]):

        # Set to training mode
        encoder.train()
        decoder.train()
        discriminator.train()
        mean_learner.train()
        decoder_projection_head.train()

        rec_losses = []
        d_losses = []
        sup_losses = []
        tot_sup_losses = []

        ys = []
        y_hats = []

        for i, (pcs, gt_labels) in enumerate(dataloader_train):

            # Configure input
            pcs = pcs.to(constants.DEVICE)
            gt_labels = gt_labels.to(constants.DEVICE)

            # Encoder forward pass
            out_labels, sup_fvs = encoder(pcs)

            with torch.no_grad():
                norm_preds = torch.nn.Softmax(dim=1)(out_labels)
                index_preds = torch.argmax(norm_preds, dim=1)
                y_hats.append(index_preds.cpu().numpy())
                ys.append(gt_labels.cpu().numpy())

            # ---------------------
            #  Train Style Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # one hot encoding of gt_labels
            oh_labels = F.one_hot(
                gt_labels, num_classes=len(config["TRAIN_CLASSES"])
            ).float()
            mus = mean_learner(oh_labels)

            # Sample noise as style_discriminator ground truth from the Prior Distribution
            z0 = (
                torch.from_numpy(
                    np.random.normal(
                        0.0,
                        1.0,
                        (pcs.shape[0], config["SUP_LATENT_DIM"]),
                    )
                )
                .to(constants.DEVICE)
                .float()
            )

            # Add the mean of the gaussian distribution of the class
            # z = z0 + mus
            z = Variable(z0 + mus)

            z.requires_grad = True

            real_logits = discriminator(z, oh_labels)
            fake_logits = discriminator(sup_fvs.detach(), oh_labels)

            # Gradient penalty term
            alphas = (
                torch.rand(size=(constants.BATCH_SIZE, 1))
                .repeat(1, config["SUP_LATENT_DIM"])
                .to(constants.DEVICE)
            )

            differences = sup_fvs.detach() - z
            interpolates = z + alphas * differences
            disc_interpolates = discriminator(interpolates, oh_labels)

            # Compute gradient of discriminator w.r.t input (i.e. interpolated codes)
            gradients = torch.autograd.grad(
                outputs=disc_interpolates,
                inputs=interpolates,
                grad_outputs=torch.ones_like(
                    disc_interpolates).to(constants.DEVICE),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            slopes = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-12)
            # test detaching also here
            gradient_penalty = ((slopes - 1) ** 2).mean()

            # WGAN Loss with Gradient Penalty
            d_loss = (
                torch.mean(fake_logits)
                - torch.mean(real_logits)
                + config["GP_WEIGHT"] * gradient_penalty
            )
            d_losses.append(d_loss.item())

            d_loss.backward()
            optimizer_D.step()

            optimizer_D.zero_grad()
            discriminator.zero_grad()

            # -----------------
            #  Train Generator (Encoder + Decoder)
            # -----------------
            optimizer_G.zero_grad()
            encoder.zero_grad()
            decoder.zero_grad()
            decoder_projection_head.zero_grad()

            # Apply decoder projection head
            projected_sup_fvs_decoder = decoder_projection_head(sup_fvs)
            rec_pcs = decoder(projected_sup_fvs_decoder)

            rec_loss = chamfer_loss(rec_pcs, pcs)
            rec_losses.append(rec_loss.item())

            synth_logits = discriminator(sup_fvs, oh_labels)

            loss_g = -torch.mean(synth_logits) * config["ADV_WEIGHT"]

            # -----------------
            # (Supervised step)
            # -----------------
            if i % config["SUPERVISION_FREQUENCY"] == 0:
                # Backward pass can be done on the discriminator loss

                # CROSS ENTROPY
                sup_loss = ce_loss(out_labels, gt_labels)
                sup_losses.append(sup_loss.item())

                tot_loss = rec_loss + loss_g + sup_loss
                tot_sup_losses.append(tot_loss.item())

            else:

                tot_loss = rec_loss + loss_g
                pass

            tot_loss.backward()
            optimizer_G.step()

            print(
                f"[Epoch {epoch}/{config['EPOCHS']}] "
                + f"[Batch {i}/{len(dataloader_train)}] "
                + f"[Chamfer Loss: {rec_loss.item():.4f}] "
                + f"[Discriminator Loss: {d_loss.item():.4f}] "
                + f"[C.E. Loss: {sup_loss.item():.4f}] ",
                # end="\r",
            )

        # ---- EVALUATION TIME! --------
        print("evaluation time")
        encoder.eval()
        decoder.eval()
        discriminator.eval()

        valid_ys = []
        valid_y_hats = []

        valid_rec_losses = []
        valid_ce_losses = []

        # Validation Set Evaluation (Accuracy, Rec Loss, CE Loss)
        with torch.no_grad():
            for i, (valid_pc, valid_gt_labels) in tqdm(enumerate(dataloader_valid)):
                valid_pc = valid_pc.to(constants.DEVICE)
                valid_gt_labels = valid_gt_labels.to(constants.DEVICE)

                valid_preds, sup_fv = encoder(valid_pc)
                projected_sup_fvs_decoder = decoder_projection_head(sup_fv)
                valid_rec_pc = decoder(projected_sup_fvs_decoder)

                valid_g_loss = chamfer_loss(valid_rec_pc, valid_pc).item()
                valid_ce_loss = ce_loss(valid_preds, valid_gt_labels).item()

                valid_rec_losses.append(valid_g_loss)
                valid_ce_losses.append(valid_ce_loss)

                norm_valid_preds = torch.nn.Softmax(dim=1)(valid_preds)
                index_valid_preds = torch.argmax(norm_valid_preds, dim=1)

                valid_y_hats.append(index_valid_preds.cpu().numpy())
                valid_ys.append(valid_gt_labels.cpu().numpy())

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
                "Discriminator Loss": np.mean(d_losses),
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
                discriminator_path = os.path.join(
                    "models", config["MODEL_NAME"], f"{config['MODEL_NAME']}_D.pt"
                )
                mean_learner_path = os.path.join(
                    "models", config["MODEL_NAME"], f"{config['MODEL_NAME']}_ML.pt"
                )
                decoder_projection_head_path = os.path.join(
                    "models", config["MODEL_NAME"], f"{config['MODEL_NAME']}_GPH.pt"
                )

                save_model(encoder, encoder_path)
                save_model(decoder, decoder_path)
                save_model(discriminator, discriminator_path)
                save_model(mean_learner, mean_learner_path)
                save_model(decoder_projection_head,
                           decoder_projection_head_path)

                # Also compute the learned centroids and save them to file
                oh_labels = F.one_hot(
                    torch.arange(0, len(config["TRAIN_CLASSES"])), num_classes=len(config["TRAIN_CLASSES"])).float().to("cuda")

                with torch.no_grad():
                    discriminator_means = mean_learner(oh_labels)

                torch.save(discriminator_means, os.path.join(
                    "models", config["MODEL_NAME"], "discriminator_means.pt"))

    run.finish()


def train_variant2(config, wandb_mode="online"):
    """
    Variant2 : No projection heads
    """

    # Just call train_AAE and setup manually supervision frequency = 1
    config["SUPERVISION_FREQUENCY"] = 1
    train_CGAAE(config)
    pass


def train_variant3(config, wandb_mode="online"):
    """
    Variant3 : CGAAE without the Decoder piece
    """
    # Save a copy of the config file as a pickle file
    os.makedirs(f"models/{config['MODEL_NAME']}", exist_ok=True)
    config_path = os.path.join("models", config["MODEL_NAME"], "config.pkl")
    with open(config_path, "wb") as f:
        pickle.dump(config, f)

    # ----- DEFINE DATA AND MODELS -----
    chamfer_loss = SeqChamferLoss()
    ce_loss = torch.nn.CrossEntropyLoss()

    # Initialize generator and style_discriminator
    encoder = (
        CGEncoder(n_out_labels=len(
            config["TRAIN_CLASSES"]), use_projection_head=False)
        .to(constants.DEVICE)
        .float()
    )

    discriminator = (
        CGDiscriminator(len(config["TRAIN_CLASSES"])).to(
            constants.DEVICE).float()
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

    # Optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(encoder.parameters()),
        lr=config["LR"],
        betas=(config["B1"], config["B1"]),
    )

    optimizer_D = torch.optim.Adam(
        itertools.chain(discriminator.parameters()),
        lr=config["LR"],
        betas=(config["B1"], config["B2"]),
    )

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

    # wandb.watch(encoder)
    # wandb.watch(discriminator)
    # wandb.watch(mean_learner)

    discriminator_means = (
        sample_distant_points(
            dimension=config["SUP_LATENT_DIM"],
            n=len(config["TRAIN_CLASSES"]),
            min_dist=10,
            sphere_radius=10,
        )
        .float()
        .to(constants.DEVICE)
    )

    # Save the discriminator means to the model folder
    torch.save(
        discriminator_means,
        os.path.join("models", config["MODEL_NAME"], "discriminator_means.pt"),
    )

    best_valid_accuracy = 0

    for epoch in range(config["EPOCHS"]):

        # Set to training mode
        encoder.train()
        discriminator.train()

        rec_losses = []
        d_losses = []
        sup_losses = []
        tot_sup_losses = []

        ys = []
        y_hats = []

        for i, (pcs, gt_labels) in enumerate(dataloader_train):

            # Configure input
            pcs = pcs.to(constants.DEVICE)
            gt_labels = gt_labels.to(constants.DEVICE)

            # Encoder forward pass
            out_labels, sup_fvs = encoder(pcs)

            with torch.no_grad():
                norm_preds = torch.nn.Softmax(dim=1)(out_labels)
                index_preds = torch.argmax(norm_preds, dim=1)
                y_hats.append(index_preds.cpu().numpy())
                ys.append(gt_labels.cpu().numpy())

            # ---------------------
            #  Train Style Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            # mean_learner.zero_grad()
            # discriminator.zero_grad()

            # one hot encoding of gt_labels
            oh_labels = F.one_hot(
                gt_labels, num_classes=len(config["TRAIN_CLASSES"])
            ).float()

            mus = torch.matmul(
                oh_labels.unsqueeze(1), discriminator_means.unsqueeze(0)
            ).squeeze()

            # Sample noise as style_discriminator ground truth from the Prior Distribution
            z0 = (
                torch.from_numpy(
                    np.random.normal(
                        0.0,
                        1.0,
                        (pcs.shape[0], config["SUP_LATENT_DIM"]),
                    )
                )
                .to(constants.DEVICE)
                .float()
            )

            # Add the mean of the gaussian distribution of the class
            # z = z0 + mus
            z = Variable(z0 + mus)

            # Normalize the feature vectors
            # z = Variable(z / torch.norm(z, dim=1).unsqueeze(1))
            z.requires_grad = True

            # Measure style_discriminator's ability to classify real from generated samples
            # In this variation we also supply the GT labels in input to the discriminator
            # to be concatenated to the current input.
            # This acts as a "mode switch"

            real_logits = discriminator(z, oh_labels)
            fake_logits = discriminator(sup_fvs.detach(), oh_labels)

            # Gradient penalty term
            alphas = (
                torch.rand(size=(constants.BATCH_SIZE, 1))
                .repeat(1, config["SUP_LATENT_DIM"])
                .to(constants.DEVICE)
            )

            differences = sup_fvs.detach() - z
            interpolates = z + alphas * differences
            disc_interpolates = discriminator(interpolates, oh_labels)

            # Compute gradient of discriminator w.r.t input (i.e. interpolated codes)
            gradients = torch.autograd.grad(
                outputs=disc_interpolates,
                inputs=interpolates,
                grad_outputs=torch.ones_like(
                    disc_interpolates).to(constants.DEVICE),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            slopes = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-12)
            # test detaching also here
            gradient_penalty = ((slopes - 1) ** 2).mean()

            # WGAN Loss with Gradient Penalty
            d_loss = (
                torch.mean(fake_logits)
                - torch.mean(real_logits)
                + config["GP_WEIGHT"] * gradient_penalty
            )
            d_losses.append(d_loss.item())

            d_loss.backward()
            optimizer_D.step()

            optimizer_D.zero_grad()
            discriminator.zero_grad()

            # print("After Discriminator backward step")
            # printGPUInfo()

            # -----------------
            #  Train Generator (Encoder + Decoder)
            # -----------------
            optimizer_G.zero_grad()
            encoder.zero_grad()

            synth_logits = discriminator(sup_fvs, oh_labels)

            # piece of adversarial loss here:
            # D(z) = likelihood that z comes from prior
            # -> we want the encoder to MAXimize this likelihood (to fool the discriminator)
            # -> MINimize: -D(z)
            loss_g = -torch.mean(synth_logits) * config["ADV_WEIGHT"]

            # -----------------
            # (Supervised step)
            # -----------------
            if i % config["SUPERVISION_FREQUENCY"] == 0:
                # Backward pass can be done on the discriminator loss

                # CROSS ENTROPY
                sup_loss = ce_loss(out_labels, gt_labels)
                sup_losses.append(sup_loss.item())

                tot_loss = loss_g + sup_loss
                tot_sup_losses.append(tot_loss.item())

            else:

                tot_loss = loss_g
                pass

            tot_loss.backward()
            optimizer_G.step()

            # print("After Generator backward step")
            # printGPUInfo()

            print(
                f"[Epoch {epoch}/{config['EPOCHS']}] "
                + f"[Batch {i}/{len(dataloader_train)}] "
                + f"[Discriminator Loss: {d_loss.item():.4f}] "
                + f"[C.E. Loss: {sup_loss.item():.4f}] ",
                # end="\r",
            )

        # ---- EVALUATION TIME! --------
        print("evaluation time")
        encoder.eval()
        discriminator.eval()

        valid_ys = []
        valid_y_hats = []

        valid_rec_losses = []
        valid_ce_losses = []

        # Validation Set Evaluation (Accuracy, Rec Loss, CE Loss)
        with torch.no_grad():
            for i, (valid_pc, valid_gt_labels) in tqdm(enumerate(dataloader_valid)):
                valid_pc = valid_pc.to(constants.DEVICE)
                valid_gt_labels = valid_gt_labels.to(constants.DEVICE)

                valid_preds, sup_fv = encoder(valid_pc)

                # print("After validation forward pass")
                # printGPUInfo()

                valid_ce_loss = ce_loss(valid_preds, valid_gt_labels).item()

                valid_ce_losses.append(valid_ce_loss)

                norm_valid_preds = torch.nn.Softmax(dim=1)(valid_preds)
                index_valid_preds = torch.argmax(norm_valid_preds, dim=1)

                valid_y_hats.append(index_valid_preds.cpu().numpy())
                valid_ys.append(valid_gt_labels.cpu().numpy())

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
                "Discriminator Loss": np.mean(d_losses),
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
                discriminator_path = os.path.join(
                    "models", config["MODEL_NAME"], f"{config['MODEL_NAME']}_D.pt"
                )
                mean_learner_path = os.path.join(
                    "models", config["MODEL_NAME"], f"{config['MODEL_NAME']}_ML.pt"
                )

                save_model(encoder, encoder_path)
                save_model(discriminator, discriminator_path)

                torch.save(discriminator_means, os.path.join(
                    "models", config["MODEL_NAME"], "discriminator_means.pt"))

    run.finish()
    pass


def train_variant4(config, wandb_mode="online", proj_head_on_discriminator=False):
    """
    Variant4 : This is the PCAA Model described in the paper
    """

    nmax_points = config["NMAX"]

    # Save a copy of the config file as a pickle file
    os.makedirs(f"models/{config['MODEL_NAME']}", exist_ok=True)
    config_path = os.path.join("models", config["MODEL_NAME"], "config.pkl")
    with open(config_path, "wb") as f:
        pickle.dump(config, f)

    # ----- DEFINE DATA AND MODELS -----
    chamfer_loss = SeqChamferLoss()
    ce_loss = torch.nn.CrossEntropyLoss()

    # Initialize generator and style_discriminator
    encoder = (
        CGEncoder(n_out_labels=len(
            config["TRAIN_CLASSES"]), use_projection_head=True, nmax_points=nmax_points)
        .to(constants.DEVICE)
        .float()
    )
    decoder = CGDecoder(
        input_dim=config["SUP_LATENT_DIM"]*2, nmax_points=nmax_points).to(constants.DEVICE).float()

    discriminator = (
        CGDiscriminator(len(config["TRAIN_CLASSES"])).to(
            constants.DEVICE).float()
    )

    decoder_projection_head = torch.nn.Sequential(
        torch.nn.Linear(config["SUP_LATENT_DIM"], config["SUP_LATENT_DIM"]*2),
        torch.nn.ELU(),
    ).to(constants.DEVICE).float()

    discriminator_projection_head = torch.nn.Sequential(
        torch.nn.Linear(config["SUP_LATENT_DIM"]*2, config["SUP_LATENT_DIM"]),
        torch.nn.ELU(),
    ).to(constants.DEVICE).float()

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

    # Optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(encoder.parameters(
        ), decoder_projection_head.parameters(), decoder.parameters()),
        lr=config["LR"],
        betas=(config["B1"], config["B2"]),
    )

    optimizer_D = torch.optim.Adam(
        itertools.chain(discriminator_projection_head.parameters(),
                        discriminator.parameters()),
        lr=config["LR"],
        betas=(config["B1"], config["B2"]),
    )

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

    discriminator_means = (
        sample_distant_points(
            dimension=config["SUP_LATENT_DIM"],
            n=len(config["TRAIN_CLASSES"]),
            min_dist=10,
            sphere_radius=10,
        )
        .float()
        .to(constants.DEVICE)
    )

    # Save the discriminator means to the model folder
    torch.save(
        discriminator_means,
        os.path.join("models", config["MODEL_NAME"], "discriminator_means.pt"),
    )

    best_valid_accuracy = 0

    for epoch in range(config["EPOCHS"]):

        # Set to training mode
        encoder.train()
        decoder.train()
        discriminator.train()

        rec_losses = []
        d_losses = []
        sup_losses = []
        tot_sup_losses = []

        ys = []
        y_hats = []

        for i, (pcs, gt_labels) in enumerate(dataloader_train):

            # Configure input
            pcs = pcs.to(constants.DEVICE)
            gt_labels = gt_labels.to(constants.DEVICE)

            # Encoder forward pass
            out_labels, sup_fvs = encoder(pcs)

            with torch.no_grad():
                norm_preds = torch.nn.Softmax(dim=1)(out_labels)
                index_preds = torch.argmax(norm_preds, dim=1)
                y_hats.append(index_preds.cpu().numpy())
                ys.append(gt_labels.cpu().numpy())

            # ---------------------
            #  Train Style Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            discriminator.zero_grad()
            discriminator_projection_head.zero_grad()

            # one hot encoding of gt_labels
            oh_labels = F.one_hot(
                gt_labels, num_classes=len(config["TRAIN_CLASSES"])
            ).float()

            # mus = mean_learner(oh_labels)
            mus = torch.matmul(
                oh_labels.unsqueeze(1), discriminator_means.unsqueeze(0)
            ).squeeze()

            # Sample noise as style_discriminator ground truth from the Prior Distribution
            z0 = (
                torch.from_numpy(
                    np.random.normal(
                        0.0,
                        1.0,
                        (pcs.shape[0], config["SUP_LATENT_DIM"]),
                    )
                )
                .to(constants.DEVICE)
                .float()
            )

            # Add the mean of the gaussian distribution of the class
            z = Variable(z0 + mus)

            # Normalize the feature vectors
            z.requires_grad = True

            if proj_head_on_discriminator:
                projected_sup_fvs_discriminator = discriminator_projection_head(
                    sup_fvs)
            else:
                projected_sup_fvs_discriminator = sup_fvs

            real_logits = discriminator(z, oh_labels)
            fake_logits = discriminator(
                projected_sup_fvs_discriminator.detach(), oh_labels)

            # Gradient penalty term
            alphas = (
                torch.rand(size=(constants.BATCH_SIZE, 1))
                .repeat(1, config["SUP_LATENT_DIM"])
                .to(constants.DEVICE)
            )

            differences = projected_sup_fvs_discriminator.detach() - z
            interpolates = z + alphas * differences
            disc_interpolates = discriminator(interpolates, oh_labels)

            # Compute gradient of discriminator w.r.t input (i.e. interpolated codes)
            gradients = torch.autograd.grad(
                outputs=disc_interpolates,
                inputs=interpolates,
                grad_outputs=torch.ones_like(
                    disc_interpolates).to(constants.DEVICE),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            slopes = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-12)
            # test detaching also here
            gradient_penalty = ((slopes - 1) ** 2).mean()

            # WGAN Loss with Gradient Penalty
            d_loss = (
                torch.mean(fake_logits)
                - torch.mean(real_logits)
                + config["GP_WEIGHT"] * gradient_penalty
            )
            d_losses.append(d_loss.item())

            d_loss.backward()
            optimizer_D.step()

            optimizer_D.zero_grad()
            discriminator.zero_grad()

            # -----------------
            #  Train Generator (Encoder + Decoder)
            # -----------------
            optimizer_G.zero_grad()
            encoder.zero_grad()
            decoder.zero_grad()
            decoder_projection_head.zero_grad()

            # Apply decoder projection head
            projected_sup_fvs_decoder = decoder_projection_head(sup_fvs)
            rec_pcs = decoder(projected_sup_fvs_decoder)

            rec_loss = chamfer_loss(rec_pcs, pcs)
            rec_losses.append(rec_loss.item())

            synth_logits = discriminator(
                projected_sup_fvs_discriminator, oh_labels)

            loss_g = -torch.mean(synth_logits) * config["ADV_WEIGHT"]

            # -----------------
            # (Supervised step)
            # -----------------
            if i % config["SUPERVISION_FREQUENCY"] == 0:
                # Backward pass can be done on the discriminator loss

                # CROSS ENTROPY
                sup_loss = ce_loss(out_labels, gt_labels)
                sup_losses.append(sup_loss.item())

                tot_loss = rec_loss + loss_g + sup_loss
                tot_sup_losses.append(tot_loss.item())

            else:

                tot_loss = rec_loss + loss_g
                pass

            tot_loss.backward()
            optimizer_G.step()

            print(
                f"[Epoch {epoch}/{config['EPOCHS']}] "
                + f"[Batch {i}/{len(dataloader_train)}] "
                + f"[Chamfer Loss: {rec_loss.item():.4f}] "
                + f"[Discriminator Loss: {d_loss.item():.4f}] "
                + f"[C.E. Loss: {sup_loss.item():.4f}] ",
                # end="\r",
            )

        # ---- EVALUATION TIME! --------
        print("evaluation time")
        encoder.eval()
        decoder.eval()
        discriminator.eval()

        valid_ys = []
        valid_y_hats = []

        valid_rec_losses = []
        valid_ce_losses = []

        # Validation Set Evaluation (Accuracy, Rec Loss, CE Loss)
        with torch.no_grad():
            for i, (valid_pc, valid_gt_labels) in tqdm(enumerate(dataloader_valid)):
                valid_pc = valid_pc.to(constants.DEVICE)
                valid_gt_labels = valid_gt_labels.to(constants.DEVICE)

                valid_preds, sup_fvs = encoder(valid_pc)
                projected_sup_fvs_decoder = decoder_projection_head(sup_fvs)
                valid_rec_pc = decoder(projected_sup_fvs_decoder)

                valid_g_loss = chamfer_loss(valid_rec_pc, valid_pc).item()
                valid_ce_loss = ce_loss(valid_preds, valid_gt_labels).item()

                valid_rec_losses.append(valid_g_loss)
                valid_ce_losses.append(valid_ce_loss)

                norm_valid_preds = torch.nn.Softmax(dim=1)(valid_preds)
                index_valid_preds = torch.argmax(norm_valid_preds, dim=1)

                valid_y_hats.append(index_valid_preds.cpu().numpy())
                valid_ys.append(valid_gt_labels.cpu().numpy())

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
                "Discriminator Loss": np.mean(d_losses),
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
                discriminator_path = os.path.join(
                    "models", config["MODEL_NAME"], f"{config['MODEL_NAME']}_D.pt"
                )
                decoder_projection_head_path = os.path.join(
                    "models", config["MODEL_NAME"], f"{config['MODEL_NAME']}_GPH.pt"
                )
                discriminator_projection_head_path = os.path.join(
                    "models", config["MODEL_NAME"], f"{config['MODEL_NAME']}_DPH.pt"
                )

                save_model(encoder, encoder_path)
                save_model(decoder, decoder_path)
                save_model(discriminator, discriminator_path)
                save_model(
                    decoder_projection_head,
                    decoder_projection_head_path)
                save_model(
                    discriminator_projection_head,
                    discriminator_projection_head_path)

    run.finish()
    pass


if __name__ == "__main__":

    model_name_base = "PCAA_Abl2_"
    n_training_classes = [2, 4, 6, 8]
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

            MSRadarDataset.generate_splits(
                train_classes=config["TRAIN_CLASSES"], seed=0, safe_mode=False
            )

            # # train variant 1
            model_name = f"{model_name_base}V1.{n_tr}.{i+1}"
            config["MODEL_NAME"] = model_name
            config["NOTES"] = "Ablation run, Variant 1"
            train_variant1(config, wandb_mode="online")

            # train variant 2
            model_name = f"{model_name_base}V2.{n_tr}.{i+1}"
            config["MODEL_NAME"] = model_name
            config["NOTES"] = "Ablation run, Variant 2"
            train_variant2(config, wandb_mode="online")

            # train variant 3
            model_name = f"{model_name_base}V3.{n_tr}.{i+1}"
            config["MODEL_NAME"] = model_name
            config["NOTES"] = "Ablation run, Variant 3"
            train_variant3(
                config,
                wandb_mode="online",
                # wandb_mode="disabled",
            )

            # train variant 4
            model_name = f"{model_name_base}V4.{n_tr}.{i+1}"
            config["MODEL_NAME"] = model_name
            config["NOTES"] = "Ablation run, Variant 4 (no projection head on Discriminator branch)"
            train_variant4(
                config,
                wandb_mode="online",
                # wandb_mode="disabled",
                proj_head_on_discriminator=False)
    pass
