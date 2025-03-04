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
from datasets import MSRadarDataset, SPLIT
from models import (
    CGEncoder,
    CGDiscriminator,
    CGDecoder,
)
from utils import (
    SeqChamferLoss,
    save_model,
    sample_distant_points,
)


def train_CGAAE(config=constants.CONFIG):
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
        CGEncoder(n_out_labels=len(config["TRAIN_CLASSES"]))
        .to(constants.DEVICE)
        .float()
    )
    decoder = CGDecoder().to(constants.DEVICE).float()
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
        itertools.chain(encoder.parameters(), decoder.parameters()),
        lr=config["LR"],
        betas=(config["B1"], config["B2"]),
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
            z = Variable(z0 + mus)

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

            rec_pcs = decoder(sup_fvs)

            rec_loss = chamfer_loss(rec_pcs, pcs)
            rec_losses.append(rec_loss.item())

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

        # ---- EVALUATION --------
        print("Evaluation")
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
                valid_rec_pc = decoder(sup_fv)

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

                save_model(encoder, encoder_path)
                save_model(decoder, decoder_path)
                save_model(discriminator, discriminator_path)

    run.finish()

    pass


if __name__ == "__main__":
    from utils import openness

    model_name_base = "PCAA_TestOpenness_"
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

            model_name = f"{model_name_base}.{n_tr}.{i+1}"

            config = constants.CONFIG
            config["MODEL_NAME"] = model_name
            config["TRAIN_CLASSES"] = random_train_classes
            config["Openness"] = openness(n_tr, len(MSRadarDataset.label_dict))

            MSRadarDataset.generate_splits(
                train_classes=config["TRAIN_CLASSES"], seed=0, safe_mode=False
            )
            train_CGAAE(config)
