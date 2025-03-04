import torch
import json
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from models import CGEncoder, CGDiscriminator
from datasets import MSRadarDataset
from constants import SPLIT
import constants
from scipy.stats import multivariate_normal


class VARIATION(Enum):
    BASE = "BASE"
    V1 = "V1"
    V2 = "V2"
    V3 = "V3"
    V4 = "V4"


def path_length_score(fvecs):
    ord = -1
    distance = 0
    for j in range(fvecs.shape[0]-1):
        distance += np.linalg.norm(fvecs[j] - fvecs[j+1], ord=ord)
    return distance


def CGAAE_inference_setup(model_name, loaders_batch_size, variation, generate_dataset=True, force_pc_subsampling=0):
    """
    Parameters
    ----------
    model_name : str
        Name of the model to load

    Returns
    -------
    encoder : pytorch model
        Encoder model

    decoder : pytorch model
        Decoder model

    discriminator : pytorch model
        Discriminator model

    train_loader : pytorch DataLoader
        Train dataset

    test_loader : pytorch DataLoader
        Test dataset

    unseen_loader : pytorch DataLoader
        Unseen dataset
    """

    # Load Config file
    model_folder = os.path.join("models", model_name)
    with open(os.path.join(model_folder, "config.pkl"), "rb") as file:
        config = pickle.load(file)

    # Generate dataset according to config file
    nmax_points = config["NMAX"]
    if generate_dataset:
        print("Generating dataset...")
        MSRadarDataset.generate_splits(
            train_classes=config["TRAIN_CLASSES"], seed=0, safe_mode=False, force_pc_subsampling=force_pc_subsampling, nmax_points=nmax_points)

    # ----- DEFINE DATA AND MODELS -----
    print("Loading model weights...")

    if variation in [VARIATION.V1, VARIATION.V2, VARIATION.V4]:
        encoder = (
            CGEncoder(n_out_labels=len(config["TRAIN_CLASSES"]), use_projection_head=True, nmax_points=nmax_points).to(
                constants.DEVICE).float())

    else:
        encoder = (
            CGEncoder(n_out_labels=len(config["TRAIN_CLASSES"]), use_projection_head=False, nmax_points=nmax_points).to(
                constants.DEVICE).float())

    discriminator = (
        CGDiscriminator(len(config["TRAIN_CLASSES"])).to(
            constants.DEVICE).float()
    )

    # Load weights
    encoder_weights = torch.load(
        f"models/{config['MODEL_NAME']}/{config['MODEL_NAME']}_E.pt")

    discriminator_weights = torch.load(
        f"models/{config['MODEL_NAME']}/{config['MODEL_NAME']}_D.pt"
    )

    encoder.load_state_dict(encoder_weights)

    discriminator.load_state_dict(discriminator_weights)

    encoder.eval()
    # decoder.eval()
    discriminator.eval()

    discriminator_means = torch.load(
        f"models/{config['MODEL_NAME']}/discriminator_means.pt"
    ).cpu()

    print("Setup complete!")

    return encoder, discriminator_means


def naive_sequential_procedure(k,
                               encoder,
                               discriminator_means,
                               figures_folder,
                               model_folder,
                               scenarios_list=constants.TRAIN_SCENARIOS,
                               seed=0,
                               unseen_valid_ratio=0.2,
                               force_pc_subsampling=0):

    rng = np.random.default_rng(seed)

    def joint_likelihood(x, means):
        n = means.shape[0]
        likelihood = 0
        for mean in means:
            gaussian_mode = multivariate_normal(mean=mean, cov=np.eye(32))
            likelihood += gaussian_mode.pdf(x)

        return likelihood / n

    # 0. Setup datasets and dataloaders
    radar_dataset_test = MSRadarDataset(
        SPLIT.TEST,
        scenarios=scenarios_list,
        subsample_factor=1.0,
        sequential=True
    )

    dataloader_test = torch.utils.data.DataLoader(
        radar_dataset_test,
        batch_size=k,
        drop_last=True,
        shuffle=False,
        num_workers=0,
    )

    radar_dataset_unseen = MSRadarDataset(
        SPLIT.UNSEEN,
        scenarios=scenarios_list,
        subsample_factor=1.0,
        sequential=True
    )

    dataloader_unseen = torch.utils.data.DataLoader(
        radar_dataset_unseen,
        batch_size=k,
        drop_last=True,
        shuffle=False,
        num_workers=0,
    )

    # 1. Find the best threshold that separates likelihoods of known vs unkown samples

    # 1.1 Take the labels of the unseen set
    all_unseen_labels_concat = np.array(
        [unseen_labels for _, unseen_labels in radar_dataset_unseen])

    # 1.2 find the indices of the samples belonging to a randomly chosen
    # set of unknown subjects (20% of the total number of unseen subjects)
    print("Choosing unseen subjects for validation...")
    unseen_subject_idxs = np.unique(all_unseen_labels_concat)

    # Choose which unseen subjects to use for "validation"
    val_unseen_subjects_idxs = rng.choice(unseen_subject_idxs, size=np.ceil(
        unseen_valid_ratio*len(unseen_subject_idxs)).astype(int), replace=False)
    n_unseen = len(all_unseen_labels_concat)
    unseen_valid_indices = np.where(
        np.isin(all_unseen_labels_concat, val_unseen_subjects_idxs))[0]
    unseen_valid_indices = np.sort(unseen_valid_indices)
    unseen_test_indices = set(range(n_unseen)) - set(unseen_valid_indices)

    # 1.3 Compute the likelihoods for known test set, unseen validation and test sets
    print("Computing likelihoods...")
    unseen_likelihoods_valid = []
    unseen_likelihoods_test = []
    test_likelihoods = []

    with torch.no_grad():
        for i, (test_pc, test_gt_labels) in enumerate(radar_dataset_test):
            test_pc = test_pc.to(constants.DEVICE).unsqueeze(0)
            test_gt_labels = test_gt_labels.to(constants.DEVICE)

            _, sup_fv = encoder(test_pc)
            test_likelihoods.append(joint_likelihood(
                sup_fv.cpu().numpy(), discriminator_means.cpu().numpy()))

        for i, (test_pc, test_gt_labels) in enumerate(radar_dataset_unseen):
            test_pc = test_pc.to(constants.DEVICE).unsqueeze(0)
            test_gt_labels = test_gt_labels.to(constants.DEVICE)

            _, sup_fv = encoder(test_pc)
            likelihood = joint_likelihood(
                sup_fv.cpu().numpy(), discriminator_means.cpu().numpy())

            if i in unseen_valid_indices:
                unseen_likelihoods_valid.append(likelihood)
            elif i in unseen_test_indices:
                unseen_likelihoods_test.append(likelihood)
            else:
                raise ValueError("Index not in validation or test set")

    unseen_likelihoods_valid = np.array(unseen_likelihoods_valid)
    unseen_likelihoods_test = np.array(unseen_likelihoods_test)
    test_likelihoods = np.array(test_likelihoods)

    # 1.4 Find the best separating threshold using the validation unseen set
    print("Finding best threshold...")
    scores = np.concatenate(
        [unseen_likelihoods_valid, test_likelihoods], axis=0)
    unkn_detection_labels = np.concatenate([np.zeros_like(
        unseen_likelihoods_valid), np.ones_like(test_likelihoods)], axis=0)

    fpr, tpr, thresholds = roc_curve(unkn_detection_labels, scores)
    best_threshold = thresholds[np.argmax(tpr - fpr)]

    # 2. Perform the unseen detection procedure on the Unseen and Seen test sets
    print("Performing unseen detection procedure...")
    openset_preds = []
    openset_labels = []
    n_labels = len(np.unique([label for _, label in radar_dataset_test]))

    with torch.no_grad():

        for i, (test_pc, test_gt_labels) in enumerate(dataloader_test):
            # ensure all labels in the batch are the same
            if len(np.unique(test_gt_labels.cpu().numpy())) != 1:
                continue
            assert len(np.unique(test_gt_labels.cpu().numpy())) == 1

            openset_labels.append(test_gt_labels.cpu().numpy()[0])
            test_pc = test_pc.to(constants.DEVICE)
            test_gt_labels = test_gt_labels.to(constants.DEVICE)

            test_preds, sup_fv = encoder(test_pc)
            norm_test_preds = torch.nn.Softmax(dim=1)(test_preds)
            index_test_preds = torch.argmax(norm_test_preds, dim=1)

            likelihoods = []

            for fv, _ in zip(sup_fv, index_test_preds):
                density = joint_likelihood(
                    fv.cpu().numpy(), discriminator_means.cpu().numpy())
                likelihoods.append(density)

            # check how many likelihoods are above threshold
            n_above_threshold = np.sum(np.array(likelihoods) > best_threshold)
            if n_above_threshold > k/2:
                # Classify as known class with the most frequent prediction
                pred_counts = np.bincount(index_test_preds.cpu().numpy())
                openset_preds.append(np.argmax(pred_counts))
                # openset_preds.append(index_test_preds[0].item())
            else:
                # Classify as unknown class
                openset_preds.append(n_labels)

            print(
                f"Prediction: {openset_preds[-1]}, True label: {openset_labels[-1]}")

        for i, (test_pc, test_gt_labels) in enumerate(dataloader_unseen):

            # ensure all labels in the batch are the same
            if len(np.unique(test_gt_labels.cpu().numpy())) != 1:
                continue
            assert len(np.unique(test_gt_labels.cpu().numpy())) == 1

            # Ensuring that we just take samples in the "test" unseen set
            if test_gt_labels[0].item() not in val_unseen_subjects_idxs:

                openset_labels.append(n_labels)
                test_pc = test_pc.to(constants.DEVICE)
                test_gt_labels = test_gt_labels.to(constants.DEVICE)

                test_preds, sup_fv = encoder(test_pc)
                norm_test_preds = torch.nn.Softmax(dim=1)(test_preds)
                index_test_preds = torch.argmax(norm_test_preds, dim=1)

                likelihoods = []

                for fv, _ in zip(sup_fv, index_test_preds):
                    density = joint_likelihood(
                        fv.cpu().numpy(), discriminator_means.cpu().numpy())
                    likelihoods.append(density)

                # check how many likelihoods are above threshold
                n_above_threshold = np.sum(
                    np.array(likelihoods) > best_threshold)
                if n_above_threshold > k/2:
                    # Classify as known class with model prediction
                    pred_counts = np.bincount(index_test_preds.cpu().numpy())
                    openset_preds.append(np.argmax(pred_counts))
                    # openset_preds.append(index_test_preds[0].item())
                else:
                    # Classify as unknown class
                    openset_preds.append(n_labels)

                print(
                    f"Prediction: {openset_preds[-1]}, True label: {n_labels}")

    openset_preds_array = np.array(openset_preds)
    openset_labels_array = np.array(openset_labels)

    # 3. Save metrics and confusion matrix
    print("Saving metrics and confusion matrix...")
    # Set global plot settings
    final_preds, final_labels = plot_confusion_matrix_cgaae(
        k, figures_folder, n_labels, openset_preds_array, openset_labels_array, "naive_seq")

    out_log = {
        "n_steps": k,
        "accuracy": np.equal(final_labels, final_preds).sum() / len(final_labels),
        "f1_micro": f1_score(final_labels, final_preds, average="micro"),
        "f1_macro": f1_score(final_labels, final_preds, average="macro"),
        "f1_weighted": f1_score(final_labels, final_preds, average="weighted"),
    }
    # save json file with log dictionary
    if force_pc_subsampling and scenarios_list == constants.TRAIN_SCENARIOS:
        # In this case we are considering point cloud subsampling, with all three scenarios
        with open(os.path.join(model_folder, f"naive_seq_log_{k}_subsampled{force_pc_subsampling}.json"), "w") as file:
            json.dump(out_log, file)
    elif not force_pc_subsampling and scenarios_list != constants.TRAIN_SCENARIOS:
        # In this case we are considering original point clouds with only one scenario
        sc = "_".join([sc.value for sc in scenarios_list])
        with open(os.path.join(model_folder, f"naive_seq_log_{k}_scenarios{sc}.json"), "w") as file:
            json.dump(out_log, file)
    else:
        # This is the default case, where we are considering original point clouds with all three scenarios
        with open(os.path.join(model_folder, f"naive_seq_log_{k}.json"), "w") as file:
            json.dump(out_log, file)

    return out_log, final_preds, final_labels


def plot_confusion_matrix_cgaae(k, figures_folder, n_labels, openset_preds_array, openset_labels_array, title_method):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 14,
        "hatch.linewidth": 0.0,
        "hatch.color": (0, 0, 0, 0.0),
    })

    # Get confusion matrix
    final_preds = openset_preds_array
    final_labels = openset_labels_array.astype(int)

    cm = confusion_matrix(final_labels, final_preds, normalize='true')

    colormap = plt.get_cmap('Blues')

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=range(n_labels + 1))
    disp.plot(cmap=colormap, values_format='.3f', ax=None)
    disp.ax_.get_images()[0].set_clim(0, 1)

    plt.xticks(range(n_labels + 1),
               [f"T{i}" for i in range(n_labels)] + ["U"], rotation=90)
    plt.yticks(range(n_labels + 1), [f"T{i}" for i in range(n_labels)] + ["U"])

    plt.savefig(os.path.join(figures_folder,
                f"openset_confusion_matrix_{title_method}_{k}.png"), dpi=300)

    return final_preds, final_labels


def CGAAE_inference(model_names, ks, force_pc_subsampling=0, scenarios_list=constants.TRAIN_SCENARIOS, variation=False):

    # Check if force_pc_subsampling and scenarios_list are both different from default
    # if yes, return error, since this would cause wrong json file to be saved
    if force_pc_subsampling and scenarios_list != constants.TRAIN_SCENARIOS:
        raise ValueError(
            "force_pc_subsampling and scenarios_list cannot be both different from default")

    out_log = {}
    batch_size = 32

    for model_name in model_names:
        print(f"Processing model {model_name}...")
        for k in ks:
            print(f"Processing k={k}...")

            out_log.setdefault(k, {})

            model_folder = os.path.join("models", model_name)
            figures_folder = os.path.join("figures", model_name)
            os.makedirs(figures_folder, exist_ok=True)

            if not variation:
                variation_name = model_name.split(".")[0][-2:]
                if variation_name == "V1":
                    variation = VARIATION.V1
                elif variation_name == "V2":
                    variation = VARIATION.V2
                elif variation_name == "V3":
                    variation = VARIATION.V3
                elif variation_name == "V4":
                    variation = VARIATION.V4
                else:
                    variation = VARIATION.BASE

            (
                encoder,
                discriminator_means,

            ) = CGAAE_inference_setup(
                model_name=model_name,
                loaders_batch_size=batch_size,
                variation=variation,
                generate_dataset=True,
                force_pc_subsampling=force_pc_subsampling,)

            # naive sequential procedure
            out_metrics, final_preds, final_labels = naive_sequential_procedure(k=k,
                                                                                encoder=encoder,
                                                                                discriminator_means=discriminator_means,
                                                                                figures_folder=figures_folder,
                                                                                model_folder=model_folder,
                                                                                scenarios_list=scenarios_list,
                                                                                seed=0,
                                                                                unseen_valid_ratio=0.2,
                                                                                force_pc_subsampling=force_pc_subsampling)
            #
            # # save final preds and final labels in models folder
            if force_pc_subsampling and scenarios_list == constants.TRAIN_SCENARIOS:
                np.save(os.path.join(model_folder,
                        f"final_preds_{k}_subsampled{force_pc_subsampling}.npy"), final_preds)
                np.save(os.path.join(model_folder,
                        f"final_labels_{k}_subsampled{force_pc_subsampling}.npy"), final_labels)
            elif not force_pc_subsampling and scenarios_list != constants.TRAIN_SCENARIOS:
                sc = "_".join([sc.value for sc in scenarios_list])
                np.save(os.path.join(model_folder,
                        f"final_preds_{k}_scenarios{sc}.npy"), final_preds)
                np.save(os.path.join(model_folder,
                        f"final_labels_{k}_scenarios{sc}.npy"), final_labels)
            elif force_pc_subsampling == 0:
                np.save(os.path.join(model_folder,
                        f"final_preds_{k}.npy"), final_preds)
                np.save(os.path.join(model_folder,
                        f"final_labels_{k}.npy"), final_labels)

            out_log[k] = {}
            f1_micro = out_metrics["f1_micro"]
            f1_macro = out_metrics["f1_macro"]
            f1_weighted = out_metrics["f1_weighted"]
            out_log[k]["f1_micro"] = f1_micro
            out_log[k]["f1_macro"] = f1_macro
            out_log[k]["f1_weighted"] = f1_weighted

            # print(out_log)

        # save json
        with open(os.path.join(model_folder, f"naive_seq_log_subsampled{force_pc_subsampling}.json"), "w") as file:
            json.dump(out_log, file)


if __name__ == "__main__":

    model_names = [
        "PCAA_Abl_V1.2.1",
        "PCAA_Abl_V1.2.2",
        "PCAA_Abl_V1.2.3",
        "PCAA_Abl_V1.2.4",
        "PCAA_Abl_V1.2.5",
        "PCAA_Abl_V1.4.1",
        "PCAA_Abl_V1.4.2",
        "PCAA_Abl_V1.4.3",
        "PCAA_Abl_V1.4.4",
        "PCAA_Abl_V1.4.5",
        "PCAA_Abl_V1.6.1",
        "PCAA_Abl_V1.6.2",
        "PCAA_Abl_V1.6.3",
        "PCAA_Abl_V1.6.4",
        "PCAA_Abl_V1.6.5",
        "PCAA_Abl_V1.8.1",
        "PCAA_Abl_V1.8.2",
        "PCAA_Abl_V1.8.3",
        "PCAA_Abl_V1.8.4",
        "PCAA_Abl_V1.8.5",
        "PCAA_Abl_V2.2.1",
        "PCAA_Abl_V2.2.2",
        "PCAA_Abl_V2.2.3",
        "PCAA_Abl_V2.2.4",
        "PCAA_Abl_V2.2.5",
        "PCAA_Abl_V2.4.1",
        "PCAA_Abl_V2.4.2",
        "PCAA_Abl_V2.4.3",
        "PCAA_Abl_V2.4.4",
        "PCAA_Abl_V2.4.5",
        "PCAA_Abl_V2.6.1",
        "PCAA_Abl_V2.6.2",
        "PCAA_Abl_V2.6.3",
        "PCAA_Abl_V2.6.4",
        "PCAA_Abl_V2.6.5",
        "PCAA_Abl_V2.8.1",
        "PCAA_Abl_V2.8.2",
        "PCAA_Abl_V2.8.3",
        "PCAA_Abl_V2.8.4",
        "PCAA_Abl_V2.8.5",
        "PCAA_Abl_V3.2.1",
        "PCAA_Abl_V3.2.2",
        "PCAA_Abl_V3.2.3",
        "PCAA_Abl_V3.2.4",
        "PCAA_Abl_V3.2.5",
        "PCAA_Abl_V3.4.1",
        "PCAA_Abl_V3.4.2",
        "PCAA_Abl_V3.4.3",
        "PCAA_Abl_V3.4.4",
        "PCAA_Abl_V3.4.5",
        "PCAA_Abl_V3.6.1",
        "PCAA_Abl_V3.6.2",
        "PCAA_Abl_V3.6.3",
        "PCAA_Abl_V3.6.4",
        "PCAA_Abl_V3.6.5",
        "PCAA_Abl_V3.8.1",
        "PCAA_Abl_V3.8.2",
        "PCAA_Abl_V3.8.3",
        "PCAA_Abl_V3.8.4",
        "PCAA_Abl_V3.8.5",
        "PCAA_Abl_V4.2.1",
        "PCAA_Abl_V4.2.2",
        "PCAA_Abl_V4.2.3",
        "PCAA_Abl_V4.2.4",
        "PCAA_Abl_V4.2.5",
        "PCAA_Abl_V4.4.1",
        "PCAA_Abl_V4.4.2",
        "PCAA_Abl_V4.4.3",
        "PCAA_Abl_V4.4.4",
        "PCAA_Abl_V4.4.5",
        "PCAA_Abl_V4.6.1",
        "PCAA_Abl_V4.6.2",
        "PCAA_Abl_V4.6.3",
        "PCAA_Abl_V4.6.4",
        "PCAA_Abl_V4.6.5",
        "PCAA_Abl_V4.8.1",
        "PCAA_Abl_V4.8.2",
        "PCAA_Abl_V4.8.3",
        "PCAA_Abl_V4.8.4",
        "PCAA_Abl_V4.8.5",

    ]

    ks = [6]

    forced_subsamplings = [0]

    for force_pc_subsampling in forced_subsamplings:

        CGAAE_inference(model_names=model_names, ks=ks,
                        force_pc_subsampling=force_pc_subsampling)
