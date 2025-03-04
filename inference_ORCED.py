import numpy as np
import os
import pickle
from scipy.stats import multivariate_normal, norm
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm

import constants
from utils import SeqChamferLoss
from datasets import MSRadarDataset, SPLIT
from models import ORCEDDecoder, ORCEDEncoder, GaussianMeanLearner
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from matplotlib import pyplot as plt


def compute_prob(mean, cov, z_test):
    """
    Computes the integral of the pdf of a multivariate normal distribution of given mean and covariance matrix,
    over a hypercube of side length square_size centered at the mean.

    Parameters:
    mean : numpy array of shape (latent_dim, 1)
        Mean of the multivariate normal distribution
    cov : numpy array of shape (latent_dim, latent_dim)
        Covariance matrix of the multivariate normal distribution

    z_test : numpy array of shape (latent_dim, 1)
        test feature vector
    """
    latent_dim = len(mean)
    mvn = multivariate_normal(mean, cov)

    # Number of points used for integration when computing cdf
    # mvn.maxpts = 100

    a_vec = mean - np.abs(z_test - mean)
    b_vec = mean + np.abs(z_test - mean)

    cdf_b = mvn.cdf(b_vec)
    cdf_a = mvn.cdf(a_vec)

    p = cdf_b - cdf_a
    return p


def ORCED_ensemble_ood_detection(
        rec_err_tr,
        f_vecs_tr,
        thresholds_g,
        gt_labels,
        pred_labels,
        x_test_prediction,
        z_test,
        re_test):
    """
    Parameters
    ----------
    rec_err_tr : numpy array of shape (n_train_samples, 1)
        Array containing reconstruction errors.
    f_vecs_tr : numpy array of shape (n_train_samples, latent_dim)
        Array containing latent vectors.
    thresholds_g : numpy array of shape (K, 1)
        Thresholds on the likelihood of the latent vectors, one for each class
    gt_labels : numpy array of shape (n_train_samples, 1)
        Array containing ground truth labels of training set
    pred_labels : numpy array of shape (n_train_samples, 1)
        Array containing predicted labels of training set
    x_tet_prediction : numpy array of shape (n_test_samples, 1)
        Array containing predicted labels of test set
    z_test : numpy array of shape (n_test_samples, latent_dim)
        Array containing test set latent vectors
    re_test : numpy array of shape (n_test_samples, 1)
        Array containing test set reconstruction errors

    """

    n_classes = len(np.unique(gt_labels))
    corr_prediction_mask = gt_labels == pred_labels
    n_test_samples = z_test.shape[0]

    means_re = []
    std_re = []
    means_z = []
    stds_z = []
    thresholds_re = []

    for k in range(n_classes):
        means_re.append(np.mean(rec_err_tr[gt_labels == k]))
        std_re.append(np.std(rec_err_tr[gt_labels == k]))
        means_z.append(np.mean(
            f_vecs_tr[corr_prediction_mask][gt_labels[corr_prediction_mask] == k], axis=0))
        stds_z.append(np.std(
            f_vecs_tr[corr_prediction_mask][gt_labels[corr_prediction_mask] == k], axis=0))

        # Compute thresohld for the rec error
        thresholds_re.append(means_re[k] + 2*std_re[k])

    p_z_ks = []
    p_re_ks = []

    for k in range(n_classes):
        # p_z_k = multivariate_normal.pdf(z_test, mean=means_z[k], cov=np.diag(stds_z[k]))
        # dists = np.linalg.norm(means_z[k] - z_test, axis=1)

        p_z_k = compute_prob(means_z[k], np.diag(stds_z[k]), z_test)

        p_re_k = norm.pdf(re_test, loc=means_re[k], scale=std_re[k])

        p_z_ks.append(p_z_k)
        p_re_ks.append(p_re_k)

    p_z_ks = np.array(p_z_ks)  # this should be of shape (K, n_test_samples)
    p_re_ks = np.array(p_re_ks)  # same as above

    # Now check the probabilities against the thresholds
    # 1) Latent space distributions
    # threshold_z_array = np.repeat(thresholds_g, n_test_samples, axis=1)
    threshold_z_array = thresholds_g
    p_zs_mask = np.less(1-p_z_ks, 1-threshold_z_array)
    latent_bools = np.sum(p_zs_mask, axis=0) == n_classes

    thresholds_re_array = np.array(
        [thresholds_re[j] for j in x_test_prediction])
    rec_err_bools = re_test > thresholds_re_array

    out_unseen_mask = np.logical_or(latent_bools, rec_err_bools)
    out_classes = torch.clone(x_test_prediction.detach().cpu())
    out_classes[out_unseen_mask] = n_classes

    return out_classes


def ORCED_inference_setup(model_name, loaders_batch_size, generate_dataset=True):
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

    mean_learner : pytorch model
        Mean learner model

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
    if generate_dataset:
        print("Generating dataset...")
        MSRadarDataset.generate_splits(
            train_classes=config["TRAIN_CLASSES"], seed=0, safe_mode=False)

    # ----- DEFINE DATA AND MODELS -----
    print("Loading model weights...")
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

    # Configure data loaders
    radar_dataset_train = MSRadarDataset(
        SPLIT.TRAIN,
        subsample_factor=config["SUBSAMPLE_FACTOR"],
    )

    dataloader_train = torch.utils.data.DataLoader(
        radar_dataset_train,
        batch_size=loaders_batch_size,
        drop_last=True,
        shuffle=False,
        num_workers=0,
    )

    radar_dataset_test = MSRadarDataset(
        SPLIT.TEST,
        subsample_factor=config["SUBSAMPLE_FACTOR"],
    )

    dataloader_test = torch.utils.data.DataLoader(
        radar_dataset_test,
        batch_size=loaders_batch_size,
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    radar_dataset_unseen = MSRadarDataset(
        SPLIT.UNSEEN,
        subsample_factor=config["SUBSAMPLE_FACTOR"],
    )

    dataloader_unseen = torch.utils.data.DataLoader(
        radar_dataset_unseen,
        batch_size=loaders_batch_size,
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    # Load weights
    encoder_weights = torch.load(
        f"models/{config['MODEL_NAME']}/{config['MODEL_NAME']}_E.pt"
    )

    decoder_weights = torch.load(
        f"models/{config['MODEL_NAME']}/{config['MODEL_NAME']}_G.pt"
    )

    mean_learner_weights = torch.load(
        f"models/{config['MODEL_NAME']}/{config['MODEL_NAME']}_ML.pt"
    )

    encoder.load_state_dict(encoder_weights)
    decoder.load_state_dict(decoder_weights)
    mean_learner.load_state_dict(mean_learner_weights)

    encoder.eval()
    decoder.eval()
    mean_learner.eval()

    oh_labels = F.one_hot(
        torch.arange(0, len(config["TRAIN_CLASSES"])), num_classes=len(config["TRAIN_CLASSES"])).float().to("cuda")

    with torch.no_grad():
        cluster_means = mean_learner(oh_labels)

    print("Setup complete!")

    return encoder, decoder, mean_learner, cluster_means, dataloader_train, dataloader_test, dataloader_unseen


def ORCED_inference(model_names):
    chamfer_loss = SeqChamferLoss()

    for model_name in model_names:
        figures_folder = os.path.join("figures", model_name)
        batch_size = 64
        os.makedirs(figures_folder, exist_ok=True)

        (
            encoder,
            decoder,
            mean_learner,
            cluster_means,
            dataloader_train,
            dataloader_test,
            dataloader_unseen,
        ) = ORCED_inference_setup(model_name=model_name,
                                  loaders_batch_size=batch_size,
                                  generate_dataset=True)

        train_feature_vectors = []
        train_rec_losses = []
        all_train_preds = []
        all_train_labels = []

        print("Computing training set statistics...")
        with torch.no_grad():
            for i, (train_pc, train_gt_labels) in tqdm(enumerate(dataloader_train)):
                train_pc = train_pc.to(constants.DEVICE)
                train_gt_labels = train_gt_labels.to(constants.DEVICE)

                # Encoder forward pass
                train_preds, sup_fvs, _, _ = encoder(train_pc)
                train_feature_vectors.append(sup_fvs.cpu().numpy())
                train_rec_pc = decoder(sup_fvs)

                train_g_loss = chamfer_loss(
                    train_rec_pc, train_pc, avg_out=False)

                train_rec_losses.append(train_g_loss.detach().cpu().numpy())

                norm_train_preds = torch.nn.Softmax(dim=1)(train_preds)
                index_train_preds = torch.argmax(norm_train_preds, dim=1)

                all_train_preds.append(index_train_preds.cpu().numpy())
                all_train_labels.append(train_gt_labels.cpu().numpy())

        rec_err_tr = np.concatenate(train_rec_losses)
        f_vecs_tr = np.concatenate(train_feature_vectors)
        gt_labels = np.concatenate(all_train_labels)
        pred_labels = np.concatenate(all_train_preds)

        # Probability thresholds for vectors of latent space
        thresholds_g = 0.95

        all_test_labels = []
        all_unseen_labels = []
        n_labels = len(np.unique(gt_labels))

        print("Computing test set statistics...")
        test_openset_predictions_array = []
        unseen_openset_predictions_array = []
        with torch.no_grad():
            for i, (test_pc, test_gt_labels) in tqdm(enumerate(dataloader_test)):
                test_pc = test_pc.to(constants.DEVICE)
                test_gt_labels = test_gt_labels
                all_test_labels.append(test_gt_labels.cpu().numpy())

                # Encoder forward pass
                test_preds, sup_fvs, _, _ = encoder(test_pc)
                # feature_vectors.append(sup_fvs.cpu().numpy())
                test_rec_pc = decoder(sup_fvs)

                test_g_loss = chamfer_loss(test_rec_pc, test_pc, avg_out=False)

                # test_rec_losses.append(test_g_loss.item())
                # test_ce_losses.append(test_ce_loss)

                norm_test_preds = torch.nn.Softmax(dim=1)(test_preds)
                index_test_preds = torch.argmax(norm_test_preds, dim=1)
                # all_test_preds.append(index_test_preds.cpu().numpy())

                test_openset_predictions = ORCED_ensemble_ood_detection(
                    rec_err_tr,
                    f_vecs_tr,
                    thresholds_g,
                    gt_labels,
                    pred_labels,
                    index_test_preds,
                    sup_fvs.detach().cpu().numpy(),
                    test_g_loss.detach().cpu().numpy())

                print(
                    f"N. of correct predictions: {torch.sum(test_gt_labels == test_openset_predictions)}")

                test_openset_predictions_array.append(test_openset_predictions)

                # if i == 0:
                #     print("Breaking from for loop (for debugging)")
                #     break

            print("Computing unseen set statistics...")

            # I have to consider a leave out label, because in the CGAAE i actually use one unseen subject
            # as validation to compute the threshold.
            # For a fair comparison I have to consider the same number of ACTUALLY unseen subjects

            leave_out_label = None

            for i, (unseen_pc, unseen_gt_labels) in tqdm(enumerate(dataloader_unseen)):
                unseen_pc = unseen_pc.to(constants.DEVICE)
                if leave_out_label is None:
                    leave_out_label = unseen_gt_labels[0].item()
                unseen_gt_labels = unseen_gt_labels.to(constants.DEVICE)
                all_unseen_labels.append(n_labels)

                # Encoder forward pass
                unseen_preds, sup_fvs, _, _ = encoder(unseen_pc)
                # feature_vectors.append(sup_fvs.cpu().numpy())
                unseen_rec_pc = decoder(sup_fvs)

                test_g_loss = chamfer_loss(
                    unseen_rec_pc, unseen_pc, avg_out=False)

                # test_rec_losses.append(test_g_loss.item())
                # test_ce_losses.append(test_ce_loss)

                norm_unseen_preds = torch.nn.Softmax(dim=1)(unseen_preds)
                index_unseen_preds = torch.argmax(norm_unseen_preds, dim=1)
                # all_unseen_preds.append(index_unseen_preds.cpu().numpy())

                # Check if the current label is equal to the leave out label
                if unseen_gt_labels[0].item() != leave_out_label:

                    unseen_openset_predictions = ORCED_ensemble_ood_detection(
                        rec_err_tr,
                        f_vecs_tr,
                        thresholds_g,
                        gt_labels,
                        pred_labels,
                        index_unseen_preds,
                        sup_fvs.detach().cpu().numpy(),
                        test_g_loss.detach().cpu().numpy())

                    unseen_openset_predictions_array.append(
                        unseen_openset_predictions)

                # if i == 0:
                #     print("Breaking from for loop (for debugging)")
                #     break

        test_openset_predictions_array = np.concatenate(
            test_openset_predictions_array)
        unseen_openset_predictions_array = np.concatenate(
            unseen_openset_predictions_array)

        # Plot confusion matrix and evaluate performances
        final_preds = np.concatenate(
            [test_openset_predictions_array, unseen_openset_predictions_array])
        final_labels = np.concatenate([np.concatenate(all_test_labels), [
                                      n_labels]*len(unseen_openset_predictions_array)])

        acc = np.equal(final_labels, final_preds).sum() / len(final_labels)
        f1_micro = f1_score(final_labels, final_preds, average="micro")
        f1_macro = f1_score(final_labels, final_preds, average="macro")
        f1_weighted = f1_score(final_labels, final_preds, average="weighted")

        # print(f"Threshold: {threshold:e}")
        print(f"Accuracy: {acc:.5f}")
        print(f"F1 Score Micro: {f1_micro:.5f}")
        print(f"F1 Score Macro: {f1_macro:.5f}")
        print(f"F1 Score Weighted: {f1_weighted:.5f}")

        cm = confusion_matrix(final_labels, final_preds, normalize='true')

        colormap = plt.get_cmap('Blues')

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=range(n_labels + 1))
        disp.plot(cmap=colormap, values_format='.3f', ax=None)
        disp.ax_.get_images()[0].set_clim(0, 1)

        plt.xticks(range(n_labels + 1),
                   [f"T{i}" for i in range(n_labels)] + ["U"], rotation=90)
        plt.yticks(range(n_labels + 1),
                   [f"T{i}" for i in range(n_labels)] + ["U"])

        plt.title(f"F1-Score micro: {f1_micro:.5f} - Accuracy: {acc:.5f}")

        plt.savefig(os.path.join(figures_folder,
                    f"openset_cnfmtrx_ensemble_ood_fixed.png"), dpi=300)
        # plt.show()

        # Save also vectors of predictions and labels
        np.save(os.path.join(figures_folder,
                "ensemble_ood_final_preds_fixed.npy"), final_preds)
        np.save(os.path.join(figures_folder,
                "ensemble_ood_final_labels_fixed.npy"), final_labels)

    pass


if __name__ == "__main__":
    ORCED_model_names = [
        "ORCED_.2.1",
        # "ORCED_.2.2",
        # "ORCED_.2.3",
        # "ORCED_.2.4",
        # "ORCED_.2.5",
        # "ORCED_.4.1",
        # "ORCED_.4.2",
        # "ORCED_.4.3",
        # "ORCED_.4.4",
        # "ORCED_.4.5",
        # "ORCED_.6.1",
        # "ORCED_.6.2",
        # "ORCED_.6.3",
        # "ORCED_.6.4",
        # "ORCED_.6.5",
        # "ORCED_.8.1",
        # "ORCED_.8.2",
        # "ORCED_.8.3",
        # "ORCED_.8.4",
        # "ORCED_.8.5",
    ]
    ORCED_inference(ORCED_model_names)
