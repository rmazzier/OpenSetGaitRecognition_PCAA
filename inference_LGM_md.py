import torch
import json
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from models import LGM_Net, LGM_Loss
from datasets import MSRadarDatasetMD
from constants import SPLIT
import constants
from scipy.stats import multivariate_normal

def plot_tsne(model, test_dataloader, unseen_dataloader, n_classes, save_dir):
    """
    model: callable that takes a batch of inputs and returns latent features [B, D]
    test_dataloader: dataloader for known (seen) classes
    unseen_dataloader: dataloader for unknown (unseen) classes
    """

    all_feats = []
    all_labels = []
    all_domains = []  # 0 for test, 1 for unseen

    # Seen data
    for x, y in test_dataloader:
        x = x.to(constants.DEVICE)
        y = y.to(constants.DEVICE)

        # Forward pass
        feats = model(x)
        all_feats.append(feats.detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())
        all_domains.append(np.zeros_like(y.detach().cpu().numpy()))  # domain = 0
    for md_unseen, y_unseen in unseen_dataloader:
        md_unseen = md_unseen.to(constants.DEVICE)
        y_unseen = y_unseen.to(constants.DEVICE)


        # Forward pass
        feats_unseen = model(md_unseen)
        all_feats.append(feats_unseen.detach().cpu().numpy())
        all_labels.append(torch.ones_like(y_unseen.detach().cpu()) * n_classes)
        all_domains.append(np.ones_like(y_unseen.detach().cpu().numpy()))  # domain = 0

    all_feats = np.concatenate(all_feats, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_domains = np.concatenate(all_domains, axis=0)

    # t-SNE projection
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_feats = tsne.fit_transform(all_feats)

    # Plot
    plt.figure(figsize=(10, 8))
    markers = ['o', 'x']  # 0 = test, 1 = unseen

    for domain in [0, 1]:
        if domain==0:
            alpha=1.0
        else:
            alpha=0.1
        idxs = (all_domains == domain)
        plt.scatter(tsne_feats[idxs, 0], tsne_feats[idxs, 1],
                    c=all_labels[idxs],
                    cmap='tab10',
                    marker=markers[domain],
                    alpha=alpha,
                    label='unseen' if domain else 'test',
                    edgecolors='k' if domain else 'none')

    plt.legend()
    plt.title('t-SNE of Latent Space')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"t-sne_latentspace.png"))


def LGM_inference_setup(model_name, generate_dataset=True):
    
    # Load Config file
    model_folder = os.path.join("models", model_name)
    with open(os.path.join(model_folder, "config.pkl"), "rb") as file:
        config = pickle.load(file)
    n_classes = len(config["TRAIN_CLASSES"])

    # Generate dataset according to config file
    nmax_points = config["NMAX"]
    if generate_dataset:
        print("Generating dataset...")
        MSRadarDatasetMD.generate_splits(
            train_classes=config["TRAIN_CLASSES"], split_seed=0, safe_mode=False)

    # ----- DEFINE DATA AND MODELS -----
    print("Loading model weights...")

    # Initialize generator and style_discriminator
    lgm_criterion = LGM_Loss(feature_dim=16, num_classes=len(config["TRAIN_CLASSES"]), margin=3, lambda_reg=0.5).to(constants.DEVICE)
    model = LGM_Net(embedding_dim=16).to(constants.DEVICE)

    # Load weights
    model_weights = torch.load(f"models/{config['MODEL_NAME']}/{config['MODEL_NAME']}.pt", weights_only=True)

    model.load_state_dict(model_weights)

    # Load class means
    class_means = torch.load(f"models/{config['MODEL_NAME']}/{config['MODEL_NAME']}_class_means.pt", weights_only=True)

    lgm_criterion.load_state_dict(class_means)

    model.eval()
    lgm_criterion.eval()

    return model, lgm_criterion, n_classes


def lgm_intruder_detection(model,
                           lgm_criterion,
                           figures_folder,
                           n_classes,
                           seed=0,):

    rng = np.random.default_rng(seed)

    # 0. Setup datasets and dataloaders
    radar_dataset_test = MSRadarDatasetMD(
        SPLIT.TEST,
        subsample_factor=1.0
    )

    dataloader_test = torch.utils.data.DataLoader(
        radar_dataset_test,
        batch_size=1,
        drop_last=True,
        shuffle=False,
        num_workers=0,
    )

    radar_dataset_unseen = MSRadarDatasetMD(
        SPLIT.UNSEEN,
        subsample_factor=1.0
    )

    dataloader_unseen = torch.utils.data.DataLoader(
        radar_dataset_unseen,
        batch_size=1,
        drop_last=True,
        shuffle=False,
        num_workers=0,
    )

    # Plot tsne
    plot_tsne(model, dataloader_test, dataloader_unseen, n_classes, figures_folder)


    # 2. Perform the unseen detection procedure on the Unseen and Seen test sets
    print("Performing unseen detection procedure...")
    openset_preds = []
    openset_scores = []
    openset_labels = []
    n_labels = len(np.unique([label for _, label in radar_dataset_test]))

    with torch.no_grad():
        
        print("Passing through test set")
        for i, (test_pc, test_gt_labels) in enumerate(dataloader_test):

            openset_labels.append(test_gt_labels.cpu().numpy()[0])
            test_pc = test_pc.to(constants.DEVICE)
            test_gt_labels = test_gt_labels.to(constants.DEVICE)

            embeddings = model(test_pc) 

            posteriors = lgm_criterion.compute_posteriors(embeddings)
            decision_score, prediction = torch.max(posteriors, dim=1)

            openset_preds.append(prediction.cpu().item())
            openset_scores.append(decision_score.cpu().item())

        print("Passing through unseen set")
        for i, (unseen_pc, unseen_gt_labels) in enumerate(dataloader_unseen):
            
            # Assignthe label for the unknown subject. e.g. with 3 known subjects, we will have labels 0, 1, 2 for known, and label 3 for unkown
            openset_labels.append(n_labels)
            unseen_pc = unseen_pc.to(constants.DEVICE)
            unseen_gt_labels = unseen_gt_labels.to(constants.DEVICE)

            embeddings = model(unseen_pc) 

            posteriors = lgm_criterion.compute_posteriors(embeddings)
            decision_score, prediction = torch.max(posteriors, dim=1)

            assert not np.isnan(posteriors.cpu().numpy()).any()
            openset_preds.append(prediction.cpu().item())
            openset_scores.append(decision_score.cpu().item())

    openset_preds_array = np.array(openset_preds)
    openset_labels_array = np.array(openset_labels)
    openset_scores_array = np.array(openset_scores)

    # Find the best threshold 
    unkn_detection_labels = np.concatenate([np.ones(len(dataloader_test)), np.zeros(len(dataloader_unseen))])
    fpr, tpr, thresholds = roc_curve(unkn_detection_labels, openset_scores_array)
    best_threshold = thresholds[np.argmax(tpr - fpr)]

    # Using the threshold, get the actual predictions 
    unknown_idxs = openset_scores_array < best_threshold

    # only at the indices corresponding to unknown subjects, assign the unkown prediction
    openset_preds_array[unknown_idxs] = n_labels

    final_labels = openset_labels_array
    final_preds = openset_preds_array

    # Save metrics and confusion matrix
    print("Saving metrics and confusion matrix...")

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
                f"openset_cnfmtrx_lgm.png"), dpi=300)
    
    metrics = {
        "accuracy": acc,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }

    return final_labels, final_preds, metrics


def LGM_inference(model_names):
   
    for model_name in model_names:
        print(f"Stating inference for model {model_name}...")

        model_folder = os.path.join("models", model_name)
        figures_folder = model_folder
        os.makedirs(figures_folder, exist_ok=True)


        (
            model,
            lgm_criterion,
            n_classes

        ) = LGM_inference_setup(
            model_name=model_name,
            generate_dataset=True)

        # lgm intruder detection criterion
        final_labels, final_preds, metrics = lgm_intruder_detection(model=model,
                                                                        lgm_criterion=lgm_criterion, 
                                                                        figures_folder=figures_folder,  
                                                                        n_classes=n_classes,
                                                                        seed=0,)
        # save json file with log dictionary
        with open(os.path.join(model_folder, f"metrics.json"), "w") as file:
            json.dump(metrics, file)

        # Save predictions and labels
        np.save(os.path.join(model_folder, "final_labels.npy"), final_labels)
        np.save(os.path.join(model_folder, "final_preds.npy"), final_preds)
        
       

if __name__ == "__main__":

    model_names = [
        "LGM_Net_MD.2.1",
        "LGM_Net_MD.2.2",
        "LGM_Net_MD.2.3",
        "LGM_Net_MD.2.4",
        "LGM_Net_MD.2.5",
        "LGM_Net_MD.4.1",
        "LGM_Net_MD.4.2",
        "LGM_Net_MD.4.3",
        "LGM_Net_MD.4.4",
        "LGM_Net_MD.4.5",
        "LGM_Net_MD.6.1",
        "LGM_Net_MD.6.2",
        "LGM_Net_MD.6.3",
        "LGM_Net_MD.6.4",
        "LGM_Net_MD.6.5",
        "LGM_Net_MD.8.1",
        "LGM_Net_MD.8.2",
        "LGM_Net_MD.8.3",
        "LGM_Net_MD.8.4",
        "LGM_Net_MD.8.5"
    ]

    LGM_inference(model_names=model_names)
