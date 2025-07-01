"""
Run the out-of-distribution (ood) algorithm presented by [1] 'Multiscenario Open-Set Gait Recognition Based on Radar Micro-Doppler Signatures'
Since Authors of [1] did not release their code, large part of the code was taken from https://github.com/loganriggs/conditionalGaussionRecreation
which reproduces [2] 'Conditional Gaussian Distribution Learning for Open Set Recognition', from which [1] takes most part 
of the neural architecture and of the ood detection algorithm.
Whenever [1] refers to [2] for specific parts of their algorithm without explicitly reporting implementation details, 
we assume that they used the same configurations of parameters, neural architectures, etc...
"""

import numpy as np
import os
import pickle
from scipy.stats import multivariate_normal, norm
import torch
from tqdm import tqdm
from sklearn.manifold import TSNE
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri, default_converter
utils = importr('utils')

utils.install_packages("mvtnorm", repos="http://cran.uk.r-project.org/")
utils.chooseBioCmirror(ind=1)
np_cv_rules = default_converter + numpy2ri.converter
mvt = importr('mvtnorm')

import constants
from datasets import MSRadarDatasetMD, SPLIT
from models import ORCED_MD
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from matplotlib import pyplot as plt

## Functions taken from qmv.py file, from LVAE refererence code repo
def multivariateGaussian(vector, mu, sigma, dim=32):
    vector = np.array(vector)
    d = (np.mat(vector - mu)) * np.mat(np.linalg.pinv(sigma)) * (np.mat(vector - mu).T)
    p = np.exp(d * (-0.5)) / (((2 * np.pi) ** int(dim/2)) * (np.linalg.det(sigma)) ** (0.5))
    p = float(p)
    return p

def multivariateGaussianNsigma(sigma,threshold=0.5, dim=32):
    """
    This function takes advantage of this method from R
    https://rdrr.io/cran/mvtnorm/man/qmvnorm.html

    This method computes the equicoordinate quantile function of a multivariate normal distribution.

    It provides a vector where all specified dimensions of the multivariate distribution share the same quantile value
    More formally the equicoordinate quantile function for a multivariate random variable v of dimension dim, 
    is a function F:[0,1]->R^{dim} that, given an input probability value p, returns a vector y \in R^{dim} such that
    y_i is the p_i-th quantile of v_i, i.e. CDF_{v_i}(y_i) = p

    It is used to compute a value n such that, if a test sample falls outside the n * std range of the distribution, it is 
    classified as unknown. The value of p is set to 0.5.

    """
    with np_cv_rules.context():
        q = np.array(mvt.qmvnorm(threshold, sigma = sigma, tail = "both.tails")[0])
    n = q[0]
    m = (np.diag(sigma) ** 0.5) * n
    d = (np.mat(m) * np.mat(np.linalg.pinv(sigma)) * (np.mat(m).T))
    p = np.exp(d * (-0.5)) / (((2 * np.pi) ** int(dim/2)) * (np.linalg.det(sigma)) ** (0.5))
    return p

### Functions for OOD detection
def ORCED_ensemble_ood_detection(
        rec_err_tr,
        f_vecs_tr,
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
    p_nsigmas = []

    for k in range(n_classes):
        p_nsigmas.append(multivariateGaussianNsigma(np.diag(stds_z[k]**2)))
        gaussian_mode = multivariate_normal(mean=means_z[k], cov=np.diag(stds_z[k]**2))
        p_test = gaussian_mode.pdf(z_test)

        p_z_ks.append(p_test)
        p_re_k = norm.pdf(re_test, loc=means_re[k], scale=std_re[k])
        p_re_ks.append(p_re_k)
    
    # [nclasses, batch_size]
    p_z_ks = np.array(p_z_ks)
    # [nclasses,]
    p_nsigmas = np.array(p_nsigmas).squeeze()   
    # [nclasses, batch_size]
    p_re_ks = np.array(p_re_ks)  # same as above

    delta = p_z_ks - p_nsigmas[:, np.newaxis]

    p_zs_mask = np.less(delta, 0)
    latent_bools = np.sum(p_zs_mask, axis=0) == n_classes

    thresholds_re_array = np.array(
        [thresholds_re[j] for j in x_test_prediction])
    rec_err_bools = re_test > thresholds_re_array

    out_unseen_mask = np.logical_or(latent_bools, rec_err_bools)
    out_classes = torch.clone(x_test_prediction.detach().cpu())
    out_classes[out_unseen_mask] = n_classes

    return out_classes


def ORCED_inference_setup(model_name, loaders_batch_size, generate_dataset=True):

    # Load Config file
    model_path = os.path.join("models", model_name)
    with open(os.path.join(model_path, "config.pkl"), "rb") as file:
        config = pickle.load(file)

    # Generate dataset according to config file
    if generate_dataset:
        print("Generating dataset...")
        MSRadarDatasetMD.generate_splits(train_classes=config["TRAIN_CLASSES"], split_seed=0, safe_mode=False)

    # ----- DEFINE DATA AND MODELS -----
    print("Loading model weights...")
    ## Test load parameters 
    model = ORCED_MD(alpha=100, beta=0, gamma=1, xi=1, tau=0.07, num_class=len(config["TRAIN_CLASSES"])).to(constants.DEVICE)
    statedict = torch.load(os.path.join(model_path, f"{config['MODEL_NAME']}.pt"), weights_only=True)
    model.load_state_dict(statedict)
    print("Model loaded successfully")

    # Configure data loaders
    radar_dataset_train = MSRadarDatasetMD(
        SPLIT.TRAIN,
        subsample_factor=config["SUBSAMPLE_FACTOR"], 
        imagenet_transform=False, 
        augmentation=False
    )

    dataloader_train = torch.utils.data.DataLoader(
        radar_dataset_train,
        batch_size=loaders_batch_size,
        drop_last=True,
        shuffle=False,
        num_workers=0,
    )

    radar_dataset_test = MSRadarDatasetMD(
        SPLIT.TEST,
        subsample_factor=1.0,
        imagenet_transform=False,
        augmentation=False
    )

    dataloader_test = torch.utils.data.DataLoader(
        radar_dataset_test,
        batch_size=loaders_batch_size,
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    radar_dataset_unseen = MSRadarDatasetMD(
        SPLIT.UNSEEN,
        subsample_factor=1,
        imagenet_transform=False,
        augmentation=False
    )

    dataloader_unseen = torch.utils.data.DataLoader(
        radar_dataset_unseen,
        batch_size=loaders_batch_size,
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

   
    model.eval()

    print("Setup complete!")
    n_classes = len(config["TRAIN_CLASSES"])
    return model, n_classes,  dataloader_train, dataloader_test, dataloader_unseen

def plot_tsne(model, test_dataloader, unseen_dataloader, n_classes, save_dir):

    all_feats = []
    all_labels = []
    all_domains = []  # 0 for test, 1 for unseen

    # Seen data
    for x, y in test_dataloader:
        x = x.to(constants.DEVICE)
        y = y.to(constants.DEVICE)
        target_en_test = torch.Tensor(y.shape[0], n_classes)
        target_en_test = target_en_test.to(constants.DEVICE)
        target_en_test.zero_()
        target_en_test.scatter_(1, y.unsqueeze(1), 1)  # one-hot encoding

        # Forward pass
        feats, mu_latent, var_latent, \
        qmu5_1, qvar5_1, qmu4_2, qvar4_2, qmu4_1, qvar4_1, qmu3_2, qvar3_2, qmu3_1, qvar3_1, \
        qmu2_2, qvar2_2, qmu2_1, qvar2_1, qmu1_2, qvar1_2, qmu1_1, qvar1_1, \
        yh, \
        test_x_re, \
        pmu5_1, pvar5_1,pmu4_2, pvar4_2, pmu4_1, pvar4_1, pmu3_2, pvar3_2, pmu3_1, pvar3_1, \
        pmu2_2, pvar2_2, pmu2_1, pvar2_1, pmu1_2, pvar1_2, pmu1_1, pvar1_1 = model.lnet(x, target_en_test)
        all_feats.append(feats.detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())
        all_domains.append(np.zeros_like(y.detach().cpu().numpy()))  # domain = 0
    for md_unseen, y_unseen in unseen_dataloader:
        md_unseen = md_unseen.to(constants.DEVICE)
        y_unseen = y_unseen.to(constants.DEVICE)

        target_en_unseen = torch.Tensor(torch.zeros(n_classes)).to(constants.DEVICE)
        target_en_unseen[0]=1
        # target_en_unseen = torch.Tensor(y_unseen.shape[0], n_classes)
        # target_en_unseen = target_en_unseen.to(constants.DEVICE)
        # target_en_unseen.zero_()
        # target_en_unseen.scatter_(1, y_unseen.unsqueeze(1), 1)  # one-hot encoding

        # Forward pass
        feats_unseen, mu_latent, var_latent, \
        qmu5_1, qvar5_1, qmu4_2, qvar4_2, qmu4_1, qvar4_1, qmu3_2, qvar3_2, qmu3_1, qvar3_1, \
        qmu2_2, qvar2_2, qmu2_1, qvar2_1, qmu1_2, qvar1_2, qmu1_1, qvar1_1, \
        yh, \
        test_x_re, \
        pmu5_1, pvar5_1,pmu4_2, pvar4_2, pmu4_1, pvar4_1, pmu3_2, pvar3_2, pmu3_1, pvar3_1, \
        pmu2_2, pvar2_2, pmu2_1, pvar2_1, pmu1_2, pvar1_2, pmu1_1, pvar1_1 = model.lnet(md_unseen, target_en_unseen)
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

def ORCED_inference(model_names):
    for model_name in model_names:
        figures_folder = os.path.join("models", model_name)
        batch_size = 32
        os.makedirs(figures_folder, exist_ok=True)

        (
            model,
            n_classes,
            dataloader_train,
            dataloader_test,
            dataloader_unseen,
        ) = ORCED_inference_setup(model_name=model_name,
                                  loaders_batch_size=batch_size,
                                  generate_dataset=True)

        ## Plot feature space to visualize if it makes sense
        plot_tsne(model, dataloader_test, dataloader_unseen, n_classes, figures_folder)
        
        train_feature_vectors = []
        train_rec_losses = []
        all_train_labels = []
        all_train_preds = []

        print("Computing training set statistics...")
        with torch.no_grad():
            for i, (train_md, train_gt_labels) in tqdm(enumerate(dataloader_train)):
                train_md = train_md.to(constants.DEVICE)
                train_gt_labels = train_gt_labels.to(constants.DEVICE)

                target_en = torch.Tensor(train_gt_labels.shape[0], n_classes)
                target_en = target_en.to(constants.DEVICE)
                target_en.zero_()
                target_en.scatter_(1, train_gt_labels.unsqueeze(1), 1)  # one-hot encoding

                # Forward pass
                latent, mu_latent, var_latent, \
                qmu5_1, qvar5_1, qmu4_2, qvar4_2, qmu4_1, qvar4_1, qmu3_2, qvar3_2, qmu3_1, qvar3_1, \
                qmu2_2, qvar2_2, qmu2_1, qvar2_1, qmu1_2, qvar1_2, qmu1_1, qvar1_1, \
                yh, \
                x_re, \
                pmu5_1, pvar5_1,pmu4_2, pvar4_2, pmu4_1, pvar4_1, pmu3_2, pvar3_2, pmu3_1, pvar3_1, \
                pmu2_2, pvar2_2, pmu2_1, pvar2_1, pmu1_2, pvar1_2, pmu1_1, pvar1_1 = model.lnet(train_md, target_en)
                
                train_feature_vectors.append(latent.cpu().numpy())
            
                # train_g_loss = chamfer_loss(
                #     train_rec_pc, train_md, avg_out=False)
                train_g_loss = torch.norm(train_md-x_re, p=1, dim=(1,2,3))
                train_rec_losses.append(train_g_loss.detach().cpu().numpy())

                # norm_train_preds = torch.nn.Softmax(dim=1)(train_preds)
                logits = model.classifier(latent)
                # norm_train_preds = torch.nn.Softmax(dim=1)(logits, dim=1)
                index_train_preds = torch.argmax(logits, dim=1)

                all_train_preds.append(index_train_preds.cpu().numpy())
                all_train_labels.append(train_gt_labels.cpu().numpy())

        rec_err_tr = np.concatenate(train_rec_losses)
        f_vecs_tr = np.concatenate(train_feature_vectors)
        gt_labels = np.concatenate(all_train_labels)
        pred_labels = np.concatenate(all_train_preds)



        all_test_labels = []
        all_unseen_labels = []
        n_labels = len(np.unique(gt_labels))

        print("Computing test set statistics...")
        test_openset_predictions_array = []
        unseen_openset_predictions_array = []
        with torch.no_grad():
            for i, (test_md, test_gt_labels) in tqdm(enumerate(dataloader_test)):
                test_md = test_md.to(constants.DEVICE)
                test_gt_labels = test_gt_labels.to(constants.DEVICE)
                all_test_labels.append(test_gt_labels.cpu().numpy())

                target_en_test = torch.Tensor(test_gt_labels.shape[0], n_classes)
                target_en_test = target_en_test.to(constants.DEVICE)
                target_en_test.zero_()
                target_en_test.scatter_(1, test_gt_labels.unsqueeze(1), 1)  # one-hot encoding

                # Forward pass
                test_latent, mu_latent, var_latent, \
                qmu5_1, qvar5_1, qmu4_2, qvar4_2, qmu4_1, qvar4_1, qmu3_2, qvar3_2, qmu3_1, qvar3_1, \
                qmu2_2, qvar2_2, qmu2_1, qvar2_1, qmu1_2, qvar1_2, qmu1_1, qvar1_1, \
                yh, \
                test_x_re, \
                pmu5_1, pvar5_1,pmu4_2, pvar4_2, pmu4_1, pvar4_1, pmu3_2, pvar3_2, pmu3_1, pvar3_1, \
                pmu2_2, pvar2_2, pmu2_1, pvar2_1, pmu1_2, pvar1_2, pmu1_1, pvar1_1 = model.lnet(test_md, target_en_test)

                test_latent = mu_latent
                logits = model.classifier(test_latent)
                index_test_preds = torch.argmax(logits, dim=1)
                test_g_loss = torch.norm(test_md-test_x_re, p=1, dim=(1,2,3))
                # all_test_preds.append(index_test_preds.cpu().numpy())

                test_openset_predictions = ORCED_ensemble_ood_detection(
                    rec_err_tr,
                    f_vecs_tr,
                    gt_labels,
                    pred_labels,
                    index_test_preds,
                    test_latent.detach().cpu().numpy(),
                    test_g_loss.detach().cpu().numpy())

                print(
                    f"N. of correct predictions: {torch.sum(test_gt_labels == test_openset_predictions.to(constants.DEVICE))}")

                test_openset_predictions_array.append(test_openset_predictions)

                # if i == 0:
                #     print("Breaking from for loop (for debugging)")
                #     break

            print("Computing unseen set statistics...")

            # I have to consider a leave out label, because in the CGAAE i actually use one unseen subject
            # as validation to compute the threshold.
            # For a fair comparison I have to consider the same number of ACTUALLY unseen subjects

            leave_out_label = None

            for i, (unseen_md, unseen_gt_labels) in tqdm(enumerate(dataloader_unseen)):
                unseen_md = unseen_md.to(constants.DEVICE)

                if leave_out_label is None:
                    leave_out_label = unseen_gt_labels[0].item()

                unseen_gt_labels = unseen_gt_labels.to(constants.DEVICE)
                all_unseen_labels.append(n_labels)

                # Select only samples not belonging to the leave_out_label
                not_leave_out_mask = unseen_gt_labels != leave_out_label
                unseen_md = unseen_md[not_leave_out_mask]

                ## NB: this one-hot encoded label is not used for anything, I just need
                target_en_unseen = torch.Tensor(unseen_gt_labels.shape[0], n_classes)
                target_en_unseen = target_en_unseen.to(constants.DEVICE)
                target_en_unseen.zero_()
                target_en_unseen[:,0] = 1
                # target_en_unseen.scatter_(1, unseen_gt_labels.unsqueeze(1), 1)  # one-hot encoding

                # Forward pass
                unseen_latent, mu_latent, var_latent, \
                qmu5_1, qvar5_1, qmu4_2, qvar4_2, qmu4_1, qvar4_1, qmu3_2, qvar3_2, qmu3_1, qvar3_1, \
                qmu2_2, qvar2_2, qmu2_1, qvar2_1, qmu1_2, qvar1_2, qmu1_1, qvar1_1, \
                yh, \
                unseen_x_re, \
                pmu5_1, pvar5_1,pmu4_2, pvar4_2, pmu4_1, pvar4_1, pmu3_2, pvar3_2, pmu3_1, pvar3_1, \
                pmu2_2, pvar2_2, pmu2_1, pvar2_1, pmu1_2, pvar1_2, pmu1_1, pvar1_1 = model.lnet(unseen_md, target_en_unseen)

                unseen_latent = mu_latent
                logits = model.classifier(unseen_latent)
                index_unseen_preds = torch.argmax(logits, dim=1)
                unseen_g_loss = torch.norm(unseen_md-unseen_x_re, p=1, dim=(1,2,3))


                unseen_openset_predictions = ORCED_ensemble_ood_detection(
                    rec_err_tr,
                    f_vecs_tr,
                    gt_labels,
                    pred_labels,
                    index_unseen_preds,
                    unseen_latent.detach().cpu().numpy(),
                    unseen_g_loss.detach().cpu().numpy())
                    
                print(f"N. of correct predictions: {torch.sum(n_classes == unseen_openset_predictions.to(constants.DEVICE))}")

                unseen_openset_predictions_array.append(unseen_openset_predictions)


        test_openset_predictions_array = np.concatenate(test_openset_predictions_array)
        unseen_openset_predictions_array = np.concatenate(unseen_openset_predictions_array)

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
        "ORCED_MD_TEST.2.1",
        "ORCED_MD_TEST.2.2",
        "ORCED_MD_TEST.2.3",
        "ORCED_MD_TEST.2.4",
        "ORCED_MD_TEST.2.5",
        "ORCED_MD_TEST.4.1",
        "ORCED_MD_TEST.4.2",
        "ORCED_MD_TEST.4.3",
        "ORCED_MD_TEST.4.4",
        "ORCED_MD_TEST.4.5",
        "ORCED_MD_TEST.6.1",
        "ORCED_MD_TEST.6.2",
        "ORCED_MD_TEST.6.3",
        "ORCED_MD_TEST.6.4",
        "ORCED_MD_TEST.6.5",
        "ORCED_MD_TEST.8.1",
        "ORCED_MD_TEST.8.2",
        "ORCED_MD_TEST.8.3",
        "ORCED_MD_TEST.8.4",
        "ORCED_MD_TEST.8.5"
    ]
    ORCED_inference(ORCED_model_names)

