import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import constants


def printGPUInfo():
    t = torch.cuda.get_device_properties(0).total_memory / 1e9
    r = torch.cuda.memory_reserved(0) / 1e9
    a = torch.cuda.memory_allocated(0) / 1e9
    f = r-a  # free inside reserved
    print(f"Reserved: {r:.3f} GB, Allocated: {a:.3f} GB, Free: {f:.3f} GB")


def plot_pointcloud(
    ax: plt.Axes,
    point_cloud: torch.Tensor,
    title="Title",
    show_axis=True,
    aspect="equal",
    point_dimension=8,
):
    """
    Plots a point cloud, given an Axis object to draw on.

    :param ax: is an Axes object obtained with:
    `ax = fig.add_subplot(m, n, 1, projection="3d")`
    :point_cloud: PyTorch array containing point cloud [n_points, 3]
    :returns: None
    """

    xs = point_cloud.cpu()[:, 0]
    ys = point_cloud.cpu()[:, 1]
    zs = point_cloud.cpu()[:, 2]

    r0 = xs - torch.min(xs)
    g0 = ys - torch.min(ys)
    b0 = zs - torch.min(zs)

    r = r0 / torch.max(r0)
    g = g0 / torch.max(g0)
    b = b0 / torch.max(b0)

    cols = torch.stack([r, g, b], dim=1)
    ax.scatter(xs, ys, zs, c=cols, s=point_dimension)
    ax.set_xlim3d(-20, 20)
    ax.set_ylim3d(-10, 10)
    ax.set_zlim3d(-10, 10)

    if aspect == "equal":
        # To make equal aspect ratio
        ax.set_box_aspect(
            (
                np.ptp(xs),
                np.ptp(zs),
                np.ptp(ys),
            )
        )
    if not show_axis:
        ax.set_axis_off()

    ax.set_title(title)


def CG_kl_divergence(mu, logvar, mu_k):
    """
    KL divergence between a normal distribution with mean mu_k and unit variance (prior),
    and a normal distribution with mean mu and std = exp(0.5 * logvar).
    From equation (6) of paper "Conditional Gaussian Distribution Learning for Open Set Recognition"

    :param mu: mean of the distribution of shape (batch_size, latent_dim)
    :param logvar: standard dev of the distribution of shape (batch_size, latent_dim)
    :param mu_k: mean of the k-th prior distribution (learned) of shape (batch_size, latent_dim)
    """
    batch_kl_div = -0.5 * torch.sum(
        1 + logvar - (mu - mu_k) ** 2 - torch.exp(logvar), axis=1
    )
    return torch.mean(batch_kl_div)


class SeqChamferLoss(torch.nn.Module):
    """Code from here:
    https://github.com/MaciejZamorski/3d-AAE/blob/master/losses/champfer_loss.py
    Adapted for point cloud sequences
    """

    def __init__(self):
        super(SeqChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def forward(self, preds, gts, avg_out=True):
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 2)
        loss_1 = torch.sum(mins, dim=2)
        mins, _ = torch.min(P, 3)
        loss_2 = torch.sum(mins, dim=2)
        if avg_out:
            return torch.mean(loss_1 + loss_2)
        else:
            return torch.mean(loss_1 + loss_2, dim=1)

    def batch_pairwise_dist(self, x, y):

        # x and y will come with shape (b_size, channels, seq_len, n_points)
        # First,reshape them to (b_size, seq_len, n_points, channels)

        x = torch.permute(x, (0, 2, 3, 1))
        y = torch.permute(y, (0, 2, 3, 1))

        _, _, num_points_x, _ = x.size()
        _, _, num_points_y, _ = y.size()

        xx = torch.matmul(x, x.permute((0, 1, 3, 2)))
        yy = torch.matmul(y, y.permute((0, 1, 3, 2)))
        zz = torch.matmul(x, y.permute((0, 1, 3, 2)))

        diag_ind_x = torch.arange(0, num_points_x).to(constants.DEVICE)
        diag_ind_y = torch.arange(0, num_points_y).to(constants.DEVICE)

        rx = xx[:, :, diag_ind_x, diag_ind_x].unsqueeze(3).expand_as(zz)
        ry = yy[:, :, diag_ind_y, diag_ind_y].unsqueeze(2).expand_as(zz)

        # P: square distance between each couple of points
        P = rx + ry - 2 * zz
        return P


def gradient_penalty(critic, z_noise, codes, latent_dim):
    alphas = (
        torch.rand(size=(constants.BATCH_SIZE, 1))
        .repeat(1, latent_dim)
        .to(constants.DEVICE)
    )

    differences = codes - z_noise
    interpolates = z_noise + alphas * differences
    disc_interpolates = critic(interpolates)

    # Compute gradient of discriminator w.r.t input (i.e. interpolated codes)
    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates).to(constants.DEVICE),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    slopes = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-12)
    gradient_penalty = ((slopes - 1) ** 2).mean()
    return gradient_penalty


def save_model(_model: torch.nn.Module, _path):
    torch.save(_model.state_dict(), _path)


def get_sorted_seq(path, subject_id, track_id):
    """Returns an array of sequentially sorted crops from a specific subject and from a
    specific track."""

    all_files = os.listdir(path)

    # Get only crops of selected subject
    subj_sel_crops = [f for f in all_files if f"subj{subject_id}" in f]

    # Retrieve unique sequence ids
    all_track_ids = np.unique([crop.split("track")[-1][:-5]
                              for crop in subj_sel_crops])

    # Get the crops of selected track (selected by index)
    selected_track = all_track_ids[track_id]
    sel_crops = [c for c in subj_sel_crops if f"track{selected_track}" in c]

    crops_ids = np.array([int(c.split("_")[0][4:]) for c in sel_crops])
    sorted_ids = np.argsort(crops_ids)

    sorted_crops = np.array(sel_crops)[sorted_ids]
    return sorted_crops


def get_subjects_from_crops(path):
    """path: directory where the crop files are stored. Returns the indices of the subjects.
    NB: Relies on naming convention: `crop{crop_n}_subj{subject_index}_track{track_id}_.npy`"""
    all_crops = os.listdir(path)
    return [int(c.split(".")[0].split("_")[1][4:]) for c in all_crops]


def get_track_dict_from_crops(path):
    """returns a dictionary where the keys are the subject indexes and values are tracks ids contained
    in the directory containing the crops"""
    all_crops = os.listdir(path)
    val_track_dict = {}

    for crop in all_crops:
        subj_index = int(crop.split(".")[0].split("_")[1][4:])
        # track_id = crop.split("_")[2][5:]
        track_id = "_".join(crop.split(".")[0].split("_")[2:])[5:]

        val_track_dict.setdefault(subj_index, set())

        val_track_dict[subj_index].add(track_id)
    return val_track_dict


def openness(n_train, n_test):
    return 1 - np.sqrt((2 * n_train) / (n_train + n_test))


def sample_distant_points(dimension, n, min_dist, sphere_radius, seed=42):
    rng = np.random.default_rng(seed)

    def sample_spherical(npoints, rng, ndim=3):
        vec = rng.standard_normal(size=(ndim, npoints))
        vec /= np.linalg.norm(vec, axis=0)
        return vec

    def farthest_point_sampling(points, n_samples, rng):
        # initialize
        n_points = points.shape[0]
        distances = np.ones(n_points) * 1e10
        farthest = rng.integers(low=0, high=n_points)
        sampled = [farthest]
        # sample
        for i in range(n_samples - 1):
            dist = np.sum((points - points[farthest]) ** 2, axis=1)
            distances = np.minimum(distances, dist)
            farthest = np.argmax(distances)
            sampled.append(farthest)
        return sampled

    npoints = 10000
    vec = sample_spherical(npoints, rng, dimension) * sphere_radius

    min_dist_ = 0
    while min_dist_ < min_dist:
        print(min_dist_)
        sampled_idxs = farthest_point_sampling(vec.T, n, rng)
        sampled_vectors = vec[:, sampled_idxs]
        dist = torch.cdist(
            torch.tensor(sampled_vectors.T), torch.tensor(sampled_vectors.T)
        )
        min_dist_ = torch.min(dist[dist > 0])

    return torch.tensor(sampled_vectors.T)
