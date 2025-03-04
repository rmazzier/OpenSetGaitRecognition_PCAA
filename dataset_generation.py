import enum
import os
import pickle
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import torch

import constants
from utils import plot_pointcloud


def crop_with_step(sequence, crop_len, step):
    """Given a numpy array of shape (n, ...), it crops it into overlapping subportions of shape (crop_len, ...), with a sliding window of step.
    Final number of obtained arrays will be ceil((n-crop_len)/step).

    Parameters:
    :param sequence: The array to crop;
    :param crop_len: The width of the sliding window;
    :param step: The step of the sliding window"""
    idxs = np.arange(len(sequence) - crop_len, step=step)
    return np.array([sequence[idx: idx + crop_len] for idx in idxs])


def gen_crops_from_track_file(
    pc_file,
    subject_index,
    target_dir,
    standardize_point_cloud=True,
    n_points=constants.NMAX,
    n_features=constants.NFEATURES,
    seq_length=constants.NSTEPS,
    crop_step=constants.CROP_STEP,
    divide_by_std=False,
    from_db=False,
    force_pc_subsampling=0,
    seed=0,
):
    """Generate .npy files from a specific pc file (i.e. a specific track, from a specific subject),
    and save them all on target_dir.

    NB: Make sure to respect the following naming conventions:
        - Folders containing track files must be named f"target{index}";
        - Output crops must be named: f"crop{crop_index}_subj{subject_index}_track{track_index}.npy"
        (example: crop0_subj0_track17_0.npy, note that track index is not a single integer
        but a string containing two numbers)

    Parameters:
    :param pc_file: The whole path for the track file to generate crops from;
    :param subject_index: The index of the subject relative to the input track file;
    :param target_dir: The whole path for the directory to save the crops in;
    :param standardize_point_cloud: If True, standardize each point cloud
    (i.e. subtract by mean, divide by standard deviation of each of the 5 features)
    :param n_points: Fixed number of points of each output point cloud;
    :param n_features: Number of features to consider. (Set to 5 to use all features. If set
    to 4, received power is ignored. If set to 3, doppler is ignored.)
    :param seq_length: The length of each output point cloud sequence crop;
    :param crop_step: The crop step used during the cropping phase;
    :param divide_by_std: True by default. If false, point cloud are just centered but not divided by
    their standard deviation during the standardization phase.
    """

    # this flag makes it so that numpy raises an error when encountering runtime overflow
    np.seterr(all="raise")

    # Set numpy rng with seed
    rng = np.random.default_rng(seed)

    # Define the path of the pc file and load it
    loaded_file = pickle.load(open(pc_file, "rb"))
    pcloud_array = np.array([])

    n_overflows = 0

    # Iterate all frames in the point cloud sequence and update pcloud_array
    for frame_idx, frame in enumerate(loaded_file):
        # Extract the data from the frame dictionary
        frame_cardinality = frame["cardinality"][0]
        frame_elements = frame["elements"]
        frame_zs = frame["z_coord"][:, np.newaxis]
        frame_dopplers = frame["dopplers"][:, np.newaxis]
        frame_powers = frame["powers"][:, np.newaxis]

        # Check if force subsampling is enabled, if yes, choose a random subset
        # of indices to keep
        if force_pc_subsampling > 0:
            frame_cardinality = force_pc_subsampling
            choices = rng.choice(
                frame_cardinality, force_pc_subsampling, replace=False
            )
            frame_elements = frame_elements[choices]
            frame_zs = frame_zs[choices]
            frame_dopplers = frame_dopplers[choices]
            frame_powers = frame_powers[choices]

        # convert to db scale
        frame_powers = 10 * np.log10(frame_powers + 1e-8)

        # Concatenate column vectors to create a matrix of shape (n_points_in_frame, n_features)
        frame_array = np.concatenate(
            [frame_elements, frame_zs, frame_dopplers, frame_powers], axis=1
        )[:, :n_features]

        # Check whether point cloud has more or less than `n_points` points
        if frame_cardinality < n_points:
            # If less than `n_points`, randomly repeat points until we arrive to `n_points`
            excess_points = n_points - frame_cardinality
            z = np.zeros((excess_points, n_features))
            choices = np.random.choice(frame_cardinality, excess_points)
            for i, c in enumerate(choices):
                z[i, :] = frame_array[c, :]
            final_frame = np.concatenate([frame_array, z], axis=0)
        else:
            # If more than `n_points`, randomly sample `n_points` points from the point cloud
            choices = np.random.choice(
                frame_cardinality, n_points, replace=False)
            final_frame = frame_array[choices, :]

        if standardize_point_cloud:
            try:
                # Subtract the mean
                ff_mean = final_frame.mean(axis=0)
                ff_std = final_frame.std(axis=0)
                final_frame = final_frame - ff_mean

                if divide_by_std:
                    final_frame = final_frame / (ff_std + 1e-8)

            except:
                # print(f"Overflow standardizing; i={frame_index};")
                n_overflows += 1
                assert False

        # Fill the corresponding row of the pcloud_array with the values for the current frame, which has shape (n_points, 5)
        if pcloud_array.shape[0] == 0:
            nff = final_frame[np.newaxis, :, :]
            pcloud_array = nff
        else:
            pcloud_array = np.concatenate(
                [pcloud_array, final_frame[np.newaxis, :, :]], 0
            )

    # Crop the pcloud_array with the chosen `seq_length` and `crop_step` parameters
    cropped_pcloud_array = crop_with_step(pcloud_array, seq_length, crop_step)

    for crop_index in range(len(cropped_pcloud_array)):
        track_index = pc_file.split("/")[-1][3:].split(".")[0]
        np.save(
            os.path.join(
                target_dir,
                f"crop{crop_index}_subj{subject_index}_track{track_index}.npy",
            ),
            cropped_pcloud_array[crop_index],
        )


def get_sorted_seq(path, subject_id, track_id):
    """Returns an array of sequentially sorted crops from a specific subject and from a specific track."""

    all_files = os.listdir(path)
    crops = [f for f in all_files if "KF" not in f]

    # Get only crops of selected subject
    subj_sel_crops = [f for f in crops if f"subj{subject_id}" in f]

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
