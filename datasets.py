from typing import List
from tqdm import tqdm
import pickle
import numpy as np
import torch
import os
from sklearn.model_selection import train_test_split
import time

from dataset_generation import gen_crops_from_track_file, get_sorted_seq
from utils import get_track_dict_from_crops, openness
from constants import SPLIT, SCENARIO
import constants


def crop_with_step(sequence, crop_len, step):
    """Given a numpy array of shape (n, ...), it crops it into overlapping subportions of shape (crop_len, ...), with a sliding window of step.
    Final number of obtained arrays will be ceil((n-crop_len)/step).

    Parameters:
    :param sequence: The array to crop;
    :param crop_len: The width of the sliding window;
    :param step: The step of the sliding window"""
    idxs = np.arange(len(sequence) - crop_len, step=step)
    return np.array([sequence[idx: idx + crop_len] for idx in idxs])


class MSRadarDataset(torch.utils.data.Dataset):

    # label_dict = {
    #     0: "target0",
    #     1: "target1",
    #     2: "target2",
    #     3: "target3",
    #     4: "target4",
    #     5: "target5",
    #     6: "target6",
    #     7: "target7",
    #     8: "target8",
    #     9: "target9",
    #     10: "target10",
    #     11: "target11",
    #     12: "target12",
    #     13: "target13",
    #     14: "target14",
    #     15: "target15",
    # }

    label_dict = {
        0: "target0",
        1: "target1",
        2: "target2",
        3: "target3",
        4: "target4",
        5: "target5",
        6: "target6",
        7: "target7",
        8: "target8",
        9: "target9",
    }

    @staticmethod
    def filename2crop(filename):
        return int(filename.split("_")[0][4:])  # cropX

    @staticmethod
    def filename2subj(filename):
        return int(filename.split("_")[1][4:])  # subjX

    @staticmethod
    def filename2track(filename):
        return filename.split("_")[-1][5:].split(".")[0]  # trackX

    @staticmethod
    def filename2scenario(filename):
        return str.join("_", filename.split("_")[2:-1])  # scenario

    @staticmethod
    def process_track(track_file_path, standardize_point_cloud, divide_by_std, force_pc_subsampling=0, nmax=constants.NMAX, rng=np.random):
        """
        This method takes as input a raw track file and performs the following preprocessing operations:
        - Convert powers to dB;
        - Standardize the point cloud (i.e. subtract mean and divide by standard deviation);
        - Randomly repeats points if the point cloud has less than `constants.NMAX` points, OR removes
        randomly some points if the point cloud has more than `constants.NMAX` points.

        Returns:
        - pcloud_array: A numpy array of shape (n_frames, n_points, n_features) containing the processed point cloud sequence.
        """
        # create rng with seed
        rng = np.random.default_rng(0)
        # open track file with pickle
        loaded_file = pickle.load(open(track_file_path, "rb"))

        pcloud_array = np.array([])

        # Iterate all frames in the point cloud sequence and update pcloud_array
        for frame in loaded_file:
            # Extract the data from the frame dictionary
            frame_cardinality = frame["cardinality"][0]
            frame_elements = frame["elements"]
            frame_zs = frame["z_coord"][:, np.newaxis]
            frame_dopplers = frame["dopplers"][:, np.newaxis]
            frame_powers = frame["powers"][:, np.newaxis]

            # Check if force subsampling is enabled, if yes, choose a random subset
            # of indices to keep
            if force_pc_subsampling > 0 and force_pc_subsampling < frame_cardinality:
                frame_cardinality = force_pc_subsampling
                choices = rng.choice(
                    frame_cardinality, force_pc_subsampling, replace=False
                )
                frame_elements = frame_elements[choices]
                frame_zs = frame_zs[choices]
                frame_dopplers = frame_dopplers[choices]
                frame_powers = frame_powers[choices]

            # convert frame powers to db
            frame_powers = 10 * np.log10(frame_powers + 1e-8)

            # Concatenate column vectors to create a matrix of shape (n_points_in_frame, n_features)
            frame_array = np.concatenate(
                [frame_elements, frame_zs, frame_dopplers, frame_powers], axis=1
            )[:, : constants.NFEATURES]

            # Check whether point cloud has more or less than nmax points
            if frame_cardinality < nmax:
                # If less than `nmax`, randomly repeat points until we arrive to `n_points`
                excess_points = nmax - frame_cardinality
                z = np.zeros((excess_points, constants.NFEATURES))
                choices = np.random.choice(frame_cardinality, excess_points)
                for i, c in enumerate(choices):
                    z[i, :] = frame_array[c, :]
                final_frame = np.concatenate([frame_array, z], axis=0)
            else:
                # If more than `n_points`, randomly sample `n_points` points from the point cloud
                choices = np.random.choice(
                    frame_cardinality, nmax, replace=False
                )
                final_frame = frame_array[choices, :]

            if standardize_point_cloud:
                # Subtract the mean
                ff_mean = final_frame.mean(axis=0)
                ff_std = final_frame.std(axis=0)
                final_frame = final_frame - ff_mean
                # Dividing by std makes all dimensions with same "extent"
                # In this case dividing by std could cause information loss?
                if divide_by_std:
                    final_frame = final_frame / (ff_std + 1e-8)

            # Fill the corresponding row of the pcloud_array with the values for the current frame, which has shape (n_points, 5)
            if pcloud_array.shape[0] == 0:
                nff = final_frame[np.newaxis, :, :]
                pcloud_array = nff
            else:
                pcloud_array = np.concatenate(
                    [pcloud_array, final_frame[np.newaxis, :, :]], 0
                )

        return pcloud_array

    @staticmethod
    def get_sorted_seq(path, subject_id, track_id):
        """Returns an array of sequentially sorted crops from a specific subject and from a specific track."""

        all_files = os.listdir(path)

        # Get only crops of selected subject
        sel_crops = [f for f in all_files if f"subj{subject_id}" in f]

        # Retrieve only crops of a specific track
        sel_crops = [c for c in sel_crops if f"track{track_id}" in c]

        crops_ids = np.array([MSRadarDataset.filename2crop(c)
                             for c in sel_crops])
        sorted_ids = np.argsort(crops_ids)

        sorted_crops = np.array(sel_crops)[sorted_ids]
        return sorted_crops

    @staticmethod
    def generate_splits(
        train_classes=[],
        train_ratio=0.8,
        valid_ratio=0.1,
        test_ratio=0.1,
        seed=0,
        safe_mode=True,
        force_pc_subsampling=0,
        nmax_points=constants.NMAX,
    ):
        """
        Generate train/validation/test splits for the multiscenario dataset,
        which are saved in the generated_dataset folder.
        The dataset is split so that each split contains the same proportion of
        walking manners (i.e. free_walk, hands_in_pocket, smartphone), resulting in an
        approximately balanced dataset.

        """
        print(
            f"Generating splits with the following parameters:\nTrain Classes: {train_classes}\nTrain ratio: {train_ratio}\nValid ratio: {valid_ratio}\nTest ratio: {test_ratio}\nSeed: {seed}"
        )
        if safe_mode:
            # Request input from user to confirm
            print("Press ENTER to confirm, or CTRL+C to abort.")
            input()

        start_time = time.time()
        train_dir = os.path.join(constants.GEN_DATA_PATH, "train")
        valid_dir = os.path.join(constants.GEN_DATA_PATH, "valid")
        test_dir = os.path.join(constants.GEN_DATA_PATH, "test")
        unseen_dir = os.path.join(constants.GEN_DATA_PATH, "unseen")

        unseen_classes = np.setdiff1d(
            list(MSRadarDataset.label_dict.keys()), train_classes
        ).tolist()

        opns = (
            openness(
                n_train=len(train_classes),
                n_test=len(MSRadarDataset.label_dict),
            )
            * 100
        )

        # create directories
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(valid_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        os.makedirs(unseen_dir, exist_ok=True)

        # empty them in case they already existed
        [os.remove(os.path.join(train_dir, f)) for f in os.listdir(train_dir)]
        [os.remove(os.path.join(valid_dir, f)) for f in os.listdir(valid_dir)]
        [os.remove(os.path.join(test_dir, f)) for f in os.listdir(test_dir)]
        [os.remove(os.path.join(unseen_dir, f))
         for f in os.listdir(unseen_dir)]

        assert train_ratio + valid_ratio + test_ratio == 1.0

        if train_classes == []:
            train_classes = list(MSRadarDataset.label_dict.keys())

        print("Generating Train, Valid and Test splits ....")
        for subj_idx in tqdm(train_classes):
            subject_dir = os.path.join(
                constants.DATA_PATH, MSRadarDataset.label_dict[subj_idx]
            )

            all_scenarios = os.listdir(subject_dir)
            assert [
                s in ["free_walk", "hands_in_pocket", "smartphone"]
                for s in all_scenarios
            ]
            for scenario in all_scenarios:
                # Assert that the folder contains only files sstarting with "pc"
                assert all(
                    [
                        x[:2] == "pc"
                        for x in np.array(
                            os.listdir(os.path.join(subject_dir, scenario))
                        )
                    ]
                ), "Invalid file in subject folder"

                all_tracks = os.listdir(os.path.join(subject_dir, scenario))

                train_tracks, valid_test_tracks = train_test_split(
                    all_tracks, train_size=train_ratio, random_state=seed
                )
                valid_tracks, test_tracks = train_test_split(
                    valid_test_tracks,
                    train_size=(valid_ratio / (valid_ratio + test_ratio)),
                    random_state=seed,
                )

                for tracks_set, target_dir in [
                    (train_tracks, train_dir),
                    (valid_tracks, valid_dir),
                    (test_tracks, test_dir),
                ]:
                    for track in tracks_set:
                        # gen_crops_from_track_file(
                        #     os.path.join(subject_dir, scenario, track),
                        #     idx,
                        #     target_dir=dir,
                        # )
                        pc_file = os.path.join(subject_dir, scenario, track)
                        pcloud_array = MSRadarDataset.process_track(
                            pc_file,
                            standardize_point_cloud=True,
                            divide_by_std=False,
                            force_pc_subsampling=force_pc_subsampling, nmax=nmax_points
                        )

                        # Crop the pcloud_array with the chosen `seq_length` and `crop_step` parameters
                        cropped_pcloud_array = crop_with_step(
                            pcloud_array,
                            crop_len=constants.NSTEPS,
                            step=constants.CROP_STEP,
                        )

                        for crop_index in range(len(cropped_pcloud_array)):
                            track_index = pc_file.split(
                                "/")[-1][5:].split(".")[0]
                            np.save(
                                os.path.join(
                                    target_dir,
                                    f"crop{crop_index}_subj{subj_idx}_{scenario}_track{track_index}.npy",
                                ),
                                cropped_pcloud_array[crop_index],
                            )

        # Generate the unknown part of the dataset

        print("Generating unseen split ...")
        for subj_idx in tqdm(unseen_classes):
            subject_dir = os.path.join(
                constants.DATA_PATH, MSRadarDataset.label_dict[subj_idx]
            )
            all_scenarios = os.listdir(subject_dir)
            assert [
                s in ["free_walk", "hands_in_pocket", "smartphone"]
                for s in all_scenarios
            ]
            for scenario in all_scenarios:
                # Assert that the folder contains only files starting with "pc"
                assert all(
                    [
                        x[:2] == "pc"
                        for x in np.array(
                            os.listdir(os.path.join(subject_dir, scenario))
                        )
                    ]
                ), "Invalid file in subject folder"

                all_tracks = os.listdir(os.path.join(subject_dir, scenario))
                for track in all_tracks:
                    # gen_crops_from_track_file(
                    #     os.path.join(subject_dir, track), subj_idx, target_dir=unseen_dir
                    # )
                    pc_file = os.path.join(subject_dir, scenario, track)
                    pcloud_array = MSRadarDataset.process_track(
                        pc_file,
                        standardize_point_cloud=True,
                        divide_by_std=False,
                        force_pc_subsampling=force_pc_subsampling, nmax=nmax_points
                    )

                    # Crop the pcloud_array with the chosen `seq_length` and `crop_step` parameters
                    cropped_pcloud_array = crop_with_step(
                        pcloud_array,
                        crop_len=constants.NSTEPS,
                        step=constants.CROP_STEP,
                    )

                    for crop_index in range(len(cropped_pcloud_array)):
                        track_index = pc_file.split("/")[-1][5:].split(".")[0]
                        np.save(
                            os.path.join(
                                unseen_dir,
                                f"crop{crop_index}_subj{subj_idx}_{scenario}_track{track_index}.npy",
                            ),
                            cropped_pcloud_array[crop_index],
                        )

        end_time = time.time()
        # Print general stats at the end
        print(
            f"~ New split created! [Time needed: {(end_time - start_time):.3f}s.] ~")
        print(f"-> Training set size: {len(os.listdir(train_dir))}")
        print(f"-> Valid set size: {len(os.listdir(valid_dir))}")
        print(f"-> Test set size: {len(os.listdir(test_dir))}")
        print(f"-> Unseen set size: {len(os.listdir(unseen_dir))}")
        print(f"-> Training Classes: {train_classes}")
        print(f"-> Unseen Classes: {unseen_classes}")
        print(f"Openness: {opns:.3f}%")
        pass

    def __init__(
        self,
        split: SPLIT,
        scenarios: List[SCENARIO] = constants.TRAIN_SCENARIOS,
        sequential=False,
        subsample_factor=1.0,
    ):
        self.filenames = []
        self.labels = []
        self.subsample_factor = subsample_factor
        self.dataset_dir = os.path.join(constants.GEN_DATA_PATH, split.value)
        self.sequential = sequential

        if self.sequential:
            filenames_list = []
            # track_dict = get_track_dict_from_crops(self.dataset_dir)
            all_crops = os.listdir(self.dataset_dir)
            track_dict = {}

            for crop in all_crops:
                subj_index = MSRadarDataset.filename2subj(crop)
                track_id = MSRadarDataset.filename2track(crop)
                track_dict.setdefault(subj_index, set())
                track_dict[subj_index].add(track_id)

            for subj_id in track_dict.keys():
                for track in track_dict[subj_id]:
                    filenames_list.append(
                        MSRadarDataset.get_sorted_seq(
                            self.dataset_dir, subj_id, track)
                    )

            self.filenames = np.concatenate(filenames_list, axis=0).tolist()
        else:
            self.filenames = os.listdir(self.dataset_dir)

        self.filenames = [
            f
            for f in self.filenames
            if MSRadarDataset.filename2scenario(f) in [s.value for s in scenarios]
        ]

        # Subsample if needed
        if self.subsample_factor < 1.0:
            print(
                f"Warning: subsampling dataset with factor {self.subsample_factor}. This is only for debugging purposes and is not reproducible (no seed)"
            )
            self.filenames = np.random.choice(
                self.filenames,
                int(len(self.filenames) * self.subsample_factor),
                replace=False,
            )

        # Get original labels: (same numbers as the one in TRAIN_CLASSES)
        self.original_labels = [
            MSRadarDataset.filename2subj(l) for l in self.filenames]
        selected_classes = list(set(self.original_labels))

        if split != SPLIT.UNSEEN:
            if selected_classes == constants.TRAIN_CLASSES:
                print(
                    f"(WARNING) {split.value} dataset: mismatch between TRAIN_CLASSES constant and classes in generated data!!"
                )
            # assert (
            #     selected_classes == constants.TRAIN_CLASSES
            # ), f"(!!!) {split.value} dataset: mismatch between TRAIN_CLASSES constant and classes in generated data!!"
        elif split == SPLIT.UNSEEN:
            if (
                selected_classes
                == np.setdiff1d(range(10), constants.TRAIN_CLASSES).tolist()
            ):
                print(
                    f"(WARNING) {split.value} dataset: mismatch between TRAIN_CLASSES constant and classes in generated data!!"
                )
            # assert selected_classes == np.setdiff1d(
            #     range(10), constants.TRAIN_CLASSES
            # ), f"(!!!) {split.value} dataset: mismatch between TRAIN_CLASSES constant and classes in generated data!!"

        # ... But labels must go from 0 to num_classes
        # if split in ["train", "valid", "test", "reid_pos"]:
        lab_dict = {}
        for i, c in enumerate(selected_classes):
            lab_dict[c] = i
        self.labels = np.array([lab_dict[j] for j in self.original_labels])

    def __getitem__(self, idx):
        path = os.path.join(self.dataset_dir, self.filenames[idx])
        pc_seq = torch.from_numpy(
            np.load(path, allow_pickle=True)).to(torch.float)

        # in pytorch channel dimension goes first
        pc_seq = torch.permute(pc_seq, (2, 0, 1))

        # normalize in range [-1,1]
        # pc_seq = pc_seq / torch.abs(pc_seq).max()

        label = torch.tensor(self.labels[idx]).type(torch.LongTensor)

        return pc_seq, label

    def __len__(self):
        return len(self.filenames)

    pass


if __name__ == "__main__":

    # MSRadarDataset.generate_splits(
    #     train_classes=constants.TRAIN_CLASSES,
    #     train_ratio=0.8,
    #     valid_ratio=0.1,
    #     test_ratio=0.1,
    #     seed=0,
    # )

    dd = MSRadarDataset(
        SPLIT.TRAIN,
        scenarios=[SCENARIO.FREE_WALK, SCENARIO.SMARTPHONE,
                   SCENARIO.HANDS_IN_POCKETS],
        sequential=False,
        subsample_factor=1.0,
    )

    loader = torch.utils.data.DataLoader(
        dd, batch_size=8, shuffle=True, num_workers=0, drop_last=False
    )

    print(len(loader))
    for data, labels in loader:
        print(data.shape)
        print(labels.shape)
        break
