import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from lib.utils.data import read_pkl, flip_data


# Refer MotionAGFormer (https://github.com/TaatiTeam/MotionAGFormer/blob/master/data/reader/motion_dataset.py)
class MotionDataset3D(Dataset):
    def __init__(self, args, subset_list, data_split, return_stats=False):
        """
        :param args: Arguments from the config file
        :param subset_list: A list of datasets
        :param data_split: Either 'train' or 'test'
        """
        np.random.seed(0)
        self.data_root = args.data_root
        self.add_velocity = args.add_velocity
        self.subset_list = subset_list
        self.data_split = data_split
        self.return_stats = return_stats

        self.flip = args.flip
        self.use_proj_as_2d = args.use_proj_as_2d

        self.file_list = self._generate_file_list()

    def _generate_file_list(self):
        file_list = []
        for subset in self.subset_list:
            data_path = os.path.join(self.data_root, subset, self.data_split)
            motion_list = sorted(os.listdir(data_path))
            for i in motion_list:
                file_list.append(os.path.join(data_path, i))
        return file_list

    @staticmethod
    def _construct_motion2d_by_projection(motion_3d):
        """Constructs 2D pose sequence by projecting the 3D pose orthographically"""
        motion_2d = np.zeros(motion_3d.shape, dtype=np.float32)
        motion_2d[:, :, :2] = motion_3d[:, :, :2]  # Get x and y from the 3D pose
        motion_2d[:, :, 2] = 1  # Set confidence score as 1
        return motion_2d

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        motion_file = read_pkl(file_path)

        motion_2d = motion_file["data_input"]
        motion_3d = motion_file["data_label"]

        if motion_2d is None or self.use_proj_as_2d:
            motion_2d = self._construct_motion2d_by_projection(motion_3d)

        if self.add_velocity:
            motion_2d_coord = motion_2d[..., :2]
            velocity_motion_2d = motion_2d_coord[1:] - motion_2d_coord[:-1]
            motion_2d = motion_2d[:-1]
            motion_2d = np.concatenate((motion_2d, velocity_motion_2d), axis=-1)

            motion_3d = motion_3d[:-1]

        if self.data_split == 'train':
            if self.flip and random.random() > 0.5:
                motion_2d = flip_data(motion_2d)
                motion_3d = flip_data(motion_3d)

        if self.return_stats:
            return torch.FloatTensor(motion_2d), torch.FloatTensor(motion_3d), motion_file['mean'], motion_file['std']
        else:
            return torch.FloatTensor(motion_2d), torch.FloatTensor(motion_3d)