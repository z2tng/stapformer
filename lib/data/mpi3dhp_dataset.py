import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from lib.utils.data import normalize_screen_coordinates, flip_data


class MPI3DHP(Dataset):
    def __init__(self, args, train=True):
        self.train = train
        self.poses_3d, self.poses_2d, self.poses_3d_valid_frames, self.seq_names = self.prepare_data(args)
        self.normalized_poses3d = self.normalize_poses()
        self.flip = args.flip
        self.left_joints = [8, 9, 10, 2, 3, 4]
        self.right_joints = [11, 12, 13, 5, 6, 7]

    def normalize_poses(self):
        normalized_poses_3d = []
        if self.train:
            for pose_sequence in self.poses_3d: # pose_sequence dim is (T, J, 3)
                width = 2048
                height = 2048
                normalized_sequence = pose_sequence.copy()
                normalized_sequence[..., :2]  = normalized_sequence[..., :2] / width * 2 - [1, height / width]
                normalized_sequence[..., 2:] = normalized_sequence[..., 2:] / width * 2

                normalized_sequence = normalized_sequence - normalized_sequence[:, 14:15, :]
                
                normalized_poses_3d.append(normalized_sequence[None, ...])
        else:
            for seq_name, pose_sequence in zip(self.seq_names, self.poses_3d): # pose_sequence dim is (T, J, 3)
                if seq_name in ["TS5", "TS6"]:
                    width = 1920
                    height = 1080
                else:
                    width = 2048
                    height = 2048
                normalized_sequence = pose_sequence.copy()
                normalized_sequence[..., :2]  = normalized_sequence[..., :2] / width * 2 - [1, height / width]
                normalized_sequence[..., 2:] = normalized_sequence[..., 2:] / width * 2

                normalized_sequence = normalized_sequence - normalized_sequence[:, 14:15, :]
                
                normalized_poses_3d.append(normalized_sequence[None, ...])

        normalized_poses_3d = np.concatenate(normalized_poses_3d, axis=0)
        
        return normalized_poses_3d

    def prepare_data(self, args):
        poses_2d, poses_3d, poses_3d_valid_frames, seq_names = [], [], [], []
        data_file = "data_train_3dhp.npz" if self.train else "data_test_3dhp.npz"
        data = np.load(os.path.join(args.data_root, data_file), allow_pickle=True)['data'].item()
        num_frames, stride = args.num_frames, args.stride if self.train else args.num_frames

        for seq in data.keys():
            if self.train:
                for cam in data[seq][0].keys():
                    anim = data[seq][0][cam]

                    data_3d_partitioned, data_2d_partitioned, _ = self.extract_poses(anim, seq, num_frames, stride)
                    poses_3d.extend(data_3d_partitioned)
                    poses_2d.extend(data_2d_partitioned)
            else:
                anim = data[seq]
                valid_frames = anim['valid']

                data_3d_partitioned, data_2d_partitioned, valid_frames_partitioned = self.extract_poses(anim, seq, num_frames, stride, valid_frames)
                poses_3d.extend(data_3d_partitioned)
                poses_2d.extend(data_2d_partitioned)
                seq_names.extend([seq] * len(data_3d_partitioned))
                poses_3d_valid_frames.extend(valid_frames_partitioned)
        
        poses_3d = np.concatenate(poses_3d, axis=0)
        poses_2d = np.concatenate(poses_2d, axis=0)
        if len(poses_3d_valid_frames) > 0:
            poses_3d_valid_frames = np.concatenate(poses_3d_valid_frames, axis=0)
        return poses_3d, poses_2d, poses_3d_valid_frames, seq_names

    def __len__(self):
        return self.poses_3d.shape[0]
    
    def extract_poses(self, anim, seq, num_frames, stride, valid_frames=None):
        data_3d = anim['data_3d']
        # data_3d -= data_3d[:, 14:15]
        # data_3d[..., 2] -= data_3d[:, 14:15, 2]
        data_3d_partitioned, valid_frames_partitioned = self.partition(data_3d, clip_length=num_frames, stride=stride, valid_frames=valid_frames)

        data_2d = anim['data_2d']
        if seq in ["TS5", "TS6"]:
            width = 1920
            height = 1080
        else:
            width = 2048
            height = 2048

        data_2d[..., :2] = normalize_screen_coordinates(data_2d[..., :2], w=width, h=height)
        # Adding 1 as confidence scores since MotionAGFormer needs (x, y, conf_score)
        confidence_scores = np.ones((*data_2d.shape[:2], 1))
        data_2d = np.concatenate((data_2d, confidence_scores), axis=-1)
        # data_2d_partitioned, _ = self.partition(data_2d, clip_length=num_frames, stride=stride)
        data_2d_partitioned, _ = self.partition(data_2d, clip_length=num_frames, stride=stride, valid_frames=valid_frames)

        return data_3d_partitioned, data_2d_partitioned, valid_frames_partitioned
    
    def partition(self, data, clip_length=243, stride=81, valid_frames=None):
        """Partitions data (n_frames, 17, 3) into list of (clip_length, 17, 3) data with given stride"""
        if valid_frames is not None:
            valid_idx = np.nonzero(valid_frames)[0]
            data = data[valid_idx, :, :]
        
        data_list, valid_list = [], []
        n_frames = data.shape[0]
        for i in range(0, n_frames, stride):
            sequence = data[i:i+clip_length]
            sequence_length = sequence.shape[0]
            if sequence_length == clip_length:
                data_list.append(sequence[None, ...])
            else:
                new_indices = self.resample(sequence_length, clip_length)
                extrapolated_sequence = sequence[new_indices]
                data_list.append(extrapolated_sequence[None, ...])

        if valid_frames is not None:
            for i in range(0, n_frames, stride):
                valid_sequence = valid_frames[i:i+clip_length]
                sequence_length = valid_sequence.shape[0]
                if sequence_length == clip_length:
                    valid_list.append(valid_sequence[None, ...])
                else:
                    new_indices = self.resample(sequence_length, clip_length)
                    extrapolated_sequence = valid_sequence[new_indices]
                    valid_list.append(extrapolated_sequence[None, ...])

        return data_list, valid_list

    @staticmethod
    def resample(original_length, target_length):
        """
        Returns an array that has indices of frames. elements of array are in range (0, original_length -1) and
        we have target_len numbers (So it interpolates the frames)
        """
        even = np.linspace(0, original_length, num=target_length, endpoint=False)
        result = np.floor(even)
        result = np.clip(result, a_min=0, a_max=original_length - 1).astype(np.uint32)
        return result

    def __getitem__(self, index):
        pose_2d = self.poses_2d[index]
        pose_3d_normalized = self.normalized_poses3d[index]
        
        if not self.train:
            valid_frames = self.poses_3d_valid_frames[index]
            pose_3d = self.poses_3d[index]
            seq_name = self.seq_names[index]
            return torch.FloatTensor(pose_2d), torch.FloatTensor(pose_3d), torch.IntTensor(valid_frames), seq_name
        
        if self.flip and random.random() > 0.5:
            pose_2d = flip_data(pose_2d, self.left_joints, self.right_joints)
            pose_3d_normalized = flip_data(pose_3d_normalized, self.left_joints, self.right_joints)

        return torch.FloatTensor(pose_2d), torch.FloatTensor(pose_3d_normalized)