# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/data_util.py

import os

from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import numpy as np
import json
import polars as pl

class Dataset_(Dataset):
    def __init__(self,
                 data_dir,
                 train,
                 load_data_in_memory=False,
                 poses=None,
                 filter_classes=None,
                 min_samples=0):
        super(Dataset_, self).__init__()
        self.data_dir = data_dir
        self.train = train
        self.load_data_in_memory = load_data_in_memory
        self.trsf_list = []
        self.pose_trsf_list = []
        self.poses = poses
        self.filter_classes = filter_classes
        self.min_samples = min_samples

        self.load_dataset()

    def load_dataset(self):
        mode = "train" if self.train == True else "valid"

        # get train/valid sample names
        self.root_ids = os.path.join(self.data_dir, 'metadata', 'splits', mode+'.json')
        with open(self.root_ids) as f:
            data_ids = json.load(f)

        # load all instances
        self.data = pl.read_csv(os.path.join(self.data_dir, 'instances.csv'))

        # filter to only use train/valid predefined samples
        predicate = pl.any_horizontal(pl.col('id') == v for v in data_ids)
        self.data = self.data.filter(predicate)

        # map signs to transform them to their numeric representations
        t_si = pl.read_csv(os.path.join(self.data_dir, 'metadata', 'sign_to_index.csv')).transpose()
        mapping = t_si.rename(t_si.head(1).to_dicts().pop()).slice(1).to_dict(as_series=False)
        mapping = {k: v[0] for k, v in mapping.items()}
        self.data = self.data.with_columns(pl.col('sign').replace(mapping).cast(pl.Int64))
        
        # get unique classes
        self.classes = self.data.unique('sign')['sign'].to_list()       

        # only keep classes with a number of samples >= min_samples
        if self.min_samples:
            samples_count__dict = {key: 0 for key in self.classes}
            for sign in self.data['sign']:
                samples_count__dict[sign] += 1
            meet_min = []
            for sign, count in samples_count__dict.items():
                if count >= self.min_samples:
                    meet_min.append(sign)

        if self.filter_classes or self.min_samples:
            if self.filter_classes:
                if self.min_samples:
                    selected_classes = list(set(self.filter_classes) & set(self.min_samples))
                else:
                    selected_classes = self.filter_classes
            else:
                selected_classes = self.min_samples
            self.data = self.data.filter(pl.col("sign").is_in(selected_classes))
            self.classes = selected_classes

    def load(self, index):
        poses_data = []
        for pose, keypoints in self.poses:
            pose_data = np.load(os.path.join(self.data_dir, 'poses', pose, self.data[index]['id'].item()+'.npy'))
            if keypoints != 'all':
                pose_data = pose_data[:,keypoints,:]
            poses_data.append(pose_data)
        value = np.concatenate(poses_data,axis=1)
        label = self.data[index, 'sign']
        return value, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
            value, label = self.load(index)
            return value, label