# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/data_util.py

import json
import math
import os
import random

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset


def normalize(data, pose):
    MEAN = {
        'pose': [0.0011, 0.1365, 0.0693],
        'right_hand': [-0.0325,  0.1191,  0.0000],
        'left_hand': [0.0251, 0.1365, 0.0000],
        'face': [-0.0024, -0.0086,  0.1384]
    }
    STD = {
        'pose': [0.0184, 0.1365, 0.0692],
        'right_hand': [0.0111, 0.0222, 42],
        'left_hand': [0.0105, 0.0198, 42],
        'face': [0.0046, 0.0086, 0.0040]
    }

    mean = np.array(MEAN[pose])
    std = np.array(STD[pose])

    mean = mean[None, None, :]
    std = std[None, None, :]

    # Normalize the data
    normalized_data = (data - mean) / std

    return normalized_data

class DropFrames(torch.nn.Module):
    """
    Sets frames from the input image to 0 with a probability of p.
    """
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, img):
        # Generate random numbers for all frames at once
        rand_nums = torch.rand(img.shape[0])

        # Create a mask where the random number is less than p
        mask = rand_nums < self.p

        # Set the frames where the mask is True to 0
        img[mask] = 0

        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"

class DropKeypoints(torch.nn.Module):
    """
    Sets keypoint blocks from the input image to 0 with a probability of p. See Cutout for more information.
    """

    def __init__(self, mask_size, p, cutout_inside=True, mask_color=0):
        super().__init__()
        self.p = p
        self.cutout_inside = cutout_inside
        self.mask_size = mask_size
        self.mask_size_half = mask_size // 2
        self.offset = 1 if mask_size % 2 == 0 else 0
        self.mask_color = mask_color

    def forward(self, img):
        if np.random.random() > self.p:
            return img

        k = img.shape[1]

        if self.cutout_inside:
            ckmin, ckmax = self.mask_size_half, k + self.offset - self.mask_size_half
        else:
            ckmin, ckmax = 0, k + self.offset

        ck = np.random.randint(ckmin, ckmax)
        kmin = ck - self.mask_size_half
        kmax = kmin + self.mask_size
        kmin = max(0, kmin)
        kmax = min(k, kmax)
        img[:,kmin:kmax] = self.mask_color
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p},block_size={self.block_size})"

class RandomCropFrames(torch.nn.Module):
    """
    Crops the random list of frames to the given size.
    """

    def __init__(self, size, window_expand=2):
        super().__init__()
        self.size = size
        self.expanded_size = self.size*window_expand

    def forward(self, img):
        offset = 0 if img.shape[0]<self.expanded_size else random.randint(0, img.shape[0]-self.expanded_size)
        frames_indexes = np.sort(np.random.choice(range(min(img.shape[0], self.expanded_size)), size=self.size, replace=True)) + offset
        return img[frames_indexes]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"

class CropFrames(torch.nn.Module):
    """
    Crops the given list of frames to the given size.
    """

    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, img):
        if img.shape[0]>self.size:
            offset = img.shape[0]-self.size
            r = random.randint(0, offset-1)
            img = img[r:-(offset-r)]
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"

class PadFrames(torch.nn.Module):
    """
    Pad the given list of frames to the given size.
    """

    def __init__(self, size, mode):
        super().__init__()
        self.size = size
        self.mode = mode

    def forward(self, img):
        if img.shape[0]<self.size:
            if self.mode == 'pad':
                img = F.pad(img, (0,0,0,0,0,self.size-img.shape[0]), 'constant', float('nan'))
            else:
                img = torch.tensor(np.pad(img, ((0, self.size-img.shape[0]),(0,0),(0,0)), self.mode))
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"

class Half(torch.nn.Module):
    """
    Change type to half.
    """

    def __init__(self):
        super().__init__()

    def forward(self, img):
        return img.half()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

class Permute(torch.nn.Module):
    """
    Permute the given dimensions.
    """

    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, img):
        return torch.permute(img, self.dims)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.dims})"

class RandomAffine(torch.nn.Module):

    def __init__(self, flip_p=0, scale = 0, rot = 0):
        super().__init__()
        self.flip_p = flip_p
        self.scale = scale
        self.rot = math.radians(rot)
                 
    def forward(self, sample):
        flip = -1 if torch.rand(1)<self.flip_p else 1
        random_scale = 1+(torch.rand(1).item()*2-1)*self.scale
        random_rot = torch.tensor((torch.rand(1).item()*2-1)*self.rot)
        mat = torch.tensor([[flip*random_scale*torch.cos(random_rot), flip*random_scale*-torch.sin(random_rot), 0], 
                        [random_scale*torch.sin(random_rot),      random_scale*torch.cos(random_rot),       0], 
                        [0,                                      0,                                       random_scale]])
        new_sample = sample.flatten(0,1).type(torch.float32) @ mat.T
        return torch.reshape(new_sample, sample.shape)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(flip_p={self.flip_p},scale={self.scale},rot={self.rot})"

class UniformSampleFrames(torch.nn.Module):
    """
    Samples `size` frames evenly spaced over the whole clip (repeating frames for short clips).
    If `speed` is set, a random sub-window covering [speed, 1] of the clip is sampled instead,
    simulating different signing speeds and inaccurate start/end boundaries.
    """

    def __init__(self, size, speed=None):
        super().__init__()
        self.size = size
        self.speed = speed

    def forward(self, img):
        n = img.shape[0]
        start, length = 0, n
        if self.speed:
            length = max(1, round(n*random.uniform(self.speed, 1.0)))
            start = random.randint(0, n-length)
        indexes = np.round(np.linspace(start, start+length-1, self.size)).astype(int)
        return img[indexes]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size},speed={self.speed})"

class ResamplePadFrames(torch.nn.Module):
    """
    Keeps the clip's own timing instead of stretching it to `size` frames: clips longer than `size`
    are evenly subsampled, shorter clips are padded by repeating the first and last frames.
    If `speed_range` is set, the clip is first resampled to a random fraction of its length
    (half of the time picking random frames instead of evenly spaced ones), and the padding
    offset is random.
    """

    def __init__(self, size, speed_range=None):
        super().__init__()
        self.size = size
        self.speed_range = speed_range

    def forward(self, img):
        n = img.shape[0]
        shift = 0.5
        if self.speed_range:
            m = max(1, int(n*random.uniform(self.speed_range[0], self.speed_range[1])))
            if random.random() < 0.5:
                indexes = np.sort(np.random.choice(n, size=m, replace=m > n))
            else:
                indexes = np.linspace(0, n-1, m).astype(int)
            img = img[indexes]
            n = m
            shift = float(np.clip(np.random.normal(0.5, 0.1), 0, 1))
        if n > self.size:
            return img[np.linspace(0, n-1, self.size).astype(int)]
        before = int((self.size-n)*shift)
        indexes = np.clip(np.arange(self.size)-before, 0, n-1)
        return img[indexes]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size},speed_range={self.speed_range})"

class HandMaskInterpolate(torch.nn.Module):
    """
    Replaces the hand keypoints of a random fraction p of the frames with a linear interpolation
    of the remaining frames, simulating frames where the hand detector failed.
    """

    def __init__(self, p, hand_slices):
        super().__init__()
        self.p = p
        self.hand_slices = hand_slices

    def forward(self, img):
        n = img.shape[0]
        k = int(self.p*n)
        if k == 0 or n-k < 2:
            return img
        masked = np.sort(np.random.choice(n, size=k, replace=False))
        kept = np.setdiff1d(np.arange(n), masked)
        out = img.clone()
        for start, end in self.hand_slices:
            hand = img[:, start:end].flatten(1).float().numpy()
            for d in range(hand.shape[1]):
                hand[masked, d] = np.interp(masked, kept, hand[kept, d])
            out[:, start:end] = torch.from_numpy(hand).reshape(n, end-start, -1).to(img.dtype)
        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"

def hand_slices(poses):
    """
    (start, end) index ranges of the hand keypoints in the concatenated keypoints of `poses`.
    """
    slices, total = [], 0
    for pose, keypoints in poses:
        size = 21 if keypoints == 'all' else len(keypoints)
        if 'hand' in pose:
            slices.append((total, total+size))
        total += size
    return slices

class RandomShearRotate(torch.nn.Module):
    """
    Random 2D shear and rotation of the x,y coordinates, both sampled from normal distributions.
    """

    def __init__(self, shear_std=0.0, rot_std=0.0):
        super().__init__()
        self.shear_std = shear_std
        self.rot_std = rot_std

    def forward(self, sample):
        sx, sy = np.random.normal(0, self.shear_std, 2) if self.shear_std else (0.0, 0.0)
        theta = np.random.normal(0, self.rot_std) if self.rot_std else 0.0
        shear = np.array([[1, sx], [sy, 1]])
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        mat = torch.eye(sample.shape[-1])
        mat[:2, :2] = torch.tensor(rot @ shear, dtype=torch.float32)
        new_sample = sample.flatten(0,1).type(torch.float32) @ mat.T
        return torch.reshape(new_sample, sample.shape)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(shear_std={self.shear_std},rot_std={self.rot_std})"

class Mirror(torch.nn.Module):
    """
    Horizontal flip with probability p: negates x and swaps left/right keypoints using `perm`.
    Assumes x is centred on the body (use with norm='shoulder').
    """

    def __init__(self, p, perm):
        super().__init__()
        self.p = p
        self.perm = torch.tensor(perm)

    def forward(self, sample):
        if random.random() < self.p:
            sample = sample[:, self.perm].clone()
            sample[..., 0] = -sample[..., 0]
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"

# MediaPipe left/right keypoint pairs, used to build the mirror permutation
MIRROR_PAIRS = {
    'pose': [(1,4),(2,5),(3,6),(7,8),(9,10),(11,12),(13,14),(15,16),(17,18),(19,20),(21,22),(23,24),(25,26),(27,28),(29,30),(31,32)],
    'face': [(61,291),(33,263),(159,386),(133,362),(145,374)],
}

def mirror_permutation(poses):
    """
    Index permutation over the concatenated keypoints of `poses` that swaps left and right.
    """
    offsets, sizes = {}, {}
    total = 0
    for pose, keypoints in poses:
        size = 21 if keypoints == 'all' and 'hand' in pose else len(keypoints)
        offsets[pose], sizes[pose] = total, size
        total += size
    perm = list(range(total))
    if 'right_hand' in offsets and 'left_hand' in offsets:
        right, left = offsets['right_hand'], offsets['left_hand']
        for i in range(21):
            perm[right+i], perm[left+i] = left+i, right+i
    for pose, keypoints in poses:
        if 'hand' in pose:
            continue
        if keypoints == 'all':
            raise NotImplementedError("mirror needs an explicit keypoint list for '{}'".format(pose))
        mirror = {}
        for a, b in MIRROR_PAIRS.get(pose, []):
            mirror[a], mirror[b] = b, a
        for i, k in enumerate(keypoints):
            m = mirror.get(k, k)
            if m not in keypoints:
                raise ValueError("mirror of keypoint {} of '{}' is not selected".format(k, pose))
            perm[offsets[pose]+i] = offsets[pose]+keypoints.index(m)
    return perm

class Dataset_(Dataset):
    def __init__(self,
                 data_dir,
                 train,
                 load_data_in_memory=False,
                 poses=None,
                 filter_classes=None,
                 map_classes=None,
                 max_len=15,
                 min_samples=None,
                 target_len=None,
                 pad_frames=False,
                 pad_mode='wrap',
                 random_crop=False,
                 drop_frame=0.0,
                 drop_keypoint=0.0,
                 block_size=9,
                 flip_p=0.0,
                 scale=0.0,
                 rot=0.0,
                 temporal_sampling="crop",
                 speed=None,
                 speed_range=None,
                 hand_mask_p=0.0,
                 norm="dataset",
                 shear_std=0.0,
                 rot_std=0.0,
                 mirror_p=0.0,
                 coords=3,
                 mode="classification"):
        super(Dataset_, self).__init__()
        self.mode = mode
        self.target_len = target_len
        self.data_dir = data_dir
        self.train = train
        self.load_data_in_memory = load_data_in_memory
        self.trsf_list = []
        self.poses = poses
        self.filter_classes = filter_classes
        self.map_classes = map_classes
        self.min_samples = min_samples
        self.max_len = max_len
        self.norm = norm
        self.coords = coords
        self.centers = []

        self.load_dataset()

        self.load_data_in_memory = load_data_in_memory

        self.trsf_list += [transforms.ToTensor()]
        self.trsf_list += [Permute([1,2,0])]
        self.trsf_list += [Half()]
        # uniform and pad sampling always return max_len frames, so no padding is needed
        if pad_frames and temporal_sampling == "crop":
            self.trsf_list += [PadFrames(self.max_len, pad_mode)]

        if self.load_data_in_memory:
            self.pre_trsf = transforms.Compose(self.trsf_list)

            self.trsf_list = []
            self.pose_data = []
            self.labels = []
            for index in range(len(self.data)):
                value, label = self.load(index)
                self.pose_data.append(self.pre_trsf(value))
                self.labels.append(label)

        if hand_mask_p > 0:
            self.trsf_list += [HandMaskInterpolate(hand_mask_p, hand_slices(self.poses))]
        if temporal_sampling == "uniform":
            self.trsf_list += [UniformSampleFrames(self.max_len, speed)]
        elif temporal_sampling == "pad":
            self.trsf_list += [ResamplePadFrames(self.max_len, speed_range)]
        elif random_crop:
            self.trsf_list += [RandomCropFrames(self.max_len)]
        else:
            self.trsf_list += [CropFrames(self.max_len)]
        if flip_p > 0 or scale > 0 or rot > 0:
            self.trsf_list += [RandomAffine(flip_p, scale, rot)]
        if shear_std > 0 or rot_std > 0:
            self.trsf_list += [RandomShearRotate(shear_std, rot_std)]
        if mirror_p > 0:
            self.trsf_list += [Mirror(mirror_p, mirror_permutation(self.poses))]
        if drop_frame>0:
            self.trsf_list += [DropFrames(drop_frame)]
        if drop_keypoint>0:
            self.trsf_list += [DropKeypoints(block_size, drop_keypoint)]
        self.trsf = transforms.Compose(self.trsf_list)

    def load_dataset(self):
        mode = "train" if self.train is True else "test"

        # get train/valid sample names
        self.root_ids = os.path.join(self.data_dir, 'metadata', 'splits', mode+'.json')
        with open(self.root_ids) as f:
            data_ids = json.load(f)

        # load all instances
        self.data = pl.read_csv(os.path.join(self.data_dir, 'instances.csv'), dtypes = {"sign": pl.Utf8})

        # filter to only use train/valid predefined samples
        self.data = self.data.filter(pl.col('id').cast(pl.String).is_in(data_ids))

        # map signs to transform them to their numeric representations
        t_si = pl.read_csv(os.path.join(self.data_dir, 'metadata', 'sign_to_index.csv'), dtypes = {"sign": pl.Utf8}).transpose()
        mapping = t_si.rename(t_si.head(1).to_dicts().pop()).slice(1).to_dict(as_series=False)
        mapping = {k: v[0] for k, v in mapping.items()}
        self.data = self.data.with_columns(pl.col('sign').replace(mapping).cast(pl.Int64))

        # get unique classes, sorted so every run maps the classes to the same indexes
        self.classes = sorted(self.data.unique('sign')['sign'].to_list())

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
                    selected_classes = sorted(set(self.filter_classes) & set(meet_min))
                else:
                    selected_classes = self.filter_classes
            else:
                selected_classes = meet_min
            self.data = self.data.filter(pl.col("sign").is_in(selected_classes))
            self.classes = selected_classes
        
        if not self.map_classes:
            self.map_classes = {i:j for i,j in zip(self.classes, range(len(self.classes)), strict=True)}

    def load_shoulder_normalized(self, index):
        """
        Centre on the shoulder midpoint and scale by the mean shoulder width of the clip.
        norm "shoulder" centres every frame, "shoulder_clip" uses the clip's mean midpoint for all frames,
        which keeps the body's movement within the clip.
        """
        sample_id = str(self.data[index]['id'].item())
        body = np.load(os.path.join(self.data_dir, 'poses', 'pose', sample_id+'.npy'))
        center = (body[:,11] + body[:,12]) / 2
        if self.norm == "shoulder_clip":
            center = np.broadcast_to(center.mean(axis=0), center.shape)
        width = np.linalg.norm(body[:,11,:2] - body[:,12,:2], axis=-1).mean() + 1e-6
        poses_data = []
        for pose, keypoints in self.poses:
            pose_data = np.load(os.path.join(self.data_dir, 'poses', pose, sample_id+'.npy'))
            if keypoints != 'all':
                pose_data = pose_data[:,keypoints,:]
            pose_data = (pose_data - center[:,None,:]) / width
            if 'hand' in pose:
                pose_data[:,:,2] = 0
            poses_data.append(pose_data)
        value = np.concatenate(poses_data,axis=1)
        label = self.data[index, 'sign']
        return value, self.map_classes[label]

    def load(self, index):
        value, label = self.load_all_coords(index)
        return value[..., :self.coords], label

    def load_all_coords(self, index):
        if self.norm in ("shoulder", "shoulder_clip"):
            return self.load_shoulder_normalized(index)
        poses_data = []
        take_center = True
        for pose, keypoints in self.poses:
            pose_data = np.load(os.path.join(self.data_dir, 'poses', pose, str(self.data[index]['id'].item())+'.npy'))
            if take_center:
                self.centers = [pose_data[0,0,0], pose_data[0,0,1], pose_data[0,0,2]]
                take_center = False
            if keypoints != 'all':
                pose_data = pose_data[:,keypoints,:]
            pose_data[:,:,0] -= self.centers[0]
            pose_data[:,:,1] -= self.centers[1]
            if 'hand' in pose:
                pose_data[:,:,2] = 0
            else:
                pose_data[:,:,2] -= self.centers[2]
            pose_data = normalize(pose_data, pose)
            poses_data.append(pose_data)
        value = np.concatenate(poses_data,axis=1)
        label = self.data[index, 'sign']
        return value, self.map_classes[label]

    def __len__(self):
        return len(self.data)

    def get_classification_item(self, index):
        if self.load_data_in_memory:
            value, label = self.pose_data[index], self.labels[index]
        else:
            value, label = self.load(index)
        return torch.flatten(self.trsf(value), 1), torch.tensor(label)

    def get_prediction_item(self, index):
        if self.load_data_in_memory:
            value = self.pose_data[index]
        else:
            value, _ = self.load(index)
        value = torch.flatten(self.trsf(value), 1)
        return value[:-self.target_len], value[-self.target_len:]
    
    def get_cond_prediction_item(self, index):
        if self.load_data_in_memory:
            value, label = self.pose_data[index], self.labels[index]
        else:
            value, label = self.load(index)
        value = torch.flatten(self.trsf(value), 1)
        return value[:-self.target_len], value[-self.target_len:], torch.tensor(label)

    def __getitem__(self, index):
        if self.mode == "classification":
            return self.get_classification_item(index)
        elif self.mode == "prediction":
            return self.get_prediction_item(index)
        elif self.mode == "cond_prediction":
            return self.get_cond_prediction_item(index)
        else:
            raise NotImplementedError

class OversamplingWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, oversampling_size=None):
        self.dataset = dataset
        self.data = self.dataset.dataset.data
        self.classes = self.dataset.dataset.classes
        label_dict = {key: [] for key in self.classes}
        for i, sign in enumerate(self.data['sign'].take(self.dataset.indices)):
            label_dict[sign].append(i)
        if oversampling_size:
            self.oversampling_size = oversampling_size
        else:
            self.oversampling_size = max([len(v) for v in label_dict.values()])
        self.num_classes = len(self.classes)
        self.indices = [item for v in label_dict.values() for item in self.multiply(v, self.oversampling_size)]

    def multiply(self, a, n):
        length = len(a)
        new_list = [0] * n
        for i in range(n):
            new_list[i] = a[i % length]
        return new_list

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


def train_val_dataset(dataset, val_split=0.25, train_size=None, random_state=42, stratify=None):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split, train_size=train_size, random_state=random_state, stratify=stratify)
    return Subset(dataset, train_idx), Subset(dataset, val_idx)