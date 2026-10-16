"""
Generator A (doc/specs/2026-10-16-generator-a-design.md): the functions shared by its models, on plain tensors.
Clips are in the box space of the classifier (pixel_coords and norm "box"), one row of keypoints x coords per frame.
"""

import json
import os
import shutil

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

# number of keypoints of each part in the files of a real dataset (MediaPipe)
PART_SIZES = {"pose": 33, "left_hand": 21, "right_hand": 21}


def collate(batch):
    """Batches clips of different lengths: (frames (B, T_max, F) padded with 0 at the end, lengths (B,), labels (B,))."""
    frames, labels = zip(*batch, strict=True)
    lengths = torch.tensor([len(f) for f in frames])
    return pad_sequence(list(frames), batch_first=True), lengths, torch.stack(list(labels))


def joint_error(pred, target, coords=2):
    """Euclidean distance of every keypoint, averaged over the keypoints: (..., keypoints * coords) -> (...)."""
    return (pred - target).unflatten(-1, (-1, coords)).norm(dim=-1).mean(dim=-1)


def masked_mean(values, mask):
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp(min=1)


def next_frame_targets(frames, lengths, pred_frames):
    """
    The targets of every position of real clips: position t predicts frames t+1 to t+x. A target frame is valid inside
    its clip, a position when it has a next frame. The end label of a position is 1 when the clip's last frame is among
    its x frames.
    """
    T = frames.shape[1]
    padded = F.pad(frames, (0, 0, 0, pred_frames))
    targets = padded.unfold(1, pred_frames, 1)[:, 1 : T + 1].permute(0, 1, 3, 2)
    t = torch.arange(T, device=frames.device)
    i = torch.arange(pred_frames, device=frames.device)
    length = lengths.to(frames.device)[:, None]
    frame_mask = (t[None, :, None] + 1 + i[None, None, :]) < length[:, :, None]
    position_mask = t[None, :] < length - 1
    end = (t[None, :] + pred_frames >= length - 1).float()
    return targets, frame_mask, position_mask, end


def rollout_targets(frames, lengths, cut, steps, pred_frames):
    """The targets of a rollout from frame `cut`: step s predicts frames cut + s*x to cut + s*x + x - 1, from position cut - 1 + s*x."""
    n = steps * pred_frames
    padded = F.pad(frames, (0, 0, 0, max(0, cut + n - frames.shape[1])))
    targets = padded[:, cut : cut + n].unflatten(1, (steps, pred_frames))
    index = cut + torch.arange(n, device=frames.device).view(steps, pred_frames)
    length = lengths.to(frames.device)[:, None]
    frame_mask = index[None] < length[:, :, None]
    position = cut - 1 + torch.arange(steps, device=frames.device) * pred_frames
    position_mask = position[None] < length - 1
    end = (position[None] + pred_frames >= length - 1).float()
    return targets, frame_mask, position_mask, end


class AutoregressiveLoss(nn.Module):
    """
    Per-keypoint position error of the predicted frames, plus `vel_weight` times the same error on their velocities
    (from the frame before them), plus `end_weight` times the binary cross entropy of the end-of-sign logit, whose rare
    positives weigh `end_pos_weight`. Returns the total and the position error.
    """

    def __init__(self, coords=2, vel_weight=1.0, end_weight=0.1, end_pos_weight=10.0):
        super().__init__()
        self.coords, self.vel_weight, self.end_weight, self.end_pos_weight = coords, vel_weight, end_weight, end_pos_weight

    def forward(self, pred, target, last, frame_mask, end_logit, end, position_mask):
        position = masked_mean(joint_error(pred, target, self.coords), frame_mask)

        def velocity_of(frames):
            return torch.cat([last.unsqueeze(-2), frames], dim=-2).diff(dim=-2)

        velocity = masked_mean(joint_error(velocity_of(pred), velocity_of(target), self.coords), frame_mask)
        pos_weight = torch.tensor(self.end_pos_weight, device=end_logit.device)
        end_loss = masked_mean(F.binary_cross_entropy_with_logits(end_logit, end, pos_weight=pos_weight, reduction="none"), position_mask)
        return position + self.vel_weight * velocity + self.end_weight * end_loss, position


def write_clip(base_dir, clip_id, frames, poses, coords=2):
    """
    Saves a generated clip (T, keypoints * coords) in the layout of a real dataset: poses/<part>/<id>.npy with every
    MediaPipe keypoint of the part and 3 channels, the generated keypoints at their indices and every other value 0.
    The box normalization finds the nose and the shoulders by their MediaPipe indices, so the full layout is needed.
    """
    frames = np.asarray(frames, dtype=np.float32).reshape(len(frames), -1, coords)
    start = 0
    for part, keypoints in poses:
        if keypoints == "all":
            raise ValueError("the generated keypoints of '{}' need explicit indices".format(part))
        full = np.zeros((len(frames), PART_SIZES[part], 3), dtype=np.float32)
        full[:, keypoints, :coords] = frames[:, start : start + len(keypoints)]
        directory = os.path.join(base_dir, "poses", part)
        os.makedirs(directory, exist_ok=True)
        np.save(os.path.join(directory, "{}.npy".format(clip_id)), full)
        start += len(keypoints)


def write_metadata(base_dir, sign_to_index, metadata, video_sizes):
    """
    Writes the metadata of a generated dataset: instances.csv from `metadata` (columns id, sign, signer, start, end),
    the list of clips as the train split, and a copy of the real dataset's `sign_to_index` file. With `video_sizes`, every
    clip gets a 1x1 video size, so pixel_coords leaves the stored box-space values unchanged.
    """
    metadata_dir = os.path.join(base_dir, "metadata")
    os.makedirs(os.path.join(metadata_dir, "splits"), exist_ok=True)
    shutil.copyfile(sign_to_index, os.path.join(metadata_dir, "sign_to_index.csv"))
    pl.DataFrame(metadata).write_csv(os.path.join(base_dir, "instances.csv"), separator=",")
    with open(os.path.join(metadata_dir, "splits", "train.json"), "w") as file:
        json.dump(metadata["id"], file)
    if video_sizes:
        n = len(metadata["id"])
        pl.DataFrame({"id": metadata["id"], "width": [1] * n, "height": [1] * n}).write_csv(os.path.join(metadata_dir, "video_sizes.csv"))
