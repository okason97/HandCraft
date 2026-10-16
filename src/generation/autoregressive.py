"""
Generator A (doc/specs/2026-10-16-generator-a-design.md): the functions shared by its models, on plain tensors.
Clips are in the box space of the classifier (pixel_coords and norm "box"), one row of keypoints x coords per frame.
"""

import json
import math
import os
import shutil

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Subset

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


def random_cut(lengths, opening_frames):
    """The frame a training rollout starts from: after opening_frames when the clips are long enough, and before the last frame of the shortest clip."""
    last = int(lengths.min()) - 1
    if last < 1:
        raise ValueError("every clip of a rollout batch needs at least 2 frames")
    first = min(opening_frames, last)
    return int(torch.randint(first, last + 1, (1,)))


def rollout(model, context, labels, steps):
    """
    Appends `steps` predictions of x frames to `context` (B, t, F), each from every frame so far. Returns the new
    frames (B, steps*x, F) and the end logit of each step (B, steps).
    """
    frames, ends = context, []
    for _ in range(steps):
        drift, end = model(frames, labels, last_only=True)
        frames = torch.cat([frames, frames[:, -1:] + drift[:, 0]], dim=1)
        ends.append(end[:, 0])
    return frames[:, context.shape[1] :], torch.stack(ends, dim=1)


@torch.no_grad()
def rollout_metrics(model, frames, lengths, labels, opening_frames, pred_frames, coords=2):
    """
    Rolls every clip of a batch out from its first k frames to the length of the longest one. Returns the per-keypoint
    error of every generated frame (B, T - k), the mask of the frames inside clips longer than k, the number of correct
    end-of-sign decisions (logit above 0 at the step that reaches the clip's last frame, below 0 before) and the number
    of decisions.
    """
    k, T = opening_frames, frames.shape[1]
    lengths = lengths.to(frames.device)
    longer = lengths > k
    if T <= k:
        return frames.new_zeros(len(frames), 0), torch.zeros(len(frames), 0, dtype=torch.bool, device=frames.device), 0, 0
    steps = math.ceil((T - k) / pred_frames)
    generated, end_logits = rollout(model, frames[:, :k], labels, steps)
    error = joint_error(generated[:, : T - k].float(), frames[:, k:].float(), coords)
    valid = (torch.arange(k, T, device=frames.device)[None] < lengths[:, None]) & longer[:, None]
    position = k - 1 + torch.arange(steps, device=frames.device) * pred_frames
    decided = (position[None] < lengths[:, None] - 1) & longer[:, None]
    ends = position[None] + pred_frames >= lengths[:, None] - 1
    correct = ((end_logits > 0) == ends) & decided
    return error, valid, int(correct.sum()), int(decided.sum())


@torch.no_grad()
def generate(model, openings, labels, source_lengths, stop, max_len):
    """
    Generates a clip from each opening (B, k, F). With stop "length" a clip reaches the length of its source clip,
    with "end_signal" it stops after the step whose end probability passes 0.5; both stop at max_len frames. Returns
    the clips, openings included, as a list of (T_b, F) tensors on the CPU.
    """
    if stop not in ("length", "end_signal"):
        raise ValueError("gen_stop must be 'length' or 'end_signal', not '{}'".format(stop))
    frames = openings
    B = len(frames)
    if stop == "length":
        lengths = source_lengths.to(frames.device).clamp(max=max_len)
    else:
        lengths = torch.full((B,), max_len, device=frames.device)
    done = torch.zeros(B, dtype=torch.bool, device=frames.device)
    while True:
        remaining = lengths[~done]
        if len(remaining) == 0 or frames.shape[1] >= int(remaining.max()):
            break
        drift, end = model(frames, labels, last_only=True)
        new = frames[:, -1:] + drift[:, 0]
        if not torch.isfinite(new).all():
            raise FloatingPointError("the generator produced a non-finite frame")
        frames = torch.cat([frames, new], dim=1)
        if stop == "end_signal":
            ended = (torch.sigmoid(end[:, 0].float()) > 0.5) & ~done
            lengths = torch.where(ended, torch.full_like(lengths, min(frames.shape[1], max_len)), lengths)
            done |= ended
    return [frames[b, : int(lengths[b])].float().cpu() for b in range(B)]


def usable_openings(dataset, labels, classes, opening_frames):
    """
    The source clips that can open a generated clip, those longer than opening_frames, as a Subset with their labels;
    the classes that still have one, and those left without any.
    """
    keep = [i for i in range(len(dataset)) if len(dataset[i][0]) > opening_frames]
    kept_labels = [labels[i] for i in keep]
    present = set(kept_labels)
    return Subset(dataset, keep), kept_labels, [c for c in classes if c in present], [c for c in classes if c not in present]
