"""
Generator A (doc/specs/2026-10-16-generator-a-design.md): the functions shared by its models, on plain tensors.
Clips are in the box space of the classifier (pixel_coords and norm "box"), one row of keypoints x coords per frame.
"""

import json
import os
import shutil

import numpy as np
import polars as pl

# number of keypoints of each part in the files of a real dataset (MediaPipe)
PART_SIZES = {"pose": 33, "left_hand": 21, "right_hand": 21}


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
