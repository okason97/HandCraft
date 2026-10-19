import json

import numpy as np
import polars as pl
from conftest import POSES, box_space_clip, make_dataset

from data.data_util import Dataset_
from generation.autoregressive import write_clip


def test_write_clip_puts_keypoints_at_their_mediapipe_indices(tmp_path):
    frames = box_space_clip(5, seed=0)
    write_clip(tmp_path, 7, frames, POSES)
    pose = np.load(tmp_path / "poses" / "pose" / "7.npy")
    left = np.load(tmp_path / "poses" / "left_hand" / "7.npy")
    assert pose.shape == (5, 33, 3) and left.shape == (5, 21, 3)
    np.testing.assert_array_equal(pose[:, 11, :2], frames.reshape(5, 29, 2)[:, 3])
    np.testing.assert_array_equal(left[:, 20, :2], frames.reshape(5, 29, 2)[:, 18])
    assert not pose[:, 1].any() and not pose[..., 2].any()


def test_written_clips_reload_unchanged_through_the_box_normalization(tmp_path):
    clips = [(box_space_clip(12, seed=1), "a"), (box_space_clip(20, seed=2), "b")]
    make_dataset(tmp_path, clips, ["a", "b"])
    options = dict(max_len=64, temporal_sampling="pad", norm="box", coords=2, pixel_coords=True, missing_hand="wrist")
    ds = Dataset_(data_dir=str(tmp_path), train=True, poses=POSES, **options)
    for i in range(len(ds)):
        value, _ = ds.load(i)
        expected = clips[int(ds.data[i]["id"].item())][0].reshape(-1, 29, 2)
        np.testing.assert_allclose(value, expected, atol=1e-5)


def test_metadata_has_the_real_lengths_splits_and_unit_video_sizes(tmp_path):
    make_dataset(tmp_path, [(box_space_clip(12, seed=1), "a"), (box_space_clip(20, seed=2), "b")], ["a", "b"])
    instances = pl.read_csv(tmp_path / "instances.csv")
    assert instances["end"].to_list() == [12, 20]
    assert json.loads((tmp_path / "metadata" / "splits" / "train.json").read_text()) == ["0", "1"]
    sizes = pl.read_csv(tmp_path / "metadata" / "video_sizes.csv")
    assert sizes["width"].to_list() == [1, 1] and sizes["height"].to_list() == [1, 1]
    assert (tmp_path / "metadata" / "sign_to_index.csv").exists()
