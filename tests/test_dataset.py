import numpy as np
import torch
from conftest import POSES, box_space_clip, make_dataset

import utils.misc as misc
from data.data_util import Dataset_, missing_classes, synthetic_dataset, train_val_dataset
from generation.autoregressive import collate

OPTIONS = dict(max_len=64, temporal_sampling="pad", norm="box", coords=2, pixel_coords=True, missing_hand="wrist")


def test_autoregressive_item_is_the_whole_clip(tmp_path):
    clips = [(box_space_clip(12, seed=1), "a"), (box_space_clip(30, seed=2), "b")]
    make_dataset(tmp_path, clips, ["a", "b"])
    ds = Dataset_(data_dir=str(tmp_path), train=True, poses=POSES, mode="autoregressive", **OPTIONS)
    for i in range(len(ds)):
        frames, label = ds[i]
        expected, sign = clips[int(ds.data[i]["id"].item())]
        assert frames.dtype == torch.float32 and frames.shape == (len(expected), 58)
        np.testing.assert_allclose(frames.numpy(), expected, atol=1e-5)
        assert int(label) == ["a", "b"].index(sign)


def test_collate_pads_to_the_longest_clip():
    batch = [(torch.ones(3, 58), torch.tensor(0)), (torch.ones(5, 58), torch.tensor(1))]
    frames, lengths, labels = collate(batch)
    assert frames.shape == (2, 5, 58)
    assert lengths.tolist() == [3, 5] and labels.tolist() == [0, 1]
    assert not frames[0, 3:].any()


def test_synthetic_dataset_keeps_the_class_indexes_of_the_real_one(tmp_path):
    real, synth = tmp_path / "real", tmp_path / "synth"
    real.mkdir(), synth.mkdir()
    signs = ["a", "b", "c"]
    make_dataset(real, [(box_space_clip(10, seed=i), sign) for i, sign in enumerate(signs)], signs)
    make_dataset(synth, [(box_space_clip(10, seed=5), "c")], signs)
    base = Dataset_(data_dir=str(real), train=True, poses=POSES, **OPTIONS)
    ds = synthetic_dataset(str(synth), base, POSES, "classification", **OPTIONS)
    _, label = ds[0]
    # built on its own, the synthetic set would number its only sign 0
    assert int(label) == base.map_classes[2] == 2
    assert missing_classes(ds, base) == [0, 1]


def test_generation_sources_exclude_the_validation_clips(tmp_path):
    # the generator trains on, and opens its clips with, the classifier's training part only
    make_dataset(tmp_path, [(box_space_clip(10, seed=i), "ab"[i % 2]) for i in range(20)], ["a", "b"])
    base = Dataset_(data_dir=str(tmp_path), train=True, poses=POSES, mode="autoregressive", **OPTIONS)
    train, valid = train_val_dataset(base, val_split=0.1, random_state=42, stratify=base.data["sign"].to_list())
    subset, _ = misc.generation_dataset(train)
    assert len(subset.indices) == 18 and not set(subset.indices) & set(valid.indices)
