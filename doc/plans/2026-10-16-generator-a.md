# Generator A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An autoregressive sign generator, with two models: a causal Transformer and a sliding siMLPe. It writes full-length synthetic clips in the data format of the baseline classifier (`stgcn/hwgat-data`), for synthetic pretraining.

**Architecture:**
- **Models:** `src/models/argen_tf.py` and `src/models/argen_mlp.py` share one forward interface. Each predicts the drift of the next x frames, plus an end-of-sign logit.
- **Shared logic:** `src/generation/autoregressive.py` holds the plain tensor functions: collate, targets, loss, rollout, validation metrics, generation, writing.
- **Wiring:** a new `--mode autoregressive` connects them to the existing training loop (`src/loader.py`, `src/worker.py`) and to the existing dataset generation (`-sd`).
- **S3 fix:** the classifier's synthetic dataset now gets the data options and class indexes of the real one.

**Tech Stack:** Python 3.12, PyTorch 2.7 (`torch.nn.TransformerEncoder`), numpy, polars, pytest (new dev dependency). Native Windows with Git Bash.

**Spec:** [doc/specs/2026-10-16-generator-a-design.md](../specs/2026-10-16-generator-a-design.md)

## Global Constraints

- Python environment: `F:/ciencia/HandCraft-venv`. Run Python as `F:/ciencia/HandCraft-venv/Scripts/python`. For `uv`, set `UV_PROJECT_ENVIRONMENT=F:\ciencia\HandCraft-venv` and `UV_CACHE_DIR=F:\ciencia\uv-cache`. Datasets, outputs and caches stay on F:, never on C:.
- Data: INCLUDE at `F:/datasets/INCLUDE_official/`, LSFB at `F:/datasets/LSFB/`.
- Every commit passes the pre-commit hook: `ruff check`, `ruff format` (line length 160), `ty check`. Never use `--no-verify`.
- Commit messages, comments and docs are in English. Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- No new runtime dependency. pytest is added only to the `dev` dependency group.
- The existing modes (`classification`, `prediction`, `cond_prediction`) and the CsiMLPe generator keep their behaviour. Every new config option defaults to a value that leaves existing configs unchanged.
- Keypoints: HWGAT's 29 in 2D, 58 values per frame, in the box space of `pixel_coords` and `norm: "box"`.
- Match the surrounding code: comments explain why, not what; names follow the existing files.

## Review Focus

1. **LSFB clips shorter than the opening.** Some LSFB clips have 6 frames, against `opening_frames` 8. They must never be used as openings, and must be left out of the validation error, without crashing (tests in Task 5).
2. **A sign with no clip longer than the opening** gets no synthetic clips and a warning; generation continues for the other signs. `ClassBatchSampler` raises on a class without clips, so these signs must be removed first (Task 5).
3. **Synthetic signs dropped by the classifier's `min_samples`.** For example, LSFB with `sd_num × batch_size` below 20 drops synthetic signs. That must produce a visible warning when the synthetic set is loaded, not a silent drop (Task 2).
4. **An end signal that never fires, or fires at the first step.** The clip must stop at `max_gen_len` in the first case and at `opening_frames + pred_frames` in the second (Task 5).
5. **A NaN or inf in a rollout** must stop generation with an error before anything broken is written (Task 5).

---

### Task 1: Test setup and the writer of generated clips

**Files:**
- Modify: `pyproject.toml` (dev group, pytest options)
- Create: `src/generation/autoregressive.py`
- Modify: `src/worker.py:835-873` (`update_metadata`, `save_metadata`)
- Create: `tests/conftest.py`, `tests/test_writer.py`

**Interfaces:**
- Produces:
  - `PART_SIZES: dict[str, int]`
  - `write_clip(base_dir, clip_id, frames, poses, coords=2) -> None`
  - `write_metadata(base_dir, sign_to_index, metadata, video_sizes) -> None`
  - test helpers `POSES`, `box_space_clip(length, seed) -> np.ndarray (length, 58)`, `make_dataset(root, clips, signs) -> None` in `tests/conftest.py`

- [ ] **Step 1: Add pytest to the dev dependencies**

Run in Git Bash from the repository root:

```bash
UV_PROJECT_ENVIRONMENT='F:\ciencia\HandCraft-venv' UV_CACHE_DIR='F:\ciencia\uv-cache' uv add --dev pytest
```

Then add to the end of `pyproject.toml`:

```toml
[tool.pytest.ini_options]
# the code is run as scripts from src/ (python src/main.py), so the tests import it the same way
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 2: Write the test helpers**

Create `tests/conftest.py`:

```python
import numpy as np
import polars as pl

from generation.autoregressive import write_clip, write_metadata

# HWGAT's 29 keypoints, as in the stgcn/hwgat-data configs
POSES = [["pose", [0, 2, 5, 11, 12, 13, 14, 15, 16]], ["left_hand", [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]], ["right_hand", [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]]]


def box_space_clip(length, seed):
    """
    A clip of the 29 keypoints in box space, shape (length, 58), whose first frame has the nose at (0.5, 1/3) and a
    shoulder width of 1/6, as every clip loaded with norm "box" has.
    """
    rng = np.random.default_rng(seed)
    clip = rng.uniform(0.2, 0.8, (length, 29, 2)).astype(np.float32)
    clip[0, 0] = (0.5, 1 / 3)  # nose
    clip[0, 3] = (0.5 + 1 / 12, 0.45)  # MediaPipe pose keypoint 11, a shoulder
    clip[0, 4] = (0.5 - 1 / 12, 0.45)  # MediaPipe pose keypoint 12, the other shoulder
    return clip.reshape(length, 58)


def make_dataset(root, clips, signs):
    """Writes `clips`, a list of (frames (T, 58), sign), as a dataset at `root`. `signs` are the signs of sign_to_index."""
    sign_to_index = root / "sign_to_index_source.csv"
    pl.DataFrame({"sign": signs, "class": list(range(len(signs)))}).write_csv(sign_to_index)
    for i, (frames, _) in enumerate(clips):
        write_clip(root, i, frames, POSES)
    metadata = {
        "id": [str(i) for i in range(len(clips))],
        "sign": [sign for _, sign in clips],
        "signer": ["Bender"] * len(clips),
        "start": [0] * len(clips),
        "end": [len(frames) for frames, _ in clips],
    }
    write_metadata(root, sign_to_index, metadata, video_sizes=True)
```

- [ ] **Step 3: Write the failing tests**

Create `tests/test_writer.py`:

```python
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
    ds = Dataset_(data_dir=str(tmp_path), train=True, poses=POSES, max_len=64, temporal_sampling="pad", norm="box", coords=2, pixel_coords=True, missing_hand="wrist")
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
```

- [ ] **Step 4: Run them to see them fail**

Run: `F:/ciencia/HandCraft-venv/Scripts/python -m pytest tests/test_writer.py -v`
Expected: FAIL. The error is `ModuleNotFoundError: No module named 'generation'`.

- [ ] **Step 5: Write the writer**

Create `src/generation/autoregressive.py`:

```python
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
```

- [ ] **Step 6: Run the tests**

Run: `F:/ciencia/HandCraft-venv/Scripts/python -m pytest tests/test_writer.py -v`
Expected: 3 passed.

- [ ] **Step 7: Use `write_metadata` in the worker**

In `src/worker.py`, add the import after `import utils.misc as misc`:

```python
import generation.autoregressive as ar
```

Replace `update_metadata` and `save_metadata` (lines 835-841 and 856-873) with:

```python
    def update_metadata(self, ids, sign, lengths=None):
        # `lengths`: the number of frames of each clip; the clips of the CsiMLPe generator all have input_size[0]
        sti = pl.read_csv(os.path.join(self.RUN.data_dir, 'metadata', 'sign_to_index.csv'))
        self.metadata['id'] = self.metadata['id'] + ids
        self.metadata['sign'] = self.metadata['sign'] + [sti.filter(sti['class'] == sign)['sign'].item() for _ in ids]
        self.metadata['signer'] = self.metadata['signer'] + ['Bender' for _ in ids]
        self.metadata['start'] = self.metadata['start'] + [0 for _ in ids]
        self.metadata['end'] = self.metadata['end'] + (list(lengths) if lengths is not None else [self.DATA.input_size[0] for _ in ids])
```

```python
    def save_metadata(self):
        # generator A saves box-space values, which pixel_coords must leave unchanged (video size 1x1)
        ar.write_metadata(
            self.base_save_dir,
            os.path.join(self.RUN.data_dir, 'metadata', 'sign_to_index.csv'),
            self.metadata,
            video_sizes=self.RUN.mode == "autoregressive",
        )
```

Remove `json` and `shutil` from the imports of `worker.py` if nothing else uses them: `grep -n "json\.\|shutil\." src/worker.py`.

- [ ] **Step 8: Run the tests and the type check**

Run: `F:/ciencia/HandCraft-venv/Scripts/python -m pytest -v`, then `UV_PROJECT_ENVIRONMENT='F:\ciencia\HandCraft-venv' uv run ty check`.
Expected: 3 passed; ty reports no errors.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml uv.lock src/generation/autoregressive.py src/worker.py tests/conftest.py tests/test_writer.py
git commit -m "Write generated clips in the layout of a real dataset" -m "Generator A saves its clips with every MediaPipe keypoint at its index, so the box normalization finds the nose and the shoulders, and a 1x1 video size, so pixel_coords leaves the box-space values unchanged. Adds pytest as a dev dependency." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: The `autoregressive` dataset mode, and the synthetic dataset of the classifier (S3)

**Files:**
- Modify: `src/data/data_util.py` (`Dataset_.get_autoregressive_item`, `__getitem__`, new `synthetic_dataset`, `missing_classes`)
- Modify: `src/generation/autoregressive.py` (`collate`)
- Modify: `src/loader.py:84-167` (shared data options, synthetic dataset)
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `make_dataset`, `box_space_clip`, `POSES` (Task 1)
- Produces:
  - `Dataset_(..., mode="autoregressive")[i] -> (frames float32 (T, keypoints*coords), label int64 tensor)`
  - `collate(batch) -> (frames (B, T_max, F) zero-padded, lengths int64 (B,), labels (B,))`
  - `synthetic_dataset(synth_dir, base_dataset, poses, mode, **options) -> Dataset_`
  - `missing_classes(synthetic, base_dataset) -> list`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dataset.py`:

```python
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
```

- [ ] **Step 2: Run them to see them fail**

Run: `F:/ciencia/HandCraft-venv/Scripts/python -m pytest tests/test_dataset.py -v`
Expected: FAIL with `ImportError: cannot import name 'missing_classes'`.

- [ ] **Step 3: Add the collate function**

In `src/generation/autoregressive.py`, add `import torch` and `from torch.nn.utils.rnn import pad_sequence` to the imports, and after `PART_SIZES`:

```python
def collate(batch):
    """Batches clips of different lengths: (frames (B, T_max, F) padded with 0 at the end, lengths (B,), labels (B,))."""
    frames, labels = zip(*batch, strict=True)
    lengths = torch.tensor([len(f) for f in frames])
    return pad_sequence(list(frames), batch_first=True), lengths, torch.stack(list(labels))
```

- [ ] **Step 4: Add the dataset mode and the synthetic dataset**

In `src/data/data_util.py`, add after `get_cond_prediction_item`:

```python
    def get_autoregressive_item(self, index):
        # generator A: the whole clip, without resampling or augmentation, (T, keypoints * coords)
        value, label = self.load(index)
        return torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32)).flatten(1), torch.tensor(label)
```

In `__getitem__`, before `else: raise NotImplementedError`:

```python
        elif self.mode == "autoregressive":
            return self.get_autoregressive_item(index)
```

After the class `Dataset_` (before `class SeededSampler`):

```python
def synthetic_dataset(synth_dir, base_dataset, poses, mode, **options):
    """
    The synthetic dataset of a pretraining, read with the data options of the real one and its classes: built on its
    own, a synthetic set missing a sign would number the classes differently from the real training set.
    """
    return Dataset_(data_dir=synth_dir, train=True, poses=poses, mode=mode, filter_classes=base_dataset.classes, map_classes=base_dataset.map_classes, **options)


def missing_classes(synthetic, base_dataset):
    """The classes of the real training set that have no clip in the synthetic one (for example after min_samples)."""
    present = set(synthetic.data["sign"].to_list())
    return [c for c in base_dataset.classes if c not in present]
```

- [ ] **Step 5: Run the tests**

Run: `F:/ciencia/HandCraft-venv/Scripts/python -m pytest tests/test_dataset.py -v`
Expected: 4 passed.

- [ ] **Step 6: Share the data options in the loader (fix S3)**

In `src/loader.py`, add `missing_classes, synthetic_dataset` to the import from `data.data_util`, and `import generation.autoregressive as ar`. Replace the construction of `base_train_dataset` (lines 84-116) with:

```python
    # the data options of the real training set; the synthetic set of a pretraining is read with the same ones
    data_options = dict(
        load_data_in_memory=cfgs.RUN.load_data_in_memory,
        max_len=cfgs.DATA.max_len,
        target_len=cfgs.DATA.target_len,
        pad_frames=cfgs.DATA.pad_frames,
        pad_mode=cfgs.DATA.pad_mode,
        min_samples=cfgs.DATA.min_samples,
        random_crop=cfgs.DATA.random_crop,
        drop_frame=cfgs.DATA.drop_frame,
        drop_keypoint=cfgs.DATA.drop_keypoint,
        block_size=cfgs.DATA.block_size,
        flip_p=cfgs.DATA.flip_p,
        scale=cfgs.DATA.scale,
        rot=cfgs.DATA.rot,
        temporal_sampling=cfgs.DATA.temporal_sampling,
        speed=cfgs.DATA.speed,
        speed_range=cfgs.DATA.speed_range,
        hand_mask_p=cfgs.DATA.hand_mask_p,
        norm=cfgs.DATA.norm,
        shear_std=cfgs.DATA.shear_std,
        rot_std=cfgs.DATA.rot_std,
        mirror_p=cfgs.DATA.mirror_p,
        coords=cfgs.DATA.coords,
        pixel_coords=cfgs.DATA.pixel_coords,
        aug_pivot=cfgs.DATA.aug_pivot,
        xflip_p=cfgs.DATA.xflip_p,
        missing_hand=cfgs.DATA.missing_hand,
        hand_mask_interp=cfgs.DATA.hand_mask_interp,
    )
    base_train_dataset = Dataset_(data_dir=cfgs.RUN.data_dir, train=True, poses=cfgs.DATA.poses, mode=cfgs.RUN.mode, **data_options)
```

Replace the synthetic dataset block (lines 142-165) with:

```python
    if cfgs.RUN.synth_dir:
        if local_rank == 0:
            logger.info("Load the synthetic dataset {dir}.".format(dir=cfgs.RUN.synth_dir))
        synth_dataset = synthetic_dataset(cfgs.RUN.synth_dir, base_train_dataset, cfgs.DATA.synth_poses, cfgs.RUN.mode, **data_options)
        missing = missing_classes(synth_dataset, base_train_dataset)
        if local_rank == 0:
            logger.info("Synthetic dataset size: {dataset_size}".format(dataset_size=len(synth_dataset)))
            if missing:
                # for example signs with fewer synthetic clips than min_samples
                logger.warning("{n} of {total} classes have no synthetic clip: {classes}".format(n=len(missing), total=len(base_train_dataset.classes), classes=missing))
    else:
        synth_dataset = None
```

- [ ] **Step 7: Check that an existing classification run still starts**

Run (Git Bash, from the repository root):

With `S` the session's scratchpad directory:

```bash
sed -E 's/^  total_steps: 400$/  total_steps: 1/' src/configs/INCLUDE/stgcn/hwgat-data.yaml > $S/smoke-stgcn.yaml
WANDB_MODE=disabled F:/ciencia/HandCraft-venv/Scripts/python src/main.py --mode classification -t --test -data F:/datasets/INCLUDE_official/ -cfg $S/smoke-stgcn.yaml -save F:/ciencia/HandCraft-outputs/smoke/ --num_workers 0 --eval_workers 0 -every 1 --print_every 1 -mpc --seed 42 2>&1 | grep -E "Epoch|Test of|Error" | tail -3
```

Expected: one epoch and a line `Test of the checkpoint of epoch 1 on 816 clips`.

- [ ] **Step 8: Commit**

```bash
git add src/data/data_util.py src/generation/autoregressive.py src/loader.py tests/test_dataset.py
git commit -m "Read synthetic datasets with the data options and classes of the real one (S3)" -m "The synthetic dataset of a pretraining ignored the normalization, sampling and augmentation options of the classifier, and numbered its classes on its own, so a synthetic set missing a sign shifted every class index. It now gets the options and the class indexes of the real training set, and the loader warns about classes without synthetic clips. Adds the autoregressive dataset mode: the whole clip and its class." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Targets and loss

**Files:**
- Modify: `src/generation/autoregressive.py`
- Modify: `src/configs/config.py` (LOSS defaults, `define_losses`)
- Test: `tests/test_loss.py`

**Interfaces:**
- Produces:
  - `joint_error(pred, target, coords=2) -> Tensor (...)`: per-joint Euclidean distance, averaged over the joints.
  - `masked_mean(values, mask) -> Tensor ()`
  - `next_frame_targets(frames, lengths, pred_frames) -> (targets (B, T, x, F), frame_mask (B, T, x), position_mask (B, T), end (B, T) float)`
  - `rollout_targets(frames, lengths, cut, steps, pred_frames) -> (targets (B, steps, x, F), frame_mask (B, steps, x), position_mask (B, steps), end (B, steps) float)`
  - `AutoregressiveLoss(coords, vel_weight, end_weight, end_pos_weight)(pred, target, last, frame_mask, end_logit, end, position_mask) -> (total, position_error)`: shapes `pred, target (..., x, F)`, `last (..., F)`, `frame_mask (..., x)`, `end_logit, end, position_mask (...)`.
  - config `LOSS.loss_type: "autoregressive"`, with `LOSS.vel_weight`, `LOSS.end_weight` and `LOSS.end_pos_weight`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_loss.py`:

```python
import torch

from generation.autoregressive import AutoregressiveLoss, joint_error, next_frame_targets, rollout_targets


def ramp(lengths, features=4):
    """Frames whose every value is the frame index, padded with 0 after each length."""
    frames = torch.zeros(len(lengths), max(lengths), features)
    for b, n in enumerate(lengths):
        frames[b, :n] = torch.arange(n, dtype=torch.float32)[:, None]
    return frames


def test_joint_error_is_the_mean_distance_over_joints():
    pred, target = torch.zeros(58), torch.zeros(58)
    pred[0], pred[1] = 3.0, 4.0
    assert torch.isclose(joint_error(pred, target), torch.tensor(5.0 / 29))


def test_next_frame_targets_and_masks():
    frames = ramp([5, 3])
    targets, frame_mask, position_mask, end = next_frame_targets(frames, torch.tensor([5, 3]), pred_frames=2)
    assert targets.shape == (2, 5, 2, 4)
    assert targets[0, 1, :, 0].tolist() == [2.0, 3.0]
    assert frame_mask[0, 3].tolist() == [True, False] and frame_mask[1, 1].tolist() == [True, False]
    assert position_mask[0].tolist() == [True, True, True, True, False]
    assert position_mask[1].tolist() == [True, True, False, False, False]
    # position t ends the clip when the clip's last frame is among its next 2 frames
    assert end[0].tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]


def test_rollout_targets_follow_the_cut():
    frames = ramp([9, 6])
    targets, frame_mask, position_mask, end = rollout_targets(frames, torch.tensor([9, 6]), cut=3, steps=2, pred_frames=2)
    assert targets[0, :, :, 0].tolist() == [[3.0, 4.0], [5.0, 6.0]]
    assert frame_mask[1].tolist() == [[True, True], [True, False]]
    assert position_mask[1].tolist() == [True, True] and end[1].tolist() == [0.0, 1.0]


def test_loss_is_zero_for_a_perfect_prediction_and_ignores_masked_frames():
    loss = AutoregressiveLoss(coords=2, vel_weight=1.0, end_weight=1.0, end_pos_weight=1.0)
    target = torch.randn(2, 3, 2, 58)
    last = torch.randn(2, 3, 58)
    frame_mask = torch.ones(2, 3, 2, dtype=torch.bool)
    frame_mask[1, 2, 1] = False
    end = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    position_mask = torch.tensor([[True, True, True], [True, True, False]])
    end_logit = (end * 2 - 1) * 50
    total, position = loss(target.clone(), target, last, frame_mask, end_logit, end, position_mask)
    assert total < 1e-6 and position < 1e-6
    pred = target.clone()
    pred[1, 2, 1] += 100.0  # a masked frame
    masked_total, _ = loss(pred, target, last, frame_mask, end_logit, end, position_mask)
    # the velocity of the masked frame is masked too; the last valid frame is unchanged
    assert masked_total < 1e-6
```

- [ ] **Step 2: Run them to see them fail**

Run: `F:/ciencia/HandCraft-venv/Scripts/python -m pytest tests/test_loss.py -v`
Expected: FAIL with `ImportError: cannot import name 'AutoregressiveLoss'`.

- [ ] **Step 3: Write the targets and the loss**

In `src/generation/autoregressive.py`, add `import torch.nn as nn` and `import torch.nn.functional as F` to the imports, and after `collate`:

```python
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
        with_last = lambda frames: torch.cat([last.unsqueeze(-2), frames], dim=-2).diff(dim=-2)  # noqa: E731
        velocity = masked_mean(joint_error(with_last(pred), with_last(target), self.coords), frame_mask)
        pos_weight = torch.tensor(self.end_pos_weight, device=end_logit.device)
        end_loss = masked_mean(F.binary_cross_entropy_with_logits(end_logit, end, pos_weight=pos_weight, reduction="none"), position_mask)
        return position + self.vel_weight * velocity + self.end_weight * end_loss, position
```

- [ ] **Step 4: Run the tests**

Run: `F:/ciencia/HandCraft-venv/Scripts/python -m pytest tests/test_loss.py -v`
Expected: 4 passed.

- [ ] **Step 5: Register the loss in the config**

In `src/configs/config.py`, after `self.LOSS.label_smoothing = 0.0` add:

```python
        # generator A (loss_type "autoregressive"): weights of the velocity and end-of-sign terms, and the weight of the
        # positive end labels, about one position in ten
        self.LOSS.vel_weight = 1.0
        self.LOSS.end_weight = 0.1
        self.LOSS.end_pos_weight = 10.0
```

Change the comment `# type of loss \in ["CCE", "motion"]` to `# type of loss \in ["CCE", "motion", "autoregressive"]`. Add `from generation.autoregressive import AutoregressiveLoss` to the imports. In `define_losses`, add to `losses_dic`:

```python
            "autoregressive": AutoregressiveLoss(
                coords=self.DATA.coords, vel_weight=self.LOSS.vel_weight, end_weight=self.LOSS.end_weight, end_pos_weight=self.LOSS.end_pos_weight
            ),
```

- [ ] **Step 6: Run every test**

Run: `F:/ciencia/HandCraft-venv/Scripts/python -m pytest -v`
Expected: 11 passed.

- [ ] **Step 7: Commit**

```bash
git add src/generation/autoregressive.py src/configs/config.py tests/test_loss.py
git commit -m "Add the targets and the loss of generator A" -m "The loss is per keypoint (the motion loss of CsiMLPe is a norm over the whole frame): position error, velocity error and a weighted binary cross entropy for the end of the sign, with the frames and positions past the end of a clip masked." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: The two models

**Files:**
- Create: `src/models/argen_tf.py`, `src/models/argen_mlp.py`
- Modify: `src/configs/config.py` (DATA and MODEL defaults)
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes:
  - `TransMLP(dims, num_layers, MODULES, MODEL)`, called as `TransMLP([x (N, dims[0], dims[1]), y (N, dims[1])]) -> (N, dims[0], dims[1])`
  - `LabelEmbedder(num_classes, hidden_size, dropout_prob)`, called as `LabelEmbedder(labels, train) -> (N, hidden_size)`
  - both from `src/models/CsiMLPe.py`
- Produces:
  - `Model(DATA, RUN, MODULES, MODEL)` in both files, with attributes `pred_frames: int` and `generation_noise: float` (0.0 by default)
  - `forward(frames (B, T, F), labels (B,), last_only=False) -> (drift (B, L, x, F), end_logit (B, L))`, with L = 1 when `last_only`
  - `DATA.pred_frames`, `DATA.opening_frames`, `DATA.gen_stop`, `DATA.max_gen_len`
  - `MODEL.context_frames`, `MODEL.generation_noise`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_models.py`:

```python
from types import SimpleNamespace

import pytest
import torch

import models.argen_mlp as argen_mlp
import models.argen_tf as argen_tf
import utils.ops as ops

DATA = SimpleNamespace(num_keypoints=29, coords=2, num_classes=5, pred_frames=3, max_gen_len=40)
RUN = SimpleNamespace(mixed_precision=False)


def build(kind):
    torch.manual_seed(0)
    if kind == "tf":
        model = argen_tf.Model(DATA=DATA, RUN=RUN, MODULES=None, MODEL=SimpleNamespace(hidden_dim=32, nheads=4, mlp_dim=64, depth=2, dropout=0.0))
    else:
        MODULES = SimpleNamespace(glinear=ops.Temporal_FC, feature_norm=ops.SLayerNorm)
        MODEL = SimpleNamespace(hidden_dim=32, depth=2, context_frames=4, class_dropout_prob=0.0, noise_scale=0.0)
        model = argen_mlp.Model(DATA=DATA, RUN=RUN, MODULES=MODULES, MODEL=MODEL)
    # the drift head starts at zero; random weights make the outputs depend on the input for these tests
    torch.nn.init.normal_(model.drift_head.weight)
    return model.eval()


@pytest.mark.parametrize("kind", ["tf", "mlp"])
def test_shapes_and_last_only(kind):
    model = build(kind)
    frames, labels = torch.randn(2, 10, 58), torch.tensor([0, 4])
    drift, end = model(frames, labels)
    assert drift.shape == (2, 10, 3, 58) and end.shape == (2, 10)
    last_drift, last_end = model(frames, labels, last_only=True)
    torch.testing.assert_close(last_drift[:, 0], drift[:, -1], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(last_end[:, 0], end[:, -1], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("kind", ["tf", "mlp"])
def test_drift_starts_at_zero(kind):
    model = build(kind)
    torch.nn.init.zeros_(model.drift_head.weight)
    drift, _ = model(torch.randn(1, 6, 58), torch.tensor([1]))
    assert not drift.any()


def test_transformer_is_causal():
    model = build("tf")
    frames, labels = torch.randn(1, 10, 58), torch.tensor([2])
    changed = frames.clone()
    changed[:, 6:] += 1.0
    drift, _ = model(frames, labels)
    drift2, _ = model(changed, labels)
    torch.testing.assert_close(drift[:, :6], drift2[:, :6])
    assert not torch.allclose(drift[:, 6:], drift2[:, 6:])


def test_mlp_sees_only_its_window():
    model = build("mlp")  # window of 4 frames
    frames, labels = torch.randn(1, 10, 58), torch.tensor([2])
    drift, _ = model(frames, labels)
    before = frames.clone()
    before[:, 3] += 1.0  # outside the window of position 7 (frames 4 to 7)
    torch.testing.assert_close(model(before, labels)[0][:, 7], drift[:, 7])
    inside = frames.clone()
    inside[:, 4] += 1.0
    assert not torch.allclose(model(inside, labels)[0][:, 7], drift[:, 7])


@pytest.mark.parametrize("kind", ["tf", "mlp"])
def test_generation_noise_changes_the_output_only_when_set(kind):
    model = build(kind)
    frames, labels = torch.randn(1, 6, 58), torch.tensor([3])
    torch.testing.assert_close(model(frames, labels)[0], model(frames, labels)[0])
    model.generation_noise = 0.5
    assert not torch.allclose(model(frames, labels)[0], model(frames, labels)[0])
```

- [ ] **Step 2: Run them to see them fail**

Run: `F:/ciencia/HandCraft-venv/Scripts/python -m pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models.argen_mlp'`.

- [ ] **Step 3: Write the causal Transformer**

Create `src/models/argen_tf.py`:

```python
"""
Generator A, causal Transformer (doc/specs/2026-10-16-generator-a-design.md): from the frames so far and the sign, the
drift of the next pred_frames frames from the last one, and the logit that the sign ends within them.
"""

import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, DATA, RUN, MODULES, MODEL):
        super().__init__()
        features = DATA.num_keypoints * DATA.coords
        self.pred_frames = DATA.pred_frames
        # standard deviation of a Gaussian noise added to the class embedding, set during generation only
        self.generation_noise = 0.0
        d = MODEL.hidden_dim
        self.embed = nn.Linear(features, d)
        self.class_embed = nn.Embedding(DATA.num_classes, d)
        # the class token and up to max_gen_len frames
        self.pos_embed = nn.Embedding(DATA.max_gen_len + 1, d)
        layer = nn.TransformerEncoderLayer(d, MODEL.nheads, MODEL.mlp_dim, MODEL.dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, MODEL.depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.drift_head = nn.Linear(d, self.pred_frames * features)
        self.end_head = nn.Linear(d, 1)
        # start by predicting no motion
        nn.init.zeros_(self.drift_head.weight)
        nn.init.zeros_(self.drift_head.bias)

    def forward(self, frames, labels, last_only=False):
        B, T, features = frames.shape
        if T + 1 > self.pos_embed.num_embeddings:
            raise ValueError("{} frames, more than max_gen_len ({})".format(T, self.pos_embed.num_embeddings - 1))
        cls = self.class_embed(labels)
        if self.generation_noise > 0:
            cls = cls + self.generation_noise * torch.randn_like(cls)
        h = torch.cat([cls[:, None], self.embed(frames)], dim=1) + self.pos_embed(torch.arange(T + 1, device=frames.device))
        mask = nn.Transformer.generate_square_subsequent_mask(T + 1, device=frames.device)
        # position t of the output (after the class token) has seen frames 0 to t
        h = self.norm(self.encoder(h, mask=mask, is_causal=True))[:, 1:]
        if last_only:
            h = h[:, -1:]
        drift = self.drift_head(h).unflatten(-1, (self.pred_frames, features))
        return drift, self.end_head(h)[..., 0]
```

- [ ] **Step 4: Write the sliding siMLPe**

Create `src/models/argen_mlp.py`:

```python
"""
Generator A, sliding siMLPe (doc/specs/2026-10-16-generator-a-design.md): from the last context_frames frames and the
sign, the drift of the next pred_frames frames from the last one, and the logit that the sign ends within them. The
blocks are those of CsiMLPe.
"""

import torch
from torch import nn

from models.CsiMLPe import LabelEmbedder, TransMLP


class Model(nn.Module):
    def __init__(self, DATA, RUN, MODULES, MODEL):
        super().__init__()
        features = DATA.num_keypoints * DATA.coords
        d, self.window, self.pred_frames = MODEL.hidden_dim, MODEL.context_frames, DATA.pred_frames
        # standard deviation of a Gaussian noise added to the class embedding, set during generation only
        self.generation_noise = 0.0
        self.embed = nn.Linear(features, d)
        # the sign is added to the features of every frame; the blocks also get it, over the frames of the window
        self.class_embed = nn.Embedding(DATA.num_classes, d)
        self.y_embedder = LabelEmbedder(DATA.num_classes, self.window, MODEL.class_dropout_prob)
        self.mlp = TransMLP(dims=[d, self.window], num_layers=MODEL.depth, MODULES=MODULES, MODEL=MODEL)
        self.temporal_out = nn.Linear(self.window, self.pred_frames)
        self.drift_head = nn.Linear(d, features)
        self.end_head = nn.Linear(d, 1)
        # start by predicting no motion
        nn.init.zeros_(self.drift_head.weight)
        nn.init.zeros_(self.drift_head.bias)

    def forward(self, frames, labels, last_only=False):
        B, T, features = frames.shape
        if last_only:
            frames = frames[:, -self.window :]
        # the window of a position near the start is padded with the first frame
        padded = torch.cat([frames[:, :1].expand(B, self.window - 1, features), frames], dim=1)
        windows = padded.unfold(1, self.window, 1)
        if last_only:
            windows = windows[:, -1:]
        L = windows.shape[1]
        windows = windows.permute(0, 1, 3, 2).reshape(B * L, self.window, features)
        labels = labels.repeat_interleave(L)
        cls = self.class_embed(labels)
        if self.generation_noise > 0:
            cls = cls + self.generation_noise * torch.randn_like(cls)
        h = (self.embed(windows) + cls[:, None]).transpose(1, 2).contiguous()
        h = self.mlp([h, self.y_embedder(labels, self.training)])
        h = self.temporal_out(h).transpose(1, 2)
        drift = self.drift_head(h).view(B, L, self.pred_frames, features)
        return drift, self.end_head(h.mean(dim=1)).view(B, L)
```

- [ ] **Step 5: Run the tests**

Run: `F:/ciencia/HandCraft-venv/Scripts/python -m pytest tests/test_models.py -v`
Expected: 8 passed.

- [ ] **Step 6: Add the config defaults**

In `src/configs/config.py`, after `self.DATA.transform = None`:

```python
        # generator A (--mode autoregressive): frames predicted per step (x), real frames that open a generated clip (k),
        # when a generated clip stops ("length": the length of its source clip, "end_signal": the learned end of the sign)
        # and the most frames a clip can have
        self.DATA.pred_frames = 4
        self.DATA.opening_frames = 8
        self.DATA.gen_stop = "length"
        self.DATA.max_gen_len = 320
```

After `self.MODEL.noise_scale = 0.0`:

```python
        # generator A: frames seen by argen_mlp (w), and the noise on the class embedding during generation
        self.MODEL.context_frames = 8
        self.MODEL.generation_noise = 0.0
```

- [ ] **Step 7: Run every test**

Run: `F:/ciencia/HandCraft-venv/Scripts/python -m pytest -v`
Expected: 19 passed.

- [ ] **Step 8: Commit**

```bash
git add src/models/argen_tf.py src/models/argen_mlp.py src/configs/config.py tests/test_models.py
git commit -m "Add the two models of generator A" -m "argen_tf: a causal torch.nn.TransformerEncoder over the frames so far with the sign as a class token. argen_mlp: the CsiMLPe blocks on the last context_frames frames, every position in one batch through unfold. Both predict the drift of the next pred_frames frames from the last one and an end-of-sign logit, starting from a zero drift." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Rollout, validation metrics and generation

**Files:**
- Modify: `src/generation/autoregressive.py`
- Test: `tests/test_rollout.py`

**Interfaces:**
- Consumes: the model interface of Task 4, and `joint_error` of Task 3.
- Produces:
  - `random_cut(lengths, opening_frames) -> int`
  - `rollout(model, context (B, t, F), labels, steps) -> (new frames (B, steps*x, F), end_logits (B, steps))`
  - `rollout_metrics(model, frames, lengths, labels, opening_frames, pred_frames, coords=2) -> (error (B, T-k), valid (B, T-k) bool, correct int, decided int)`
  - `generate(model, openings (B, k, F), labels, source_lengths (B,), stop, max_len) -> list of B tensors (T_b, F) on the CPU`
  - `usable_openings(dataset, labels, classes, opening_frames) -> (Subset, labels, kept classes, dropped classes)`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_rollout.py`:

```python
import pytest
import torch
from torch import nn

from generation.autoregressive import generate, random_cut, rollout, rollout_metrics, usable_openings


class ConstantDrift(nn.Module):
    """Predicts that every value grows by `step` per frame, with a constant end logit; records its inputs."""

    def __init__(self, pred_frames, step=1.0, end=-5.0, value=None):
        super().__init__()
        self.pred_frames, self.step, self.end, self.value, self.calls = pred_frames, step, end, value, []

    def forward(self, frames, labels, last_only=False):
        self.calls.append(frames.clone())
        B, T, features = frames.shape
        L = 1 if last_only else T
        drift = self.step * torch.arange(1, self.pred_frames + 1, dtype=frames.dtype).view(1, 1, -1, 1).expand(B, L, -1, features)
        if self.value is not None:
            drift = torch.full_like(drift, self.value)
        return drift, torch.full((B, L), self.end)


def ramp(n, features=4):
    return torch.arange(n, dtype=torch.float32)[None, :, None].expand(1, n, features).clone()


def test_rollout_feeds_its_own_frames_back():
    model = ConstantDrift(pred_frames=2)
    context = ramp(3)
    new, ends = rollout(model, context, torch.tensor([0]), steps=2)
    assert new[0, :, 0].tolist() == [3.0, 4.0, 5.0, 6.0]
    assert ends.shape == (1, 2)
    torch.testing.assert_close(model.calls[1][:, :3], context)
    torch.testing.assert_close(model.calls[1][:, 3:], new[:, :2])


def test_random_cut_stays_inside_the_shortest_clip():
    for _ in range(50):
        cut = random_cut(torch.tensor([20, 6]), opening_frames=8)
        assert 5 <= cut <= 5
    with pytest.raises(ValueError):
        random_cut(torch.tensor([1, 6]), opening_frames=8)


def test_rollout_metrics_masks_short_and_padded_clips():
    frames = torch.zeros(3, 12, 4)
    lengths = torch.tensor([12, 9, 3])  # the third is not longer than the 4 opening frames
    for b, n in enumerate(lengths.tolist()):
        frames[b, :n] = ramp(n)[0]
    error, valid, correct, decided = rollout_metrics(ConstantDrift(pred_frames=2), frames, lengths, torch.zeros(3, dtype=torch.long), 4, 2, coords=2)
    assert error.shape == (3, 8)
    assert valid[0].all() and valid[1].tolist() == [True] * 5 + [False] * 3 and not valid[2].any()
    assert error[valid].abs().max() < 1e-6
    # the end logit is always negative: right on every step but the last of each clip
    assert decided == 4 + 3 and correct == decided - 2


def test_generate_stops_at_the_source_length():
    clips = generate(ConstantDrift(pred_frames=3), ramp(4).expand(2, 4, 4), torch.tensor([0, 1]), torch.tensor([9, 15]), "length", max_len=12)
    assert [len(c) for c in clips] == [9, 12]
    assert clips[0][:, 0].tolist() == list(range(9))


def test_generate_stops_at_the_end_signal_or_the_cap():
    opening = ramp(4)
    assert len(generate(ConstantDrift(pred_frames=3, end=5.0), opening, torch.tensor([0]), torch.tensor([30]), "end_signal", max_len=20)[0]) == 7
    assert len(generate(ConstantDrift(pred_frames=3, end=-5.0), opening, torch.tensor([0]), torch.tensor([30]), "end_signal", max_len=20)[0]) == 20


def test_generate_refuses_non_finite_frames():
    with pytest.raises(FloatingPointError):
        generate(ConstantDrift(pred_frames=2, value=float("nan")), ramp(4), torch.tensor([0]), torch.tensor([10]), "length", max_len=20)


def test_usable_openings_drops_short_clips_and_empty_classes():
    dataset = [(torch.zeros(n, 4), torch.tensor(c)) for n, c in [(5, 0), (10, 1), (3, 2), (4, 1)]]
    subset, labels, kept, dropped = usable_openings(dataset, [0, 1, 2, 1], [0, 1, 2], opening_frames=4)
    assert list(subset.indices) == [0, 1] and labels == [0, 1]
    assert kept == [0, 1] and dropped == [2]
```

- [ ] **Step 2: Run them to see them fail**

Run: `F:/ciencia/HandCraft-venv/Scripts/python -m pytest tests/test_rollout.py -v`
Expected: FAIL with `ImportError: cannot import name 'generate'`.

- [ ] **Step 3: Write the rollout functions**

In `src/generation/autoregressive.py`, add `import math` and `from torch.utils.data import Subset` to the imports, and at the end of the file:

```python
def random_cut(lengths, opening_frames):
    """The frame a training rollout starts from: after opening_frames when the clips are long enough, and before the last frame of the shortest clip."""
    last = int(lengths.min()) - 1
    if last < 1:
        raise ValueError("every clip of a rollout batch needs at least 2 frames")
    first = min(opening_frames, last)
    return int(torch.randint(first, last + 1, (1,)))


def rollout(model, context, labels, steps):
    """Appends `steps` predictions of x frames to `context` (B, t, F), each from every frame so far. Returns the new frames (B, steps*x, F) and the end logit of each step (B, steps)."""
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
```

- [ ] **Step 4: Run the tests**

Run: `F:/ciencia/HandCraft-venv/Scripts/python -m pytest tests/test_rollout.py -v`
Expected: 7 passed.

- [ ] **Step 5: Run every test**

Run: `F:/ciencia/HandCraft-venv/Scripts/python -m pytest -v`
Expected: 26 passed.

- [ ] **Step 6: Commit**

```bash
git add src/generation/autoregressive.py tests/test_rollout.py
git commit -m "Add the rollout, validation metrics and generation of generator A" -m "A rollout appends the model's own frames x at a time. The validation rolls every clip out from its opening and masks the frames past its end and the clips not longer than the opening. Generation stops at the source length or at the end signal, at most at max_gen_len, and refuses non-finite frames. Clips not longer than the opening cannot open a generated clip; signs left without one are reported." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Wiring, configs and an end-to-end run

**Files:**
- Modify: `src/worker.py` (mode binding, train and evaluate steps, generation)
- Modify: `src/loader.py` (collate functions, generation branch)
- Create: `src/configs/INCLUDE/argen/tf.yaml`, `src/configs/INCLUDE/argen/mlp-w4.yaml`, `src/configs/INCLUDE/argen/mlp-w8.yaml`, `src/configs/INCLUDE/stgcn/hwgat-data-pretrain.yaml`

**Interfaces:**
- Consumes: everything in `generation.autoregressive`, through `import generation.autoregressive as ar`, and the models of Task 4.
- Produces:
  - `--mode autoregressive`, for training (`-t`), testing (`--test`) and generation (`-sd --sd_num N`)
  - the generated dataset in `<save>/generated_datasets/<run_name>/`
  - `WORKER.autoregressive_train_step`, `WORKER.autoregressive_evaluate_step`, `WORKER.save_autoregressive_dataset`

- [ ] **Step 1: Bind the mode in the worker**

In `src/worker.py`, in `__init__`, before `else: raise NotImplementedError` of the mode binding:

```python
        elif cfgs.RUN.mode == "autoregressive":
            self.train_step = self.autoregressive_train_step
            self.evaluate_step = self.autoregressive_evaluate_step
```

- [ ] **Step 2: Add the training step**

After `cond_prediction_train_step`:

```python
    def autoregressive_train_step(self, step):
        """
        One epoch of generator A. With rollout_steps 1 every position of the real clips predicts its next frames in one
        pass; with more, the model continues from a random cut on its own frames, and the loss covers every step.
        """
        self.model.train()
        misc.toggle_grad(model=self.model, grad=True)
        x, r = self.DATA.pred_frames, self.OPTIMIZATION.rollout_steps
        valid_loss = misc.AverageMeter()
        for frames, lengths, labels in self.train_iter:
            self.OPTIMIZATION.optimizer.zero_grad()
            frames = frames.to(self.local_rank, non_blocking=True).float()
            lengths = lengths.to(self.local_rank, non_blocking=True)
            labels = labels.to(self.local_rank, non_blocking=True)
            with torch.autocast("cuda", enabled=self.RUN.mixed_precision):
                if r == 1:
                    drift, end_logit = self.model(frames, labels)
                    targets, frame_mask, position_mask, end = ar.next_frame_targets(frames, lengths, x)
                    pred, last = frames[:, :, None] + drift, frames
                else:
                    cut = ar.random_cut(lengths, self.DATA.opening_frames)
                    generated, end_logit = ar.rollout(self.model, frames[:, :cut], labels, r)
                    targets, frame_mask, position_mask, end = ar.rollout_targets(frames, lengths, cut, r, x)
                    pred = generated.unflatten(1, (r, x))
                    # the frame before each step: the last real one, then the last generated one of the previous step
                    last = torch.cat([frames[:, cut - 1 : cut], generated[:, x - 1 : (r - 1) * x : x]], dim=1)
            # in float32: the loss sums over many frames
            loss, _ = self.loss(pred.float(), targets, last.float(), frame_mask, end_logit.float(), end, position_mask)
            if self.RUN.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            self.optimizer_step()
            valid_loss.update(loss.item(), frames.size(0))

        if self.RUN.empty_cache:
            torch.cuda.empty_cache()
        return None, None, valid_loss.avg
```

- [ ] **Step 3: Add the evaluation step**

After `cond_prediction_evaluate_step`:

```python
    @torch.no_grad()
    def autoregressive_evaluate_step(self, dataloader):
        """
        Rolls every clip out from its first opening_frames frames to its real length. The loss, which selects the best
        checkpoint, is the mean per-keypoint error of the generated frames; the error per rollout step and the accuracy
        of the end signal are logged.
        """
        self.model.eval()
        x, k = self.DATA.pred_frames, self.DATA.opening_frames
        error_sum = torch.zeros(0, dtype=torch.float64)
        error_count = torch.zeros(0, dtype=torch.float64)
        correct = decided = too_short = 0
        for frames, lengths, labels in dataloader:
            frames = frames.to(self.local_rank).float()
            labels = labels.to(self.local_rank)
            with torch.autocast("cuda", enabled=self.RUN.mixed_precision):
                error, valid, c, d = ar.rollout_metrics(self.model, frames, lengths, labels, k, x, self.DATA.coords)
            correct, decided, too_short = correct + c, decided + d, too_short + int((lengths <= k).sum())
            # sums per rollout step: generated frame j belongs to step j // x
            step_index = (torch.arange(error.shape[1]) // x).repeat(error.shape[0])
            weights = valid.cpu().flatten().double()
            sums = torch.bincount(step_index, weights=error.cpu().flatten().double() * weights)
            counts = torch.bincount(step_index, weights=weights)
            n = max(len(sums), len(error_sum))
            error_sum = torch.nn.functional.pad(error_sum, (0, n - len(error_sum))) + torch.nn.functional.pad(sums, (0, n - len(sums)))
            error_count = torch.nn.functional.pad(error_count, (0, n - len(error_count))) + torch.nn.functional.pad(counts, (0, n - len(counts)))

        mean = float(error_sum.sum() / error_count.sum().clamp(min=1))
        if self.logger:
            per_step = (error_sum / error_count.clamp(min=1)).tolist()
            self.logger.info(
                "Rollout error per step: {steps} | end accuracy {acc:.2f} | clips not longer than the opening: {short}".format(
                    steps=" ".join("{:.4f}".format(e) for e in per_step[:12]), acc=100 * correct / max(decided, 1), short=too_short
                )
            )
        return None, None, mean, mean
```

- [ ] **Step 4: Add the generation**

After `save_dataset`:

```python
    @torch.no_grad()
    def save_autoregressive_dataset(self, dataloader, signs):
        """
        Generator A: a clip from the opening frames of every clip of `dataloader`, whose batches each hold clips of one
        class (misc.ClassBatchSampler), saved in the layout of a real dataset. `signs` maps a class index to its sign.
        """
        if self.global_rank == 0:
            self.logger.info("Generate {n} clips per class for {c} classes.".format(n=self.RUN.sd_num * self.OPTIMIZATION.batch_size, c=len(dataloader.batch_sampler.classes)))
        self.model.eval()
        model = misc.peel_model(self.model)
        model.generation_noise = self.MODEL.generation_noise
        k = self.DATA.opening_frames
        ids, lengths_out, current = [], [], None
        for frames, lengths, labels in dataloader:
            label = int(labels[0])
            if label != current:
                if current is not None:
                    self.update_metadata(ids, signs[current], lengths_out)
                    ids, lengths_out = [], []
                current = label
            frames = frames.to(self.local_rank).float()
            labels = labels.to(self.local_rank)
            with torch.autocast("cuda", enabled=self.RUN.mixed_precision):
                clips = ar.generate(self.model, frames[:, :k], labels, lengths, self.DATA.gen_stop, self.DATA.max_gen_len)
            for clip in clips:
                ar.write_clip(self.base_save_dir, self.gen_curr_id, clip.numpy(), self.DATA.poses, self.DATA.coords)
                ids.append(str(self.gen_curr_id))
                lengths_out.append(len(clip))
                self.gen_curr_id += 1
        if current is not None:
            self.update_metadata(ids, signs[current], lengths_out)
        model.generation_noise = 0.0
        if self.global_rank == 0:
            self.logger.info("Dataset saved: {n} clips.".format(n=self.gen_curr_id))
```

- [ ] **Step 5: Wire the loader**

In `src/loader.py`, after the `worker_options` function, add:

```python
    # generator A batches whole clips of different lengths
    collate = ar.collate if cfgs.RUN.mode == "autoregressive" else None
```

Add `collate_fn=collate,` to the `train_dataloader`, `valid_dataloader` and `test_dataloader` constructors. Replace the body of `if cfgs.RUN.save_dataset:` with:

```python
    if cfgs.RUN.save_dataset:
        if global_rank == 0:
            print(""), logger.info("-" * 80)
        # one loader for every class: its batches come class by class, sd_num batches each (starting the loader
        # workers for every class took about 35 s per class on Windows)
        gen_dataset, gen_labels = misc.generation_dataset(train_dataset.dataset if isinstance(train_dataset, SeededDataset) else train_dataset)
        classes = [base_train_dataset.map_classes[sign] for sign in base_train_dataset.classes]
        if cfgs.RUN.mode == "autoregressive":
            # a generated clip opens with real frames: its source clip must be longer than them
            gen_dataset, gen_labels, classes, dropped = ar.usable_openings(gen_dataset, gen_labels, classes, cfgs.DATA.opening_frames)
            if dropped and global_rank == 0:
                logger.warning("{n} classes have no clip longer than {k} frames and get no generated clips: {c}".format(n=len(dropped), k=cfgs.DATA.opening_frames, c=dropped))
        gen_dataloader = DataLoader(
            dataset=gen_dataset,
            batch_sampler=misc.ClassBatchSampler(gen_labels, classes, cfgs.OPTIMIZATION.batch_size, cfgs.RUN.sd_num, cfgs.RUN.seed),
            pin_memory=True,
            collate_fn=collate,
            **worker_options(cfgs.RUN.num_workers),
            generator=torch.Generator().manual_seed(cfgs.RUN.seed),
        )
        worker.prepare_generation()
        signs = {base_train_dataset.map_classes[sign]: sign for sign in base_train_dataset.classes}
        if cfgs.RUN.mode == "autoregressive":
            worker.save_autoregressive_dataset(gen_dataloader, signs)
        else:
            worker.save_dataset(gen_dataloader, signs)
        worker.save_metadata()
```

- [ ] **Step 6: Write the configs**

Create `src/configs/INCLUDE/argen/tf.yaml`:

```yaml
# Generator A, causal Transformer (doc/specs/2026-10-16-generator-a-design.md): HWGAT's 29 keypoints in 2D in the box
# space of stgcn/hwgat-data, 4 frames predicted per step, rollouts opened with 8 real frames.
DATA:
  name: "INCLUDE"
  poses: [["pose",[0,2,5,11,12,13,14,15,16]],["left_hand",[0,4,5,8,9,12,13,16,17,20]],["right_hand",[0,4,5,8,9,12,13,16,17,20]]]
  num_keypoints: 29
  num_classes: 262
  min_samples: 0
  oversample: False
  max_len: 64
  input_size: [64, 29, 2]
  transform: "none"
  coords: 2
  norm: "box"
  pixel_coords: True
  missing_hand: "wrist"
  test_drop_last: False
  pred_frames: 4
  opening_frames: 8
  gen_stop: "length"
  max_gen_len: 320
MODEL:
  backbone: "argen_tf"
  hidden_dim: 256
  nheads: 8
  mlp_dim: 1024
  depth: 4
  dropout: 0.1
  generation_noise: 0.0
LOSS:
  loss_type: "autoregressive"
  vel_weight: 1.0
  end_weight: 0.1
  end_pos_weight: 10.0
OPTIMIZATION:
  type_: "RAdam"
  batch_size: 16
  lr: 0.001
  weight_decay: 0.0001
  beta1: 0.9
  beta2: 0.999
  total_steps: 200
  lrscheduler: "OneCycle"
  max_lr: 0.001
  pct_start: 0.1
  rollout_steps: 1
```

Create `src/configs/INCLUDE/argen/mlp-w8.yaml` with the same `DATA`, `LOSS` and `OPTIMIZATION` sections and the header comment `# Generator A, sliding siMLPe on the last 8 frames (doc/specs/2026-10-16-generator-a-design.md), with the data of argen/tf.` The `MODEL` section is:

```yaml
MODEL:
  backbone: "argen_mlp"
  hidden_dim: 256
  depth: 6
  context_frames: 8
  use_spatial_fc: False
  feature_norm: "slayernorm"
  noise_scale: 0.0
  class_dropout_prob: 0.0
  dropout: 0.1
  generation_noise: 0.0
```

Create `src/configs/INCLUDE/argen/mlp-w4.yaml` as a copy of `mlp-w8.yaml` with `context_frames: 4` and `last 4 frames` in the header comment.

Add `rollout_steps` to `src/configs/config.py`, after `self.OPTIMIZATION.synth_total_steps = 1`:

```python
        # generator A: steps of x frames a training rollout continues on the model's own frames (1 = real context only)
        self.OPTIMIZATION.rollout_steps = 1
```

Create `src/configs/INCLUDE/stgcn/hwgat-data-pretrain.yaml`:

```bash
{ echo "# stgcn/hwgat-data pretrained for 40 epochs on a synthetic dataset (-s_data) of generator A, read with the same keypoints."; tail -n +5 src/configs/INCLUDE/stgcn/hwgat-data.yaml | sed 's/^  test_drop_last: False$/  test_drop_last: False\n  synth_poses: [["pose",[0,2,5,11,12,13,14,15,16]],["left_hand",[0,4,5,8,9,12,13,16,17,20]],["right_hand",[0,4,5,8,9,12,13,16,17,20]]]/; s/^  total_steps: 400$/  total_steps: 400\n  synth_total_steps: 40/'; } > src/configs/INCLUDE/stgcn/hwgat-data-pretrain.yaml
grep -n "synth_poses\|synth_total_steps" src/configs/INCLUDE/stgcn/hwgat-data-pretrain.yaml
```

Expected: one line for each option.

- [ ] **Step 7: Run every test and the linters**

Run: `F:/ciencia/HandCraft-venv/Scripts/python -m pytest -v`, then `UV_PROJECT_ENVIRONMENT='F:\ciencia\HandCraft-venv' uv run ruff check && UV_PROJECT_ENVIRONMENT='F:\ciencia\HandCraft-venv' uv run ty check`.
Expected: 26 passed; no lint or type errors.

- [ ] **Step 8: End-to-end run on INCLUDE**

In Git Bash, with `S` the session's scratchpad directory:

```bash
for cfg in tf mlp-w4; do
  sed -E 's/^  total_steps: 200$/  total_steps: 2/' src/configs/INCLUDE/argen/$cfg.yaml > $S/smoke-$cfg.yaml
  WANDB_MODE=disabled F:/ciencia/HandCraft-venv/Scripts/python src/main.py --mode autoregressive -t --test -sd --sd_num 1 \
    -data F:/datasets/INCLUDE_official/ -cfg $S/smoke-$cfg.yaml -save F:/ciencia/HandCraft-outputs/smoke-argen/ \
    --num_workers 0 --eval_workers 0 -every 1 --print_every 1 -mpc --seed 42 > $S/smoke-$cfg.out 2>&1
  echo "$cfg exit $?"; grep -E "Epoch|Rollout error|Test of|Dataset saved|Error" $S/smoke-$cfg.out | sed -E 's/^\[INFO\] [0-9-]+ //' | tail -6
done
```

Expected for each config:
- exit 0
- 2 epochs with a validation line and a `Rollout error per step` line
- a `Test of the checkpoint` line
- `Dataset saved: 4192 clips.` (262 signs × 16)

- [ ] **Step 9: Pretrain the classifier on the generated clips for one epoch**

```bash
D=$(ls -d /f/ciencia/HandCraft-outputs/smoke-argen/generated_datasets/*smoke-tf* | tail -1); D="F:${D#/f}"
sed -E 's/^  total_steps: 400$/  total_steps: 1/; s/^  synth_total_steps: 40$/  synth_total_steps: 1/' src/configs/INCLUDE/stgcn/hwgat-data-pretrain.yaml > $S/smoke-pretrain.yaml
WANDB_MODE=disabled F:/ciencia/HandCraft-venv/Scripts/python src/main.py --mode classification -t --test -data F:/datasets/INCLUDE_official/ -s_data "$D/" \
  -cfg $S/smoke-pretrain.yaml -save F:/ciencia/HandCraft-outputs/smoke-argen/ --num_workers 0 --eval_workers 0 -every 1 --print_every 1 -mpc --seed 42 > $S/smoke-pretrain.out 2>&1
echo "exit $?"; grep -E "Synthetic|Pretrain|Epoch|Test of|no synthetic" $S/smoke-pretrain.out | sed -E 's/^\[INFO\] [0-9-]+ //' | tail -6
```

Expected:
- exit 0
- `Synthetic dataset size: 4192` and no warning about missing classes
- a `Pretrain epoch` line, one training epoch and a `Test of the checkpoint` line

- [ ] **Step 10: Commit**

```bash
git add src/worker.py src/loader.py src/configs/config.py src/configs/INCLUDE/argen src/configs/INCLUDE/stgcn/hwgat-data-pretrain.yaml
git commit -m "Train, test and run generator A with --mode autoregressive" -m "The worker trains generator A on real context or on rollouts, selects the checkpoint by the rollout error from the opening frames, and generates clips from the openings of the training clips into a regular dataset. Configs for the causal Transformer and the sliding siMLPe on INCLUDE, and stgcn/hwgat-data-pretrain to pretrain the baseline classifier on the generated clips." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Documentation

**Files:**
- Modify: `doc/configuration.md`, `doc/models.md`, `doc/synthetic-data.md`, `doc/training.md`, `doc/development.md`, `doc/roadmap.md`

- [ ] **Step 1: Document the options**

In `doc/configuration.md`:
- Add rows in the DATA table for these options, with the defaults and the meanings of the comments in `config.py`:
  - `pred_frames` (4)
  - `opening_frames` (8)
  - `gen_stop` (`"length"`)
  - `max_gen_len` (320)
- MODEL table: add `context_frames` (8) and `generation_noise` (0.0).
- LOSS table: add `"autoregressive"` to the values of `loss_type`, and add rows for `vel_weight` (1.0), `end_weight` (0.1) and `end_pos_weight` (10.0).
- OPTIMIZATION table: add `rollout_steps` (1).
- Add `autoregressive` to the values of `--mode`.
- For `synth_poses`, say that the synthetic set is now read with every data option of `DATA` and the class indexes of the real training set.

- [ ] **Step 2: Document the models and the workflow**

- `doc/models.md`: add a section "Generation (`--mode autoregressive`)" with two rows:
  - `argen_tf`: the causal Transformer with the class token, `hidden_dim`, `nheads`, `mlp_dim`, `depth`.
  - `argen_mlp`: CsiMLPe blocks on the last `context_frames` frames.

  Both rows say that the models predict the drift of the next `pred_frames` frames and an end-of-sign logit, and link the spec.
- `doc/synthetic-data.md`: add a section "Generator A". Give the three commands, written out in full:
  1. Train with the end-to-end command of Task 6 Step 8, with the real config and without `-sd`.
  2. Generate with `-ckpt <checkpoint dir> -best -sd --sd_num N` and no `-t`.
  3. Pretrain with `-s_data` and `stgcn/hwgat-data-pretrain.yaml`.

  Also list what is written: the layout, `video_sizes.csv` 1×1, the real lengths in `instances.csv`.
- `doc/training.md`: add `autoregressive` to the list of modes, with what its validation logs (rollout error per step, end accuracy).
- `doc/development.md`: add a section "Tests" with `F:/ciencia/HandCraft-venv/Scripts/python -m pytest` (or `uv run pytest`), and the note that `tests/` imports the code from `src/` as `python src/main.py` does.

- [ ] **Step 3: Update the roadmap**

In `doc/roadmap.md`:
- Mark S3 **fixed**: the synthetic dataset gets the data options and class indexes of the real one.
- Mark S5's loss and metric part **fixed for generator A**: per keypoint.
- Replace "proposed, not confirmed" with "decided" for the output and keypoint rows of generator A.
- Mark tests 2.3 to 2.6 **next**: the code exists, the runs are next.

- [ ] **Step 4: Commit**

```bash
git add doc
git commit -m "Document generator A" -m "Options, models, the training, generation and pretraining commands, the tests, and the roadmap: S3 is fixed and the generator A tests are next." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
