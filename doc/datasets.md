# Datasets

HandCraft trains on keypoints, not on video. Every dataset is converted to the same layout, the one used by [LSFB](https://lsfb.info.unamur.be/):

```
<HANDCRAFT_DATA>/<dataset>/
├── instances.csv               # id,sign,signer,start,end  (one row per clip)
├── metadata/
│   ├── sign_to_index.csv       # sign,class
│   └── splits/{train,test}.json   # lists of clip ids
└── poses/{pose,right_hand,left_hand,face}/<id>.npy
```

Each `.npy` file is a `(frames, keypoints, 3)` array of MediaPipe landmarks: 33 for `pose`, 21 per hand and 478 for `face`. The config option `DATA.poses` selects which keypoints a model uses.

There is no separate validation split on disk. Training holds out a stratified 10% of `train.json` for validation (see [training.md](training.md#validation)).

| Dataset | Language | Signs | Clips | Source | Keypoints |
|---|---|---|---|---|---|
| LSFB | French Belgian | 4,657 (610 after `min_samples`) | 120,739 | `lsfb-dataset` package | provided by the dataset |
| INCLUDE | Indian | 263 (262 on Zenodo) | 4,292 (4,284 on Zenodo) | [Zenodo 4010759](https://zenodo.org/record/4010759) | extracted here |
| DiSPLaY | Persian medical | 54 | 1,728 | [IEEE DataPort](https://doi.org/10.21227/5gsb-fb69) | extracted here |

The steps below assume `HANDCRAFT_DATA` is set (see [installation.md](installation.md#paths)).

## MediaPipe models

INCLUDE and DiSPLaY need three MediaPipe models to extract keypoints. Download them once:

```bash
uv sync --extra extract
./scripts/data/download_mediapipe_models.sh $HANDCRAFT_DATA/mediapipe
```

The script checks the files against the checksums of the models used for the published keypoints.

## LSFB

```bash
uv sync --extra lsfb
uv run python scripts/data/LSFB/setup_lsfb.py -data_dir $HANDCRAFT_DATA
```

The script downloads the isolated-sign poses (no video) into `$HANDCRAFT_DATA/LSFB` with the [`lsfb-dataset`](https://github.com/lsfb-team/lsfb-dataset) package, then removes from both splits the clips that are empty or longer than 60 frames.

## INCLUDE

```bash
# 1. download and extract the videos into INCLUDE/original (44 zip files, 57 GB)
./scripts/data/INCLUDE/download_data.sh $HANDCRAFT_DATA/INCLUDE
# 2. rename the videos, write the metadata and our random split, extract the keypoints
uv run python scripts/data/INCLUDE/format.py -data_dir $HANDCRAFT_DATA/INCLUDE -model_dir $HANDCRAFT_DATA/mediapipe
# 3. create the official split in a second directory that reuses the keypoints
uv run python scripts/data/INCLUDE/make_official_split.py -data_dir $HANDCRAFT_DATA/INCLUDE -out_dir $HANDCRAFT_DATA/INCLUDE_official
```

Step 2 produces the 70/30 random split used in the HandCraft paper. Step 3 produces the official split used by other papers. See [scripts/data/INCLUDE/README.md](../scripts/data/INCLUDE/README.md) for the script options and for how the two splits differ.

Keypoint extraction runs MediaPipe on every frame on the CPU and takes several hours for the whole dataset. `-workers N` extracts N videos in parallel, one process each, and gives the same keypoints as one worker (see [Keypoint extraction](#keypoint-extraction)). `extract_keypoints.py` repeats only that step and takes the same option.

## DiSPLaY

The dataset is distributed through IEEE DataPort and needs a (free) account, so the download cannot be scripted.

1. Download the eleven files `Signs(1-5).zip` to `Signs(51-55).zip` from <https://doi.org/10.21227/5gsb-fb69>.
2. Extract them into `$HANDCRAFT_DATA/DiSPLaY/original/`, so that the clips are at `original/Signs(1-5)/Sign_01_Performer_01_1/` and so on. Each clip has the folders `01 Times` and `02 Color Frames`, which are the only ones used.
3. Write the metadata and a random 70/30 split, and extract the keypoints:

```bash
uv run python scripts/data/DiSPLaY/format.py -data_dir $HANDCRAFT_DATA/DiSPLaY -model_dir $HANDCRAFT_DATA/mediapipe
```

`scripts/data/DiSPLaY/extract_keypoints.py` repeats only the extraction. Both scripts take `-workers N` to extract N clips in parallel.

## Keypoint extraction

[`scripts/data/mediapipe_keypoints.py`](../scripts/data/mediapipe_keypoints.py) is shared by INCLUDE and DiSPLaY. For every frame it detects the pose, then the hands and the face in crops around the pose landmarks. Frames without a detection are filled by linear interpolation, and each track is smoothed with a Savitzky-Golay filter (window 15, order 3).

MediaPipe processes the frames of a clip one by one, so a clip uses only part of the CPU. `-workers N` runs N clips at the same time, each in its own process with its own landmarkers; every clip is still processed in order by one process, so the keypoints are identical to a single-worker run. On a 12-core machine, 6 workers extracted about 3.5 times more clips per minute than 1 worker; each worker uses about 200 to 300 MB of memory. The MediaPipe GPU delegate is not an alternative: the Windows wheels are built without it, and under WSL2 it ran about 5 times slower than the CPU.

Re-extracting one INCLUDE video with the pinned MediaPipe models gave the keypoints used for the published results (the pose identical, the hands and face within 0.002). The complete dataset extracted again in October 2026 nevertheless gives clearly better results than the earlier one (see [results-include.md](results-include.md#current-baseline-october-2026)), so the two differ in more than that video showed. The earlier keypoint files are not available to compare.

## Differences from the data used for the published results

Regenerating the datasets with these scripts does not give exactly the data the existing results were trained on, because the scripts fix problems the original data has. See [reproducibility.md](reproducibility.md#known-issues-in-the-data) for the details and their effect.
