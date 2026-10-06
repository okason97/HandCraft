# INCLUDE data processing

Scripts to download [INCLUDE](https://zenodo.org/record/4010759) (Indian Sign Language, 263 signs, 4292 videos) and convert it into the layout HandCraft's data loader ([data_util.py](../data_util.py)) expects:

```
<data_dir>/
├── instances.csv               # id,sign,signer,start,end
├── metadata/
│   ├── sign_to_index.csv       # sign,class
│   └── splits/{train,test}.json
├── poses/{pose,right_hand,left_hand,face}/<id>.npy
└── raw/<Category>_<sign>#<video>.{MOV,MP4}
```

The scripts need `polars`, `scikit-learn`, `mediapipe`, `opencv-python`, `scipy` and `wget` (only for `download.py`). Keypoint extraction also needs the MediaPipe models `pose_landmarker_heavy.task`, `hand_landmarker.task` and `face_landmarker.task` in one directory (`-model_dir`, default `/disco1/models/mediapipe`).

## Usage

1. **Download** the videos into `<data_dir>/original/<Category>/<N>. <sign>/...`:

   ```bash
   ./download_data.sh <data_dir>
   ```

   This downloads every zip with `wget` and extracts it into `<data_dir>/original`. As an alternative, `download.py` downloads the same zips from the Zenodo listing in `files.json` (it doesn't extract them):

   ```bash
   python download.py -out_dir <data_dir>
   ```

2. **Format** the dataset and extract keypoints:

   ```bash
   python format.py -data_dir <data_dir> -model_dir <mediapipe_models>
   ```

   Steps:
   - moves every video from `original/` to `raw/` and renames it to `<Category>_<sign>#<video>` (for example, `original/Places/19. House/Extra/MVI_3439.MOV` becomes `raw/Places_House#MVI_3439.MOV`)
   - writes `instances.csv` (signer is a placeholder, and `start`/`end` are always `0`/`1`) and `metadata/sign_to_index.csv` (classes in alphabetical order)
   - writes a random train/test split to `metadata/splits/` (see below). Change it with `-test_size` (default `0.3`) and `-seed` (default `42`)
   - runs the keypoint extraction

3. **Re-extract keypoints only** (for example, after changing the MediaPipe models):

   ```bash
   python extract_keypoints.py -data_dir <data_dir> -model_dir <mediapipe_models>
   ```

   For each video in `raw/`, it detects the pose and then the hands and face in crops around the pose landmarks. Missing detections are filled in by linear interpolation, and a Savitzky-Golay filter (window 15, order 3) smooths each track. Each track is saved as a `(frames, keypoints, 3)` array: 33 pose, 21 per hand and 478 face keypoints.

Then train with `-data <data_dir>` as with any other dataset.

## Train/test split: official vs. the one used here

`format.py` does **not** use the official INCLUDE split. The Zenodo README mentions a `Train_Test_Split` folder, but the Zenodo record doesn't contain it. The official split is in the authors' repository, [AI4Bharat/INCLUDE](https://github.com/AI4Bharat/INCLUDE), under `train_test_paths/`, as lists of paths relative to the dataset root (for example, `Seasons/63. Winter/MVI_4997.MOV`):

| | Official INCLUDE | Official INCLUDE-50 | This repo (`format.py`) |
|---|---|---|---|
| Source | `include_{train,val,test}.txt` | `include50_{train,val,test}.txt` | `sklearn.train_test_split` |
| Classes | 263 | 50 | all signs found in `original/` |
| Train / val / test | 3127 / 348 / 817 | 689 / 77 / 192 | 70% / – / 30% |
| Validation set | yes | yes | no |
| Deterministic | yes | yes | yes, for a given `-seed` and file list |

What this means in practice:
- **Results are not comparable to published INCLUDE / INCLUDE-50 numbers.** Those numbers use the official test set, and our test set is a different, larger random sample.
- The split is not stratified, so the number of test samples per sign varies and a sign can be missing from one side. In the split currently at `/disco2/datasets/INCLUDE`, 2 of the 262 signs have no test samples.
- There is no INCLUDE-50 subset and no validation set.
- The split currently at `/disco2/datasets/INCLUDE` (2979 train / 1278 test) was made by an earlier version of `format.py`. That version didn't sort the file list or the classes, so running the current script again produces a **different** split and a different `sign_to_index.csv`. To keep results comparable with existing checkpoints, keep the existing `metadata/` files.
- That earlier version also mishandled videos in `Extra/` subfolders: they were written to `raw/` as `<sign>#Extra` without an extension, so they overwrote each other and were never processed. The official lists contain 27 such videos; only 16 files survived in `raw/`, and none of them are in the existing split. The current `format.py` keeps them.
- The 8 videos of `Days_and_Time/Second (Number)` are missing from `/disco2/datasets/INCLUDE`, which is why it has 262 signs instead of 263. That folder has no `<N>. ` prefix; `format.py` now names it `Days_and_Time_Second_(Number)` (the earlier version would have produced `Days_and_Time_econd_(Number)`).

### Difficulty: official split vs. ours

Measured on the split currently at `/disco2/datasets/INCLUDE` (4257 videos with keypoints) and the official lists mapped to the same ids. In the official lists, 35 videos have no keypoints here: 25 `Extra/` videos, the 8 `Second (Number)` videos and 2 others. Only one of them is in the official test set, so it has 816 test videos here instead of 817.

**How the samples are distributed**

| | Official (train+val / test) | Ours (train / test) |
|---|---|---|
| Test fraction | 19% | 30% |
| Test videos per sign | min 1, median 3, max 7 | min 0, median 5, max 11 |
| Test fraction per sign | 7–33% (sd 0.04), roughly stratified | 0–64% (sd 0.12), not stratified |
| Train videos per sign | min 3, median 13 | min 2, median 11 |
| Signs without test videos | 0 | 2 |

**Recording sessions.** Neither split holds out signers or recording sessions. The camera numbers the files (`MVI_xxxx`) in recording order. For 86% of official test videos (76% of ours), the next or previous file number is a training video of the same sign, so the model has seen the same signer, session and background. Both splits measure recognition of signers already seen in training, not new ones.

**Empirical difficulty.** Same simple classifier on both splits. Features: upper-body and hand keypoints, centred on the shoulders, scaled by shoulder width and resampled to 16 frames, then standardized and reduced to 256 dimensions with PCA. Classifiers: logistic regression and a 1-nearest-neighbour baseline. The "random" rows average 5 seeds (± is the standard deviation):

| Split | Train / test | Logistic regression | 1-NN |
|---|---|---|---|
| Official | 3441 / 816 | **0.697** | **0.578** |
| Random stratified, same size as official | 3441 / 816 | 0.700 ± 0.008 | 0.564 ± 0.006 |
| Random stratified, 30% | 2979 / 1278 | 0.669 ± 0.016 | 0.539 ± 0.012 |
| Random non-stratified, 30% (how `format.py` splits) | 2979 / 1278 | 0.652 ± 0.005 | 0.522 ± 0.004 |
| Ours | 2979 / 1278 | **0.643** | **0.530** |

Conclusions:
- **Our split is harder than the official one:** accuracy is about 5 points lower. Roughly 3 points come from the smaller training set (30% instead of 19% held out) and 1.5–2 from not stratifying, which leaves some signs with very few training videos. What remains is within the seed-to-seed variation.
- **The official split is no harder than a random stratified split of the same size.** It is not a signer- or session-independent benchmark, so it isn't harder in that way.
- Expect higher numbers on the official split than on ours for the same model. With a simple baseline the difference is about 5 points; with stronger models it may differ in size, but it should go the same way.
- **Don't evaluate a model trained on our split on the official test set.** 577 of the 816 official test videos are in our training set, and 1039 of our test videos are in the official train or val sets. To report official numbers, retrain on the official train/val lists.

## Transformer-SL results on the official split

### Setting up the official split

`format.py` writes our random split. To train on the official split, `make_official_split.py` creates a second data directory that reuses the keypoints and only replaces the split files:

```bash
python src/data/INCLUDE/make_official_split.py -data_dir <data_dir> -out_dir <official_dir>
```

It downloads the official lists from [AI4Bharat/INCLUDE](https://github.com/AI4Bharat/INCLUDE) (pinned to a commit), links `poses`, `instances.csv` and `sign_to_index.csv` from `<data_dir>`, and writes `metadata/splits/train.json` (official train + val lists) and `test.json`, keeping only the videos that have keypoints. On `/disco2` this is `/disco2/datasets/INCLUDE_official` (3441 train+val / 816 test). Use `-dataset include50` for the INCLUDE-50 split.

Train and evaluate with `test_script.sh`. It takes the same arguments as `script.sh` and also passes `--test`, so the best checkpoint (lowest validation loss) is evaluated on the test set. The results below use seeds 42 and 43:

```bash
./test_script.sh classification ViT official-nm-nodct INCLUDE -data /disco1/datasets/INCLUDE_official/ --seed 42
./test_script.sh classification stgcn official-lr5 INCLUDE -data /disco1/datasets/INCLUDE_official/ --seed 42
```

The log is written to `logs/INCLUDE/<model>-<config>/test0.out`, so run one seed at a time or copy the log between runs.

### Evaluation protocol

- The pipeline carves a stratified 10% validation set out of `train.json`; it doesn't use the official validation list.
- Signs with fewer than 5 training videos are dropped (`min_samples: 5`), and the test loader drops the last incomplete batch. Together this leaves out 16 of the 816 official test videos, so 800 are evaluated.
- The "Best Top 1-acc" printed during training is **validation** accuracy. Only the `Test Top 1-acc` printed after "End of training" is test accuracy. The INCLUDE numbers in the HandCraft paper (86.4% and 87.1% for Transformer-SL) match validation accuracies in `logs/INCLUDE`, not test accuracies.

### Baseline: official split vs. ours

`original-pad-128x2` (the paper's Transformer-SL config without synthetic data), seed 42:

| | Official split | Our split |
|---|---|---|
| Test top-1 | 84.8% (800 videos) | 81.0% (1232 videos) |
| Test top-10 | 97.4% | 97.0% |
| Validation top-1 | 84.8% | 86.1% |

### Closing the gap with published results

The papers that beat us on INCLUDE use the same kind of Transformer but a different data pipeline: their plain Transformer baselines reach 90.4% ([OpenHands](https://arxiv.org/abs/2110.05877)) and 94.9% ([HWGAT](https://arxiv.org/abs/2407.14224)) on the official split. These configs test their main differences one at a time. Each changes only the listed options from `original-pad-128x2`; the options are documented in [config.py](../../configs/config.py).

| Config | Change | Seed 42 | Seed 43 | Mean |
|---|---|---|---|---|
| `official-base` | none | 82.0 | 84.3 | 83.1 |
| `official-shoulder` | `norm: shoulder`: centre each frame on the shoulder midpoint, scale by shoulder width | 84.0 | 84.6 | 84.3 |
| `official-uniform` | `temporal_sampling: uniform`: 32 frames evenly spaced over the whole clip instead of a random 32-frame crop | 87.3 | 84.8 | 86.0 |
| `official-aug` | `shear_std: 0.1`, `rot_std: 0.1`, `mirror_p: 0.5` (flip that swaps left/right keypoints) | 79.1 | 79.5 | 79.3 |
| `official-all` | all of the above, plus `speed: 0.8` (random 80–100% sub-window during training) and `label_smoothing: 0.1` | **90.8** | **91.6** | **91.2** |

Test top-1 accuracy (%) on the official test set.

- **The combination matters more than any single change:** together they add 8 points.
- **Uniform sampling is the largest single gain.** The random crop sees about half of a typical clip (median 63 frames), and with `crop` the window is random at test time too, so crop-mode scores vary by about ±1.5 points between runs. Uniform sampling is deterministic at test time.
- **Augmentation only helps with shoulder normalization.** On its own it costs about 4 points in both seeds, likely because flip, shear and rotation pivot around the first frame's nose rather than the body. The ablations below show the flip itself hurts even with shoulder normalization.
- The existing `flip_p` option only negates x and doesn't swap left/right keypoints; use `mirror_p` instead.

Compared with published test accuracy on the official split:

| Model | Top-1 |
|---|---|
| HWGAT | 97.7 |
| HWGAT's Transformer baseline | 94.9 |
| **ST-GCN, `stgcn/official-lr5`** | **94.3** |
| SL-GCN (OpenHands) | 93.5 |
| Transformer-SL, `official-nm-nodct` | 93.2 |
| Transformer-SL, `official-all-nomirror` | 92.3 |
| Transformer-SL, `official-all` | 91.2 |
| OpenHands Transformer | 90.4 |
| LSTM (Khartheesvar et al., 2024) | 87.4 |
| Transformer-SL, `official-base` | 83.1 |

OpenHands selects checkpoints on the test set, so its numbers are somewhat optimistic. HWGAT uses a 10% validation split like ours. The remaining differences with HWGAT's Transformer are tested in the last section.

### Ablations of `official-all`

Each config changes one thing from `official-all`:

| Config | Change | Seed 42 | Seed 43 | Mean | vs `all` |
|---|---|---|---|---|---|
| `official-all` | – | 90.8 | 91.6 | 91.2 | – |
| `official-all-nospeed` | no `speed` (whole clip, deterministic sampling in training too) | 85.6 | 85.8 | 85.7 | −5.5 |
| `official-all-2d` | `coords: 2` (x, y only; `input_size[2]: 2`) | 89.4 | 89.0 | 89.2 | −2.0 |
| `official-all-nols` | no `label_smoothing` | 90.3 | 89.1 | 89.7 | −1.5 |
| `official-all-64` | `max_len: 64` (`input_size[0]: 64`) | 92.0 | 89.3 | 90.6 | −0.6 |
| `official-all-big` | `depth: 4`, `hidden_dim: 128`, `mlp_dim: 256` | 91.6 | 91.1 | 91.4 | +0.2 |
| **`official-all-nomirror`** | no `mirror_p` | **92.6** | **91.9** | **92.3** | **+1.1** |

Test top-1 accuracy (%) on the official test set.

- **The speed sub-window is the most important component** (−5.5 without it). Without it, the combined config is no better than uniform sampling alone: the model sees the same 32 frames of each clip every epoch, so the random windows are the main source of temporal variety.
- **The left/right flip hurts** (+1.1 without it, both seeds). A mirrored sign is performed with the other dominant hand, which likely isn't a faithful sample of the same sign in INCLUDE. Use `official-all-nomirror`.
- **Label smoothing and depth both help** (−1.5 and −2.0 without them).
- **64 frames and a moderately bigger model give no reliable gain.** The spread between seeds (up to 2.7 points for `all-64`) is larger than the differences. `all-big` is still far smaller than HWGAT's Transformer; that size is tested below as `nm-wide`.
- With two seeds per config, differences under about 1 point aren't meaningful.

### Remaining differences with HWGAT's Transformer

HWGAT's Transformer baseline (94.9%) differs from ours in more ways than the changes above. Each config here changes one thing from `official-all-nomirror`:

| Config | Change | Seed 42 | Seed 43 | Mean | vs `all-nomirror` |
|---|---|---|---|---|---|
| `official-all-nomirror` | – | 92.6 | 91.9 | 92.3 | – |
| **`official-nm-nodct`** | `transform: "none"`: raw frames with the learned positional embedding, no DCT | **93.1** | **93.3** | **93.2** | **+0.9** |
| `official-nm-handmask` | `hand_mask_p: 0.2`: hands in a random 20% of the frames replaced by linear interpolation | 91.8 | 92.1 | 91.9 | −0.3 |
| `official-nm-clipnorm` | `norm: "shoulder_clip"`: centre and scale once per clip instead of every frame | 92.5 | 91.4 | 91.9 | −0.3 |
| `official-nm-wide` | HWGAT's size (`depth: 3`, `hidden_dim: 512`, `nheads: 8`, `mlp_dim: 2048`, `dropout: 0.1`) with `AdamW` at 5e-4 | 91.8 | 91.8 | 91.8 | −0.5 |
| `official-nm-pad64` | `temporal_sampling: "pad"`, `speed_range: [0.5, 1.5]`, 64 frames: random speed, then short clips padded with their edge frames instead of stretched | 92.6 | 90.4 | 91.5 | −0.8 |
| `official-nm-lowlr` | learning rate 1e-3 instead of 1e-2 | 91.5 | 90.3 | 90.9 | −1.4 |

Test top-1 accuracy (%) on the official test set.

- **Removing the DCT helps** (+0.9, both seeds agree). `official-nm-nodct` is the best Transformer config at 93.2%.
- **None of the other HWGAT differences help here.** The wide model, per-clip normalization, hand masking and HWGAT's temporal pipeline are all within noise of the baseline or slightly below it.
- **The high learning rate is right for the small model:** 1e-3 costs 1.4 points.
- These were tested one at a time on our pipeline. They might still help in combination, or with HWGAT's longer schedule, which weren't tested.

### ST-GCN

[stgcn.py](../../models/stgcn.py) is the ST-GCN implementation from the [sl-hwgat](https://github.com/suvajit-patra/sl-hwgat) repository (MIT), wrapped to fit our model interface (`backbone: "stgcn"`). It uses HWGAT's 29-keypoint skeleton: nose, eyes, shoulders, elbows and wrists, plus 10 keypoints per hand (wrist, fingertips and finger bases). The configs are in `src/configs/INCLUDE/stgcn/` and use the same data pipeline as `official-nm-nodct`.

| Config | Change | Seed 42 | Seed 43 | Mean |
|---|---|---|---|---|
| **`stgcn/official-lr5`** | ST-GCN, 2D keypoints, learning rate 5e-3 | **94.0** | **94.6** | **94.3** |
| `stgcn/official-64` | learning rate 1e-3, 64 frames | 92.1 | 91.9 | 92.0 |
| `stgcn/official` | learning rate 1e-3 | 91.3 | 91.3 | 91.3 |
| `stgcn/official-3d` | learning rate 1e-3, with the depth coordinate | 87.5 | 90.3 | 88.9 |
| `ViT/official-kp29` | Transformer (`official-nm-nodct`) on the 29 keypoints | 91.1 | 92.8 | 91.9 |
| `ViT/official-nm-nodct` | Transformer on our 60 keypoints, for reference | 93.1 | 93.3 | 93.2 |

Test top-1 accuracy (%) on the official test set.

- **ST-GCN with a learning rate of 5e-3 is the best model so far: 94.3%**, 1.1 points above the best Transformer and above the published SL-GCN result (93.5%).
- **The learning rate matters a lot:** 5e-3 is 3 points better than 1e-3, the value OpenHands uses. Higher values and the combination with 64 frames weren't tested.
- **The gain comes from the graph model, not the keypoints.** On the same 29 keypoints the Transformer scores 91.9%, 1.3 points below its result on our 60 keypoints.
- **Depth hurts ST-GCN** (−2.4 at the same learning rate), the opposite of the Transformer.
- 64 frames gives a small gain at learning rate 1e-3 (+0.7).
- HWGAT reports 96.7% for ST-GCN on INCLUDE, so a gap of about 2.4 points to their training setup remains.
