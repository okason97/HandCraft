# Results on the official INCLUDE split

The current baseline is in the first section, [Current baseline](#current-baseline-october-2026): ST-GCN at 97.2%. The sections after it are earlier experiments on an older extraction of the dataset; their numbers are lower and only comparable with each other.

All numbers are top-1 accuracy on the official test set, mean of seeds 42 and 43 unless stated. How the official split differs from the random split of the HandCraft paper, and which one is harder, is analysed in [scripts/data/INCLUDE/README.md](../scripts/data/INCLUDE/README.md).

The runs of the earlier sections were made before the class list was sorted (see [reproducibility.md](reproducibility.md#class-indexes)). Their accuracy is not affected by that, but rerunning a config with the same seed will not give exactly the same number.

## Current baseline (October 2026)

The dataset was downloaded and its keypoints extracted again in October 2026, and the training code changed (see [reproducibility.md](reproducibility.md)). The runs of this section use that data and commit `0776df9`. **Every other section of this page was measured on the earlier extraction and is not comparable with this one**: the same config, `stgcn/official-lr5`, went from 94.3% there to 96.7% here.

| Config | Data pipeline | Training settings | Seed 42 | Seed 43 | Mean |
|---|---|---|---|---|---|
| **`stgcn/hwgat-data`** | HWGAT's | ours | **97.55** | **96.81** | **97.2** |
| `stgcn/hwgat-data-linear` | HWGAT's, hand masking refilled linearly instead of with a spline | ours | 97.30 | 97.43 | 97.4 |
| `stgcn/official-lr5` | ours | ours | 96.19 | 97.17 | 96.7 |
| `stgcn/official-lr5-augval` | ours, validation clips augmented | ours | 96.81 | 97.17 | 97.0 |
| `stgcn/hwgat-full` | HWGAT's | the defaults of HWGAT's code | 94.98 | 94.61 | 94.8 |

Test top-1 accuracy (%) on the official test set, of the checkpoint with the lowest validation loss. Published results on the same split: HWGAT 97.7, HWGAT's ST-GCN 96.7, SL-GCN (OpenHands) 93.5.

- **ST-GCN with HWGAT's data pipeline and our training settings reaches 97.2%**, above HWGAT's own ST-GCN result (96.7%) and 0.5 points below HWGAT (97.7%). `stgcn/hwgat-data` is the baseline classifier.
- **HWGAT's data pipeline is worth about half a point** over ours (97.2 against 96.7). With two seeds, whose results differ by up to 1 point, this is within the noise.
- **The default training settings of HWGAT's code are worse for this model**: 94.8% with AdamW at 5e-4, batch size 4, a repeating cosine schedule and 500 epochs, against 97.2% with RAdam and Lookahead at 5e-3, batch size 16, a one-cycle schedule and 400 epochs, on the same data. This is not a reproduction of their ST-GCN result (96.7%): their paper gives a learning rate of 1e-4 and up to 4,000 epochs with early stopping, their schedule is stepped once per epoch and ours every batch, they train in full precision, and their keypoints are extracted differently.
- **Spline or linear hand masking makes no measurable difference** (97.2 against 97.4).
- **Augmenting the validation clips makes no measurable difference** (97.0 against 96.7). With seed 43 both runs selected the same epoch, so they tested the same checkpoint.

What is and is not the same between these configs:

- `hwgat-*` configs keep all 262 signs and score the 816 test clips. `official-lr5` drops the signs with fewer than 5 training clips (`min_samples: 5`, 253 signs) and scores 814 clips. The other 2 clips could change its accuracy by at most 0.25 points.
- HWGAT's keypoints come from MediaPipe Holistic without smoothing, ours from separate pose and hand models with interpolation and smoothing. The validation split also differs (ours is 10% of the official train and validation lists).
- The Zenodo copy of INCLUDE lacks the 8 videos of one sign, one of them in the test set.

Each run is on Weights & Biases in the project `handcraft-stgcn-INCLUDE`, with its config.

## Setting up the official split

`format.py` writes our random split. To train on the official split, `make_official_split.py` creates a second data directory that reuses the keypoints and only replaces the split files:

```bash
python scripts/data/INCLUDE/make_official_split.py -data_dir <data_dir> -out_dir <official_dir>
```

It downloads the official lists from [AI4Bharat/INCLUDE](https://github.com/AI4Bharat/INCLUDE) (pinned to a commit), links `poses`, `instances.csv` and `sign_to_index.csv` from `<data_dir>`, and writes `metadata/splits/train.json` (official train + val lists) and `test.json`, keeping only the videos that have keypoints. With the keypoints used for these results it has 3441 train+val and 816 test clips. Use `-dataset include50` for the INCLUDE-50 split.

Train and evaluate with `scripts/run/test.sh`. It takes the same arguments as `scripts/run/train.sh` and also passes `--test`, so the best checkpoint (lowest validation loss) is evaluated on the test set. The results below use seeds 42 and 43:

```bash
./scripts/run/test.sh classification ViT official-nm-nodct INCLUDE -data $HANDCRAFT_DATA/INCLUDE_official/ --seed 42
./scripts/run/test.sh classification stgcn official-lr5 INCLUDE -data $HANDCRAFT_DATA/INCLUDE_official/ --seed 42
```

Each run writes its own numbered log, `logs/INCLUDE/<model>-<config>/testN.out`.

## Evaluation protocol

This describes the earlier experiments. Since October 2026 every validation and test clip is scored, and the logs name validation and test separately (see [training.md](training.md#validation)).

- The pipeline carves a stratified 10% validation set out of `train.json`; it doesn't use the official validation list.
- Signs with fewer than 5 training videos are dropped (`min_samples: 5`), and the test loader drops the last incomplete batch. Together this leaves out 16 of the 816 official test videos, so 800 are evaluated.
- In the logs of these runs, the "Best Top 1-acc" printed during training is **validation** accuracy. Only the `Test Top 1-acc` printed after "End of training" is test accuracy. The INCLUDE numbers in the HandCraft paper (86.4% and 87.1% for Transformer-SL) match validation accuracies in `logs/INCLUDE`, not test accuracies.

## Baseline: official split vs. ours

`original-pad-128x2` (the paper's Transformer-SL config without synthetic data), seed 42:

| | Official split | Our split |
|---|---|---|
| Test top-1 | 84.8% (800 videos) | 81.0% (1232 videos) |
| Test top-10 | 97.4% | 97.0% |
| Validation top-1 | 84.8% | 86.1% |

## Closing the gap with published results

The papers that beat us on INCLUDE use the same kind of Transformer but a different data pipeline: their plain Transformer baselines reach 90.4% ([OpenHands](https://arxiv.org/abs/2110.05877)) and 94.9% ([HWGAT](https://arxiv.org/abs/2407.14224)) on the official split. These configs test their main differences one at a time. Each changes only the listed options from `original-pad-128x2`; the options are documented in [config.py](../src/configs/config.py).

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

## Ablations of `official-all`

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

## Remaining differences with HWGAT's Transformer

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

## ST-GCN

These are the earlier experiments on the older extraction; the current numbers are in [Current baseline](#current-baseline-october-2026).

[stgcn.py](../src/models/stgcn.py) is the ST-GCN implementation from the [sl-hwgat](https://github.com/suvajit-patra/sl-hwgat) repository (MIT), wrapped to fit our model interface (`backbone: "stgcn"`). It uses HWGAT's 29-keypoint skeleton: nose, eyes, shoulders, elbows and wrists, plus 10 keypoints per hand (wrist, fingertips and finger bases). The configs are in `src/configs/INCLUDE/stgcn/` and use the same data pipeline as `official-nm-nodct`.

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
