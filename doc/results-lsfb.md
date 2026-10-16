# Results on LSFB

All numbers are top-1 accuracy (%) on the LSFB test split, of the checkpoint with the lowest validation loss. The setup is the one of the HandCraft paper: the 610 signs with at least 20 training clips, clips of at most 60 frames, and 10% of the training clips for validation (47,115 training, 5,235 validation and 40,059 test clips). Runs are from October 2026, with commits `ab43ad2` to `4464666`.

The configs are the INCLUDE configs of [results-include.md](results-include.md#current-baseline-october-2026) with the LSFB signs and fewer epochs: an LSFB epoch has 15 times more clips than an INCLUDE one. The HWGAT data pipeline needs the size of the videos for `pixel_coords`, which LSFB does not distribute; every clip is assumed to be 600×480, the size of the dataset's preview clip (`scripts/data/video_sizes.py -size 600 480`).

## Summary

| Model | Config | Backbone parameters | Epochs | Seed 42 | Seed 43 | Mean |
|---|---|---|---|---|---|---|
| **ST-GCN** | `stgcn/hwgat-data-e80` | 3.1M | 80 | **62.06** | – | – |
| ST-GCN | `stgcn/hwgat-data` | 3.1M | 40 | 59.95 | 60.46 | 60.2 |
| ST-GCN | `stgcn/official-lr5` (our older data pipeline) | 3.1M | 40 | 59.59 | 60.58 | 60.1 |
| HWGAT | `hwgat/hwgat-data-lr5e-4-e160` | 9.7M | 160 | 51.02 | – | – |
| Mamba-SL | `mamba/hwgat-data-lr5e-3-e80` | 0.07M | 80 | 50.18 | – | – |
| HWGAT | `hwgat/hwgat-data-lr5e-4-e80` | 9.7M | 80 | 46.64 | – | – |
| Transformer-SL | `ViT/hwgat-data-inorm-big-lr5e-3-e80` | 3.2M | 80 | 46.16 | – | – |
| Transformer-SL | `ViT/hwgat-data-inorm-lr5e-3-e80` | 0.1M | 80 | 45.45 | – | – |
| HWGAT | `hwgat/hwgat-data-lr5e-4` | 9.7M | 40 | 37.49 | 37.15 | 37.3 |

Published results on the same split: Transformer-SL 55.0 and Mamba-SL 54.5 with synthetic pretraining (HandCraft paper), Transformer 54.4 (Fink et al., 2023), ConvAtt 42.7. The HandCraft paper's numbers are validation accuracies (see [reproducibility.md](reproducibility.md)); on LSFB our validation accuracies are 5 to 9 points above the test accuracies of the same checkpoints, so they are not directly comparable with the table.

- **ST-GCN is the best model: 62.1%**, 11 points above every other model, with the data pipeline and training settings that are the baseline on INCLUDE.
- **The data pipeline matters less than on INCLUDE**: HWGAT's pipeline and ours give the same accuracy (60.2 and 60.1).
- **Every model was still improving at the end of its run.** The best checkpoints are at or near the last epoch, except HWGAT at 160 epochs (epoch 94).
- **The test set is harder than the validation split**: every checkpoint scores 5 to 9 points less on test than on validation (62.1 against 68.7 for ST-GCN). The validation clips come from the training split.

## ST-GCN

| Config | Epochs | Seed 42 | Seed 43 | Mean | Best epoch |
|---|---|---|---|---|---|
| `stgcn/hwgat-data` | 40 | 59.95 | 60.46 | 60.2 | 40, 39 |
| `stgcn/official-lr5` | 40 | 59.59 | 60.58 | 60.1 | 40, 39 |
| `stgcn/hwgat-data-e80` | 80 | 62.06 | – | – | 79 |

`official-lr5` does not oversample the small classes, unlike its INCLUDE version: with oversampling an LSFB epoch would have 958,000 clips. Twice the epochs adds 1.9 points.

## HWGAT

| Config | Peak learning rate | Epochs | Test top-1 | Best epoch |
|---|---|---|---|---|
| `hwgat/hwgat-data-lr5e-4` | 5e-4 | 40 | 37.3 (37.49, 37.15) | 35 |
| `hwgat/hwgat-data-lr1e-3` | 1e-3 | 40 | 37.83 | 37 |
| `hwgat/hwgat-data-lr2e-3` | 2e-3 | 40 | 38.12 | 35 |
| `hwgat/hwgat-data-lr5e-4-e80` | 5e-4 | 80 | 46.64 | 72 |
| `hwgat/hwgat-data-lr5e-4-e160` | 5e-4 | 160 | 51.02 | 94 |

- **The number of epochs is what limits HWGAT on LSFB**, not the learning rate: from 5e-4 to 2e-3 changes less than 1 point, 40 to 80 epochs adds 9.3, and 80 to 160 another 4.4.
- At 160 epochs it is still 11 points below ST-GCN at 80. On INCLUDE the two are within a point (96.4 and 97.2).

## Transformer-SL and Mamba-SL

The configs are the best ones on INCLUDE ([results-include.md](results-include.md#the-transformer-and-mamba-sl)): HWGAT's data pipeline with `input_norm`, and the size of the HandCraft paper's INCLUDE models. Two variants each, at the learning rate of ST-GCN (5e-3): the paper's size, and a backbone of the size of ST-GCN's (Transformer: hidden size 256, 4 layers of 8 heads, MLP 1024; Mamba-SL: hidden size 256, 7 layers).

| Config | Backbone parameters | Test top-1 | Best epoch |
|---|---|---|---|
| `mamba/hwgat-data-lr5e-3-e80` | 0.07M | **50.18** | 79 |
| `mamba/hwgat-data-big-lr5e-3-e80` | 3.1M | diverged at epoch 8 | – |
| `ViT/hwgat-data-inorm-big-lr5e-3-e80` | 3.2M | 46.16 | 77 |
| `ViT/hwgat-data-inorm-lr5e-3-e80` | 0.1M | 45.45 | 80 |

- **Mamba-SL is the better of the two, at 50.2%**, close to HWGAT trained twice as long, with a backbone 140 times smaller.
- **Size does not help the Transformer**: 30 times more backbone parameters give 0.7 points, within the difference between two seeds.
- **The large Mamba-SL diverged** when the one-cycle schedule reached its peak of 5e-3: its loss became NaN at epoch 8 and never recovered. Until then it was ahead of the small one (30.9 against 27.9 validation top-1 at epoch 7). Training has no gradient clipping. It uses `mamba_scan: "log"` ([configuration.md](configuration.md)), which makes it about 3.6 times faster.
- Two earlier runs were stopped to rerun at 5e-3: the Transformer at its INCLUDE learning rate (1e-2) was at 25.2 validation top-1 at epoch 19, against 32.8 at 5e-3 at the same epoch.

## Open points

- One seed for most of the 80 and 160 epoch configs.
- The test split has 40,059 clips; the HandCraft paper reports 39,831. The difference is in the filtering of the clips and has to be found before comparing with published numbers.
- The large Mamba-SL with gradient clipping or a lower learning rate, and the small one with more epochs.
- The video size of 600×480 is an assumption: the size of the other clips could not be checked.
