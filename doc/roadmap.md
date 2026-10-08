# Plans and remaining tests

The plans for the classifier and the generator, and every test that has not been run or verified yet, as of 2026-10-08. Results that exist are in [results-include.md](results-include.md); known problems of earlier results are in [reproducibility.md](reproducibility.md).

Status of a test: **next** (implemented or being implemented, not run yet), **planned** (needs code or data first).

## Plan for the classifier

Goal: the strongest classifier we can train on each dataset, evaluated like the papers we compare with. Its data configuration is the format the new generator has to produce, so this comes first.

1. **Match HWGAT's data configuration on INCLUDE.** HWGAT is the best published model on INCLUDE (97.7%), and its ST-GCN baseline (96.7%) uses the same model code as ours (94.3%). The differences left are in the data pipeline and the training settings. Both were ported as config options and are measured separately (tests 1.1 to 1.5).
2. **Evaluate like they do.** Validation without augmentation, every class and every test clip.
3. **Close what remains.** In this order, each only if a gap is left: keypoints extracted like theirs (1.6), their training settings one at a time, the HWGAT model itself (1.7).
4. **Repeat on LSFB** (section 3). The pipeline changes that helped on INCLUDE were never tried there, and LSFB has no test-set result yet.
5. **DiSPLaY** when its data is available (section 4).

The configuration that wins on INCLUDE and LSFB is the baseline for every synthetic data test.

## Plan for the generator

Goal: synthetic clips of full length, in the data format of the baseline classifier, that improve it when used for pretraining.

The current generator (CsiMLPe) predicts 16 frames from 16 real frames in one pass; a forward and a reversed model are joined into clips of exactly 32 frames, shorter than every real INCLUDE clip (33 to 154 frames). It stays in the repository unchanged. Two new generators are added next to it, each with its own model and configs.

**Generator A: autoregressive.**

| Decision | State |
|---|---|
| Predicts the next x frames from the previous frames and the sign, as the drift from the last known frame; applied repeatedly to build a clip of any length | decided |
| x is a config value and is tuned by experiment | decided |
| A clip starts from the real opening frames of a training clip of that sign | decided |
| Clip length: both stopping at the length of the source clip and a learned end-of-sign signal are implemented and compared | decided |
| Robustness to its own errors: a setting for the number of rollout steps during training (1 = real context only), tuned together with x | decided |
| Output: saved as a regular dataset in raw coordinates, so any classifier config can pretrain on it with its own normalization and sampling | proposed, not confirmed |
| Keypoints: the 29 of HWGAT and of our ST-GCN, with depth available, as the first config. The keypoint set is a config value | proposed, not confirmed |
| Normalization inside the generator, and whether it uses the DCT | open: follows the classifier baseline |

**Generator B: reference-guided.** A second real clip of the same sign is given to the model as a guide, to pull the generated frames towards a realistic shape of the sign. It is built and tested whatever the result of generator A. Its design starts after generator A.

Both are measured by the test accuracy of the baseline classifier with and without pretraining on their clips (section 2).

## Known problems in the code

Found in a review of the whole pipeline on 2026-10-08. Each was confirmed by reading the code. Status: **fixed** (on the current branch), **next**, **later**.

### Classifier training

| # | Problem | Status |
|---|---|---|
| T1 | The best checkpoint is the one with the lowest validation loss; the "Best Top 1-acc" log line is the accuracy of that epoch. Kept as the criterion; the log line now says so | fixed |
| T2 | Validation always dropped its last incomplete batch, and the test set did unless `test_drop_last: False`. Every clip is now scored | fixed |
| T3 | Mixed precision could not be turned off: autocast was on in both branches, `-mpc` only switched the loss scaler | fixed |
| T4 | After synthetic pretraining, the real training had to beat the best validation loss of the pretrained model, or no best checkpoint was saved and the final test failed | fixed |
| T5 | A resumed run reloaded its loss and metric history from the wrong directory, and without `--seed` it built a different train/validation split. It now takes the seed of the checkpoint | fixed |
| T6 | On wandb, validation is logged one epoch before the training metrics of the same epoch, and synthetic pretraining reuses the steps of the real training | next |
| T7 | Evaluation builds autograd graphs (`torch.no_grad()` missing): memory and time only | next |

### Synthetic data

| # | Problem | Status |
|---|---|---|
| S1 | The reversed generator (`-tg -r_ckpt`) is loaded but never used: both "reversed" generations call the forward model on time-flipped input. Affects every twin-generated dataset, including those of the published results | next |
| S2 | Synthetic clips are re-centred on the right shoulder instead of the nose when loaded (their pose file has only the 6 selected body keypoints), which shifts them about 5 standard deviations in x | next |
| S3 | The synthetic pretraining dataset ignores the classifier's data options (normalization, sampling, coordinates, class mapping) | next |
| S4 | Dataset generation crashed on one GPU after the resume change (the seeded training dataset was passed to it) | fixed |
| S5 | The generator's embedding noise is also applied at validation and generation; classifier-free guidance, `class_emb_size`, `init` and the DiT weight initialization are never used; "MPJPE" is a norm over the whole frame, not per joint | next |
| S6 | Dataset generation failed for classes with fewer training clips than the batch size (most INCLUDE classes with batch size 16), and was slow on Windows: it started new data loader workers for every class (about 35 s per class). It now uses one loader for every class and repeats the clips of small classes; 260 classes take about 75 s | fixed |

### Problems behind earlier results

| # | Problem | Affected | Status |
|---|---|---|---|
| E1 | The conv1d attention layers attend across the clips of a batch instead of across time (`batch_first` missing) | every conv1d config with attention (the default), including the ConvAtt results | next |
| E2 | `apply_ema` trains the EMA copy directly and never averages | the LSFB `-ema` configs | next |
| E3 | `--dset_used 0.3` trains on 70% of the data | the training-set size experiments | next |
| E4 | With `-l`, `drop_frame` and `drop_keypoint` overwrite the cached clips permanently | LSFB configs with `drop_frame` | next |
| E5 | The `LSFB/seccon` configs train conv1d, and `seccon.py` cannot run | seccon results | next |
| E6 | The padding mask is applied to DCT coefficients as if they were frames, so short clips lose their high frequencies | `original-pad*`, `official-base/aug/shoulder`, conv1d `*-pad` | next |
| E7 | Mislabelled configs: `INCLUDE/mamba/original.yaml` is an LSFB copy (`min_samples: 20`), `DiSPLaY/siMLPe/*` are named LSFB, `DiSPLaY/ViT/original-pad-1024X2.yaml` starts with `SDATA:`, the ViT synth25 and synth50 configs pretrain for 75 epochs | those configs | next |
| E8 | `official-aug` mirrors with the left/right swap under the dataset normalization, which puts the mirrored hands far from the real data | `official-aug` | next |

### Latent problems and unused options

- Crashes in combinations no config uses yet: hand masking or mirroring with a non-hand part given as `"all"`, `mirror_p` with the 10-keypoint hands, `RandomAffine` with 2D input, multi-GPU training (several places), late dropout on backbones without it.
- Unused options: `class_emb_size`, `init`, `guidance_scale`, `apply_rflip`, `lecam_*`, `drop_path` for ViT and ST-GCN.
- Training batches are float32 and evaluation batches half precision for configs with shear or rotation; `temporal_sampling: "crop"` evaluates on a random crop.

## 1. Classifier on INCLUDE

All runs use the official split, seeds 42 and 43, and report test top-1. Our best so far is `stgcn/official-lr5`, 94.3%.

| # | Test | Config | Status | Needs |
|---|---|---|---|---|
| 1.1 | Old best config on this machine (PyTorch 2.7, regenerated keypoints, 4,284 clips) | `stgcn/official-lr5` | next | – |
| 1.2 | HWGAT data pipeline with our training settings | `stgcn/hwgat-data` | next | – |
| 1.3 | HWGAT data pipeline and HWGAT training settings (AdamW 5e-4, batch 4, repeating cosine, 500 epochs, label smoothing 0.01) | `stgcn/hwgat-full` | next | – |
| 1.4 | Hand masking refilled with a spline, as HWGAT does, instead of linear interpolation | variant of 1.2 or 1.3 | next | option being added |
| 1.5 | Effect of validating without augmentation on 1.1 | `stgcn/official-lr5` | next | fix being added |
| 1.6 | Keypoints extracted like HWGAT: MediaPipe Holistic, not smoothed, missing hands filled when loading | new keypoints | planned | re-extract the 4,284 videos (about 3 hours); check that Holistic exists in MediaPipe 0.10.35 |
| 1.7 | The HWGAT model itself | new backbone | planned | port the model from [sl-hwgat](https://github.com/suvajit-patra/sl-hwgat) |

Changes in the evaluation that these runs include:

- **Validation without augmentation.** The validation clips are taken from the training set and were augmented like it. HWGAT validates without augmentation. This changes which checkpoint is selected, so it is measured on its own (1.5).
- **Every class and every test clip.** The `hwgat-*` configs keep all 262 signs and score all 816 test clips. The older configs drop the signs with fewer than 5 training clips and the last incomplete batch (814 clips with the current keypoints).

Not covered by 1.1 to 1.7. Each was tested at most on the Transformer and one at a time:

| Test | Status |
|---|---|
| ST-GCN with a learning rate above 5e-3 | planned |
| ST-GCN with 64 frames at learning rate 5e-3 (with our uniform sampling) | planned |
| HWGAT's training settings on our data pipeline (the reverse of 1.2) | planned |
| Batch size, optimizer, schedule and number of epochs separately, if 1.3 differs from 1.2 | planned |
| The winning data pipeline on the Transformer (`ViT/official-kp29`) | planned |
| More than two seeds for the final configs: differences under about 1 point are not meaningful with two | planned |

## 2. Synthetic data

| # | Test | Status | Needs |
|---|---|---|---|
| 2.1 | Current generator (CsiMLPe, forward and reversed), regenerated with the fixed class numbering, then pretraining the paper's Transformer config: does synthetic pretraining still help? | planned | train both generators, generate, pretrain |
| 2.2 | The same on the official split and the baseline classifier | planned | the generated clips are 32 frames in the old input format, which the `official-*` and `stgcn` configs cannot read; needs generator A or a format conversion |
| 2.3 | Generator A: number of frames predicted per step (x) | planned | generator A |
| 2.4 | Generator A: training on its own rollouts against real context only | planned | generator A |
| 2.5 | Generator A: stopping at the source clip's length against the learned end-of-sign signal | planned | generator A |
| 2.6 | Generator A: error per rollout step (how fast the quality decays), for each x | planned | generator A |
| 2.7 | Generator B against generator A | planned | generator B |
| 2.8 | Baseline classifier pretrained on each synthetic dataset against no pretraining, on INCLUDE | planned | 2.3 to 2.7, section 1 |
| 2.9 | The same on LSFB | planned | 2.8, section 3 |

## 3. LSFB

The dataset is downloaded (120,739 clips; 65,620 in the training split and 50,849 in the test split before the class filter) and its splits are filtered. Nothing has been run on this machine. The paper compares with a Transformer at 54.4% (Fink et al., 2023).

| # | Test | Status | Needs |
|---|---|---|---|
| 3.1 | Test accuracy of the paper's configs. The paper's numbers (55.0%) are validation accuracy | planned | – |
| 3.2 | The INCLUDE pipeline changes of `official-nm-nodct`: uniform sampling with the speed window, shoulder normalization, label smoothing, no DCT | planned | new LSFB configs |
| 3.3 | ST-GCN with our best INCLUDE settings (`stgcn/official-lr5`) | planned | new LSFB config |
| 3.4 | ST-GCN with the HWGAT data pipeline, with our training settings and with theirs (tests 1.2 and 1.3 on LSFB) | planned | new LSFB configs; the video sizes for `pixel_coords` (see below) |
| 3.5 | Spline against linear hand masking (test 1.4 on LSFB) | planned | 3.4 |
| 3.6 | Validation without augmentation (test 1.5 on LSFB) | planned | – |
| 3.7 | Synthetic pretraining with a regenerated dataset | planned | section 2 |

Differences from INCLUDE to take into account:

- **Size.** About 12 times more clips than INCLUDE, so an epoch takes proportionally longer. HWGAT's 500 epochs at batch size 4 is probably not affordable on LSFB; the number of epochs has to be scaled.
- **Clips are short.** The paper's setup drops clips longer than 60 frames and signs with fewer than 20 clips (610 signs). With 64 input frames every clip is padded, never subsampled.
- **Keypoints come with the dataset**, already interpolated and smoothed, in [0, 1] coordinates, and without missing hands. `pixel_coords` needs the size of the videos, which is not in the downloaded metadata: it has to be taken from the LSFB documentation or from the videos.
- **Whether the signs with few clips are kept** changes the number of classes and the accuracy, and has to match the paper we compare with.

## 4. DiSPLaY

The existing results were obtained on keypoints scrambled in time, so all of them have to be redone.

| # | Test | Status | Needs |
|---|---|---|---|
| 4.1 | Download the full dataset and extract the keypoints with the corrected script | planned | manual download from IEEE DataPort (account needed); about 5 hours of extraction with one worker, less with `-workers` |
| 4.2 | Every classifier and synthetic pretraining result of the paper | planned | 4.1 |
| 4.3 | The baseline classifier configuration of sections 1 and 3 | planned | 4.1 |

## 5. Mamba-SL

| # | Test | Status | Needs |
|---|---|---|---|
| 5.1 | Mamba-SL on the improved pipeline, on any dataset | planned | Linux: `mamba-ssm` does not build on Windows and has to be rebuilt for PyTorch 2.7 |

## 6. Code and environment

| # | Test | Status | Needs |
|---|---|---|---|
| 6.1 | `scripts/dev/regression_check.sh` on this machine | planned | the `INCLUDE_official` directory now exists; the script has to be run from Git Bash |
| 6.2 | `uv sync` and a training run on Linux with the PyTorch 2.7 lock | planned | a Linux machine |
| 6.3 | Multi-GPU training (`-DDP`) with PyTorch 2.7 | planned | a machine with two GPUs |
| 6.4 | Loading checkpoints saved with PyTorch 2.4 | planned | an old checkpoint |
| 6.5 | `scripts/data/DiSPLaY/format.py` on the real dataset layout (tested only on a hand-made copy) | planned | 4.1 |
| 6.6 | Memory use of `-l` (whole dataset in memory) on LSFB | planned | – |
