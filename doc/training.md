# Training, validation and testing

Everything runs through `src/main.py`. Four shell scripts wrap it with the usual arguments:

| Script | What it does | Log file |
|---|---|---|
| `scripts/run/train.sh` | trains, validating after every epoch | `runN.out` |
| `scripts/run/test.sh` | trains, then evaluates the best checkpoint on the test set | `testN.out` |
| `scripts/run/eval.sh` | evaluates a saved checkpoint on the test set, without training | `evalN.out` |
| `scripts/run/generate_dataset.sh` | generates a synthetic dataset with trained generators | `generateN.out` |

They are in `scripts/run/` and are run from the root of the repository. All four take the same arguments:

```bash
./scripts/run/train.sh <mode> <model> <config> <dataset> [extra arguments for src/main.py]
```

| Argument | Values |
|---|---|
| `mode` | `classification` (sign recognition), `cond_prediction` (sign generation conditioned on the sign), `prediction` (unconditional motion prediction) |
| `model` | a folder of `src/configs/<dataset>/`: `ViT`, `mamba`, `stgcn`, `conv1d`, `seccon`, `CsiMLPe`, `siMLPe` |
| `config` | a file of `src/configs/<dataset>/<model>/`, without `.yaml` |
| `dataset` | `LSFB`, `INCLUDE` or `DiSPLaY` |

The scripts read the data from `$HANDCRAFT_DATA/<dataset>/` and write to `$HANDCRAFT_SAVE/<dataset>/<model>-<config>/`. Both directories can also be given with `--data-root <dir>` and `--save-root <dir>` (see [installation.md](installation.md#paths)). Logs go to `./logs/<dataset>/<model>-<config>/`, numbered so that a new run never overwrites an earlier one.

Extra arguments are passed to `src/main.py`. The most useful ones:

| Argument | Effect |
|---|---|
| `--seed 42` | fixes the random seed. Without it a random seed is used and the run is not reproducible |
| `-data <dir>` | reads the dataset from another directory, for example the official INCLUDE split |
| `-s_data <dir>` | pretrains on a synthetic dataset first (see [synthetic-data.md](synthetic-data.md)) |
| `-ckpt <dir>` | loads a checkpoint directory |
| `--reverse` | trains a generator that predicts backwards in time |

[configuration.md](configuration.md) lists every flag.

## Classification

Train a sign recognition model and validate it after every epoch:

```bash
./scripts/run/train.sh classification ViT original-pad-128x2 INCLUDE --seed 42
```

Train and also report the accuracy on the test set:

```bash
./scripts/run/test.sh classification ViT original-pad-128x2 INCLUDE --seed 42
```

Evaluate a model trained earlier, without training:

```bash
./scripts/run/eval.sh classification ViT original-pad-128x2 INCLUDE \
    -ckpt $HANDCRAFT_SAVE/INCLUDE/ViT-original-pad-128x2/checkpoints/<run name>/
```

For the official INCLUDE split, add `-data $HANDCRAFT_DATA/INCLUDE_official/` and use one of the `official-*` configs (see [results-include.md](results-include.md)):

```bash
./scripts/run/test.sh classification stgcn official-lr5 INCLUDE -data $HANDCRAFT_DATA/INCLUDE_official/ --seed 42
```

## Generation

The generator is a conditional motion predictor: given the first frames of a clip and the sign, it predicts the following frames. `DATA.max_len` is the clip length and `DATA.target_len` the number of predicted frames (32 and 16 in the provided configs).

```bash
# train, validating after every epoch
./scripts/run/train.sh cond_prediction CsiMLPe depth_big_noise_0.1 INCLUDE --seed 42
# train and also evaluate on the test set
./scripts/run/test.sh cond_prediction CsiMLPe depth_big_noise_0.1 INCLUDE --seed 42
```

During training, animations of generated clips are saved to `figures/` every validation.

Producing a full synthetic dataset needs a second generator trained backwards in time; the whole pipeline is in [synthetic-data.md](synthetic-data.md).

## Validation

Training holds out a stratified 10% of the training split for validation. The split depends on `--seed`. The log has one line per epoch for the training and, every `-every` epochs (1 in the scripts) and after the last epoch, one for the validation:

```
Epoch  331/400 | 1:02:11 | lr 0.000412 | train loss 0.4391 | top1 97.12 | top10 99.94
Epoch  331/400 | valid loss 0.6074 | top1 84.82 | top10 97.92 | best: epoch 331 (loss 0.6074, top1 84.82) *
```

- Epochs are counted from 1 in every line. The first line has the time since the start of the run and the learning rate at the end of the epoch.
- `best` is the epoch with the lowest validation loss so far, with its loss and its accuracy (not the highest accuracy). A `*` marks the epochs that become the best one; their checkpoint is kept and tested at the end.
- Every validation clip is evaluated, without augmentation.
- Generation models show `mpjpe` instead of the accuracies.
- With synthetic pretraining, the pretraining epochs come first as `Pretrain epoch   12/75 | ...`, followed by one validation line.

Logs written before this format show the validation as `Test Top 1-acc ...` and `Best Top 1-acc ...` lines, with epochs counted from 0.

## Testing

With `--test` (`scripts/run/test.sh` and `scripts/run/eval.sh`), the best checkpoint is loaded after training and evaluated on `test.json`. Its result is the last line of the log, which `scripts/run/test.sh` prints when it finishes:

```
Test of the checkpoint of epoch 331 on 816 clips | test loss 0.5712 | top1 94.24 | top10 99.39
```

Three details of the evaluation:

- Signs with fewer than `DATA.min_samples` clips in the training split are removed from training, validation and test.
- Every test clip is evaluated. `DATA.test_drop_last: True` drops the last incomplete batch, as runs before October 2026 did.
- With `DATA.temporal_sampling: "crop"` (the default, used by the configs of the paper), clips longer than `max_len` are cropped at a random position at test time too, so the test accuracy changes slightly between evaluations of the same checkpoint. `"uniform"` and `"pad"` are deterministic: `scripts/run/eval.sh` then reproduces the test result of the training run exactly.

## Outputs

`$HANDCRAFT_SAVE/<dataset>/<model>-<config>/` contains, for every run (named `<dataset>-<config>-train-<timestamp>`):

| Folder | Contents |
|---|---|
| `checkpoints/<run name>/` | `model=<model>-best-weights-step=N.pth` (lowest validation loss) and `model=<model>-current-weights-step=N.pth` (last epoch) |
| `logs/` | the Python log of the run |
| `statistics/<run name>/` | training and validation metrics as `.npy` |
| `figures/<run name>/` | animations of generated clips (generation modes) |
| `generated_datasets/<run name>/` | synthetic datasets written by `scripts/run/generate_dataset.sh` |
| `wandb/` | the Weights & Biases run |

## Multiple GPUs and other options

`src/main.py` supports distributed training with `-DDP`, synchronized batch norm with `-sync_bn`, and loading the dataset in memory with `-l`. The results in this documentation were obtained on a single GPU per run, selected with `CUDA_VISIBLE_DEVICES`.
