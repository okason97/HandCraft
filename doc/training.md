# Training, validation and testing

Everything runs through `src/main.py`. Four shell scripts wrap it with the usual arguments:

| Script | What it does | Log file |
|---|---|---|
| `script.sh` | trains, validating after every epoch | `runN.out` |
| `test_script.sh` | trains, then evaluates the best checkpoint on the test set | `testN.out` |
| `eval_script.sh` | evaluates a saved checkpoint on the test set, without training | `evalN.out` |
| `gdataset_script.sh` | generates a synthetic dataset with trained generators | `generateN.out` |

All four take the same arguments:

```bash
./script.sh <mode> <model> <config> <dataset> [extra arguments for src/main.py]
```

| Argument | Values |
|---|---|
| `mode` | `classification` (sign recognition), `cond_prediction` (sign generation conditioned on the sign), `prediction` (unconditional motion prediction) |
| `model` | a folder of `src/configs/<dataset>/`: `ViT`, `mamba`, `stgcn`, `conv1d`, `seccon`, `CsiMLPe`, `siMLPe` |
| `config` | a file of `src/configs/<dataset>/<model>/`, without `.yaml` |
| `dataset` | `LSFB`, `INCLUDE` or `DiSPLaY` |

The scripts read the data from `$HANDCRAFT_DATA/<dataset>/` and write to `$HANDCRAFT_SAVE/<dataset>/<model>-<config>/` (see [installation.md](installation.md#paths)). Logs go to `./logs/<dataset>/<model>-<config>/`, numbered so that a new run never overwrites an earlier one.

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
./script.sh classification ViT original-pad-128x2 INCLUDE --seed 42
```

Train and also report the accuracy on the test set:

```bash
./test_script.sh classification ViT original-pad-128x2 INCLUDE --seed 42
```

Evaluate a model trained earlier, without training:

```bash
./eval_script.sh classification ViT original-pad-128x2 INCLUDE \
    -ckpt $HANDCRAFT_SAVE/INCLUDE/ViT-original-pad-128x2/checkpoints/<run name>/
```

For the official INCLUDE split, add `-data $HANDCRAFT_DATA/INCLUDE_official/` and use one of the `official-*` configs (see [results-include.md](results-include.md)):

```bash
./test_script.sh classification stgcn official-lr5 INCLUDE -data $HANDCRAFT_DATA/INCLUDE_official/ --seed 42
```

## Generation

The generator is a conditional motion predictor: given the first frames of a clip and the sign, it predicts the following frames. `DATA.max_len` is the clip length and `DATA.target_len` the number of predicted frames (32 and 16 in the provided configs).

```bash
# train, validating after every epoch
./script.sh cond_prediction CsiMLPe depth_big_noise_0.1 INCLUDE --seed 42
# train and also evaluate on the test set
./test_script.sh cond_prediction CsiMLPe depth_big_noise_0.1 INCLUDE --seed 42
```

During training, animations of generated clips are saved to `figures/` every validation.

Producing a full synthetic dataset needs a second generator trained backwards in time; the whole pipeline is in [synthetic-data.md](synthetic-data.md).

## Validation

Training holds out a stratified 10% of the training split for validation. The split depends on `--seed`. After every `-every` epochs (1 in the scripts) the model is evaluated on it and the log shows two lines:

```
Test Top 1-acc 84.8214    Test Top 10-acc 97.9167    Test Loss 0.6074
Best Top 1-acc 84.8214    Best Top 10-acc 97.9167    Best Loss (Step: 330): 0.6074
```

**During training both lines are validation metrics**, despite the word "Test". The checkpoint with the lowest validation loss is kept as the best one. Generation models report the loss and the MPJPE (mean per joint position error) instead of accuracies.

## Testing

With `--test` (`test_script.sh` and `eval_script.sh`), the best checkpoint is loaded after training and evaluated on `test.json`. Its result is the `Test ...` line **after** `End of training!` in the log; `test_script.sh` prints it when it finishes.

Two details of the evaluation:

- Signs with fewer than `DATA.min_samples` clips in the training split are removed from training, validation and test.
- The test loader drops the last incomplete batch, so up to `batch_size - 1` test clips are not evaluated.

## Outputs

`$HANDCRAFT_SAVE/<dataset>/<model>-<config>/` contains, for every run (named `<dataset>-<config>-train-<timestamp>`):

| Folder | Contents |
|---|---|
| `checkpoints/<run name>/` | `model=<model>-best-weights-step=N.pth` (lowest validation loss) and `model=<model>-current-weights-step=N.pth` (last epoch) |
| `logs/` | the Python log of the run |
| `statistics/<run name>/` | training and validation metrics as `.npy` |
| `figures/<run name>/` | animations of generated clips (generation modes) |
| `generated_datasets/<run name>/` | synthetic datasets written by `gdataset_script.sh` |
| `wandb/` | the Weights & Biases run |

## Multiple GPUs and other options

`src/main.py` supports distributed training with `-DDP`, synchronized batch norm with `-sync_bn`, and loading the dataset in memory with `-l`. The results in this documentation were obtained on a single GPU per run, selected with `CUDA_VISIBLE_DEVICES`.
