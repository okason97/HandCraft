# Installation

## Requirements

- Linux or Windows with an NVIDIA GPU. The locked PyTorch build is 2.7.1 for CUDA 12.8, which also supports Blackwell GPUs (RTX 50xx); the wheels bundle the CUDA libraries, so only the NVIDIA driver is needed (version 570 or newer).
- [uv](https://docs.astral.sh/uv/getting-started/installation/), which installs Python and every dependency.
- `wget` and `unzip` for the dataset download scripts.

The experiments of the HandCraft paper ran on two RTX 3070 (8 GB) with PyTorch 2.4.1. A classification model uses under 1 GB of GPU memory.

## Install

```bash
git clone https://github.com/okason97/HandCraft.git
cd HandCraft
uv sync
```

`uv sync` creates `.venv` with the exact versions in `uv.lock`. Dependencies are declared in `pyproject.toml`:

| Group | Install with | Used for |
|---|---|---|
| core | `uv sync` | training, validation, testing and dataset generation |
| `extract` | `uv sync --extra extract` | keypoint extraction for INCLUDE and DiSPLaY (MediaPipe, OpenCV) |
| `lsfb` | `uv sync --extra lsfb` | downloading LSFB |
| dev | included by default | `ruff` and `ty` |

`uv sync --all-extras` installs everything.

Run the commands of this documentation inside the environment: prefix them with `uv run` (`uv run ./scripts/run/train.sh ...`) or activate it once with `source .venv/bin/activate` (`.venv\Scripts\activate` on Windows).

To keep the environment (about 6 GB) and uv's download cache outside the repository, for example on another disk, set `UV_PROJECT_ENVIRONMENT` and `UV_CACHE_DIR` before `uv sync` and `uv run`.

### Windows

Training, testing, dataset generation and keypoint extraction run on Windows. The run scripts are bash scripts: run them from Git Bash, or call `python src/main.py` directly. Two things differ from Linux:

- The `mamba` backbone is not available: `mamba-ssm` builds only on Linux.
- Distributed training (`-DDP`) needs `--backend gloo`; the default `nccl` backend does not exist on Windows. A single GPU does not use either.

### Mamba

The `mamba` backbone needs `mamba-ssm`, which is compiled against the local CUDA toolkit and is not in the lock file. Install it into the environment when needed:

```bash
uv pip install mamba-ssm
```

## Paths

The run scripts need two directories, given as environment variables or as options, and stop with a message if one is missing:

| Variable | Option | Contents |
|---|---|---|
| `HANDCRAFT_DATA` | `--data-root <dir>` | one directory per dataset (`LSFB`, `INCLUDE`, `DiSPLaY`, ...) |
| `HANDCRAFT_SAVE` | `--save-root <dir>` | checkpoints, statistics and generated datasets |

```bash
export HANDCRAFT_DATA=/path/to/datasets
export HANDCRAFT_SAVE=/path/to/outputs
```

or, for a single run, anywhere in the arguments (an option takes precedence over the variable):

```bash
./scripts/run/train.sh classification ViT original-pad-128x2 INCLUDE --data-root /path/to/datasets --save-root /path/to/outputs
```

The dataset and development scripts take their directories as arguments or from `HANDCRAFT_DATA` only, as shown in their sections.

## Weights & Biases

Training logs to [Weights & Biases](https://wandb.ai). Either log in once with `wandb login`, or disable the upload:

```bash
export WANDB_MODE=offline
```

Runs go to the project `handcraft-<backbone>-<dataset>` (the run scripts pass `--project handcraft-<model>-<dataset>`) of the entity in `WANDB_ENTITY`, or of the account's default entity when it is not set. Set it to log to a team:

```bash
export WANDB_ENTITY=<team>
```

What a run records:

| Key | Contents |
|---|---|
| `train_loss`, `train_top1`, `train_top10`, `epoch`, `lr` | Training metrics of every epoch, its number and the learning rate at its end |
| `valid_loss`, `valid_top1`, `valid_top10`, `valid_mpjpe` | Validation metrics of every epoch, on the 10% of the training split held out for validation |
| `test_loss`, `test_top1`, `test_top10`, `test_mpjpe` | The final evaluation of the best checkpoint on the test split (only with `--test`) |
| `pretrain_train_loss`, `pretrain_train_top1`, `pretrain_train_top10`, `pretrain_epoch`, `pretrain_valid_*` | The same for the epochs of the pretraining on synthetic data, and the validation at its end |
| config | Every option of the `DATA`, `MODEL`, `LOSS`, `OPTIMIZATION` and `RUN` sections, including the seed and the data directory |

The training and the validation of an epoch are logged at the same step. With synthetic pretraining, the pretraining epochs take the first steps and the training on the real data continues after them.

The run name is `<dataset>-<config>-seed<seed>-train-<timestamp>`.

Runs made before this was introduced logged the validation metrics as `test_*` and did not log the test set, so their `test_top1` is validation accuracy (see [reproducibility.md](reproducibility.md)).

## Docker

The `Dockerfile` installs the same locked environment. The image sets `HANDCRAFT_DATA=/data` and `HANDCRAFT_SAVE=/outputs`; mount the datasets and the output directory there:

```bash
docker build -t handcraft .
docker run --gpus all -it \
    -v /path/to/datasets:/data \
    -v /path/to/outputs:/outputs \
    handcraft bash
```

The Dockerfile was rewritten for uv and has not been built yet; the uv environment it installs is the one used for every run in this documentation.
