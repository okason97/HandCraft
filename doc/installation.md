# Installation

## Requirements

- Linux with an NVIDIA GPU. The locked PyTorch build is 2.4.1 for CUDA 12.1; the wheels bundle the CUDA libraries, so only the NVIDIA driver is needed.
- [uv](https://docs.astral.sh/uv/getting-started/installation/), which installs Python and every dependency.
- `wget` and `unzip` for the dataset download scripts.

The experiments in this repository ran on two RTX 3070 (8 GB). A classification model uses under 1 GB of GPU memory.

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

Run the commands of this documentation inside the environment: prefix them with `uv run` (`uv run ./scripts/run/train.sh ...`) or activate it once with `source .venv/bin/activate`.

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
