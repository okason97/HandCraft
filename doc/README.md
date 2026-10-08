# HandCraft documentation

| Document | Contents |
|---|---|
| [installation.md](installation.md) | Requirements, installing with uv, Docker, Weights & Biases |
| [datasets.md](datasets.md) | Downloading and processing LSFB, INCLUDE and DiSPLaY; the data layout |
| [training.md](training.md) | The run scripts; training, validation and testing of classification and generation models; outputs |
| [synthetic-data.md](synthetic-data.md) | Training the generators, generating a synthetic dataset and pretraining a classifier on it |
| [configuration.md](configuration.md) | Every config file option and command line flag |
| [models.md](models.md) | The available backbones |
| [results-include.md](results-include.md) | Experiments on the official INCLUDE split and comparison with published results |
| [reproducibility.md](reproducibility.md) | Seeds, what is deterministic, and known issues in the data and in earlier results |
| [development.md](development.md) | Linting, type checking, and how changes are checked for regressions |
| [roadmap.md](roadmap.md) | Plans for the classifier and the generator, and every test that has not been run yet |

The shortest path from a fresh clone to a trained and tested model is in the [README](../README.md#getting-started).

## Repository layout

```
src/                 library code: main.py, loader.py, worker.py, models/, data/data_util.py, configs/
scripts/
├── run/             train.sh, test.sh, eval.sh, generate_dataset.sh (train, test, evaluate, generate)
│                    experiments.sh (the commands of past experiments)
│                    common.sh (reads the data and output directories; sourced by the others)
├── data/            download_mediapipe_models.sh, mediapipe_keypoints.py (shared keypoint extraction)
│   ├── LSFB/        setup_lsfb.py
│   ├── INCLUDE/     download, format, keypoint extraction and official split scripts
│   └── DiSPLaY/     format and keypoint extraction scripts
└── dev/             regression_check.sh
doc/                 this documentation
```

All scripts are run from the root of the repository.
