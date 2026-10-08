# Reproducibility

## Seeds

Pass `--seed <n>` to every run. Without it `src/main.py` draws a random seed and the run cannot be repeated.

With a seed, a run is deterministic: two runs of the same command on the same machine give identical validation and test metrics, and `scripts/run/generate_dataset.sh` writes identical synthetic clips. This was checked for classification (Transformer, ST-GCN, conv1d, old and new data pipelines), both generators, dataset generation and synthetic pretraining.

The seed fixes:

- the 10% validation split taken from the training split,
- the model initialization, the batch order and the data augmentation,
- cuDNN, which is set to its deterministic mode.

Keep `--num_workers` the same between runs you want to compare: the augmentation random numbers are drawn inside the data loader workers. Results can still differ between GPU models, driver versions and PyTorch versions.

### Same seed, different numbers after the PyTorch 2.7 update

The commit "Support Windows and Blackwell GPUs" changed two things that change the random numbers a seed produces, so a seed gives different (but still repeatable) results before and after it:

- PyTorch went from 2.4.1 to 2.7.1, with other GPU kernels.
- The validation and test data loaders keep their worker processes between evaluations. The validation split comes from the training split and applies its augmentations, so before this change every validation re-seeded the workers from the main process and drew another random number from it.

Two runs with the same seed after the commit still give identical metrics (checked with a Transformer on INCLUDE on Windows).

### Class indexes

Until commit "Sort the class list so class indexes are the same in every run", the list of classes came from an unordered operation, so every process numbered the signs differently. Two consequences for anything produced before it:

- Runs with the same seed did not give the same result.
- A checkpoint loaded by a different process than the one that trained it saw another numbering. Training followed by testing in one run (`-t --test`) was not affected. Generating a synthetic dataset was: `scripts/run/generate_dataset.sh` loads the generators in a new process, so every sign was generated with the class embedding of a different sign. The synthetic datasets used for the published synthetic pretraining results have this problem.

Checkpoints saved before the fix cannot be evaluated or used for generation with the current code, because their class numbering is unknown.

## Validation and test accuracy

The metrics logged during training are validation metrics, on 10% of the training split (see [training.md](training.md#validation)). `scripts/run/train.sh` does not evaluate on the test set; `scripts/run/test.sh` does.

The INCLUDE and LSFB accuracies in the logs of the HandCraft paper experiments are validation accuracies: none of those runs was evaluated on the test split. For INCLUDE, the paper's Transformer-SL result (86.4%) is the validation accuracy of `ViT/original-pad-128x2`; the same config gets 81.0% on the test set of the paper's split and 84.8% on the official test set. Published results of other papers are test accuracies.

## Known issues in the data

The processing scripts in this repository fix problems that the datasets used for the existing results have. Regenerating a dataset therefore gives better, but not identical, data.

### INCLUDE

- **Missing videos.** An earlier `format.py` gave every video of an `Extra/` subfolder the same name, so they overwrote each other. The keypoints used for the existing results lack 25 such videos, the 8 videos of `Second (Number)` and 2 others: 4,257 of 4,292 clips, and 262 of 263 signs. The current script keeps every video of the Zenodo record, but the record has no `Second (Number)` folder: the official lists name 4,292 videos and 263 signs, and the Zenodo zips contain 4,284 videos and 262 signs. A regenerated dataset therefore has 4,284 clips and 262 signs, and its official split 3,468 train (train and val lists) and 816 test videos.
- **The random split cannot be regenerated.** The paper's 70/30 split was drawn from an unsorted file list, so it depended on the file system. The current script sorts the list and gives a different, reproducible split. Its split files are the only record of the original one.
- **The official split is reproducible.** `make_official_split.py` writes the same split files that were used for [results-include.md](results-include.md).

### DiSPLaY

- **Frames out of order.** The original script read the frames of a clip in the order returned by the file system, which is not the frame order. The keypoint sequences used for the existing DiSPLaY results are scrambled in time and then smoothed, which removes most of the motion: the stored wrist trajectory of a clip has less than half of the real range. The current script reads the frames by frame number. Most of the motion information in those sequences is lost, so the existing DiSPLaY results are not comparable with results on correctly extracted data.
- **Sign names.** The stored `instances.csv` names the signs `01`, `02`, ... and `sign_to_index.csv` names them `1`, `2`, ..., so the mapping between them never matched and the class of a sign was its number. The current script writes `1`, `2`, ... in both.
- **Splits.** The random split depended on the file system order and on Python's hash seed. The current script sorts the clips and the signs.

### LSFB

`setup_lsfb.py` removed entries from a list while iterating over it, which can skip entries. No empty or too long clip was found in a sample of 8,000 clips of the stored splits, so it had no visible effect. It is fixed.

## Reproducing the documented results

| Result | Can it be reproduced from a fresh clone? |
|---|---|
| Official INCLUDE split ([results-include.md](results-include.md)) | Yes, within seed noise. Exact numbers need the original keypoint files: regenerated keypoints include 35 more clips, and the class index fix changes the random numbers a run draws |
| Paper's INCLUDE split | Only with the original `metadata/splits/*.json` files |
| DiSPLaY | The pipeline runs, but on corrected data the results will differ |
| LSFB | Expected to: the dataset and its split are provided by `lsfb-dataset`. It was not downloaded again to check |

The original keypoints and split files are not in the repository. Publishing them (for example as a Zenodo record) would make the first two rows exactly reproducible.
