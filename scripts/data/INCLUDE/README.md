# INCLUDE data processing

Scripts to download [INCLUDE](https://zenodo.org/record/4010759) (Indian Sign Language, 263 signs, 4292 videos) and convert it into the layout HandCraft's data loader ([data_util.py](../../../src/data/data_util.py)) expects:

```
<data_dir>/
├── instances.csv               # id,sign,signer,start,end
├── metadata/
│   ├── sign_to_index.csv       # sign,class
│   └── splits/{train,test}.json
├── poses/{pose,right_hand,left_hand,face}/<id>.npy
└── raw/<Category>_<sign>#<video>.{MOV,MP4}
```

The scripts need the `extract` dependencies (`uv sync --extra extract`) and the MediaPipe models, which `scripts/data/download_mediapipe_models.sh <dir>` downloads; pass that directory as `-model_dir`. The steps for all the datasets are in [doc/datasets.md](../../../doc/datasets.md).

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
   - runs the keypoint extraction. `-workers N` (default `1`) extracts N videos in parallel and gives the same keypoints

3. **Re-extract keypoints only** (for example, after changing the MediaPipe models):

   ```bash
   python extract_keypoints.py -data_dir <data_dir> -model_dir <mediapipe_models> [-workers N]
   ```

   For each video in `raw/`, it detects the pose and then the hands and face in crops around the pose landmarks. Missing detections are filled in by linear interpolation, and a Savitzky-Golay filter (window 15, order 3) smooths each track. Each track is saved as a `(frames, keypoints, 3)` array: 33 pose, 21 per hand and 478 face keypoints.

Then train as with any other dataset (see [doc/training.md](../../../doc/training.md)).

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
- The split is not stratified, so the number of test samples per sign varies and a sign can be missing from one side. In the split used for the published results, 2 of the 262 signs have no test samples.
- There is no INCLUDE-50 subset and no validation set.
- The split used for the published results (2979 train / 1278 test) was made by an earlier version of `format.py`. That version didn't sort the file list or the classes, so running the current script again produces a **different** split and a different `sign_to_index.csv`. To keep results comparable with existing checkpoints, keep the existing `metadata/` files.
- That earlier version also mishandled videos in `Extra/` subfolders: they were written to `raw/` as `<sign>#Extra` without an extension, so they overwrote each other and were never processed. The official lists contain 27 such videos; only 16 files survived in `raw/`, and none of them are in the existing split. The current `format.py` keeps them.
- The 8 videos of `Days_and_Time/Second (Number)` are missing from the data used for the published results, which is why it has 262 signs instead of 263. That folder has no `<N>. ` prefix; `format.py` now names it `Days_and_Time_Second_(Number)` (the earlier version would have produced `Days_and_Time_econd_(Number)`).

### Difficulty: official split vs. ours

Measured on the data used for the published results (4257 videos with keypoints) and the official lists mapped to the same ids. In the official lists, 35 videos have no keypoints here: 25 `Extra/` videos, the 8 `Second (Number)` videos and 2 others. Only one of them is in the official test set, so it has 816 test videos here instead of 817.

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

## Results on the official split

The experiments on the official split (data pipeline changes, ablations, ST-GCN and the comparison with published results) are in [doc/results-include.md](../../../doc/results-include.md).
