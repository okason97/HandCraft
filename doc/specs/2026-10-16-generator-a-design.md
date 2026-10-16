# Generator A: autoregressive sign generation

Design approved on 2026-10-16. The test numbers refer to [roadmap.md](../roadmap.md#2-synthetic-data).

## Goal

Synthetic clips of full length that raise the test accuracy of the baseline classifier (`stgcn/hwgat-data`) when it is pretrained on them and then fine-tuned on the real data. Success is a higher mean test top-1 than the same classifier without pretraining, over seeds 42 and 43, on INCLUDE first and then on LSFB.

The current generator (CsiMLPe) stays unchanged. Generator A is new code next to it, with two interchangeable models.

## Decisions

| Decision | Choice |
|---|---|
| Keypoints | HWGAT's 29 (nose, eyes, shoulders, elbows, wrists, 10 per hand), in 2D: 58 values per frame. The keypoint set is the `DATA.poses` of the config |
| Coordinates | The classifier's box space: `pixel_coords` and `norm: "box"` |
| Prediction | The drift of the next x frames from the last known frame, and an end-of-sign logit. x is a config value |
| Models | A causal Transformer over the whole context (`argen_tf`) and a sliding siMLPe on the last w frames (`argen_mlp`, w = 4 or 8) |
| Start of a clip | The first k real frames of a training clip of the sign (`opening_frames`) |
| End of a clip | The length of the source clip, or the learned end-of-sign signal, by config |
| Robustness to its own errors | r rollout steps in training (`rollout_steps`; 1 = real context only) |
| Use by the classifier | Pretraining on the synthetic clips, optimizer reset, fine-tuning on real data (the existing `-s_data` path) |
| Development dataset | INCLUDE, then LSFB |
| Not in this version | Stochastic outputs, mixing synthetic and real clips in one training, augmentation in the generator, the DCT |

## Reuse

The design adds as little code as possible. Existing code and libraries used:

| Need | Reused |
|---|---|
| Training loop, checkpoints, exact resume, wandb, schedulers, early stopping | `loader.py` and `worker.py`, with new step functions for the new mode, as `cond_prediction` has |
| Clips in box space with pixel coordinates | `Dataset_.load_box_normalized` (`src/data/data_util.py`) |
| Train and validation split | the seeded stratified split of `loader.py`, the same as the classifier's |
| Choosing the source clips of each sign | `misc.generation_dataset` and `misc.ClassBatchSampler` |
| Writing a dataset | `worker.prepare_generation`, `update_metadata`, `save_metadata`; `save_keypoints` gets the full-layout writing below |
| siMLPe blocks with class conditioning | `TransMLP`, `MLPblock`, `LabelEmbedder` of `src/models/CsiMLPe.py` |
| Causal Transformer | `torch.nn.TransformerEncoder` with `nn.Transformer.generate_square_subsequent_mask` (the ViT `Encoder` has no causal mask) |
| Class weighting of the end loss | `F.binary_cross_entropy_with_logits(pos_weight=...)` |
| Tests | pytest, added to the `dev` dependency group |

## Components

### Data: `--mode autoregressive` in `Dataset_`

`__getitem__` returns the whole clip in box space, shape (T, 58), and its class. A collate function pads a batch to its longest clip and returns a mask of the real frames. No temporal resampling or augmentation.

### Models

Both have the usual `Model(DATA, RUN, MODULES, MODEL)` constructor and the same forward:

```
forward(frames, labels, last_only=False) -> drift (B, L, x, 58), end_logit (B, L)
```

The output at position t is the drift of frames t+1 to t+x from frame t, and the logit that the clip ends within those frames. With `last_only`, only the last position is computed (L = 1), which is what a rollout needs.

No padding mask is needed. Clips of a batch are padded at the end, and through the causal mask (Transformer) or the window (MLP), padding only affects positions after the end of a clip, whose targets are masked in the loss.

- **`src/models/argen_tf.py`**
  - Each frame goes through a linear layer from 58 to `hidden_dim`, plus a learned position embedding with `max_gen_len + 1` positions.
  - The sign is a class token at the start.
  - `torch.nn.TransformerEncoder` with a causal mask.
  - Two linear heads: one for the x × 58 drift, one for the end logit.
  - Defaults: hidden 256, 4 layers, 8 heads, MLP 1024.
  - With real context, one forward pass gives the predictions for every position of the clip.
- **`src/models/argen_mlp.py`**
  - At each position, the last w frames, padded with the first frame when there are fewer.
  - A linear layer over the features plus a learned embedding of the sign, then the CsiMLPe `TransMLP` over the w frames, whose blocks also get the sign through `LabelEmbedder`.
  - A temporal linear layer from w to x frames gives the drift; the pooled features give the end logit.
  - The w-frame windows of a clip are built with `unfold`, so every position is computed in one batch.

### Rollout, losses and writing: `src/generation/autoregressive.py`

Plain functions on tensors:

- **`rollout(model, opening, labels, steps)`** appends x predicted frames per step, with an optional cap. Training uses it with gradients through every step; validation and generation use it under `no_grad`.
- **`autoregressive_loss(...)`** sums three terms, each with a config weight:
  - the mean per-joint L2 error of the predicted positions
  - the same error on frame-to-frame velocities
  - the end-of-sign binary cross entropy, with positive weight `end_pos_weight`

  Targets past the end of the clip are masked. The end label of a position is 1 when the clip ends within its x frames.
- **`write_clip(...)`** puts the 29 keypoints back at their MediaPipe indices: 33 pose points and 21 per hand, 3 channels. Unused points and z are 0.

### Training step

- **With r = 1:** one forward pass on the real clip, with the loss at every position.
- **With r > 1:** a random cut per clip, at least `opening_frames` into it. From there, r rollout steps of x frames on the model's own output, with the loss on every step.

### Validation and testing

- Each clip of the validation (or test) split is rolled out from its first k frames to its real length. Clips of k frames or fewer have nothing to predict and are left out; their number is logged.
- Reported:
  - the mean per-joint error over all generated frames
  - the error per rollout step (test 2.6)
  - the accuracy of the end signal at the real end
- The checkpoint with the lowest overall rollout error is kept.

### Generation (`-sd`)

For each sign, `sd_num × batch_size` clips are generated (the existing `--sd_num` flag). Each one is built as follows:
- **Source clip:** a training clip of the sign, chosen by `ClassBatchSampler`. Clips with fewer than k + 1 frames are skipped. A sign with none left gets no clips, with a warning.
- **Opening:** its first k real frames.
- **Rollout:** until the source length (`gen_stop: "length"`) or until the end probability passes 0.5 (`gen_stop: "end_signal"`). Either way, it is capped at `max_gen_len` frames (320 by default, twice the longest INCLUDE clip).
- **Noise:** optional Gaussian noise on the class embedding, during generation only (`generation_noise`).

A NaN in a rollout stops the generation with an error.

### Written dataset

The layout of a real dataset, so any classifier config reads it:

- `poses/pose/<id>.npy` (T, 33, 3), `poses/left_hand/<id>.npy` and `poses/right_hand/<id>.npy` (T, 21, 3)
- `instances.csv` with the real length of each clip
- `metadata/splits/train.json`, plus `metadata/sign_to_index.csv` copied from the real dataset
- `metadata/video_sizes.csv` with 1×1 for every clip

The opening frames are real, so the first frame keeps its nose at (0.5, 1/3) and its shoulder width at 1/6. The box normalization on reload computes the same box, and the stored values come back unchanged (to 1e-6).

### Pretraining (fix S3)

The synthetic `Dataset_` built for `-s_data` in `loader.py` gets every data option of the real one: normalization, pixel coordinates, coordinates, temporal sampling, speed, hand masking, augmentations, missing hands. It also gets the class list and class indexes of the real training set: built on its own, a synthetic set missing a sign would number the classes differently. The keypoints it reads are `DATA.synth_poses`, set to the classifier's `poses` in the pretraining configs.

## Validation leak

The classifier selects its checkpoint on 10% of the training clips, drawn with the seed of the run. The generator uses the same split with the same seed: it trains only on the classifier's training part, and the openings come only from it.

## Configs

`src/configs/INCLUDE/argen/`:

| File | Model |
|---|---|
| `tf.yaml` | `argen_tf` |
| `mlp-w4.yaml` | `argen_mlp`, w = 4 |
| `mlp-w8.yaml` | `argen_mlp`, w = 8 |

The data section of all three is that of `stgcn/hwgat-data` without temporal sampling and augmentation.

New options:

| Section | Options |
|---|---|
| `DATA` | `pred_frames` (x), `opening_frames` (k), `gen_stop`, `max_gen_len` |
| `MODEL` | `context_frames` (w, MLP only), `generation_noise` |
| `LOSS` | `loss_type: "autoregressive"`, `vel_weight`, `end_weight`, `end_pos_weight` |
| `OPTIMIZATION` | `rollout_steps` (r) |

They are documented in `doc/configuration.md`.

## Testing

pytest, in `tests/`:

| Test | What it checks |
|---|---|
| Writer round trip | A written clip reloads through `Dataset_` with `norm: "box"` and `pixel_coords` to the stored values |
| Rollout | With r = 1 the model only sees real frames; with r = 2 its first prediction is the context of the second |
| Causal mask | The Transformer's outputs at position t do not change when frames after t change |
| Window | The MLP's output at t depends only on frames t−w+1 to t |
| Loss masking | Targets past the end of a clip do not change the loss |
| No leak | No opening comes from a validation clip |

Plus an end-to-end smoke run on INCLUDE: train a few epochs, generate, pretrain `stgcn/hwgat-data` for one epoch on the result.

## Experiments

On INCLUDE, in this order:

1. Both models at r = 1 with x ∈ {1, 4, 8}, and w ∈ {4, 8} for the MLP, compared by the rollout error per step (tests 2.3 and 2.6).
2. The best of step 1 at r ∈ {2, 4} (test 2.4).
3. Stopping at the source length against the end signal (test 2.5).
4. `stgcn/hwgat-data` pretrained on the clips of the best generator, against no pretraining, with seeds 42 and 43 (test 2.8).
5. The same on LSFB (test 2.9).

Generator B (reference-guided) gets its own design after these.
