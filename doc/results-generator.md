# Results of generator A

Generator A ([design](specs/2026-10-16-generator-a-design.md)) on the official INCLUDE split, seed 42 (commit `8be6005`). Every run trains for 200 epochs on the training part of the classifier's split. The checkpoint with the lowest validation rollout error is evaluated on the 816 test clips.

The **rollout error** is the mean per-keypoint distance between the generated and the real frames. Each clip is rolled out from its first 8 real frames to its real length. The unit is the width of the classifier's box, 6 shoulder widths, so an error of 0.05 is 0.3 shoulder widths. Every configuration rolls out the same frames, so the errors compare directly.

## Frames predicted per step (test 2.3)

Test rollout error (validation in brackets):

| Model | 1 frame per step | 4 frames per step | 8 frames per step |
|---|---|---|---|
| Transformer (`argen/tf*`) | 0.0604 (0.0580) | 0.0503 (0.0503) | **0.0449** (0.0442) |
| siMLPe, 8-frame window (`argen/mlp-w8*`) | 0.0880 (0.0885) | 0.0665 (0.0669) | 0.0605 (0.0606) |
| siMLPe, 4-frame window (`argen/mlp-w4*`) | 0.1070 (0.1057) | 0.0788 (0.0791) | 0.0707 (0.0710) |

- **The Transformer predicting 8 frames per step is the best configuration: 0.045**, 26% below the best siMLPe.
- **More frames per step is better for every model.** One frame per step is the worst: the model predicts its own input most often, so its errors pile up over the most steps.
- **The siMLPe needs its wider window.** 8 frames beat 4 at every step size.
- Test and validation errors agree, within 0.004.

## Error along the rollout (test 2.6)

Test rollout error of the frame that comes a given number of frames after the opening. The last two horizons are measured on fewer clips. Only 218 of the 816 test clips are longer than 72 frames, and 17 are longer than 104. Those clips are a different population, so the curve is not comparable past frame +32.

| Config | Mean | +1 | +4 | +8 | +16 | +32 | +64 | +96 |
|---|---|---|---|---|---|---|---|---|
| `tf-x1` | 0.0604 | 0.0026 | 0.0137 | 0.0310 | 0.0494 | 0.0833 | 0.0614 | 0.1046 |
| `tf` (x = 4) | 0.0503 | 0.0032 | 0.0149 | 0.0310 | 0.0453 | 0.0652 | 0.0574 | 0.0612 |
| **`tf-x8`** | **0.0449** | 0.0032 | 0.0147 | 0.0298 | **0.0413** | **0.0603** | **0.0494** | **0.0587** |
| `mlp-w8-x1` | 0.0880 | **0.0014** | **0.0101** | **0.0279** | 0.0563 | 0.1077 | 0.1568 | 0.1914 |
| `mlp-w8` (x = 4) | 0.0665 | 0.0020 | 0.0114 | 0.0284 | 0.0498 | 0.0791 | 0.1032 | 0.1947 |
| `mlp-w8-x8` | 0.0605 | 0.0026 | 0.0121 | 0.0306 | 0.0478 | 0.0769 | 0.0833 | 0.1257 |
| `mlp-w4-x1` | 0.1070 | 0.0133 | 0.0592 | 0.1116 | 0.1496 | 0.1213 | 0.0769 | 0.0792 |
| `mlp-w4` (x = 4) | 0.0788 | 0.0020 | 0.0118 | 0.0353 | 0.0668 | 0.0966 | 0.1062 | 0.1363 |
| `mlp-w4-x8` | 0.0707 | 0.0046 | 0.0190 | 0.0410 | 0.0635 | 0.0851 | 0.0940 | 0.1424 |

- **The siMLPe predicts the first frames best, the Transformer the long run.** Over the first 8 frames the siMLPe with an 8-frame window has the lowest error. After 16 frames the Transformer is better, and its error grows much more slowly: 0.060 at frame +32, against 0.077 to 0.108 for the siMLPe.
- The errors grow fastest over the first 16 frames, about the length of a movement of the sign.
- `mlp-w4-x1` is the exception: it drifts away within the first steps, and its error is already 0.11 at frame +8.
