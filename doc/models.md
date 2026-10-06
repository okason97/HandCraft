# Models

`MODEL.backbone` selects a file of [`src/models/`](../src/models/). Every backbone defines a `Model(DATA, RUN, MODULES, MODEL)` class and receives a batch of shape `(batch, frames, keypoints × coordinates)`.

## Sign recognition (`--mode classification`)

| Backbone | Description | Notes |
|---|---|---|
| `ViT` | Transformer-SL: each frame is embedded with a linear layer, a class token is added, and a Transformer encoder classifies the sequence | The model of the HandCraft paper. Size is set by `depth`, `nheads`, `hidden_dim`, `mlp_dim` and `representation_size` |
| `mamba` | Mamba-SL: the same tokenization with a Mamba state space model, the class token at the end of the sequence | Needs `mamba-ssm` (see [installation.md](installation.md#mamba)) |
| `minimamba` | A pure PyTorch Mamba implementation | No extra dependency, slower |
| `stgcn` | ST-GCN: spatio-temporal graph convolutions over the skeleton | Adapted from [sl-hwgat](https://github.com/suvajit-patra/sl-hwgat). Requires the 29 keypoints and raw frames; see below |
| `conv1d` | 1D convolutional blocks over time, optionally followed by Transformer layers (`apply_attn`) | The ConvAtt family |
| `seccon` | A variant of `conv1d` | |

### ST-GCN requirements

`stgcn` has a fixed skeleton graph, so the config must use exactly these keypoints, and frames that are not padded or DCT-transformed:

```yaml
DATA:
  poses: [["pose",[0,2,5,11,12,13,14,15,16]],["left_hand",[0,4,5,8,9,12,13,16,17,20]],["right_hand",[0,4,5,8,9,12,13,16,17,20]]]
  num_keypoints: 29
  input_size: [32, 29, 2]
  coords: 2
  transform: "none"
  temporal_sampling: "uniform"
MODEL:
  backbone: "stgcn"
  representation_size: 256
  dropout: 0.05
```

The keypoints are the nose, eyes, shoulders, elbows and wrists, plus the wrist, fingertips and finger bases of each hand. `src/configs/INCLUDE/stgcn/` has working configs.

## Sign generation (`--mode cond_prediction` and `prediction`)

| Backbone | Mode | Description |
|---|---|---|
| `CsiMLPe` | `cond_prediction` | CMLPe, the generator of the HandCraft paper: an MLP-based motion predictor conditioned on the sign through adaptive layer normalization. Works on the DCT of the input frames. `noise_scale` adds noise for sample diversity |
| `siMLPe` | `prediction` | The unconditional motion predictor it is based on |

Both predict the last `DATA.target_len` frames of a clip from the frames before them, or the first ones from the frames after them with `--reverse`.

## Adding a backbone

Add `src/models/<name>.py` with a `Model` class with the signature above and a `forward(x, masks=None)` method (`forward(x, labels)` for conditional generation), and set `MODEL.backbone: "<name>"`. New config options have to be added to `src/configs/config.py` first.
