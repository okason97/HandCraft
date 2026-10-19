# Models

`MODEL.backbone` selects a file of [`src/models/`](../src/models/). Every backbone defines a `Model(DATA, RUN, MODULES, MODEL)` class and receives a batch of shape `(batch, frames, keypoints × coordinates)`.

## Sign recognition (`--mode classification`)

| Backbone | Description | Notes |
|---|---|---|
| `ViT` | Transformer-SL: each frame is embedded with a linear layer, a class token is added, and a Transformer encoder classifies the sequence | The model of the HandCraft paper. Size is set by `depth`, `nheads`, `hidden_dim`, `mlp_dim` and `representation_size` |
| `mamba` | Mamba-SL: the same tokenization with a Mamba state space model, the class token at the end of the sequence | Uses `mamba-ssm` when installed, otherwise its pure PyTorch copy `mamba_ref.py` (see [installation.md](installation.md#mamba)) |
| `minimamba` | A different pure PyTorch Mamba model ([mamba-minimal](https://github.com/johnma2006/mamba-minimal)), not Mamba-SL | No extra dependency, slower |
| `stgcn` | ST-GCN: spatio-temporal graph convolutions over the skeleton | Adapted from [sl-hwgat](https://github.com/suvajit-patra/sl-hwgat). Requires the 29 keypoints and raw frames; see below |
| `hwgat` | HWGAT: graph attention inside 4 overlapping windows of 16 keypoints (head, one arm, one hand), over blocks of 2 frames, in 3 stages | Adapted from [sl-hwgat](https://github.com/suvajit-patra/sl-hwgat). Same requirements as `stgcn`; see below |
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

### HWGAT

`hwgat` uses the same 29 keypoints and the same requirements as `stgcn`; the number of frames must be divisible by `hwgat_temporal_patch_size` squared (64 with the defaults). The model builds its 4 windows from the 29 keypoints itself. Its size is set by `hwgat_embed_dim`, `hwgat_temporal_patch_size`, `hwgat_depths`, `hwgat_heads`, `hwgat_ff_ratio`, `hwgat_attn_dropout` and `hwgat_pe`, which default to the values of HWGAT's code (9.9M parameters), and its dropout by `dropout`. With the same weights it gives the same outputs as the original model. `src/configs/INCLUDE/hwgat/` has a working config.

## Sign generation (`--mode cond_prediction` and `prediction`)

| Backbone | Mode | Description |
|---|---|---|
| `CsiMLPe` | `cond_prediction` | CMLPe, the generator of the HandCraft paper: an MLP-based motion predictor conditioned on the sign through adaptive layer normalization. Works on the DCT of the input frames. `noise_scale` adds noise for sample diversity |
| `siMLPe` | `prediction` | The unconditional motion predictor it is based on |

Both predict the last `DATA.target_len` frames of a clip from the frames before them, or the first ones from the frames after them with `--reverse`.

## Generator A (`--mode autoregressive`)

Both models predict, from the frames so far and the sign, the drift of the next `DATA.pred_frames` frames from the last one and the logit that the sign ends within them. They start by predicting no motion. The design is in [specs/2026-10-16-generator-a-design.md](specs/2026-10-16-generator-a-design.md).

| Backbone | Description |
|---|---|
| `argen_tf` | A causal Transformer (`torch.nn.TransformerEncoder`) over every frame so far, with the sign as a class token. Size: `hidden_dim`, `nheads`, `mlp_dim`, `depth` |
| `argen_mlp` | The CsiMLPe blocks on the last `context_frames` frames, with the sign added to the features of every frame and given to the blocks. Size: `hidden_dim`, `depth` |

## Adding a backbone

Add `src/models/<name>.py` with a `Model` class with the signature above and a `forward(x, masks=None)` method (`forward(x, labels)` for conditional generation), and set `MODEL.backbone: "<name>"`. New config options have to be added to `src/configs/config.py` first.
