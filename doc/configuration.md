# Configuration

A run is configured by a YAML file in `src/configs/<dataset>/<model>/` and by command line flags. The YAML file has four sections, `DATA`, `MODEL`, `LOSS` and `OPTIMIZATION`. An option that is not in the file keeps the default of [`src/configs/config.py`](../src/configs/config.py); an option that does not exist there is an error.

In this code a "step" is one epoch: one pass over the training set followed, every `-every` steps, by a validation.

## DATA: Data, sampling and augmentation

| Option | Default | Description |
|---|---|---|
| `name` | `"LSFB"` | Dataset name, used in run names and logs |
| `input_size` | `[15, 32, 3]` | `[frames, keypoints, coordinates]` of the model input; must match `max_len`, the keypoints selected by `poses` and `coords` |
| `num_classes` | `10` | Overwritten at run time with the number of signs that pass `min_samples` |
| `min_samples` | `10` | Signs with fewer training clips are dropped |
| `max_len` | `15` | Number of frames given to the model |
| `target_len` | `8` | Generation: number of frames to predict (the last `target_len` of `max_len`) |
| `pad_frames` | `True` | Pad clips shorter than `max_len` (with `temporal_sampling: "crop"`) |
| `pad_mode` | `'wrap'` | `"pad"` marks padded frames so attention masks them; `"wrap"`, `"edge"` and `"mean"` are numpy padding modes |
| `oversample` | `False` | Repeat clips so every sign has as many training clips as the most frequent one |
| `poses` | `[]` | List of `[part, keypoints]`, with part in `pose`, `right_hand`, `left_hand`, `face` and keypoints a list of MediaPipe indexes or `"all"` |
| `synth_poses` | `[["pose","all"],["right_hand","all"],["left_hand","all"], ["face","all"]]` | Keypoints read from a synthetic dataset (`-s_data`) |
| `num_keypoints` | `91` | Total number of keypoints selected by `poses` |
| `flip_p` | `0.0` | Probability of negating x. It does not swap left and right keypoints; prefer `mirror_p` |
| `scale` | `0.0` | Random scaling of ±`scale` |
| `random_crop` | `False` | With `"crop"` sampling, pick random frames from a window instead of consecutive ones |
| `drop_frame` | `0.0` | Probability of zeroing a frame |
| `drop_keypoint` | `0.0` | Probability of zeroing a block of `block_size` keypoints |
| `block_size` | `5` | Size of the keypoint block dropped by `drop_keypoint` |
| `rot` | `0.0` | Random rotation of ±`rot` degrees |
| `shear_std` | `0.0` | Standard deviation of a random shear |
| `rot_std` | `0.0` | Standard deviation (radians) of a random rotation |
| `mirror_p` | `0.0` | Probability of a horizontal flip that also swaps left/right keypoints (use with norm "shoulder") |
| `temporal_sampling` | `"crop"` | `"crop"`: random window of consecutive frames, also at test time. `"uniform"`: frames evenly spaced over the whole clip, deterministic at test time. `"pad"`: keep the clip timing, subsample long clips and pad short ones with their edge frames |
| `speed` | `None` | For uniform sampling: minimum fraction of the clip covered by the random training window (None = whole clip) |
| `speed_range` | `None` | For pad sampling: [min, max] random speed factor applied to the clip length during training (None = off) |
| `hand_mask_p` | `0.0` | Fraction of training frames whose hand keypoints are replaced by interpolation |
| `norm` | `"dataset"` | `"dataset"`: subtract the first frame's nose and divide by dataset constants. `"shoulder"`: centre every frame on the shoulder midpoint and scale by the mean shoulder width. `"shoulder_clip"`: the same with one centre per clip. `"box"`: the per-clip normalization of HWGAT, a box 6 shoulder widths wide placed so the nose of the first frame is at (0.5, 1/3) |
| `coords` | `3` | Number of coordinates per keypoint used as input (2 = x,y; 3 = x,y,z), must match input_size[2] |
| `pixel_coords` | `False` | For norm "box": convert x and y to pixels before normalizing, so the box is square in the video. Needs `metadata/video_sizes.csv`, written by `scripts/data/video_sizes.py` |
| `aug_pivot` | `None` | `[mean, std]` of a random pivot for `shear_std` and `rot_std`, as in HWGAT: the shear and the rotation are applied around their own pivot and the shear only moves y. `None` applies them around the origin |
| `xflip_p` | `0.0` | Probability of reflecting x without swapping left and right keypoints, around 0.5 for norm "box" and around 0 otherwise |
| `missing_hand` | `None` | For norm "box": `"wrist"` places a hand that was not detected in any frame of the clip at its wrist |
| `test_drop_last` | `True` | Drop the last incomplete batch of the test set. `False` evaluates every test clip |
| `transform` | `None` | `"DCT"` applies a discrete cosine transform over the frames; anything else leaves the frames unchanged |
| `batch_size` | `128` | Set from `OPTIMIZATION.batch_size` |

## MODEL: Model

| Option | Default | Description |
|---|---|---|
| `backbone` | `"conv1d"` | File of `src/models/` to use: `ViT`, `mamba`, `minimamba`, `stgcn`, `conv1d`, `seccon`, `CsiMLPe`, `siMLPe` |
| `apply_sn` | `False` | Whether to apply spectral normalization |
| `act_fn` | `"ReLU"` | Type of activation function in ["ReLU", "Leaky_ReLU", "ELU", "GELU"] |
| `feature_norm` | `"batchnorm"` | Feature normalization method in ["batchnorm", "layernorm", "slayernorm", "tlayernorm", None] |
| `apply_attn` | `True` | Whether to apply transformer layers |
| `apply_ema` | `False` | Use exponential moving average |
| `ema_beta` | `0.9999` | Ema parameters |
| `ema_update_after_step` | `10` | EMA: steps before the first update |
| `ema_update_every` | `1` | EMA: update interval |
| `ema_power` | `0.9` | EMA: warm-up exponent |
| `nheads` | `4` | Transformer layer number of heads |
| `embed_size` | `64` | Embeding size |
| `bias` | `True` | Bias for linear layers (MAMBA) |
| `class_emb_size` | `32` | Class embeding size |
| `class_dropout_prob` | `0.0` | Class dropout probability for classifier-free guidance |
| `conv_dim` | `64` | Base channel for the classifier architecture |
| `conv_bias` | `True` | Convolutional bias for the classifier architecture (MAMBA) |
| `hidden_dim` | `64` | Hidden dimension for the classifier architecture |
| `mlp_dim` | `64` | Hidden size of the transformer feed-forward layers |
| `k_size` | `17` | Kernel size for the convolutions |
| `stride` | `1` | Stride size for the convolutions |
| `expand_ratio` | `2` | Expand ratio of channels for conv1d block |
| `depth` | `4` | Number of layers |
| `dropout` | `0.8` | Dropout ratio of the model layers |
| `drop_path` | `0.2` | Drop path ratio of the model convolutional blocks |
| `late_dropout` | `None` | Apply dropout when the training reaches the indicated epoch https://arxiv.org/abs/2303.01500 |
| `late_drop_path` | `None` | Drop path to switch to at `late_drop_path_step` |
| `late_dropout_step` | `None` | Epoch at which `late_dropout` starts |
| `late_drop_path_step` | `None` | Epoch at which `late_drop_path` starts |
| `init` | `"ortho"` | Weight initialization method in ["ortho", "N02", "glorot", "xavier"] |
| `temporal_fc_in` | `False` | Use temporal dimension for the input fully conected layer |
| `temporal_fc_out` | `False` | Use temporal dimension for the output fully conected layer |
| `use_spatial_fc` | `True` | SiMLPe: fully connected layers over the keypoints instead of over time |
| `apply_dw_conv` | `False` | Use depthwise convolution |
| `noise_scale` | `0.0` | Generator noise value |
| `representation_size` | `None` | Size of the layer before the classifier (ViT), or of the output features (stgcn) |
| `d_state` | `16` | MAMBA latent state dim |
| `dt_rank` | `'auto'` | Mamba: rank of Δ |

## LOSS: Loss

| Option | Default | Description |
|---|---|---|
| `loss_type` | `"CCE"` | `"CCE"` (cross entropy) for classification, `"motion"` for generation |
| `lecam_ema_start_iter` | `"N/A"` | Unused |
| `lecam_ema_decay` | `"N/A"` | Unused |
| `relative_motion` | `True` | Use relative motion for motion loss |
| `label_smoothing` | `0.0` | Label smoothing for the CCE loss |

## OPTIMIZATION: Optimization

| Option | Default | Description |
|---|---|---|
| `type_` | `"RAdam"` | Type of the optimizer for training in ["SGD", "RMSprop", "Adam", "RAdam", "AdamW"] |
| `lrscheduler` | `None` | `"OneCycle"`, `"Cosine"` or none |
| `cosine_epochs` | `20` | For `"Cosine"`: epochs from the initial learning rate down to 0. The rate then rises again and the cycle repeats |
| `max_lr` | `0.1` | Peak learning rate of OneCycle |
| `pct_start` | `0.3` | Fraction of training spent increasing the learning rate (OneCycle) |
| `batch_size` | `128` | Batch size |
| `lr` | `0.0002` | Learning rate. With OneCycle the schedule is defined by `max_lr` |
| `weight_decay` | `0.0` | Weight decay strength |
| `lookahead` | `True` | Wrap the optimizer in Lookahead (k=5, alpha=0.5) |
| `momentum` | `"N/A"` | Momentum value for SGD and RMSprop optimizers |
| `nesterov` | `False` | Nesterov value for SGD optimizer |
| `alpha` | `"N/A"` | Alpha value for RMSprop optimizer |
| `beta1` | `0.9` | Adam, RAdam and AdamW beta 1 |
| `beta2` | `0.999` | Adam, RAdam and AdamW beta 2 |
| `total_steps` | `10` | Number of training epochs |
| `synth_total_steps` | `1` | Number of pretraining epochs on the synthetic dataset (`-s_data`) |

## Command line flags

Flags of `src/main.py`. The run scripts set `--mode`, `-t`, `-data`, `-cfg`, `-save`, `--project`, `--num_workers 4`, `--prefetch_factor 2`, `-every 1`, `--print_every 1` and `-mpc`; anything given after the dataset name is appended.

| Flag | Default | Description |
|---|---|---|
| `--entity` | `None` | Weights & Biases entity. Default: `WANDB_ENTITY`, or the account's default entity |
| `--project` | `None` | Weights & Biases project. Default: `handcraft-<backbone>-<dataset>` |
| `-cfg, --cfg_file` | required | Config file |
| `-data, --data_dir` | required | Dataset directory |
| `-s_data, --synth_dir` | `None` | Synthetic dataset to pretrain on (classification) |
| `-save, --save_dir` | `"./"` | Output directory |
| `-ckpt, --ckpt_dir` | `None` | Checkpoint directory to load |
| `-r_ckpt, --r_ckpt_dir` | `None` | Checkpoint directory of the reversed generator (with `-tg`) |
| `-best, --load_best` | `off` | Load the best checkpoint of `-ckpt` instead of the last one |
| `--seed` | `-1` | Random seed; `-1` picks a random one |
| `-DDP, --distributed_data_parallel` | `off` | Distributed training over the visible GPUs |
| `--backend` | `"nccl"` | Backend for distributed training: `nccl` or `gloo` |
| `--mode` | `"classification"` | `classification`, `prediction` or `cond_prediction` |
| `--reverse` | `off` | Generation: predict backwards in time |
| `-tn, --total_nodes` | `1` | Distributed training: number of nodes |
| `-cn, --current_node` | `0` | Distributed training: rank of this node |
| `--num_workers` | `8` | Data loader workers |
| `--prefetch_factor` | `2` | Batches prefetched per worker |
| `-sync_bn, --synchronized_bn` | `off` | Synchronized batch norm |
| `-mpc, --mixed_precision` | `off` | Mixed precision training |
| `-t, --train` | `off` | Train the model |
| `--test` | `off` | Evaluate the best checkpoint on the test set |
| `-ss, --save_samples` | `off` | Save animations of generated clips |
| `-sd, --save_dataset` | `off` | Generate a synthetic dataset |
| `-tg, --twin_generator` | `off` | Also use the reversed generator when generating |
| `--ss_num` | `1` | Number of clips saved by `-ss` |
| `--sd_num` | `10` | Batches generated per sign by `-sd` |
| `-empty_cache, --empty_cache` | `off` | Empty the CUDA cache after every step |
| `-l, --load_data_in_memory` | `off` | Load the whole dataset in memory |
| `--print_every` | `5` | Log the training metrics every N epochs |
| `-every, --save_every` | `5` | Validate and save a checkpoint every N epochs |
| `--dset_used` | `1.0` | Fraction (below 1) or number (above 1) of training clips to use |
