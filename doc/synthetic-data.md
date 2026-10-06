# Synthetic data pretraining

HandCraft generates synthetic sign clips with a conditional motion predictor (CMLPe, `CsiMLPe` in the code) and pretrains the recognition model on them. The pipeline has four steps. The commands use INCLUDE; LSFB and DiSPLaY have the same configs.

## 1. Train the forward generator

Given the first 16 frames of a clip and its sign, it predicts the next 16.

```bash
./script.sh cond_prediction CsiMLPe depth_big_noise_0.1 INCLUDE --seed 42
```

## 2. Train the reversed generator

The same model trained backwards in time: given the last 16 frames, it predicts the first 16.

```bash
./script.sh cond_prediction CsiMLPe depth_big_noise_0.1-reversed INCLUDE --reverse --seed 42
```

Each run writes its checkpoints to `$HANDCRAFT_SAVE/INCLUDE/CsiMLPe-<config>/checkpoints/<run name>/`.

## 3. Generate the dataset

```bash
./gdataset_script.sh cond_prediction CsiMLPe depth_big_noise_0.1-reversed INCLUDE \
    -ckpt $HANDCRAFT_SAVE/INCLUDE/CsiMLPe-depth_big_noise_0.1/checkpoints/<forward run name>/ \
    -tg -r_ckpt $HANDCRAFT_SAVE/INCLUDE/CsiMLPe-depth_big_noise_0.1-reversed/checkpoints/<reversed run name>/ \
    --sd_num 100 --seed 42
```

`-ckpt` is the forward generator, `-tg -r_ckpt` adds the reversed one. For every sign, `--sd_num` batches of real training clips are taken. From each clip:

- the forward generator predicts the second half from the real first half,
- the reversed generator predicts the first half from the real second half,

and the two generated halves are joined into one fully synthetic clip. The result is written in the standard dataset layout to

```
$HANDCRAFT_SAVE/INCLUDE/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/<generated dataset>/
```

with `sd_num × batch_size` clips per sign (1,600 with the values above). The directory is named `<reversed config>-train-<timestamp of the forward run>`; it is the only entry of `generated_datasets/` after one generation.

## 4. Pretrain and train the classifier

Pass the generated dataset with `-s_data` and use a config that sets `OPTIMIZATION.synth_total_steps`:

```bash
./test_script.sh classification ViT original-pad-synth75-475 INCLUDE \
    -s_data $HANDCRAFT_SAVE/INCLUDE/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/<generated dataset> --seed 42
```

The model is trained for `synth_total_steps` epochs on the synthetic clips, the optimizer and the learning rate schedule are reset, and it is then trained for `total_steps` epochs on the real clips as usual.

## Things to know

- **Train the generators and generate on the same data as the classifier.** A generator trained on a split that contains the classifier's test clips leaks them into pretraining. The official INCLUDE split shares 577 test clips with the training set of our random split, so a generator trained on one must not be used for the other.
- **The generated clips are stored in the classifier's input space.** They use the dataset-level normalization (`DATA.norm: "dataset"`) and the keypoints of the generator's `DATA.poses`, and are 32 consecutive frames long. Classifier configs that use `norm: "shoulder"`, `temporal_sampling: "uniform"` or a different keypoint set (the `official-*` and `stgcn` configs) do not match that representation, so the existing pipeline cannot pretrain them yet.
- **Class conditioning needed a fix.** Until the class list was sorted (see [reproducibility.md](reproducibility.md#class-indexes)), every process numbered the signs differently. Step 3 runs in a different process than steps 1 and 2, so each sign was generated with the embedding of another sign. Datasets generated before that fix are affected.
