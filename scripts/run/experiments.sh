#!/bin/bash
# nohup

# Example scripts
# ./scripts/run/train.sh cond_prediction CsiMLPe depth_big-reversed --reverse
# ./scripts/run/train.sh cond_prediction CsiMLPe depth_big_noise_0.1
# ./scripts/run/generate_dataset.sh cond_prediction CsiMLPe depth_big-reversed -ckpt $HANDCRAFT_SAVE/LSFB/CsiMLPe-depth_big/checkpoints/LSFB-depth_big-train-2024_06_27_15_41_40/ -tg -r_ckpt $HANDCRAFT_SAVE/LSFB/CsiMLPe-depth_big-reversed/checkpoints/LSFB-depth_big-reversed-train-2024_08_28_14_03_47/ --sd_num 2 

#./scripts/run/train.sh cond_prediction CsiMLPe depth_big_noise_0.1-reversed --reverse
#./scripts/run/generate_dataset.sh cond_prediction CsiMLPe depth_big_noise_0.1-reversed -ckpt $HANDCRAFT_SAVE/LSFB/CsiMLPe-depth_big_noise_0.1/checkpoints/LSFB-depth_big_noise_0.1-train-2024_08_29_09_08_38/ -tg -r_ckpt $HANDCRAFT_SAVE/LSFB/CsiMLPe-depth_big_noise_0.1-reversed/checkpoints/LSFB-depth_big_noise_0.1-reversed-train-2024_09_04_14_48_40/ --sd_num 5
#./scripts/run/train.sh classification conv1d DCT-depth6-oc-pad LSFB
#./scripts/run/train.sh classification conv1d DCT-depth4-oc-pad INCLUDE
#./scripts/run/train.sh classification conv1d DCT-depth4-oc-pad DiSPLaY
#./scripts/run/train.sh classification ViT original-pad LSFB
#./scripts/run/train.sh classification ViT original-pad INCLUDE
#./scripts/run/train.sh classification ViT original-pad DiSPLaY 

# MAMBA
#./scripts/run/train.sh classification mamba original128-pad LSFB
#./scripts/run/train.sh classification mamba original128-edge LSFB
#./scripts/run/train.sh classification mamba original128-mean LSFB
#./scripts/run/train.sh classification mamba original128-wrap LSFB
#./scripts/run/train.sh classification mamba original64-pad INCLUDE
#./scripts/run/train.sh classification mamba original64-edge INCLUDE
#./scripts/run/train.sh classification mamba original64-mean INCLUDE
#./scripts/run/train.sh classification mamba original64-wrap INCLUDE
#./scripts/run/train.sh classification mamba original64-pad DiSPLaY 
#./scripts/run/train.sh classification mamba original64-edge DiSPLaY 
#./scripts/run/train.sh classification mamba original64-mean DiSPLaY 
#./scripts/run/train.sh classification mamba original64-wrap DiSPLaY 

# ViT
#./scripts/run/train.sh classification ViT original-pad LSFB
#./scripts/run/train.sh classification ViT original-edge LSFB
#./scripts/run/train.sh classification ViT original-mean LSFB
#./scripts/run/train.sh classification ViT original-wrap LSFB
#./scripts/run/train.sh classification ViT original-edge INCLUDE
#./scripts/run/train.sh classification ViT original-pad INCLUDE
#./scripts/run/train.sh classification ViT original-mean INCLUDE
#./scripts/run/train.sh classification ViT original-wrap INCLUDE
#./scripts/run/train.sh classification ViT original-pad DiSPLaY 
#./scripts/run/train.sh classification ViT original-edge DiSPLaY 
#./scripts/run/train.sh classification ViT original-mean DiSPLaY 
#./scripts/run/train.sh classification ViT original-wrap DiSPLaY 

# Conv1D
#./scripts/run/train.sh classification conv1d DCT-depth7-oc-pad LSFB
#./scripts/run/train.sh classification conv1d DCT-depth7-oc-edge LSFB
#./scripts/run/train.sh classification conv1d DCT-depth7-oc-mean LSFB
#./scripts/run/train.sh classification conv1d DCT-depth7-oc-wrap LSFB
#./scripts/run/train.sh classification conv1d DCT-depth4-oc-pad INCLUDE
#./scripts/run/train.sh classification conv1d DCT-depth4-oc-edge INCLUDE
#./scripts/run/train.sh classification conv1d DCT-depth4-oc-mean INCLUDE
#./scripts/run/train.sh classification conv1d DCT-depth4-oc-wrap INCLUDE
#./scripts/run/train.sh classification conv1d DCT-depth4-oc-pad DiSPLaY 
#./scripts/run/train.sh classification conv1d DCT-depth4-oc-edge DiSPLaY 
#./scripts/run/train.sh classification conv1d DCT-depth4-oc-mean DiSPLaY 
#./scripts/run/train.sh classification conv1d DCT-depth4-oc-wrap DiSPLaY 

#CsiMLPe
#./scripts/run/train.sh cond_prediction CsiMLPe depth_big_noise_0.1 INCLUDE
#./scripts/run/train.sh cond_prediction CsiMLPe depth_big_noise_0.1-reversed INCLUDE --reverse
#./scripts/run/generate_dataset.sh cond_prediction CsiMLPe depth_big_noise_0.1-reversed INCLUDE -ckpt $HANDCRAFT_SAVE/INCLUDE/CsiMLPe-depth_big_noise_0.1/checkpoints/INCLUDE-depth_big_noise_0.1-train-2024_11_11_22_23_33/ -tg -r_ckpt $HANDCRAFT_SAVE/INCLUDE/CsiMLPe-depth_big_noise_0.1-reversed/checkpoints/INCLUDE-depth_big_noise_0.1-reversed-train-2024_11_11_22_41_48/ --sd_num 100
#./scripts/run/train.sh cond_prediction CsiMLPe depth_big_noise_0.1 DiSPLaY
#./scripts/run/train.sh cond_prediction CsiMLPe depth_big_noise_0.1-reversed DiSPLaY --reverse
#./scripts/run/generate_dataset.sh cond_prediction CsiMLPe depth_big_noise_0.1-reversed DiSPLaY -ckpt $HANDCRAFT_SAVE/DiSPLaY/CsiMLPe-depth_big_noise_0.1/checkpoints/DiSPLaY-depth_big_noise_0.1-train-2024_11_11_22_59_58/ -tg -r_ckpt $HANDCRAFT_SAVE/DiSPLaY/CsiMLPe-depth_big_noise_0.1-reversed/checkpoints/DiSPLaY-depth_big_noise_0.1-reversed-train-2024_11_11_23_13_29/ --sd_num 100

# MAMBA
#./scripts/run/train.sh classification mamba original128-pad LSFB
#./scripts/run/train.sh classification mamba original128-edge LSFB
#./scripts/run/train.sh classification mamba original128-mean LSFB
#./scripts/run/train.sh classification mamba original128-wrap LSFB
#./scripts/run/train.sh classification mamba original64-pad INCLUDE
#./scripts/run/train.sh classification mamba original64-edge INCLUDE
#./scripts/run/train.sh classification mamba original64-mean INCLUDE
#./scripts/run/train.sh classification mamba original64-wrap INCLUDE
#./scripts/run/train.sh classification mamba original64-pad DiSPLaY 
#./scripts/run/train.sh classification mamba original64-edge DiSPLaY 
#./scripts/run/train.sh classification mamba original64-mean DiSPLaY 
#./scripts/run/train.sh classification mamba original64-wrap DiSPLaY 

# ViT
#./scripts/run/train.sh classification ViT original-pad LSFB 
#./scripts/run/train.sh classification ViT original-edge LSFB
#./scripts/run/train.sh classification ViT original-mean LSFB
#./scripts/run/train.sh classification ViT original-wrap LSFB
#./scripts/run/train.sh classification ViT original-edge INCLUDE
#./scripts/run/train.sh classification ViT original-pad INCLUDE
#./scripts/run/train.sh classification ViT original-mean INCLUDE
#./scripts/run/train.sh classification ViT original-wrap INCLUDE
#./scripts/run/train.sh classification ViT original-pad DiSPLaY 
#./scripts/run/train.sh classification ViT original-edge DiSPLaY 
#./scripts/run/train.sh classification ViT original-mean DiSPLaY 
#./scripts/run/train.sh classification ViT original-wrap DiSPLaY 

# Conv1D INCREASE K SIZE! didnt work :( good bye conv1d, you had a good life
#./scripts/run/train.sh classification conv1d DCT-depth7-oc-pad LSFB
#./scripts/run/train.sh classification conv1d DCT-depth7-oc-edge LSFB
#./scripts/run/train.sh classification conv1d DCT-depth7-oc-mean LSFB
#./scripts/run/train.sh classification conv1d DCT-depth7-oc-wrap LSFB
#./scripts/run/train.sh classification conv1d DCT-depth4-oc-pad INCLUDE
#./scripts/run/train.sh classification conv1d DCT-depth4-oc-edge INCLUDE
#./scripts/run/train.sh classification conv1d DCT-depth4-oc-mean INCLUDE
#./scripts/run/train.sh classification conv1d DCT-depth4-oc-wrap INCLUDE
#./scripts/run/train.sh classification conv1d DCT-depth4-oc-pad DiSPLaY 
#./scripts/run/train.sh classification conv1d DCT-depth4-oc-edge DiSPLaY 
#./scripts/run/train.sh classification conv1d DCT-depth4-oc-mean DiSPLaY 
#./scripts/run/train.sh classification conv1d DCT-depth4-oc-wrap DiSPLaY 

# BIGGER WINDOWS

# MAMBA
#./scripts/run/train.sh classification mamba original64-pad-w64 INCLUDE
#./scripts/run/train.sh classification mamba original64-pad-w128 INCLUDE
#./scripts/run/train.sh classification mamba original64-pad-w64 DiSPLaY
#./scripts/run/train.sh classification mamba original64-pad-w128 DiSPLaY

# ViT
#./scripts/run/train.sh classification ViT original-pad-w64 INCLUDE
#./scripts/run/train.sh classification ViT original-pad-w128 INCLUDE
#./scripts/run/train.sh classification ViT original-pad-w64 DiSPLaY
#./scripts/run/train.sh classification ViT original-pad-w128 DiSPLaY

# SYNTH TRAIN

# MAMBA
#./scripts/run/train.sh classification mamba original128-pad-synth5 LSFB -s_data $HANDCRAFT_SAVE/LSFB/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/depth_big_noise_0.1-reversed-train-2024_08_29_09_08_38
#./scripts/run/train.sh classification mamba original128-pad-synth10 LSFB -s_data $HANDCRAFT_SAVE/LSFB/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/depth_big_noise_0.1-reversed-train-2024_08_29_09_08_38
#./scripts/run/train.sh classification mamba original64-pad-synth50 INCLUDE -s_data $HANDCRAFT_SAVE/INCLUDE/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/depth_big_noise_0.1-reversed-train-2024_11_11_22_23_33
#./scripts/run/train.sh classification mamba original64-pad-synth50 DiSPLaY -s_data $HANDCRAFT_SAVE/DiSPLaY/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/depth_big_noise_0.1-reversed-train-2024_11_11_22_59_58

# ViT
#scripts/run/train.sh classification ViT original-pad-synth10 LSFB -s_data $HANDCRAFT_SAVE/LSFB/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/depth_big_noise_0.1-reversed-train-2024_08_29_09_08_38
#./scripts/run/train.sh classification ViT original-pad-synth25 INCLUDE -s_data $HANDCRAFT_SAVE/INCLUDE/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/depth_big_noise_0.1-reversed-train-2024_11_11_22_23_33
#./scripts/run/train.sh classification ViT original-pad-synth50 INCLUDE -s_data $HANDCRAFT_SAVE/INCLUDE/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/depth_big_noise_0.1-reversed-train-2024_11_11_22_23_33
#./scripts/run/train.sh classification ViT original-pad-synth75 INCLUDE -s_data $HANDCRAFT_SAVE/INCLUDE/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/depth_big_noise_0.1-reversed-train-2024_11_11_22_23_33
#./scripts/run/train.sh classification ViT original-pad-synth25-425 INCLUDE -s_data $HANDCRAFT_SAVE/INCLUDE/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/depth_big_noise_0.1-reversed-train-2024_11_11_22_23_33
#./scripts/run/train.sh classification ViT original-pad-synth50-450 INCLUDE -s_data $HANDCRAFT_SAVE/INCLUDE/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/depth_big_noise_0.1-reversed-train-2024_11_11_22_23_33
#./scripts/run/train.sh classification ViT original-pad-synth75-475 INCLUDE -s_data $HANDCRAFT_SAVE/INCLUDE/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/depth_big_noise_0.1-reversed-train-2024_11_11_22_23_33
#./scripts/run/train.sh classification ViT original-pad-synth50 DiSPLaY -s_data $HANDCRAFT_SAVE/DiSPLaY/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/depth_big_noise_0.1-reversed-train-2024_11_11_22_59_58

# SYNTH TRAIN PRETRAIN STEPS
# 50 dio el mejor de los resultados, 75 deja muy pocos steps para el real train y 25 muy pocos para el fake train
#./scripts/run/train.sh classification mamba original64-pad-synth25 INCLUDE -s_data $HANDCRAFT_SAVE/INCLUDE/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/depth_big_noise_0.1-reversed-train-2024_11_11_22_23_33
#./scripts/run/train.sh classification mamba original64-pad-synth50 INCLUDE -s_data $HANDCRAFT_SAVE/INCLUDE/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/depth_big_noise_0.1-reversed-train-2024_11_11_22_23_33
#./scripts/run/train.sh classification mamba original64-pad-synth75 INCLUDE -s_data $HANDCRAFT_SAVE/INCLUDE/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/depth_big_noise_0.1-reversed-train-2024_11_11_22_23_33

# DA

# MAMBA
#./scripts/run/train.sh classification mamba original64-pad-da INCLUDE
#./scripts/run/train.sh classification mamba original64-pad-da DiSPLaY

# ViT
#./scripts/run/train.sh classification ViT original-pad-da INCLUDE
#./scripts/run/train.sh classification ViT original-pad-da DiSPLaY

# BIGGER MODELS?
#./scripts/run/train.sh classification mamba original64-pad-64x1-o1024 DiSPLaY
#./scripts/run/train.sh classification mamba original64-pad-128x1-o1024 DiSPLaY
#./scripts/run/train.sh classification mamba original64-pad-64x1-o2048 DiSPLaY
#./scripts/run/train.sh classification mamba original64-pad-64x2-o1024 DiSPLaY # WINNER!!!!!
#./scripts/run/train.sh classification ViT original-pad-128X2 DiSPLaY # WINNER!!!!!
#./scripts/run/train.sh classification ViT original-pad-256X2 DiSPLaY
#./scripts/run/train.sh classification ViT original-pad-512X2 DiSPLaY
#./scripts/run/train.sh classification ViT original-pad-1024X2 DiSPLaY

#./scripts/run/train.sh classification mamba original64-pad-64x2-o1024 INCLUDE # WINNER!!!!!
#./scripts/run/train.sh classification ViT original-pad-128x2 INCLUDE # WINNER!!!!!

#different LR?
#./scripts/run/train.sh classification ViT original-pad-lr01-mlr1 INCLUDE # falla, lr demasiado alto
#./scripts/run/train.sh classification ViT original-pad-lr001-mlr01 INCLUDE 
#./scripts/run/train.sh classification ViT original-pad-lr0001-mlr001 INCLUDE 

#different rep size
#./scripts/run/train.sh classification ViT original-pad-rs128 INCLUDE 
#./scripts/run/train.sh classification ViT original-pad-rs256 INCLUDE 
#./scripts/run/train.sh classification ViT original-pad-rs512 INCLUDE 
#./scripts/run/train.sh classification ViT original-pad-rs1024 INCLUDE 


#./scripts/run/train.sh classification mamba original64-pad-synth25 INCLUDE -s_data $HANDCRAFT_SAVE/INCLUDE/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/depth_big_noise_0.1-reversed-train-2024_11_11_22_23_33
#./scripts/run/train.sh classification mamba original64-pad-synth50 INCLUDE -s_data $HANDCRAFT_SAVE/INCLUDE/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/depth_big_noise_0.1-reversed-train-2024_11_11_22_23_33
#./scripts/run/train.sh classification mamba original128-pad-synth5 LSFB -s_data $HANDCRAFT_SAVE/LSFB/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/depth_big_noise_0.1-reversed-train-2024_08_29_09_08_38
#./scripts/run/train.sh classification mamba original128-pad LSFBs
#./scripts/run/train.sh classification ViT original-pad-synth5 LSFB -s_data $HANDCRAFT_SAVE/LSFB/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/depth_big_noise_0.1-reversed-train-2024_08_29_09_08_38

#./scripts/run/train.sh classification ViT original-pad-1x128-1024 LSFB
#./scripts/run/train.sh classification ViT original-pad-2x256-1024 LSFB
#./scripts/run/train.sh classification ViT original-pad-1x512-1024 LSFB

#./scripts/run/train.sh classification ViT original-pad-lr0 LSFB
#./scripts/run/train.sh classification ViT original-pad-lr1 LSFB
#./scripts/run/train.sh classification ViT original-pad-lr2 LSFB
#./scripts/run/train.sh classification ViT original-pad-lr3 LSFB


#./scripts/run/train.sh classification ViT original-pad-2x256-1024-lr0 LSFB
#./scripts/run/train.sh classification ViT original-pad-2x256-1024-lr1 LSFB
#./scripts/run/train.sh classification ViT original-pad-2x256-1024-lr2 LSFB
#./scripts/run/train.sh classification ViT original-pad-2x256-1024-lr3 LSFB

#./scripts/run/train.sh classification ViT original-pad-2x256-1024-lr1-rotate LSFB
#./scripts/run/train.sh classification ViT original-pad-2x256-1024-lr1-scale LSFB
#./scripts/run/train.sh classification ViT original-pad-2x256-1024-lr1-da LSFB
#./scripts/run/train.sh classification ViT original-pad-2x256-1024-lr1-synth5 LSFB
#./scripts/run/train.sh classification ViT original-pad-synth5-da LSFB -s_data $HANDCRAFT_SAVE/LSFB/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/depth_big_noise_0.1-reversed-train-2024_08_29_09_08_38
#./scripts/run/train.sh classification ViT original-pad-synth5 LSFB -s_data $HANDCRAFT_SAVE/LSFB/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/depth_big_noise_0.1-reversed-train-2024_08_29_09_08_38

#./scripts/run/train.sh classification mamba original128-pad-synth5-da LSFB -s_data $HANDCRAFT_SAVE/LSFB/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/depth_big_noise_0.1-reversed-train-2024_08_29_09_08_38
#./scripts/run/train.sh classification mamba original128-pad LSFB

#./scripts/run/train.sh classification ViT original-pad-synth25-425-da INCLUDE -s_data $HANDCRAFT_SAVE/INCLUDE/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/depth_big_noise_0.1-reversed-train-2024_11_11_22_23_33
#./scripts/run/train.sh classification ViT original-pad-rs1024-da INCLUDE 

# INCLUDE official split (see scripts/data/INCLUDE/README.md), results are the mean of seeds 42 and 43
#./scripts/run/test.sh classification ViT official-base INCLUDE -data $HANDCRAFT_DATA/INCLUDE_official/ --seed 42
#./scripts/run/test.sh classification ViT official-all-nomirror INCLUDE -data $HANDCRAFT_DATA/INCLUDE_official/ --seed 42
#./scripts/run/test.sh classification ViT official-nm-nodct INCLUDE -data $HANDCRAFT_DATA/INCLUDE_official/ --seed 42
#./scripts/run/test.sh classification ViT official-nm-nodct INCLUDE -data $HANDCRAFT_DATA/INCLUDE_official/ --seed 43
#./scripts/run/test.sh classification stgcn official-lr5 INCLUDE -data $HANDCRAFT_DATA/INCLUDE_official/ --seed 42
#./scripts/run/test.sh classification stgcn official-lr5 INCLUDE -data $HANDCRAFT_DATA/INCLUDE_official/ --seed 43

./scripts/run/train.sh classification ViT original-pad-2x256-1024-lr1-ema LSFB
./scripts/run/train.sh classification ViT original-pad-2x256-1024-lr1-latedrop LSFB
./scripts/run/train.sh classification ViT original-pad-4x256-1024-lr1-highdrop LSFB