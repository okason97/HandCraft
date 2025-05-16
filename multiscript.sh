#!/bin/bash
# nohup

# Example scripts
# ./script.sh cond_prediction CsiMLPe depth_big-reversed --reverse
# ./script.sh cond_prediction CsiMLPe depth_big_noise_0.1
# ./gdataset_script.sh cond_prediction CsiMLPe depth_big-reversed -ckpt /disco2/models/HandCraft/samples/LSFB/CsiMLPe-depth_big/checkpoints/LSFB-depth_big-train-2024_06_27_15_41_40/ -tg -r_ckpt /disco2/models/HandCraft/samples/LSFB/CsiMLPe-depth_big-reversed/checkpoints/LSFB-depth_big-reversed-train-2024_08_28_14_03_47/ --sd_num 2 

#./script.sh cond_prediction CsiMLPe depth_big_noise_0.1-reversed --reverse
#./gdataset_script.sh cond_prediction CsiMLPe depth_big_noise_0.1-reversed -ckpt /disco1/models/HandCraft/samples/LSFB/CsiMLPe-depth_big_noise_0.1/checkpoints/LSFB-depth_big_noise_0.1-train-2024_08_29_09_08_38/ -tg -r_ckpt /disco1/models/HandCraft/samples/LSFB/CsiMLPe-depth_big_noise_0.1-reversed/checkpoints/LSFB-depth_big_noise_0.1-reversed-train-2024_09_04_14_48_40/ --sd_num 5
#./script.sh classification conv1d DCT-depth6-oc-pad LSFB
#./script.sh classification conv1d DCT-depth4-oc-pad INCLUDE
#./script.sh classification conv1d DCT-depth4-oc-pad DiSPLaY
#./script.sh classification ViT original-pad LSFB
#./script.sh classification ViT original-pad INCLUDE
#./script.sh classification ViT original-pad DiSPLaY 

# MAMBA
#./script.sh classification mamba original128-pad LSFB
#./script.sh classification mamba original128-edge LSFB
#./script.sh classification mamba original128-mean LSFB
#./script.sh classification mamba original128-wrap LSFB
#./script.sh classification mamba original64-pad INCLUDE
#./script.sh classification mamba original64-edge INCLUDE
#./script.sh classification mamba original64-mean INCLUDE
#./script.sh classification mamba original64-wrap INCLUDE
#./script.sh classification mamba original64-pad DiSPLaY 
#./script.sh classification mamba original64-edge DiSPLaY 
#./script.sh classification mamba original64-mean DiSPLaY 
#./script.sh classification mamba original64-wrap DiSPLaY 

# ViT
#./script.sh classification ViT original-pad LSFB
#./script.sh classification ViT original-edge LSFB
#./script.sh classification ViT original-mean LSFB
#./script.sh classification ViT original-wrap LSFB
#./script.sh classification ViT original-edge INCLUDE
#./script.sh classification ViT original-pad INCLUDE
#./script.sh classification ViT original-mean INCLUDE
#./script.sh classification ViT original-wrap INCLUDE
#./script.sh classification ViT original-pad DiSPLaY 
#./script.sh classification ViT original-edge DiSPLaY 
#./script.sh classification ViT original-mean DiSPLaY 
#./script.sh classification ViT original-wrap DiSPLaY 

# Conv1D
#./script.sh classification conv1d DCT-depth7-oc-pad LSFB
#./script.sh classification conv1d DCT-depth7-oc-edge LSFB
#./script.sh classification conv1d DCT-depth7-oc-mean LSFB
#./script.sh classification conv1d DCT-depth7-oc-wrap LSFB
#./script.sh classification conv1d DCT-depth4-oc-pad INCLUDE
#./script.sh classification conv1d DCT-depth4-oc-edge INCLUDE
#./script.sh classification conv1d DCT-depth4-oc-mean INCLUDE
#./script.sh classification conv1d DCT-depth4-oc-wrap INCLUDE
#./script.sh classification conv1d DCT-depth4-oc-pad DiSPLaY 
#./script.sh classification conv1d DCT-depth4-oc-edge DiSPLaY 
#./script.sh classification conv1d DCT-depth4-oc-mean DiSPLaY 
#./script.sh classification conv1d DCT-depth4-oc-wrap DiSPLaY 

#CsiMLPe
#./script.sh cond_prediction CsiMLPe depth_big_noise_0.1 INCLUDE
#./script.sh cond_prediction CsiMLPe depth_big_noise_0.1-reversed INCLUDE --reverse
#./gdataset_script.sh cond_prediction CsiMLPe depth_big_noise_0.1-reversed INCLUDE -ckpt /disco1/models/HandCraft/samples/INCLUDE/CsiMLPe-depth_big_noise_0.1/checkpoints/INCLUDE-depth_big_noise_0.1-train-2024_11_11_22_23_33/ -tg -r_ckpt /disco1/models/HandCraft/samples/INCLUDE/CsiMLPe-depth_big_noise_0.1-reversed/checkpoints/INCLUDE-depth_big_noise_0.1-reversed-train-2024_11_11_22_41_48/ --sd_num 100
#./script.sh cond_prediction CsiMLPe depth_big_noise_0.1 DiSPLaY
#./script.sh cond_prediction CsiMLPe depth_big_noise_0.1-reversed DiSPLaY --reverse
#./gdataset_script.sh cond_prediction CsiMLPe depth_big_noise_0.1-reversed DiSPLaY -ckpt /disco1/models/HandCraft/samples/DiSPLaY/CsiMLPe-depth_big_noise_0.1/checkpoints/DiSPLaY-depth_big_noise_0.1-train-2024_11_11_22_59_58/ -tg -r_ckpt /disco1/models/HandCraft/samples/DiSPLaY/CsiMLPe-depth_big_noise_0.1-reversed/checkpoints/DiSPLaY-depth_big_noise_0.1-reversed-train-2024_11_11_23_13_29/ --sd_num 100

# MAMBA
#./script.sh classification mamba original128-pad LSFB
#./script.sh classification mamba original128-edge LSFB
#./script.sh classification mamba original128-mean LSFB
#./script.sh classification mamba original128-wrap LSFB
#./script.sh classification mamba original64-pad INCLUDE
#./script.sh classification mamba original64-edge INCLUDE
#./script.sh classification mamba original64-mean INCLUDE
#./script.sh classification mamba original64-wrap INCLUDE
#./script.sh classification mamba original64-pad DiSPLaY 
#./script.sh classification mamba original64-edge DiSPLaY 
#./script.sh classification mamba original64-mean DiSPLaY 
#./script.sh classification mamba original64-wrap DiSPLaY 

# ViT
#./script.sh classification ViT original-pad LSFB 
#./script.sh classification ViT original-edge LSFB
#./script.sh classification ViT original-mean LSFB
#./script.sh classification ViT original-wrap LSFB
#./script.sh classification ViT original-edge INCLUDE
#./script.sh classification ViT original-pad INCLUDE
#./script.sh classification ViT original-mean INCLUDE
#./script.sh classification ViT original-wrap INCLUDE
#./script.sh classification ViT original-pad DiSPLaY 
#./script.sh classification ViT original-edge DiSPLaY 
#./script.sh classification ViT original-mean DiSPLaY 
#./script.sh classification ViT original-wrap DiSPLaY 

# Conv1D INCREASE K SIZE! didnt work :( good bye conv1d, you had a good life
#./script.sh classification conv1d DCT-depth7-oc-pad LSFB
#./script.sh classification conv1d DCT-depth7-oc-edge LSFB
#./script.sh classification conv1d DCT-depth7-oc-mean LSFB
#./script.sh classification conv1d DCT-depth7-oc-wrap LSFB
#./script.sh classification conv1d DCT-depth4-oc-pad INCLUDE
#./script.sh classification conv1d DCT-depth4-oc-edge INCLUDE
#./script.sh classification conv1d DCT-depth4-oc-mean INCLUDE
#./script.sh classification conv1d DCT-depth4-oc-wrap INCLUDE
#./script.sh classification conv1d DCT-depth4-oc-pad DiSPLaY 
#./script.sh classification conv1d DCT-depth4-oc-edge DiSPLaY 
#./script.sh classification conv1d DCT-depth4-oc-mean DiSPLaY 
#./script.sh classification conv1d DCT-depth4-oc-wrap DiSPLaY 

# BIGGER WINDOWS

# MAMBA
#./script.sh classification mamba original64-pad-w64 INCLUDE
#./script.sh classification mamba original64-pad-w128 INCLUDE
#./script.sh classification mamba original64-pad-w64 DiSPLaY
#./script.sh classification mamba original64-pad-w128 DiSPLaY

# ViT
#./script.sh classification ViT original-pad-w64 INCLUDE
#./script.sh classification ViT original-pad-w128 INCLUDE
#./script.sh classification ViT original-pad-w64 DiSPLaY
#./script.sh classification ViT original-pad-w128 DiSPLaY

# SYNTH TRAIN

# MAMBA
#./script.sh classification mamba original128-pad-synth5 LSFB -s_data /disco1/models/HandCraft/samples/LSFB/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/depth_big_noise_0.1-reversed-train-2024_08_29_09_08_38
#./script.sh classification mamba original128-pad-synth10 LSFB -s_data /disco1/models/HandCraft/samples/LSFB/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/depth_big_noise_0.1-reversed-train-2024_08_29_09_08_38
#./script.sh classification mamba original64-pad-synth50 INCLUDE -s_data /disco1/models/HandCraft/samples/INCLUDE/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/depth_big_noise_0.1-reversed-train-2024_11_11_22_23_33
#./script.sh classification mamba original64-pad-synth50 DiSPLaY -s_data /disco1/models/HandCraft/samples/DiSPLaY/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/depth_big_noise_0.1-reversed-train-2024_11_11_22_59_58

# ViT
#/script.sh classification ViT original-pad-synth10 LSFB -s_data /disco1/models/HandCraft/samples/LSFB/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/depth_big_noise_0.1-reversed-train-2024_08_29_09_08_38
#./script.sh classification ViT original-pad-synth25 INCLUDE -s_data /disco1/models/HandCraft/samples/INCLUDE/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/depth_big_noise_0.1-reversed-train-2024_11_11_22_23_33
#./script.sh classification ViT original-pad-synth50 INCLUDE -s_data /disco1/models/HandCraft/samples/INCLUDE/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/depth_big_noise_0.1-reversed-train-2024_11_11_22_23_33
#./script.sh classification ViT original-pad-synth75 INCLUDE -s_data /disco1/models/HandCraft/samples/INCLUDE/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/depth_big_noise_0.1-reversed-train-2024_11_11_22_23_33
#./script.sh classification ViT original-pad-synth25-425 INCLUDE -s_data /disco1/models/HandCraft/samples/INCLUDE/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/depth_big_noise_0.1-reversed-train-2024_11_11_22_23_33
#./script.sh classification ViT original-pad-synth50-450 INCLUDE -s_data /disco1/models/HandCraft/samples/INCLUDE/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/depth_big_noise_0.1-reversed-train-2024_11_11_22_23_33
#./script.sh classification ViT original-pad-synth75-475 INCLUDE -s_data /disco1/models/HandCraft/samples/INCLUDE/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/depth_big_noise_0.1-reversed-train-2024_11_11_22_23_33
#./script.sh classification ViT original-pad-synth50 DiSPLaY -s_data /disco1/models/HandCraft/samples/DiSPLaY/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/depth_big_noise_0.1-reversed-train-2024_11_11_22_59_58

# SYNTH TRAIN PRETRAIN STEPS
# 50 dio el mejor de los resultados, 75 deja muy pocos steps para el real train y 25 muy pocos para el fake train
#./script.sh classification mamba original64-pad-synth25 INCLUDE -s_data /disco1/models/HandCraft/samples/INCLUDE/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/depth_big_noise_0.1-reversed-train-2024_11_11_22_23_33
#./script.sh classification mamba original64-pad-synth50 INCLUDE -s_data /disco1/models/HandCraft/samples/INCLUDE/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/depth_big_noise_0.1-reversed-train-2024_11_11_22_23_33
#./script.sh classification mamba original64-pad-synth75 INCLUDE -s_data /disco1/models/HandCraft/samples/INCLUDE/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/depth_big_noise_0.1-reversed-train-2024_11_11_22_23_33

# DA

# MAMBA
#./script.sh classification mamba original64-pad-da INCLUDE
#./script.sh classification mamba original64-pad-da DiSPLaY

# ViT
#./script.sh classification ViT original-pad-da INCLUDE
#./script.sh classification ViT original-pad-da DiSPLaY

# BIGGER MODELS?
#./script.sh classification mamba original64-pad-64x1-o1024 DiSPLaY
#./script.sh classification mamba original64-pad-128x1-o1024 DiSPLaY
#./script.sh classification mamba original64-pad-64x1-o2048 DiSPLaY
#./script.sh classification mamba original64-pad-64x2-o1024 DiSPLaY # WINNER!!!!!
#./script.sh classification ViT original-pad-128X2 DiSPLaY # WINNER!!!!!
#./script.sh classification ViT original-pad-256X2 DiSPLaY
#./script.sh classification ViT original-pad-512X2 DiSPLaY
#./script.sh classification ViT original-pad-1024X2 DiSPLaY

#./script.sh classification mamba original64-pad-64x2-o1024 INCLUDE # WINNER!!!!!
#./script.sh classification ViT original-pad-128x2 INCLUDE # WINNER!!!!!

#different LR?
#./script.sh classification ViT original-pad-lr01-mlr1 INCLUDE # falla, lr demasiado alto
#./script.sh classification ViT original-pad-lr001-mlr01 INCLUDE 
#./script.sh classification ViT original-pad-lr0001-mlr001 INCLUDE 

#different rep size
#./script.sh classification ViT original-pad-rs128 INCLUDE 
#./script.sh classification ViT original-pad-rs256 INCLUDE 
#./script.sh classification ViT original-pad-rs512 INCLUDE 
#./script.sh classification ViT original-pad-rs1024 INCLUDE 


#./script.sh classification mamba original64-pad-synth25 INCLUDE -s_data /disco1/models/HandCraft/samples/INCLUDE/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/depth_big_noise_0.1-reversed-train-2024_11_11_22_23_33
#./script.sh classification mamba original64-pad-synth50 INCLUDE -s_data /disco1/models/HandCraft/samples/INCLUDE/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/depth_big_noise_0.1-reversed-train-2024_11_11_22_23_33
#./script.sh classification mamba original128-pad-synth5 LSFB -s_data /disco1/models/HandCraft/samples/LSFB/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/depth_big_noise_0.1-reversed-train-2024_08_29_09_08_38
#./script.sh classification mamba original128-pad LSFBs
#./script.sh classification ViT original-pad-synth5 LSFB -s_data /disco1/models/HandCraft/samples/LSFB/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/depth_big_noise_0.1-reversed-train-2024_08_29_09_08_38

#./script.sh classification ViT original-pad-1x128-1024 LSFB
#./script.sh classification ViT original-pad-2x256-1024 LSFB
#./script.sh classification ViT original-pad-1x512-1024 LSFB

#./script.sh classification ViT original-pad-lr0 LSFB
#./script.sh classification ViT original-pad-lr1 LSFB
#./script.sh classification ViT original-pad-lr2 LSFB
#./script.sh classification ViT original-pad-lr3 LSFB


#./script.sh classification ViT original-pad-2x256-1024-lr0 LSFB
#./script.sh classification ViT original-pad-2x256-1024-lr1 LSFB
#./script.sh classification ViT original-pad-2x256-1024-lr2 LSFB
#./script.sh classification ViT original-pad-2x256-1024-lr3 LSFB

#./script.sh classification ViT original-pad-2x256-1024-lr1-rotate LSFB
#./script.sh classification ViT original-pad-2x256-1024-lr1-scale LSFB
#./script.sh classification ViT original-pad-2x256-1024-lr1-da LSFB
#./script.sh classification ViT original-pad-2x256-1024-lr1-synth5 LSFB
#./script.sh classification ViT original-pad-synth5-da LSFB -s_data /disco1/models/HandCraft/samples/LSFB/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/depth_big_noise_0.1-reversed-train-2024_08_29_09_08_38
#./script.sh classification ViT original-pad-synth5 LSFB -s_data /disco1/models/HandCraft/samples/LSFB/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/depth_big_noise_0.1-reversed-train-2024_08_29_09_08_38

#./script.sh classification mamba original128-pad-synth5-da LSFB -s_data /disco1/models/HandCraft/samples/LSFB/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/depth_big_noise_0.1-reversed-train-2024_08_29_09_08_38
#./script.sh classification mamba original128-pad LSFB

#./script.sh classification ViT original-pad-synth25-425-da INCLUDE -s_data /disco1/models/HandCraft/samples/INCLUDE/CsiMLPe-depth_big_noise_0.1-reversed/generated_datasets/depth_big_noise_0.1-reversed-train-2024_11_11_22_23_33
#./script.sh classification ViT original-pad-rs1024-da INCLUDE 

./script.sh classification ViT original-pad-2x256-1024-lr1-ema LSFB
./script.sh classification ViT original-pad-2x256-1024-lr1-latedrop LSFB
./script.sh classification ViT original-pad-4x256-1024-lr1-highdrop LSFB