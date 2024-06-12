#!/bin/bash
# nohup

# $1 = mode (prediction, cond_prediction, classification), $2 = model (conv1d/siMLPe/CsiMLPe), $3 = config
# Create folder
echo making dir $2-$3
mkdir -p ./logs/LSFB/$2-$3/
mkdir -p /mnt/sda2/models/HandCraft/samples/LSFB/$2-$3/
touch ./logs/LSFB/$2-$3/run0.out 
touch ./logs/LSFB/$2-$3/run0.err

# Train Model
echo Training
python src/main.py --mode $1 -t -data /mnt/sda2/datasets/isolated-cont-sl/LSFB/ -cfg ./src/configs/LSFB/$2/$3.yaml -save /mnt/sda2/models/HandCraft/samples/LSFB/$2-$3/ --project handcraft-$2 --num_workers 4 --prefetch_factor 2 -every 1 --print_every 1 -mpc > ./logs/LSFB/$2-$3/run0.out 2> ./logs/LSFB/$2-$3/run0.err

# Save dataset
#python src/main.py --mode $1 -sd --sd_num 10 -data /mnt/sda2/datasets/isolated-cont-sl/LSFB/ -cfg ./src/configs/LSFB/$2/$3.yaml -save /mnt/sda2/models/HandCraft/samples/LSFB/$2-$3/ -ckpt /mnt/sda2/models/HandCraft/samples/LSFB/$2-$3/checkpoints/LSFB-vanilla-train-2024_05_21_19_17_26/ -best --project handcraft-$2 --num_workers 4 --prefetch_factor 2 -mpc > ./logs/LSFB/$2-$3/run0.out 2> ./logs/LSFB/$2-$3/run0.err