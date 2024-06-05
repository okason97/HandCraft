#!/bin/bash
# nohup

# Create folder
#mkdir -p ./logs/LSFB/conv1d-$1/
#mkdir -p /mnt/sda2/models/HandCraft/samples/LSFB/conv1d-$1/
mkdir -p ./logs/LSFB/siMLPe-$1/
mkdir -p /mnt/sda2/models/HandCraft/samples/LSFB/siMLPe-$1/
touch ./logs/LSFB/siMLPe-$1/run0.out 
touch ./logs/LSFB/siMLPe-$1/run0.err
#mkdir -p ./logs/LSFB/CsiMLPe-$1/
#mkdir -p /mnt/sda2/models/HandCraft/samples/LSFB/CsiMLPe-$1/

# Train Model
# python src/main.py -t -l -data /mnt/sda2/datasets/isolated-cont-sl/LSFB/ -cfg ./src/configs/LSFB/conv1d/$1.yaml -save /mnt/sda2/models/HandCraft/samples/LSFB/conv1d-$1/ --project handcraft --num_workers 4 --prefetch_factor 2 -every 1 --print_every 1 -mpc > ./logs/LSFB/conv1d-$1/run0.out 2> ./logs/LSFB/conv1d-$1/run0.err
#python src/main.py --mode prediction -t -data /mnt/sda2/datasets/isolated-cont-sl/LSFB/ -cfg ./src/configs/LSFB/siMLPe/$1.yaml -save /mnt/sda2/models/HandCraft/samples/LSFB/siMLPe-$1/ --project handcraft-siMLPe --num_workers 4 --prefetch_factor 2 -every 1 --print_every 1 -mpc > ./logs/LSFB/siMLPe-$1/run0.out 2> ./logs/LSFB/siMLPe-$1/run0.err
#python src/main.py --mode cond_prediction -t -data /mnt/sda2/datasets/isolated-cont-sl/LSFB/ -cfg ./src/configs/LSFB/CsiMLPe/$1.yaml -save /mnt/sda2/models/HandCraft/samples/LSFB/CsiMLPe-$1/ --project handcraft-CsiMLPe --num_workers 4 --prefetch_factor 2 -every 1 --print_every 1 -mpc > ./logs/LSFB/CsiMLPe-$1/run0.out 2> ./logs/LSFB/CsiMLPe-$1/run0.err
python src/main.py --mode prediction -t -data /mnt/sda2/datasets/isolated-cont-sl/LSFB/ -cfg ./src/configs/LSFB/siMLPe/$1.yaml -save /mnt/sda2/models/HandCraft/samples/LSFB/siMLPe-$1/ --project handcraft-siMLPe --num_workers 4 --prefetch_factor 2 -every 1 --print_every 1 -mpc > ./logs/LSFB/siMLPe-$1/run0.out 2> ./logs/LSFB/siMLPe-$1/run0.err

# Save dataset
#python src/main.py --mode cond_prediction -sd --sd_num 10 -data /mnt/sda2/datasets/isolated-cont-sl/LSFB/ -cfg ./src/configs/LSFB/CsiMLPe/$1.yaml -save /mnt/sda2/models/HandCraft/samples/LSFB/CsiMLPe-$1/ -ckpt /mnt/sda2/models/HandCraft/samples/LSFB/CsiMLPe-vanilla/checkpoints/LSFB-vanilla-train-2024_05_21_19_17_26/ -best --project handcraft-CsiMLPe --num_workers 4 --prefetch_factor 2 -mpc > ./logs/LSFB/CsiMLPe-$1/run0.out 2> ./logs/LSFB/CsiMLPe-$1/run0.err