#!/bin/bash
# nohup

mkdir -p ./logs/LSFB/conv1d-$1/s
mkdir -p /mnt/sda2/models/HandCraft/samples/LSFB/conv1d-$1/
# python src/main.py -t -data /mnt/sda2/datasets/isolated-cont-sl/LSFB/ -cfg ./src/configs/LSFB/conv1d/$1.yaml -save /mnt/sda2/models/HandCraft/samples/LSFB/conv1d-$1/ --project handcraft --num_workers 4 --prefetch_factor 20 -every 1 --print_every 1 -mpc > ./logs/LSFB/conv1d-$1/run0.out 2> ./logs/LSFB/conv1d-$1/run0.err
python src/main.py -t -l -data /mnt/sda2/datasets/isolated-cont-sl/LSFB/ -cfg ./src/configs/LSFB/conv1d/$1.yaml -save /mnt/sda2/models/HandCraft/samples/LSFB/conv1d-$1/ --project handcraft --num_workers 4 --prefetch_factor 2 -every 1 --print_every 1 -mpc > ./logs/LSFB/conv1d-$1/run0.out 2> ./logs/LSFB/conv1d-$1/run0.err