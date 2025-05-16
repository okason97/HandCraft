#!/bin/bash
# nohup

# $1 = mode (prediction, cond_prediction, classification), $2 = model (conv1d/siMLPe/CsiMLPe), $3 = config
# Create folder
echo making dir $2-$3
mkdir -p ./logs/$4/$2-$3/
mkdir -p /disco1/models/HandCraft/samples/$4/$2-$3/
touch ./logs/$4/$2-$3/run0.out 
touch ./logs/$4/$2-$3/run0.err

# Save dataset
echo Saving generated dataset
python src/main.py --mode $1 -sd -data /disco1/datasets/$4/ -cfg ./src/configs/$4/$2/$3.yaml -save /disco1/models/HandCraft/samples/$4/$2-$3/ -best --project handcraft-$2-$4-generate --num_workers 4 --prefetch_factor 2 -mpc "${@:5}" > ./logs/$4/$2-$3/run0.out 2> ./logs/$4/$2-$3/run0.err