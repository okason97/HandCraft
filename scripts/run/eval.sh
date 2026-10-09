#!/bin/bash
# nohup

# $1 = mode (prediction, cond_prediction, classification), $2 = model (conv1d/siMLPe/CsiMLPe/ViT/mamba/stgcn), $3 = config, $4 = dataset
# Any other argument is passed to src/main.py (--seed 42, -data <dir>, -s_data <dir>, ...)
# The directory with the datasets and the directory for checkpoints and generated datasets are read from
# HANDCRAFT_DATA and HANDCRAFT_SAVE, or from --data-root <dir> and --save-root <dir> anywhere in the arguments
source "$(dirname "$0")/common.sh"

# Create folder
echo making dir $2-$3
mkdir -p ./logs/$4/$2-$3/
mkdir -p $SAVE_DIR/$4/$2-$3/

# Use the next free log number so previous runs (other seeds) are not overwritten
n=0
while [ -e ./logs/$4/$2-$3/eval$n.out ]; do n=$((n+1)); done
LOG=./logs/$4/$2-$3/eval$n
echo logging to $LOG.out

# Evaluate a trained model on the test set, without training
# The checkpoint directory is passed with -ckpt <save dir>/checkpoints/<run name>/
echo Testing
python src/main.py --mode $1 --test -data $DATA_DIR/$4/ -cfg ./src/configs/$4/$2/$3.yaml -save $SAVE_DIR/$4/$2-$3/ -best --project handcraft-$2-$4 --num_workers 4 --prefetch_factor 2 -mpc "${@:5}" > $LOG.out 2> $LOG.err

# Show the test results
grep "Test of the checkpoint" $LOG.out | tail -1
