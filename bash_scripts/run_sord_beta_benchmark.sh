#!/bin/bash

# SORD and Beta under the same optimization conditions as the paper's
# OrdinalMLP: [100]-ReLU MLP, Adam, lr=0.01, batch=4096, 500 epochs,
# training-loss early stopping with patience 200, and no validation set.

set -u

export MODELS="sord beta"
export DATASETS="sicap crowdgleason"
export FEATURES="CONCHV15 GIGAPATH PHIKONV2 UNI UNI2 VIRCHOW VIRCHOW2"
export SEEDS="0 1 2 3 4"

export EPOCHS=500
export BATCH_SIZE=4096
export LEARNING_RATE=0.01
export PATIENCE=200

export WANDB_PROJECT=${WANDB_PROJECT:-"OGP"}
export PYTHON_BIN=${PYTHON_BIN:-"/home/fjaviersaezm/.conda/envs/GPFLOW/bin/python"}
export MAX_PROCESSES=${MAX_PROCESSES:-4}
export PERCENTAGE=100
export WANDB_ENABLED=true
export SAVE_WEIGHTS=true

bash bash_scripts/run_ordinal_baselines.sh
