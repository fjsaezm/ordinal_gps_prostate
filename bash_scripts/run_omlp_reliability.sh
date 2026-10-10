#!/bin/bash

# Retrain the ten VIRCHOW2 OMLP folds needed by the reliability diagrams.
# The epoch counts reproduce the historical OMLP configurations: 500 for
# SICAPv2 and 5000 for CrowdGleason. Environment variables may override the
# scheduler settings in the same way as run_ordinal_baselines.sh.

set -u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

export MODELS=ordinal_mlp
export DATASETS=sicap
export FEATURES=VIRCHOW2
export SEEDS=${SEEDS:-"0 1 2 3 4"}
export EPOCHS=${SICAP_EPOCHS:-500}
export BATCH_SIZE=${BATCH_SIZE:-4096}
export LEARNING_RATE=${LEARNING_RATE:-0.01}
export PATIENCE=${PATIENCE:-200}
export WANDB_PROJECT=${WANDB_PROJECT:-OGP}
export SAVE_WEIGHTS=true
export PYTHON_BIN=${PYTHON_BIN:-/home/fjaviersaezm/.conda/envs/GPFLOW_GPU/bin/python}

# The GPU environment carries the CUDA 11 runtime expected by TensorFlow 2.8,
# but invoking its Python executable directly does not add that directory to
# the dynamic-linker search path.
python_env_dir=$(cd "$(dirname "$PYTHON_BIN")/.." && pwd)
if [ -d "$python_env_dir/lib" ]; then
    export LD_LIBRARY_PATH="$python_env_dir/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

cd "$REPO_ROOT" || exit 1
bash bash_scripts/run_ordinal_baselines.sh

export DATASETS=crowdgleason
export EPOCHS=${CROWDGLEASON_EPOCHS:-5000}
bash bash_scripts/run_ordinal_baselines.sh
