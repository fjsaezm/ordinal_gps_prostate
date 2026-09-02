#!/bin/bash

# Script to run experiments using multiple sklearn models, datasets, and folds
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=5,6

# List of models and datasets
model_names=("RLOrdinal" "LogisticRegression" "MLP" "SVM" "KNN" "NaiveBayes" "RandomForest" "OrdinalSVGP" "SVGP_multiclass")
#dataset_names=("sicap" "crowdgleason")  # Add your dataset names here
model_names=("SVM" "RandomForest" "MLP")
dataset_names=("sicap-features_CONCHV15")
fold_values=(0)

# Path to the Python script
PYTHON_SCRIPT="src/experiments/main.py"

# Loop over models, datasets, and folds
for model_name in "${model_names[@]}"; do
    for dataset in "${dataset_names[@]}"; do
        for fold in "${fold_values[@]}"; do
            echo "Running model=${model_name}, dataset=${dataset}, fold=${fold}"

            COMMAND="nohup python ${PYTHON_SCRIPT} \
                --model_name ${model_name} \
                --fold ${fold} \
                --dataset ${dataset} \
                --batch_size 4096 \
                --epochs 5000 \
                --patience 100 \
                --learning_rate 0.1 \
                --num_inducing 100 \
                --init_inducing_points random \
                --lengthscale 5.0 \
                > logs/${model_name}_${dataset}_fold${fold}.out 2>&1"

            echo "Command: ${COMMAND}"
            eval "${COMMAND}"

            # Optional: add a short sleep to avoid collisions
            sleep 1
        done
    done
done

echo "All experiments have been launched."
