#!/bin/bash

# Script to run baseline model experiments using for loops instead of Hydra
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6,7

echo "Running baseline model experiments with for loops..."

# Define experiment parameters
WANDB_PROJECT="OGP"
DATASET="crowdgleason"
FEATURES=("UNI2" "PHIKONV2"  "VIRCHOW" "VIRCHOW2" "UNI" "GIGAPATH" "CONCHV15") # CONCHV15 done

MODELS=("rl_ordinal" "mlp" "logistic_regression" "knn" "random_forest" "svm")
seeds=(0 1 2 3 4)
LEARNING_RATE=0.01
EPOCHS=500
percentages=(100)

# Create output directory
mkdir -p logs

echo "=== Running Comprehensive Baseline Sweep ==="

# Main experiment loops
for feature in "${FEATURES[@]}"; do
    echo "Processing features: ${feature}"
    
    for model in "${MODELS[@]}"; do
        echo "  Running model: ${model}"

        for percentage in "${percentages[@]}"; do
                echo "    Data percentage: ${percentage}%"

                for seed in "${seeds[@]}"; do

                    # Create unique output directory for this combination
                    exp_output_dir="logs/${DATASET}/${model}/"
                    mkdir -p ${exp_output_dir}

                    # Run experiment
                    python -u src/experiments/main.py \
                        wandb.project=${WANDB_PROJECT} \
                        dataset=${DATASET} \
                        dataset.features=${feature} \
                        dataset.percentage=${percentage} \
                        model=${model} \
                        seed=${seed} \
                        training.learning_rate=${LEARNING_RATE} \
                        training.epochs=${EPOCHS} \
                        > ${exp_output_dir}/feat${feature}_seed${seed}_${percentage}.log 2>&1 &
                    
                    # Optional: Wait a bit between launches to avoid resource conflicts
                    sleep 2
                done
        done
        wait
    done
    
    # Optional: Wait for all jobs in this feature group to complete before starting next
    # wait
done

echo "All baseline model experiments launched in background."
echo "Check logs/ directory for individual experiment logs."
echo "Results will be saved in ${OUTPUT_DIR}/"
echo ""
echo "To monitor progress, use:"
echo "  tail -f logs/*.log"
echo "  ps aux | grep python"
