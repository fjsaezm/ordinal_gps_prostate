#!/bin/bash

# Script to run SVGP experiments using OmegaConf/Hydra with GPU allocation
# Usage: ./run_multiclass_svgp


export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Remove the hardcoded CUDA_VISIBLE_DEVICES to allow GPU management to work
export CUDA_VISIBLE_DEVICES=0
source "bash_scripts/GPU_management.sh"

MODEL_NAME=${1:-"svgp_multiclass"}
WANDB_PROJECT="OGP"
mkdir -p logs


# Function to run a single experiment
run_experiment() {
    local dataset=$1
    local feature=$2
    local gpu_id=$3
    local model_name=$4
    local class_weights=$5
    local seed=$6
    local lr=$7
    local percentage=$8
    local noise_level=$9
    
    mkdir -p logs/${dataset}/${model_name}/

    local log_file="logs/${dataset}/${model_name}/${feature}_seed${seed}_${percentage}.log"
    echo "Running ${model_name}: dataset=$dataset, feature=$feature on GPU $gpu_id"
    
    CUDA_VISIBLE_DEVICES=$gpu_id nohup python -u src/experiments/main.py \
        wandb.project=$WANDB_PROJECT \
        dataset=$dataset \
        dataset.features=$feature \
        dataset.percentage=$percentage \
        dataset.noise_level=$noise_level \
        model=$model_name \
        seed=$seed \
        training.learning_rate=$lr \
        training.epochs=500 \
        training.patience=50 \
        training.batch_size=4096 \
        training.log_interval=10 \
        training.class_weights=None \
        training.lr_scheduler=None\
        training.lr_scheduler_last_epoch=2000 \
        model.inducing_points.num_inducing=50 \
        model.inducing_points.init_method=kmeans \
        model.inner_mean_function=zero \
        model.kernel.type=RBF \
        model.kernel.lengthscale=scale \
        > $log_file 2>&1 &
        
    
    local pid=$!
    echo "  Started with PID $pid, GPU $gpu_id, log: $log_file"
    
    # Brief pause to avoid overwhelming the system
    sleep 20
}

echo "Starting ${MODEL_NAME} experiments with GPU allocation..."

# Define experiment parameters
datasets=("crowdgleason_EM")
features=( "CONCHV15"  "UNI2" "PHIKONV2"  "VIRCHOW" "VIRCHOW2" "UNI" "GIGAPATH")


seeds=(0 1 2 3 4)
percentages=(100)
class_weight="None"
noise_levels=(0)


# Counter for active jobs
active_jobs=0
max_jobs=4


# Run experiments
for dataset in "${datasets[@]}"; do
    for feature in "${features[@]}"; do
        for noise_level in "${noise_levels[@]}"; do

            for percentage in "${percentages[@]}"; do
                for seed in "${seeds[@]}"; do
                    gpu_id=$(wait_for_gpu)
                    # Learning rate dependent on dataset
                    if [ "$dataset" == "crowdgleason" ]; then
                        lr=0.01
                    elif [ "$dataset" == "crowdgleason_EM" ]; then
                        lr=0.01
                    else
                        lr=0.1
                    fi

                    # Run the experiment
                    run_experiment $dataset $feature $gpu_id $MODEL_NAME $class_weight $seed $lr $percentage $noise_level
                done
            done
        done
        wait
    done
done
       
# Wait for all background jobs to complete
wait
echo "All experiments completed!"
