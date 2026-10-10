#!/bin/bash

# Script to run SVGP experiments using OmegaConf/Hydra with GPU allocation
# Usage: ./run_ordinal_svgp.sh [model_name]
# Example: ./run_ordinal_svgp.sh ordinal_svgp
#          ./run_ordinal_svgp.sh svgp_multiclass


export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Remove the hardcoded CUDA_VISIBLE_DEVICES to allow GPU management to work
export CUDA_VISIBLE_DEVICES=6,7
source "bash_scripts/GPU_management.sh"

MODEL_NAME=${1:-"ordinal_svgp"}
WANDB_PROJECT="OGP"
mkdir -p logs


# Function to run a single experiment
run_experiment() {
    local dataset=$1
    local feature=$2
    local gpu_id=$3
    local model_name=$4
    local class_weights=$5
    local train_likelihood_params=$6
    local seed=$7
    local lr=$8
    local percentage=$9
    local likelihood=${10}
    local bin_size=${11}
    local bin_range=${12}
    local predict_using_sigma=${13}
    local lr_scheduler=${14}

    mkdir -p logs/${dataset}/${model_name}/

    local log_file="logs/${dataset}/${model_name}/${feature}_seed${seed}_${percentage}_a${bin_size}_r${bin_range}.log"

    echo "Running ${model_name}: dataset=$dataset, feature=$feature on GPU $gpu_id"
    
    CUDA_VISIBLE_DEVICES=$gpu_id nohup python -u src/experiments/main.py \
        wandb.project=$WANDB_PROJECT \
        dataset=$dataset \
        dataset.features=$feature \
        dataset.percentage=$percentage \
        model=$model_name \
        seed=$seed \
        training.learning_rate=$lr \
        training.epochs=5000 \
        training.patience=100 \
        training.batch_size=4096 \
        training.log_interval=10 \
        training.class_weights=$class_weights \
        training.lr_scheduler=$lr_scheduler \
        training.lr_scheduler_last_epoch=2000 \
        model.inducing_points.num_inducing=500 \
        model.inducing_points.init_method=kmeans \
        model.inner_mean_function=zero \
        model.kernel.type=RBF \
        model.kernel.lengthscale=scale \
        model.train_likelihood_params=$train_likelihood_params \
        model.likelihood=$likelihood \
        model.predict_likelihood_variance=$predict_using_sigma \
        model.bin_size=$bin_size \
        model.bin_range=$bin_range \
        model.standardize_bins=True \
        > $log_file 2>&1 &
        
    
    local pid=$!
    echo "  Started with PID $pid, GPU $gpu_id, log: $log_file"
    
    # Brief pause to avoid overwhelming the system
    sleep 5
}

echo "Starting ${MODEL_NAME} experiments with GPU allocation..."

# Define experiment parameters
datasets=( "sicap" "crowdgleason")
features=( "CONCHV15" "GIGAPATH" "VIRCHOW" "VIRCHOW2" "UNI2" "PHIKONV2"  "UNI" )


class_weights=( "None" )
train_likelihood_params=( True )
seeds=(0 1 2 3 4)
#lr=0.1 # DEFINED INSIDE LOOP. DEPENDS ON DATASET.
percentages=(100)
likelihoods=("Ordinal_trainable")
bin_sizes=(1.0) # Used as alphas
bin_ranges=(1.0) # Used to scale
lr_scheduler="cosine"


# Counter for active jobs
active_jobs=0
max_jobs=4

# Run experiments
for dataset in "${datasets[@]}"; do
    for feature in "${features[@]}"; do
        for percentage in "${percentages[@]}"; do
                for seed in "${seeds[@]}"; do
                    for likelihood in "${likelihoods[@]}"; do
                        for bin_size in "${bin_sizes[@]}"; do
                            for bin_range in "${bin_ranges[@]}"; do
                                for class_weight in "${class_weights[@]}"; do
                                    for train_likelihood_param in "${train_likelihood_params[@]}"; do
                                        # Get an available GPU
                                        gpu_id=$(wait_for_gpu)
                                        # Learning rate dependent on dataset
                                        if [ "$dataset" == "crowdgleason" ]; then
                                            lr=0.01
                                            predict_using_sigma=false
                                        else
                                            lr=0.1
                                            predict_using_sigma=true
                                        fi

                                        # Run the experiment
                                        run_experiment $dataset $feature $gpu_id $MODEL_NAME $class_weight $train_likelihood_param $seed $lr $percentage $likelihood $bin_size $bin_range $predict_using_sigma $lr_scheduler

                                done
                            done
                        done
                    done
                done
            done
        done
        wait
    done
done
       
# Wait for all background jobs to complete
wait
echo "All experiments completed!"
