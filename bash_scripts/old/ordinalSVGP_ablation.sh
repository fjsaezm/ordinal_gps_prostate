#!/bin/bash

# Script to run experiments using multiple sklearn models, datasets, and folds

# List of models and datasets
model_names=("OrdinalSVGP")
dataset_names=("crowdgleason_UNI_full")
fold_values=(0 1 2 3)

# Values to vary
num_inducing_values=(200)
init_inducing_points_values=("kmeans")
kernel_types=("RBF")
lengthscale_values=("scale")
learning_rate_values=(0.01)
prior_temp_values=(1.0)  # <<< New parameter

# Path to the Python script
PYTHON_SCRIPT="src/experiments/main.py"

total_combinations=$(( ${#model_names[@]} * ${#dataset_names[@]} * ${#fold_values[@]} * ${#num_inducing_values[@]} * ${#init_inducing_points_values[@]} * ${#kernel_types[@]} * ${#lengthscale_values[@]} * ${#learning_rate_values[@]} * ${#prior_temp_values[@]} ))
echo "Total combinations to run: ${total_combinations}"
run_configs=0

# Loop over all hyperparameter combinations
for model_name in "${model_names[@]}"; do
    for dataset in "${dataset_names[@]}"; do
        for fold in "${fold_values[@]}"; do
            for num_inducing in "${num_inducing_values[@]}"; do
                for init_inducing in "${init_inducing_points_values[@]}"; do
                    for kernel_type in "${kernel_types[@]}"; do
                        for lengthscale in "${lengthscale_values[@]}"; do
                            for learning_rate in "${learning_rate_values[@]}"; do
                                for prior_temp in "${prior_temp_values[@]}"; do

                                    run_configs=$((run_configs + 1))
                                    echo "Running configuration ${run_configs}/${total_combinations}"

                                    echo "Running model=${model_name}, dataset=${dataset}, fold=${fold}, num_inducing=${num_inducing}, init_inducing=${init_inducing}, kernel_type=${kernel_type}, lengthscale=${lengthscale}, lr=${learning_rate}, prior_temp=${prior_temp}"

                                    LOG_FILE="logs/${model_name}_${dataset}_fold${fold}_inducing${num_inducing}_${init_inducing}_${kernel_type}_lengthscale${lengthscale}_lr${learning_rate}_temp${prior_temp}.out"

                                    COMMAND="nohup python ${PYTHON_SCRIPT} \
                                        --model_name ${model_name} \
                                        --fold ${fold} \
                                        --dataset ${dataset} \
                                        --batch_size 4096 \
                                        --epochs ${epochs} \
                                        --patience 50 \
                                        --learning_rate ${learning_rate} \
                                        --num_inducing ${num_inducing} \
                                        --init_inducing_points ${init_inducing} \
                                        --kernel_type ${kernel_type} \
                                        --lengthscale ${lengthscale} \
                                        --prior_temp ${prior_temp} \
                                        > ${LOG_FILE} 2>&1"

                                    echo "Command: ${COMMAND}"
                                    eval "${COMMAND}"

                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

echo "All experiments have been launched."
