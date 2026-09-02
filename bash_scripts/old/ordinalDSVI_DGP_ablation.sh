#!/bin/bash

# Script to run experiments using multiple sklearn models, datasets, and folds
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1



# List of models and datasets
model_names=("OrdinalDSVI_DGP")
model_names=("OrdinalSVGP")
dataset_names=("sicap_UNI_full")
dataset_names=("crowdgleason_UNI_full")
fold_values=(0 1 2 3 4)
batch_size=1024
epochs=5000


# Values to vary
num_inducing_values=(200)
init_inducing_points_values=("kmeans")
kernel_types=("RBF")
lengthscale_values=("scale")
learning_rate_values=(0.1 0.01)
prior_temp_values=(1.0) 
num_layers_values=(2 3)  # <-- Added num_layers values
patience=100

# Path to the Python script
PYTHON_SCRIPT="src/experiments/main.py"

total_combinations=$(( ${#model_names[@]} * ${#dataset_names[@]} * ${#fold_values[@]} * ${#num_inducing_values[@]} * ${#init_inducing_points_values[@]} * ${#kernel_types[@]} * ${#lengthscale_values[@]} * ${#learning_rate_values[@]} * ${#prior_temp_values[@]} * ${#num_layers_values[@]} ))
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
                                    for num_layers in "${num_layers_values[@]}"; do

                                        run_configs=$((run_configs + 1))
                                        echo "Running configuration ${run_configs}/${total_combinations}"

                                        echo "Running model=${model_name}, dataset=${dataset}, fold=${fold}, num_inducing=${num_inducing}, init_inducing=${init_inducing}, kernel_type=${kernel_type}, lengthscale=${lengthscale}, lr=${learning_rate}, prior_temp=${prior_temp}, num_layers=${num_layers}"

                                        LOG_FILE="logs/${model_name}_${dataset}_fold${fold}_inducing${num_inducing}_${init_inducing}_${kernel_type}_lengthscale${lengthscale}_lr${learning_rate}_temp${prior_temp}_layers${num_layers}.out"

                                        COMMAND="nohup python ${PYTHON_SCRIPT} \
                                            --model_name ${model_name} \
                                            --fold ${fold} \
                                            --dataset ${dataset} \
                                            --batch_size ${batch_size} \
                                            --epochs ${epochs} \
                                            --patience ${patience} \
                                            --learning_rate ${learning_rate} \
                                            --num_inducing ${num_inducing} \
                                            --init_inducing_points ${init_inducing} \
                                            --kernel_type ${kernel_type} \
                                            --lengthscale ${lengthscale} \
                                            --prior_temp ${prior_temp} \
                                            --num_layers ${num_layers} \
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
done

echo "All experiments have been launched."
