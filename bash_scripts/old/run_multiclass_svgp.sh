#!/bin/bash

# Este script ejecuta train2.py con diferentes combinaciones de parámetros para modelos ordinales y multiclase.

# Definir los valores para cada parámetro
epochs_values=(10000)
learning_rate_values=(0.01)
num_inducing_values=(100)
fold_values=(1)
kernel_type_values=("Matern32")
inducing_points_method=("kmeans")
lengthscale_values=(1.0)  # ← AÑADIDO
DATASET_NAME="grx"

model_names=("SVGP_multiclass")
PYTHON_SCRIPT="train2.py"

echo "Iniciando la ejecución de experimentos para dataset: $DATASET_NAME"
echo "-------------------------------------"

for model_name in "${model_names[@]}"; do
    for epochs in "${epochs_values[@]}"; do
        for lr in "${learning_rate_values[@]}"; do
            for num_ind in "${num_inducing_values[@]}"; do
                for fold in "${fold_values[@]}"; do
                    for kernel_type in "${kernel_type_values[@]}"; do
                        for inducing_method in "${inducing_points_method[@]}"; do
                            for lengthscale in "${lengthscale_values[@]}"; do
                                echo "Modelo: ${model_name} | dataset=${DATASET_NAME} | epochs=${epochs}, lr=${lr}, num_inducing=${num_ind}, fold=${fold}, kernel_type=${kernel_type}, inducing_method=${inducing_method}, lengthscale=${lengthscale}"

                                COMMAND="nohup python ${PYTHON_SCRIPT} \
                                    --model_name ${model_name} \
                                    --dataset_name ${DATASET_NAME} \
                                    --epochs ${epochs} \
                                    --learning_rate ${lr} \
                                    --num_inducing ${num_ind} \
                                    --fold ${fold} \
                                    --kernel_type ${kernel_type} \
                                    --inducing_points_method ${inducing_method} \
                                    --lengthscale ${lengthscale}"

                                echo "Ejecutando: ${COMMAND}"
                                eval "${COMMAND}"
                                echo "-------------------------------------"
                            done
                        done
                    done
                done
            done
        done
    done
done

echo "Todos los experimentos han sido lanzados."

#nohup bash run_Multiclase.sh &