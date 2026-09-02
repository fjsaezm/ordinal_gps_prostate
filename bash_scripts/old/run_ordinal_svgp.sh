#!/bin/bash

# Este script ejecuta train2.py con diferentes combinaciones de parámetros.

# Definir los valores para cada parámetro
epochs_values=(10000)
learning_rate_values=(0.1)
num_inducing_values=(500)
fold_values=(0 1 2 3 4)
kernel_type_values=("RBF")
inducing_points_method=("random")  # Aquí defines los métodos
dataset_name="grx"  # ← CORREGIDO: no uses paréntesis ni espacios

# Nombre del script Python a ejecutar
PYTHON_SCRIPT="train2.py"

# Crear carpeta de logs si no existe
mkdir -p logs

echo "Iniciando la ejecución de experimentos..."
echo "-----------------------------------------"

# Bucle anidado para iterar sobre todas las combinaciones de parámetros
for epochs in "${epochs_values[@]}"; do
    for lr in "${learning_rate_values[@]}"; do
        for num_ind in "${num_inducing_values[@]}"; do
            for fold in "${fold_values[@]}"; do
                for kernel_type in "${kernel_type_values[@]}"; do
                    for method in "${inducing_points_method[@]}"; do
                        echo "Ejecutando con: epochs=${epochs}, lr=${lr}, num_inducing=${num_ind}, fold=${fold}, kernel_type=${kernel_type}, inducing_points_method=${method}"

                        # Construir el comando completo
                        COMMAND="nohup python ${PYTHON_SCRIPT} \
                        --dataset_name ${dataset_name} \
                        --epochs ${epochs} \
                        --learning_rate ${lr} \
                        --num_inducing ${num_ind} \
                        --fold ${fold} \
                        --kernel_type ${kernel_type} \
                        --inducing_points_method ${method}"

                        echo "EJECUTANDO: ${COMMAND}"
                        eval "${COMMAND}"
                        echo "-----------------------------------------"
                    done
                done
            done
        done
    done
done

echo "Todos los experimentos han sido lanzados."
#nohup bash run_all.sh &
