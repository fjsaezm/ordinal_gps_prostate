#!/bin/bash

# Launch SORD, Beta, POE, CORE, DRF, and CORF through the standard Hydra entry point.
# Environment variables can override every sweep dimension. For example:
#
# GPU_IDS="0,1,2,3" MAX_PROCESSES=4 SKIP_EXISTING=true \
#     bash bash_scripts/run_ordinal_baselines.sh
#
# Set IGNORE_EXTERNAL_GPU_PROCESSES=true to use the selected GPUs even when
# nvidia-smi reports memory allocated by processes that this launcher did not start.

set -u

WANDB_PROJECT=${WANDB_PROJECT:-"OGP"}
MODELS_STRING=${MODELS:-"sord beta poe core drf corf"}
DATASETS_STRING=${DATASETS:-"sicap crowdgleason"}
FEATURES_STRING=${FEATURES:-"UNI2 PHIKONV2 VIRCHOW VIRCHOW2 UNI GIGAPATH CONCHV15"}
SEEDS_STRING=${SEEDS:-"0 1 2 3 4"}
PYTHON_BIN=${PYTHON_BIN:-"python"}
EPOCHS=${EPOCHS:-500}
BATCH_SIZE=${BATCH_SIZE:-128}
LEARNING_RATE=${LEARNING_RATE:-0.0001}
PATIENCE=${PATIENCE:-50}
PERCENTAGE=${PERCENTAGE:-100}
WANDB_ENABLED=${WANDB_ENABLED:-true}
SAVE_WEIGHTS=${SAVE_WEIGHTS:-true}

# Scheduling options.
# GPU_IDS accepts comma- or space-separated physical GPU indices. When omitted,
# inherit the launcher's CUDA_VISIBLE_DEVICES, or use every GPU from nvidia-smi.
GPU_IDS_STRING=${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-}}
MAX_PROCESSES_VALUE=${MAX_PROCESSES:-}
SKIP_EXISTING=${SKIP_EXISTING:-false}
IGNORE_EXTERNAL_GPU_PROCESSES=${IGNORE_EXTERNAL_GPU_PROCESSES:-false}
MIN_FREE_GPU_MEMORY_PERCENT=${MIN_FREE_GPU_MEMORY_PERCENT:-60}
GPU_POLL_INTERVAL=${GPU_POLL_INTERVAL:-30}

is_true() {
    case "${1,,}" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

read -r -a models <<< "$MODELS_STRING"
read -r -a datasets <<< "$DATASETS_STRING"
read -r -a features <<< "$FEATURES_STRING"
read -r -a seeds <<< "$SEEDS_STRING"

if [ -z "$GPU_IDS_STRING" ]; then
    GPU_IDS_STRING=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | paste -sd, -)
fi

GPU_IDS_STRING=${GPU_IDS_STRING//,/ }
read -r -a raw_gpu_ids <<< "$GPU_IDS_STRING"
gpu_ids=()
for gpu_id in "${raw_gpu_ids[@]}"; do
    gpu_id=${gpu_id//[[:space:]]/}
    if [ -n "$gpu_id" ]; then
        gpu_ids+=("$gpu_id")
    fi
done

if [ "${#gpu_ids[@]}" -eq 0 ]; then
    echo "Error: no GPUs found. Set GPU_IDS to the physical GPU indices to use." >&2
    exit 1
fi

if [ -z "$MAX_PROCESSES_VALUE" ]; then
    MAX_PROCESSES_VALUE=${#gpu_ids[@]}
fi
if ! [[ "$MAX_PROCESSES_VALUE" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: MAX_PROCESSES must be a positive integer." >&2
    exit 1
fi
if ! [[ "$MIN_FREE_GPU_MEMORY_PERCENT" =~ ^[0-9]+$ ]] \
        || [ "$MIN_FREE_GPU_MEMORY_PERCENT" -gt 100 ]; then
    echo "Error: MIN_FREE_GPU_MEMORY_PERCENT must be an integer from 0 to 100." >&2
    exit 1
fi

mkdir -p logs/ordinal_baselines

# A selected GPU is assigned to at most one child of this launcher at a time.
# PIDs already present on a GPU are considered external and are handled by
# IGNORE_EXTERNAL_GPU_PROCESSES.
declare -A gpu_pids=()

refresh_launcher_jobs() {
    local gpu_id pid
    for gpu_id in "${gpu_ids[@]}"; do
        pid=${gpu_pids[$gpu_id]:-}
        if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null || true
            unset 'gpu_pids[$gpu_id]'
        fi
    done
}

launcher_job_count() {
    local gpu_id count=0
    for gpu_id in "${gpu_ids[@]}"; do
        if [ -n "${gpu_pids[$gpu_id]:-}" ]; then
            count=$((count + 1))
        fi
    done
    echo "$count"
}

gpu_has_enough_free_memory() {
    local requested_gpu=$1
    local used total

    if is_true "$IGNORE_EXTERNAL_GPU_PROCESSES"; then
        return 0
    fi

    read -r used total < <(
        nvidia-smi --query-gpu=memory.used,memory.total \
            --format=csv,noheader,nounits -i "$requested_gpu" 2>/dev/null \
            | tr -d ' ' | tr ',' ' '
    )
    if ! [[ "${used:-}" =~ ^[0-9]+$ && "${total:-}" =~ ^[1-9][0-9]*$ ]]; then
        return 1
    fi

    [ $(( (total - used) * 100 )) -ge $(( MIN_FREE_GPU_MEMORY_PERCENT * total )) ]
}

gpu_has_external_compute_processes() {
    local requested_gpu=$1
    local process_rows

    if is_true "$IGNORE_EXTERNAL_GPU_PROCESSES"; then
        return 1
    fi

    process_rows=$(nvidia-smi -i "$requested_gpu" \
        --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)
    [ -n "$process_rows" ]
}

choose_gpu() {
    local gpu_id
    refresh_launcher_jobs

    if [ "$(launcher_job_count)" -ge "$MAX_PROCESSES_VALUE" ]; then
        return 1
    fi

    for gpu_id in "${gpu_ids[@]}"; do
        if [ -z "${gpu_pids[$gpu_id]:-}" ] \
                && ! gpu_has_external_compute_processes "$gpu_id" \
                && gpu_has_enough_free_memory "$gpu_id"; then
            echo "$gpu_id"
            return 0
        fi
    done
    return 1
}

wait_for_gpu_slot() {
    local selected_gpu
    while true; do
        if selected_gpu=$(choose_gpu); then
            echo "$selected_gpu"
            return 0
        fi
        echo "No selected GPU is currently available; checking again in ${GPU_POLL_INTERVAL}s." >&2
        sleep "$GPU_POLL_INTERVAL"
    done
}

echo "Selected GPUs: ${gpu_ids[*]}"
echo "Maximum concurrent launcher jobs: $MAX_PROCESSES_VALUE"
echo "Skip existing logs: $SKIP_EXISTING"
echo "Ignore external GPU processes: $IGNORE_EXTERNAL_GPU_PROCESSES"

launched=0
skipped=0
for dataset in "${datasets[@]}"; do
    for feature in "${features[@]}"; do
        for model in "${models[@]}"; do
            for seed in "${seeds[@]}"; do
                log_file="logs/ordinal_baselines/${dataset}_${feature}_${model}_seed${seed}.log"
                if is_true "$SKIP_EXISTING" && [ -e "$log_file" ]; then
                    echo "Skipping existing run: model=${model}, dataset=${dataset}, feature=${feature}, seed=${seed}"
                    skipped=$((skipped + 1))
                    continue
                fi

                gpu_id=$(wait_for_gpu_slot)
                echo "Launching model=${model}, dataset=${dataset}, feature=${feature}, seed=${seed}, GPU=${gpu_id}"
                CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" -u src/experiments/main.py \
                    wandb.project="$WANDB_PROJECT" \
                    dataset="$dataset" \
                    dataset.features="$feature" \
                    dataset.percentage="$PERCENTAGE" \
                    model="$model" \
                    seed="$seed" \
                    training.epochs="$EPOCHS" \
                    training.batch_size="$BATCH_SIZE" \
                    training.learning_rate="$LEARNING_RATE" \
                    training.patience="$PATIENCE" \
                    training.log_interval=10 \
                    training.lr_scheduler=none \
                    wandb.enabled="$WANDB_ENABLED" \
                    output.save_weights="$SAVE_WEIGHTS" \
                    > "$log_file" 2>&1 &
                gpu_pids[$gpu_id]=$!
                launched=$((launched + 1))
            done
        done
    done
done

for gpu_id in "${gpu_ids[@]}"; do
    pid=${gpu_pids[$gpu_id]:-}
    if [ -n "$pid" ]; then
        wait "$pid" || true
    fi
done

echo "All ordinal baseline runs completed. Launched: $launched; skipped: $skipped."
