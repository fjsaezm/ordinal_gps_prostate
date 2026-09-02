#!/bin/bash

# GPU helper functions extracted from run scripts
# Provides: get_free_gpu, wait_for_gpu

#!/bin/bash

# Return the first GPU index with at least `threshold` percent free memory
get_free_gpu() {
    local threshold=${1:-60}  # GPU memory free threshold (%)

    # Query GPU memory usage: index, memory.used, memory.total (in MiB)
    gpu_info=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null)

    # If nvidia-smi failed or returned nothing, fallback to GPU 0
    if [ -z "$gpu_info" ]; then
        echo -1
        return
    fi

    while IFS=',' read -r gpu_id mem_used mem_total; do
        gpu_id=$(echo $gpu_id | tr -d ' ')
        mem_used=$(echo $mem_used | tr -d ' ')
        mem_total=$(echo $mem_total | tr -d ' ')

        if [ "$mem_total" -gt 0 ]; then
            # Use bc for floating point calculation to avoid integer division issues
            usage_percent=$(echo "scale=2; $mem_used * 100 / $mem_total" | bc)
            free_percent=$(echo "scale=2; 100 - $usage_percent" | bc)
            # Use bc for comparison to handle floating point
            is_free=$(echo "$free_percent >= $threshold" | bc -l)
            if [ "$is_free" -eq 1 ]; then
                echo $gpu_id
                return
            fi
        fi
    done <<< "$gpu_info"

    # No GPU met the threshold
    echo -1
}

# Wait until a GPU becomes available (or timeout) and return its index.
# Usage: wait_for_gpu [max_wait_seconds] [sleep_interval] [threshold]
wait_for_gpu() {
    local max_wait=${1:-3600}
    local sleep_interval=${2:-30}
    local threshold=${3:-60}
    local waited=0

    while [ $waited -lt $max_wait ]; do
        local gpu_id=$(get_free_gpu $threshold)
        if [ "$gpu_id" != "-1" ]; then
            echo $gpu_id
            return
        fi
        echo "No GPU available with >${threshold}% free memory. Waiting ${sleep_interval}s..."
        sleep $sleep_interval
        waited=$((waited + sleep_interval))
    done

    echo "Timeout waiting for available GPU. Using GPU 0 as fallback."
    echo 0
}

export -f get_free_gpu
export -f wait_for_gpu
