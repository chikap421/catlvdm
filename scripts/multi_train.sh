#!/bin/bash

PATH_TO_LOG_DIR="/path/to/log_dir"
PATH_TO_CONFIG="configs/t2v_train_deepspeed.yaml"

# Get number of available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)

# Configuration file
CONFIG="$PATH_TO_CONFIG"
TECHNIQUES=("uniform" "gaussian") # Specify the techniques you want to run
NOISES=("0.025" "0.05" "0.075" "0.1" "0.15" "0.2") # Specify the noise ratios you want to run (2.5% = 0.025, 5% = 0.05, etc.)

JOB_COUNT=0

LOG_DIR_BASE="$PATH_TO_LOG_DIR"

declare -a JOBS_TO_RUN=()

for TECHNIQUE in "${TECHNIQUES[@]}"; do
    for NOISE in "${NOISES[@]}"; do
        # Only allow none_0.0
        if [[ "$TECHNIQUE" == "none" && "$NOISE" != "0.0" ]]; then
            continue
        fi

        # Disallow 0.0 for other techniques
        if [[ "$TECHNIQUE" != "none" && "$NOISE" == "0.0" ]]; then
            continue
        fi
        
        FOLDER_NAME="${TECHNIQUE}_$(python3 -c "val=round(${NOISE} * 100, 1); print(int(val) if val.is_integer() else val)")"
        FULL_LOG_DIR="${LOG_DIR_BASE}/${FOLDER_NAME}"
        if [ ! -d "$FULL_LOG_DIR" ]; then
            JOBS_TO_RUN+=("${TECHNIQUE},${NOISE}")
        else
            echo "Skipping existing folder: $FULL_LOG_DIR"
        fi
    done
done

declare -a GPU_PIDS
GPU_PIDS=()

for JOB in "${JOBS_TO_RUN[@]}"; do
    while true; do
        job_launched=false
        for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
            if [ -z "${GPU_PIDS[gpu]}" ] || ! kill -0 "${GPU_PIDS[gpu]}" 2>/dev/null; then
                IFS=',' read TECHNIQUE NOISE <<< "$JOB"

                FOLDER_NAME="${TECHNIQUE}_$(python3 -c "val=round(${NOISE} * 100, 1); print(int(val) if val.is_integer() else val)")"
                FULL_LOG_DIR="${LOG_DIR_BASE}/${FOLDER_NAME}"

                sed -i -E "/embedder:/,/}/{s/('noise_type':\s*)'.*'/\1'${TECHNIQUE}'/; s/('noise_ratio':\s*)[0-9.]+/\1${NOISE}/}" "$CONFIG"
                sed -i -E "/clip_visual:/,/}/{s/('noise_type':\s*)'.*'/\1'${TECHNIQUE}'/; s/('noise_ratio':\s*)[0-9.]+/\1${NOISE}/}" "$CONFIG"
                echo "Updated noise settings: noise_type=${TECHNIQUE}, noise_ratio=${NOISE}"

                MASTER_PORT=$((RANDOM % 10000 + 20000))
                export MASTER_PORT=$MASTER_PORT

                CUDA_VISIBLE_DEVICES=$gpu bash scripts/train_deepspeed.sh &
                GPU_PIDS[gpu]=$!
                sleep 120
                job_launched=true
                break
            fi
        done
        if $job_launched; then
            break
        fi
        sleep 10
    done
done

wait