#!/bin/bash

# Activate Conda environment
eval "$(conda shell.bash hook)"
conda activate /sciclone/home/ccmaduabuchi/miniconda3/envs/demo

# Navigate to project directory
cd /sciclone/home/ccmaduabuchi/DEMO

# Set cache directories
export TRITON_CACHE_DIR=/sciclone/home/ccmaduabuchi/.triton_cache
export TRANSFORMERS_CACHE=/sciclone/home/ccmaduabuchi/.cache/huggingface
export TORCH_HOME=/sciclone/home/ccmaduabuchi/.cache/torch
export HF_HOME=/sciclone/home/ccmaduabuchi/.cache/huggingface
export XDG_CACHE_HOME=/sciclone/home/ccmaduabuchi/.cache
mkdir -p $TRANSFORMERS_CACHE $TORCH_HOME $HF_HOME $XDG_CACHE_HOME

# Set CUDA paths
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$CUDA_HOME/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH

# Get number of available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)

# Configuration file
CONFIG="configs/t2v_train_deepspeed.yaml"
TECHNIQUES=("uniform" "gaussian")
NOISES=("0.025" "0.05" "0.075" "0.1" "0.15" "0.2")

JOB_COUNT=0

LOG_DIR_BASE=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['log_dir'])")

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