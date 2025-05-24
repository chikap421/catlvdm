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

CONFIG="configs/t2v_inference_deepspeed2.yaml"

# Study: Varying DDIM steps
TECHNIQUE=("gaussian")
NOISES=("20")
GUIDANCES=(8.0 9.0 10.0 11 12 13 14 15)

for TECH in "${TECHNIQUE[@]}"; do
    echo "[`date '+%Y-%m-%d %H:%M:%S'`] >>> Running technique: $TECH"
    if [ "$TECH" = "results_2M_train" ]; then
        for GS in "${GUIDANCES[@]}"; do
            OUTPUT_FOLDER="/sciclone/data10/ccmaduabuchi/inference/guidance_scale/msrvtt/${TECH}_guidance${GS}"

            if [ -d "$OUTPUT_FOLDER" ]; then
                echo "[`date '+%Y-%m-%d %H:%M:%S'`] Skipping ${OUTPUT_FOLDER} — already exists."
                continue
            fi

            sed -i -E "s|'resume_checkpoint': '[^']+',|'resume_checkpoint': '/sciclone/data10/ccmaduabuchi/workspace/corruption/${TECH}/checkpoints/${TECH}.pth',|g" "$CONFIG"
            sed -i -E "s|'pretrained': '[^']+_motion_encoder\.pth',|'pretrained': '/sciclone/data10/ccmaduabuchi/workspace/corruption/${TECH}/checkpoints/${TECH}_motion_encoder.pth',|g" "$CONFIG"
            sed -i -E "s|guide_scale: [0-9.]+|guide_scale: ${GS}|g" "$CONFIG"
            sed -i -E "s|ddim_timesteps: [0-9]+|ddim_timesteps: 50|g" "$CONFIG"
            sed -i -E "s|^(\s*log_dir:\s*\").*(\")|\1${OUTPUT_FOLDER}\2|g" "$CONFIG"

            echo "[`date '+%Y-%m-%d %H:%M:%S'`] Running ${TECH} with guide_scale=${GS}"
            bash scripts/inference_deepspeed2.sh
        done
    else
        for NOISE in "${NOISES[@]}"; do
            for GS in "${GUIDANCES[@]}"; do
                OUTPUT_FOLDER="/sciclone/data10/ccmaduabuchi/inference/guidance_scale/msrvtt/${TECH}_${NOISE}_guidance${GS}"

                if [ -d "$OUTPUT_FOLDER" ]; then
                    echo "[`date '+%Y-%m-%d %H:%M:%S'`] Skipping ${OUTPUT_FOLDER} — already exists."
                    continue
                fi

                sed -i -E "s|'resume_checkpoint': '[^']+',|'resume_checkpoint': '/sciclone/data10/ccmaduabuchi/workspace/corruption/${TECH}_${NOISE}/checkpoints/${TECH}_${NOISE}.pth',|g" "$CONFIG"
                sed -i -E "s|'pretrained': '[^']+_motion_encoder\.pth',|'pretrained': '/sciclone/data10/ccmaduabuchi/workspace/corruption/${TECH}_${NOISE}/checkpoints/${TECH}_${NOISE}_motion_encoder.pth',|g" "$CONFIG"
                sed -i -E "s|guide_scale: [0-9.]+|guide_scale: ${GS}|g" "$CONFIG"
                sed -i -E "s|ddim_timesteps: [0-9]+|ddim_timesteps: 50|g" "$CONFIG"
                sed -i -E "s|^(\s*log_dir:\s*\").*(\")|\1${OUTPUT_FOLDER}\2|g" "$CONFIG"

                echo "[`date '+%Y-%m-%d %H:%M:%S'`] Running ${TECH}_${NOISE} with guide_scale=${GS}"
                bash scripts/inference_deepspeed2.sh
            done
        done
    fi
done