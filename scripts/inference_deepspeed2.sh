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


deepspeed --master_port $(shuf -i 20000-30000 -n 1) inference.py --cfg configs/t2v_inference_deepspeed2.yaml

