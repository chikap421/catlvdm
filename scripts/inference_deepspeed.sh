#!/bin/bash
source ~/miniconda3/bin/activate ~/miniconda3/envs/vgen12.3
export CUDA_HOME="$HOME/miniconda3/envs/vgen12.3"
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/vgen12.3/lib:$LD_LIBRARY_PATH"

# cd ..

deepspeed --master_port 29501 inference.py --cfg configs/t2v_inference_deepspeed.yaml

