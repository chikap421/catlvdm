#!/bin/bash
source ~/miniconda3/bin/activate ~/miniconda3/envs/demo
export CUDA_HOME="$HOME/miniconda3/envs/demo"
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/demo/lib:$LD_LIBRARY_PATH"

# cd ..

deepspeed --master_port 29502 --include="localhost:5" inference.py --cfg configs/t2v_inference_deepspeed.yaml

