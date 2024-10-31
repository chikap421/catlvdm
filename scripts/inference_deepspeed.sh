#!/bin/bash

# Uncomment and adjust as you may need

#source ~/miniconda3/bin/activate ~/miniconda3/envs/demo
#export CUDA_HOME="$HOME/miniconda3/envs/demo"
#export LD_LIBRARY_PATH="$HOME/miniconda3/envs/demo/lib:$LD_LIBRARY_PATH"


deepspeed --master_port 29502 inference.py --cfg configs/t2v_inference_deepspeed.yaml

