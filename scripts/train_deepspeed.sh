#!/bin/bash

# Uncomment and adjust as you may need

#source ~/miniconda3/bin/activate ~/miniconda3/envs/vgen12.3
#export CUDA_HOME="$HOME/miniconda3/envs/vgen12.3"
#export LD_LIBRARY_PATH="$HOME/miniconda3/envs/vgen12.3/lib:$LD_LIBRARY_PATH"


deepspeed --master_port 29501 train.py --cfg configs/t2v_train_deepspeed.yaml

