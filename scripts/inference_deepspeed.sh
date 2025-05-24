#!/bin/bash



deepspeed --master_port $(shuf -i 20000-30000 -n 1) inference.py --cfg configs/t2v_inference_deepspeed.yaml

