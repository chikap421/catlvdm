#!/bin/bash


deepspeed --master_port $(shuf -i 20000-30000 -n 1) train.py --cfg configs/t2v_train_deepspeed.yaml
