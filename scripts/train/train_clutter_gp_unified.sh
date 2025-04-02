#!/bin/bash
# set -x
gpu=1

data_path=PATH_OF_DATA
log_suffix=a2

CUDA_VISIBLE_DEVICES=$gpu python a2/train/main.py --lr 1e-4 --use_rope --data_path $data_path --log_suffix $log_suffix
