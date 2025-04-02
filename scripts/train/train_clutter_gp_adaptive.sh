#!/bin/bash
# set -x
gpu=0

data_path=PATH_OF_ADAPTIVE_DATA
model_path=model_path=a2_pretrained/checkpoints/sl_checkpoint_199.pth
log=a2-adaptive

CUDA_VISIBLE_DEVICES=$gpu python a2/train/main.py --lr 1e-5 --sample_num 100 --use_rope --adaptive --data_path $data_path --load_model --model_path $model_path --log_suffix $log-place-sl-adaptive-lr-1e-5