#!/bin/bash
# set -x
gpu=0

model_path=a2_pretrained/checkpoints/sl_checkpoint_199.pth
log=a2

echo $model_path
echo seen
CUDA_VISIBLE_DEVICES=$gpu python a2/evaluate/test_pick.py --use_rope --load_model --model_path $model_path --log_suffix grasp-$log --testing_case_dir testing_cases/grasp_testing_cases/seen
echo unseen
CUDA_VISIBLE_DEVICES=$gpu python a2/evaluate/test_pick.py --use_rope --load_model --model_path $model_path --log_suffix grasp-$log-unseen --testing_case_dir testing_cases/grasp_testing_cases/unseen
