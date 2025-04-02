#!/bin/bash
# set -x
gpu=1

model_path=a2_pretrained/checkpoints/sl_checkpoint_199.pth
log=a2

echo $model_path
echo seen
CUDA_VISIBLE_DEVICES=$gpu python a2/evaluate/test_place.py --use_rope --load_model --model_path $model_path --log_suffix place-$log --testing_case_dir testing_cases/place_testing_cases/seen --action_var
echo unseen
CUDA_VISIBLE_DEVICES=$gpu python a2/evaluate/test_place.py --unseen --use_rope --load_model --model_path $model_path --log_suffix place-$log-unseen --testing_case_dir testing_cases/place_testing_cases/unseen --action_var