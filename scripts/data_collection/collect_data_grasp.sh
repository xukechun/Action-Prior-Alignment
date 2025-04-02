#!/bin/bash
# set -x
gpu=0

echo collect data for grasp
CUDA_VISIBLE_DEVICES=$gpu python data_collection/collect_data_grasp.py --log_suffix collect-data-grasp