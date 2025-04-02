#!/bin/bash
# set -x
gpu=0

echo collect data for place
CUDA_VISIBLE_DEVICES=$gpu python data_collection/collect_data_place.py --log_suffix collect-data-place