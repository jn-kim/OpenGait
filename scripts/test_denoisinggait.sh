#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate opengait

export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m torch.distributed.launch --nproc_per_node=4 opengait/main.py \
    --cfgs ./configs/denoisinggait/denoisinggait_ccpg.yaml \
    --phase test \
    --log_to_file

