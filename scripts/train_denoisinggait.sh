#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate opengait

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

python -m torch.distributed.launch --nproc_per_node=4 opengait/main.py \
    --cfgs ./configs/denoisinggait/denoisinggait_ccpg.yaml \
    --phase train