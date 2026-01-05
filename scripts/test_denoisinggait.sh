#!/bin/bash

# Test script for DenoisingGait on CCPG dataset
# 체크포인트: GaitBasefusion_p0.2_threshold0.1_test_daFalse_8416-80000.pt
# 설정 파일에서 evaluator_cfg.restore_hint: 80000으로 자동 로드됨

eval "$(conda shell.bash hook)"
conda activate opengait

export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m torch.distributed.launch --nproc_per_node=4 opengait/main.py \
    --cfgs ./configs/denoisinggait/denoisinggait_ccpg.yaml \
    --phase test \
    --log_to_file

