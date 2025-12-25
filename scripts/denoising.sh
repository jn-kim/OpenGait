#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate opengait

export CUDA_VISIBLE_DEVICES=0,1,2,3

cd /home/aix23907/OpenGait

python datasets/pretreatment_denoisinggait.py \
    --input_path /data0/aix23907/gait/CCPG_RGB_256x128 \
    --output_path /data0/aix23907/gait/CCPG_DenoisingGait \
    --sd_model_path runwayml/stable-diffusion-v1-5 \
    --timestep 700 \
    --batch_size 8 \
    --num_gpus 4

