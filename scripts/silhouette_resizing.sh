#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate opengait

python datasets/pretreatment_silhouette_96x48.py \
    --input_path /data0/aix23907/gait/CCPG_D_MASK_FACE_SHOE \
    --output_path /data0/aix23907/gait/CCPG_DenoisingGait

