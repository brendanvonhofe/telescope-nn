#!/bin/bash

source /home/bread/anaconda3/bin/activate telescope

python src/train.py \
    --batch_size=16 \
    --threads=8 \
    --stage=0 \
    --save_dir=models/testing \
    --epochs=20 \
    --checkpoint=fresh \
    --weighted_loss=0

conda deactivate
