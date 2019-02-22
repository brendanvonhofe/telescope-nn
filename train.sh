#!/bin/bash

source /home/bread/anaconda3/bin/activate telescope

python src/train.py \
    --batch_size=16 \
    --threads=8 \
    --stage=2 \
    --save_dir=models/feb2219 \
    --epochs=20 \
    --checkpoint=models/feb222019

conda deactivate