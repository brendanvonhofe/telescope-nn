#!/bin/bash

source /home/bread/anaconda3/bin/activate telescope

python src/train.py \
    --checkpoint=fresh \
    --save_dir=models/cyclic_0 \
    --stage=0 \
    --early_cutoff=100 \
    --epochs=-1 \
    --batch_size=16 \
    --threads=8 \
    --weighted_loss=0

python src/train.py \
    --checkpoint=cyclic_0 \
    --save_dir=models/cyclic_1 \
    --stage=1 \
    --early_cutoff=100 \
    --epochs=-1 \
    --batch_size=16 \
    --threads=8 \
    --weighted_loss=0 

python src/train.py \
    --checkpoint=cyclic_1 \
    --save_dir=models/cyclic_2 \
    --stage=2 \
    --early_cutoff=100 \
    --epochs=-1 \
    --batch_size=16 \
    --threads=8 \
    --weighted_loss=0 

conda deactivate
