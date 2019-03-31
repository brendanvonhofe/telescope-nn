#!/bin/bash

python src/train.py \
    --checkpoint=fresh \
    --save_dir=models/telescope_0 \
    --stage=0 \
    --early_cutoff=100 \
    --epochs=-1 \
    --batch_size=16 \
    --threads=8 \
    --weighted_loss=0

python src/train.py \
    --checkpoint=telescope_0 \
    --save_dir=models/telescope_1 \
    --stage=1 \
    --early_cutoff=25 \
    --epochs=-1 \
    --batch_size=16 \
    --threads=8 \
    --weighted_loss=0 

python src/train.py \
    --checkpoint=telescope_1 \
    --save_dir=models/telescope_2 \
    --stage=2 \
    --early_cutoff=125 \
    --epochs=-1 \
    --batch_size=16 \
    --threads=8 \
    --weighted_loss=0 

python src/train.py \
    --checkpoint=telescope_2 \
    --save_dir=models/telescope_3 \
    --stage=2 \
    --early_cutoff=250 \
    --epochs=-1 \
    --batch_size=16 \
    --threads=8 \
    --weighted_loss=1