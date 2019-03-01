#!/bin/bash

source /home/bread/anaconda3/bin/activate telescope

python src/train.py \
    --batch_size=16 \
    --threads=8 \
    --stage=0 \
    --save_dir=models/vgg_es \
    --epochs=2000 \
    --checkpoint=fresh \
    --weighted_loss=0

python src/train.py \
    --batch_size=16 \
    --threads=8 \
    --stage=1 \
    --save_dir=models/vgg_es \
    --epochs=2000 \
    --checkpoint=vgg_es \
    --weighted_loss=0

python src/train.py \
    --batch_size=16 \
    --threads=8 \
    --stage=2 \
    --save_dir=models/vgg_es \
    --epochs=2000 \
    --checkpoint=vgg_es \
    --weighted_loss=0

conda deactivate
