#!/bin/bash

# python src/train.py \
#     --checkpoint=fresh \
#     --save_dir=models/nudataset_0 \
#     --stage=0 \
#     --early_cutoff=100 \
#     --epochs=-1 \
#     --batch_size=16 \
#     --threads=8 \
#     --weighted_loss=0

# python src/train.py \
#     --checkpoint=nudataset_0 \
#     --save_dir=models/nudataset_1 \
#     --stage=1 \
#     --early_cutoff=25 \
#     --epochs=-1 \
#     --batch_size=16 \
#     --threads=8 \
#     --weighted_loss=0 

# python src/train.py \
#     --checkpoint=nudataset_1 \
#     --save_dir=models/nudataset_2 \
#     --stage=2 \
#     --early_cutoff=125 \
#     --epochs=-1 \
#     --batch_size=16 \
#     --threads=8 \
#     --weighted_loss=0 

python src/train.py \
    --checkpoint=nudataset_2 \
    --save_dir=models/nudataset_3 \
    --stage=2 \
    --early_cutoff=150 \
    --epochs=-1 \
    --batch_size=16 \
    --threads=8 \
    --weighted_loss=0 

# python src/train.py \
#     --checkpoint=cyclic_2 \
#     --save_dir=models/test \
#     --stage=2 \
#     --early_cutoff=-1 \
#     --epochs=1 \
#     --batch_size=16 \
#     --threads=8 \
#     --weighted_loss=1 
