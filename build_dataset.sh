#!/bin/bash

mkdir data/processed
mkdir data/processed/train
mkdir data/processed/val
mkdir data/processed/train/fg
mkdir data/processed/train/bg
mkdir data/processed/train/mattes
mkdir data/processed/val/fg
mkdir data/processed/val/bg
mkdir data/processed/val/mattes

unzip data/raw/Adobe_Deep_Matting_Dataset.zip
cp Combined_Dataset/Training_set/Adobe-licensed\ images/alpha/* data/processed/train/mattes
cp Combined_Dataset/Training_set/Adobe-licensed\ images/fg/* data/processed/train/fg
cp Combined_Dataset/Training_set/Other/fg/* data/processed/train/fg
cp Combined_Dataset/Training_set/Other/alpha/* data/processed/train/mattes

rm -r Combined_Dataset

gsutil -m rsync gs://images.cocodataset.org/train2014 data/processed/train/bg

file="src/utils/coco_removed_ims_2d.txt"
while FS= read line
do
	rm "data/processed/train/bg/$line"
done <"$file"

python src/utils/trainval_split.py