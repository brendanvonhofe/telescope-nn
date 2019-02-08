#!/bin/bash
file="coco_removed_ims_2d.txt"
while FS= read line
do
	echo "$line"
done <"$file"
