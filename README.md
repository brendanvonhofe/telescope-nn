# Telescope Neural Network

Neural network that performs image matting from a user-defined trimap.

Currently using the dataset compiled from:
Xu, Ning, et al. “Deep Image Matting.” 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, doi:10.1109/cvpr.2017.41.

https://arxiv.org/pdf/1703.03872.pdf

## Requirements
* Ubuntu 16.04
* Nvidia GeForce GTX 1080 Ti or better
* Cuda 9
* Conda
* Python 3


## Install
Place the ```Adobe_Deep_Matting_Dataset.zip``` in the ```data/raw``` directory.

Need gsutil rsync to ```build_dataset.sh```. Run the following
```
curl https://sdk.cloud.google.com | bash
```
Create environment and install dependencies with
```
conda env create -f env.yml
```

Run ```build_dataset.sh``` and then ```train.sh```.
