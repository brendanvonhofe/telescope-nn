# Telescope Neural Network

Neural network that performs image matting from a user-defined trimap.

Currently using the dataset compiled from:
Xu, Ning, et al. “Deep Image Matting.” 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, doi:10.1109/cvpr.2017.41.

https://arxiv.org/pdf/1703.03872.pdf

## Requirements
* Conda
* Python 3
* Ubuntu

## Install
Create environment and install dependencies with
```
conda env create -f env.yml
conda activate telescope
```
Play with the model by running the ```user_interface.ipynb``` notebook
```
jupyter notebook
```
Test the model with individual images/trimaps with ```telescope.py```.
For example
```
python telescope.py samples/images/head.png samples/trimaps/head.png
```