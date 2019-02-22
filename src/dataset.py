import os
import numpy as np
import cv2
import torch
from skimage import io, util, morphology
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random

class MatteDataset(Dataset):    
    def __init__(self, filenames, root_dir, fg_path, transform=None):
        self.fns = filenames
        self.root_dir = root_dir
        self.transform = transform
        self.fg_path = fg_path
        self.fgs = os.listdir(self.fg_path)
        
    def __len__(self):
        return len(self.fns)
    
    def __getitem__(self, idx):
        # Read in foreground, background and alpha mask for foreground
        fn = self.fgs[np.random.randint(0, len(self.fgs))] # Random foreground
        fg = io.imread(self.fg_path/fn)
        mask = io.imread(self.root_dir/'mattes'/fn)
        bg = io.imread(self.root_dir/'bg'/self.fns[idx])
        
        # Resize to fit be same as background
        bg = bg.astype(np.float64)
        fg = cv2.resize(fg, (bg.shape[1], bg.shape[0])).astype(np.float64)
        mask = cv2.resize(mask, (bg.shape[1], bg.shape[0])).astype(np.uint8)
        
        # Composite image
        alpha = np.array([mask/255]*3).transpose(1,2,0).astype(np.float64)
        foreground = np.multiply(alpha, fg)
        background = np.multiply(1.0 - alpha, bg)
        image = cv2.add(foreground, background).astype(np.uint8)
                
        # Create trimap from mask by randomly dilating
        if(len(mask.shape) == 2):
            mask = np.expand_dims(mask, -1)
        trimap = gen_trimap(mask)
        
        # Pad images 
        if(image.shape[0] < 333 or image.shape[1] < 333):
            image = cv2.resize(image, (333, 333))
            trimap = np.expand_dims(cv2.resize(trimap, (333, 333)), -1)
            mask = np.expand_dims(cv2.resize(mask, (333, 333)), -1)
            fg = cv2.resize(fg, (333, 333))
            bg = cv2.resize(bg, (333, 333))

        # Concatenate image and trimap to make (h, w, 4) shape input
        im_map = np.concatenate((image, trimap), axis=2)

        # Scale to between 0 and 1
        im_map = np.multiply(im_map, (1/255))
        mask = np.multiply(mask, (1/255))
        
        # Package input, label and fg/bg to reconstruct image to calculate loss
        sample = {'im_map': im_map, 'mask': mask, 'bg': bg, 'fg': fg}
        if self.transform:
            sample = self.transform(sample)
    
        return sample

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        im_map, mask, bg, fg = sample['im_map'], sample['mask'], sample['bg'], sample['fg']

        h, w = im_map.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        im_map = im_map[top: top + new_h, left: left + new_w]
        mask = mask[top: top + new_h, left: left + new_w]
        bg = bg[top: top + new_h, left: left + new_w]
        fg = fg[top: top + new_h, left: left + new_w]

        return {'im_map': im_map, 'mask': mask, 'bg': bg, 'fg': fg}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        im_map, mask, bg, fg = sample['im_map'], sample['mask'], sample['bg'], sample['fg']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        im_map = im_map.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))
        bg = bg.transpose((2, 0, 1))
        fg = fg.transpose((2, 0, 1))
        return {'im_map': torch.from_numpy(im_map).float(),
                'mask': torch.from_numpy(mask).float(),
                'bg': torch.from_numpy(bg).float(),
                'fg': torch.from_numpy(fg).float()}

def getTrainValSplit(path):
    fns = np.array(os.listdir(path))
    val_idxs = [i for i in range(0,int(len(fns)/80))]
    val_fns = fns[:int(len(fns)/80)]
    train_idxs = list(set([i for i in range(0,len(fns))]).difference(set(val_idxs)))
    train_fns = fns[train_idxs]
    return train_fns, val_fns

def getTransforms():
    return transforms.Compose([RandomCrop(320), ToTensor()])

def gen_trimap(alpha):
    k_size = random.choice(range(1, 5))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    dilated = cv2.dilate(alpha, kernel, iterations=np.random.randint(1, 20))
    
    trimap = np.zeros(alpha.shape)
    trimap.fill(128)
    trimap[alpha >= 255] = 255
    trimap[dilated <= 0] = 0

    return trimap.astype(np.uint8)