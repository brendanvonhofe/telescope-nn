import os
import numpy as np
import cv2
import torch
import random
from skimage import io, util, morphology
from torch.utils.data import Dataset
from pathlib import Path
from albumentations.augmentations.transforms import RandomCrop, Resize, Flip

class MatteDataset(Dataset):    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.fg_fns = os.listdir(Path(root_dir/'fg'))
        self.bg_fns = os.listdir(Path(root_dir/'bg'))
        self.rc320 = RandomCrop(height=320, width=320, always_apply=True)
        self.rc480 = RandomCrop(height=480, width=480, always_apply=True)
        self.rc640 = RandomCrop(height=640, width=640, always_apply=True)
        self.resize = Resize(height=320, width=320, always_apply=True)
        self.flip = Flip(p=.75)
        
    def __len__(self):
        return len(self.fg_fns)
    
    def __gentrimap__(self, alpha):
        k_size = random.choice(range(1, 5))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        dilated = cv2.dilate(alpha, kernel, iterations=np.random.randint(1, 20))

        trimap = np.zeros(alpha.shape)
        trimap.fill(128)
        trimap[alpha >= 255] = 255
        trimap[dilated <= 0] = 0

        return trimap.astype(np.uint8)
    
    def __totensor__(self, sample):
        im_map, mask, bg, fg = sample['inputs'], sample['mask'], sample['bg'], sample['fg']

        # swap color axis: numpy image: H x W x C, torch image: C X H X W
        im_map, mask = im_map.transpose((2, 0, 1)), mask.transpose((2, 0, 1))
        bg, fg = bg.transpose((2, 0, 1)), fg.transpose((2, 0, 1))
        return {'inputs': torch.from_numpy(im_map).float(),
                'mask': torch.from_numpy(mask).float(),
                'bg': torch.from_numpy(bg).float(),
                'fg': torch.from_numpy(fg).float()}
    
    def __getitem__(self, idx):
        # Returns image, mask, trimap, bg, and fg with values in [0, 1] as np.float32
        
        bg_fn = self.bg_fns[np.random.randint(0, len(self.bg_fns))] # Random background
        bg = io.imread(self.root_dir/'bg'/bg_fn).astype(np.uint8)
        fg = io.imread(self.root_dir/'fg'/self.fg_fns[idx]).astype(np.uint8)
        mask = io.imread(self.root_dir/'mattes'/self.fg_fns[idx]).astype(np.uint8)
        
        shape = (fg.shape[0], fg.shape[1])
        
        # CROPPING
        if(shape[0] > 640 and shape[1] > 640):
            # Randomly crop 320x320, 480x480, or 640x640
            r = np.random.randint(0, 3)
            if(r == 0):
                cropped = self.rc320(image=fg, mask=mask)
                fg, mask = cropped['image'], cropped['mask']
            elif(r == 1):
                cropped = self.rc480(image=fg, mask=mask)
                fg, mask = cropped['image'], cropped['mask']
            else:
                cropped = self.rc640(image=fg, mask=mask)
                fg, mask = cropped['image'], cropped['mask']
        elif(shape[0] > 480 and shape[1] > 480):
            # Randomly crop 320x320 or 480x480
            r = np.random.randint(0, 2)
            if(r == 0):
                cropped = self.rc320(image=fg, mask=mask)
                fg, mask = cropped['image'], cropped['mask']
            else:
                cropped = self.rc480(image=fg, mask=mask)
                fg, mask = cropped['image'], cropped['mask']
        elif(shape[0] > 320 and shape[1] > 320):
            cropped = self.rc320(image=fg, mask=mask)
            fg, mask = cropped['image'], cropped['mask']
            
        # Resize to 320x320
        resized = self.resize(image=fg, mask=mask)
        fg, mask = resized['image'], resized['mask']
            
        # FLIPPING (and other augmentations)
        flipped = self.flip(image=fg, mask=mask)
        fg, mask = flipped['image'], flipped['mask']
        
        # CREATE IMAGE and TRIMAP, SCALE to [0,1]
        trimap = np.expand_dims(self.__gentrimap__(mask).astype(np.float32) * (1./255), -1)
        bg = self.resize(image=bg)['image']
        mask = np.expand_dims(mask.astype(np.float32) * (1./255), -1)
        image = cv2.add(fg*mask, bg*(1-mask)) * (1./255)
        bg, fg = bg * (1./255), fg * (1./255)
        
        # SEND
        inputs = np.concatenate((image, trimap), axis=2)
        sample = self.__totensor__({'inputs': inputs, 'mask': mask, 'bg': bg, 'fg': fg})
        sample['bg_fn'] = bg_fn

        return sample

class SampleDataset(Dataset):
    def __init__(self):
        self.dir = Path('../samples')
        self.images = os.listdir(str(self.dir/'images'))
        self.trimaps = os.listdir(str(self.dir/'trimaps'))
        self.masks = os.listdir(str(self.dir/'masks'))
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        sample = {'image': self.dir/'images'/self.images[idx], 'trimap': self.dir/'trimaps'/self.trimaps[idx], 'mask': self.dir/'masks'/self.masks[idx]}

        return sample