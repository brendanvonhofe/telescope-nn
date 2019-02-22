from pathlib import Path
import json # For config file
import time
import copy
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter

from dataset import getTrainValSplit, getTransforms, MatteDataset
from architecture.linknet import LinkNet34
from architecture.refinement_layer import MatteRefinementLayer

PATH = Path('../data/processed/train/')
VAL = Path('../data/processed/val/')
BG = PATH/'bg'
FG = PATH/'fg'
MASKS = PATH/'mattes'
MODELS = Path('../models')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Training on device: ', device)

def main():
    
    # CONFIGURATION

    # Load configuration file
    with open("utils/config.json", "r") as read_file:
        config = json.load(read_file)

    # Set config variables
    batch_size = config['batch_size']
    pretrained = config['pretrained_model']
    pretrained_r = config['pretrained_refine'] # Refinement network
    savename = config['savename']
    savename_r = config['savename_r'] # Refinement metwork
    iterations = config['iterations'] # Using iterations, not epochs

    # LOAD DATASET

    data_transform = getTransforms() # Consider adding additional transforms

    image_datasets = {'train': MatteDataset(root_dir=PATH, transform=data_transform),
                  'val': MatteDataset(root_dir=VAL, transform=data_transform)}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                shuffle=True, num_workers=8)
                for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    # SETUP MODEL

    telescope = LinkNet34(1)
    refinement = MatteRefinementLayer()
    if(len(pretrained)):
        print("Loading weights from", pretrained)
        telescope.load_state_dict(torch.load(MODELS/pretrained))
        refinement.load_state_dict(torch.load(MODELS/pretrained_r))
    telescope = telescope.to(device)
    refinement = refinement.to(device)

    criterion = dim_loss_weighted()
    criterion_r = ap_loss()
    optimizer = optim.Adam(telescope.parameters(), lr=1e-5)
    optimizer_r = optim.Adam(refinement.parameters(), lr=1e-5)
    model = telescope
    model_r = refinement


    # TRAIN

    print(datetime.now())
    since = time.time()

    it = 0

    writer = SummaryWriter('../logs/' + savename)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_model_wts_r = copy.deepcopy(model_r.state_dict())
    best_loss = 50000 # CHECK THIS
    
    num_epochs = int(iterations / 250)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                model_r.train()
            else:
                model.eval()   # Set model to evaluate mode
                model_r.eval()

            running_loss = []

            # Iterate over data.
            for i, sample in enumerate(dataloaders[phase]):
                if(phase=='train'):
                    it += 1
                if(i != 0 and i % 250 == 0 and phase == 'train'):
                    break
                inputs, labels, fg, bg = sample['im_map'], sample['mask'], sample['fg'], sample['bg']
                inputs = inputs.to(device)
                labels = labels.to(device)
                fg = fg.to(device)
                bg = bg.to(device)
                trimap = inputs[:,3,:,:]

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs) # Output is single channel matte
                    r_inputs = torch.cat((inputs[:,:3,:,:], outputs), 1)
                    r_outputs = model_r(r_inputs)
                    # _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels, fg, bg, trimap)
                    loss_r = criterion_r(r_outputs, labels, trimap)
                    # loss = criterion(outputs, labels, fg, bg)
                    running_loss.append(loss_r.item()/batch_size)
                    print("Loss at step {}: {}".format(i, loss_r/batch_size))

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        writer.add_scalar('train_loss', loss_r/batch_size, it)
                        # loss.backward()
                        loss_r.backward()
                        optimizer.step()
                        optimizer_r.step()

            epoch_loss = np.array(running_loss).mean()
            writer.add_scalar(phase + "epoch_loss", epoch_loss, it)
            
            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                best_model_wts_r = copy.deepcopy(model_r.state_dict())
                
            print("Saving model at", savename)
            torch.save(model.state_dict(), MODELS/savename)
            torch.save(model_r.state_dict(), MODELS/savename_r)

    print(datetime.now())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))

    writer.close()
    # load best model weights
    model.load_state_dict(best_model_wts)
    model_r.load_state_dict(best_model_wts_r)

    torch.save(telescope.state_dict(), MODELS/savename)

def composite(fg, bg, alpha):
    foreground = torch.mul(alpha, fg)
    background = torch.mul(1.0 - alpha, bg)
    return torch.add(foreground, background)

class _Loss(nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction

def alpha_pred_loss(p_mask, gt_mask, eps=1e-6):
    return torch.sqrt(gt_mask.sub(p_mask).pow(2).sum() + eps)

def alpha_pred_loss_weighted(p_mask, gt_mask, trimap, eps=1e-6):
    sqr_diff = gt_mask.sub(p_mask).pow(2)
    unknown = torch.eq(trimap, torch.FloatTensor(np.ones(gt_mask.shape)*(128./255)).to(device)).float()
    return torch.sqrt(torch.mul(sqr_diff, unknown).sum() + eps)

class ap_loss(_Loss):
    def __init__(self, eps=1e-6):
        super(ap_loss, self).__init__()
        self.eps = eps

    def forward(self, p_mask, gt_mask, trimap):
        return alpha_pred_loss_weighted(p_mask, gt_mask, trimap, self.eps)

def compositional_loss(p_mask, gt_mask, fg, bg, eps=1e-6):
    gt_comp = composite(fg, bg, gt_mask)
    p_comp = composite(fg, bg, p_mask)
    return torch.sqrt(gt_comp.sub(p_comp).pow(2).sum() + eps)

def compositional_loss_weighted(p_mask, gt_mask, fg, bg, trimap, eps=1e-6):
    gt_comp = composite(fg, bg, gt_mask)
    p_comp = composite(fg, bg, p_mask)
    bs, h, w = trimap.shape
    ones = torch.FloatTensor(np.ones(trimap.shape)*(128./255)).to(device)
    unknown = torch.eq(trimap, ones).float().expand(3, bs, h, w).contiguous().view(bs,3,h,w)
    s_diff = gt_comp.sub(p_comp).pow(2)
    return torch.sqrt(torch.mul(s_diff, unknown).sum() + eps)
    
class dim_loss(_Loss):
    def __init__(self, eps=1e-6, w=0.5):
        super(dim_loss, self).__init__()
        self.eps = eps
        self.w = w
        
    def forward(self, p_mask, gt_mask, fg, bg):
        return self.w * alpha_pred_loss(p_mask, gt_mask, self.eps) + \
               (1-self.w) * compositional_loss(p_mask, gt_mask, fg, bg, self.eps)
    
class dim_loss_weighted(_Loss):
    def __init__(self, eps=1e-6, w=0.5):
        super(dim_loss_weighted, self).__init__()
        self.eps = eps
        self.w = w
        
    def forward(self, p_mask, gt_mask, fg, bg, trimap):
        return self.w * alpha_pred_loss_weighted(p_mask, gt_mask, trimap, self.eps) + \
               (1-self.w) * compositional_loss_weighted(p_mask, gt_mask, fg, bg, trimap, self.eps)

if __name__ == "__main__":
    main()