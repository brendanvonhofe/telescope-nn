from pathlib import Path
import json # For config file
import time
import copy
import os
import argparse
from datetime import datetime
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter

from dataset import getTransforms, MatteDataset
from architecture.linknet import LinkNet34
from architecture.vgg16 import DeepMattingVGG
from architecture.refinement_layer import MatteRefinementLayer

PATH = Path('data/processed/train/')
VAL = Path('data/processed/val/')
BG = PATH/'bg'
FG = PATH/'fg'
MASKS = PATH/'mattes'
MODELS = Path('models')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Training on device:', device)

def get_args():
    # Config settings and hyperparameters
    parser = argparse.ArgumentParser(description='Deep Image Matting training')
    parser.add_argument('--batch_size', type=int, required=True, help='training batch size')
    parser.add_argument('--threads', type=int, required=True, help='number of workers for dataset generation')
    parser.add_argument('--stage', type=int, required=True, help='0 -> just enc-dec, 1 -> just refine, 2 -> both')
    parser.add_argument('--save_dir', type=str, required=True, help='dst to save checkpoint')
    parser.add_argument('--epochs', type=int, default=-1, help='number of epochs to train for, -1 -> train current stage')
    parser.add_argument('--checkpoint', type=str, default='vgg16', help='directory to load checkpoints from')
    args = parser.parse_args()
    print(args)
    return args

def main():
    # Load args
    args = get_args()

    # Load Dataset 
    data_transform = getTransforms() # Consider adding additional transforms
    image_datasets = {'train': MatteDataset(root_dir=PATH, transform=data_transform),
                  'val': MatteDataset(root_dir=VAL, transform=data_transform)}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                                shuffle=True, num_workers=args.threads)
                for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    # Setup Model
    epoch = 0
    # encdec = LinkNet34(1)
    encdec = DeepMattingVGG()
    if(args.checkpoint != 'vgg16'):
        save_name = os.listdir(args.checkpoint + '/encdec/')[0]
        print("Loading encoder-decoder:", save_name)
        ckpt = torch.load(args.checkpoint + '/encdec/' + save_name)
        epoch = ckpt['epoch']
        encdec.load_state_dict(ckpt['state_dict'])
        # encdec.load_state_dict(ckpt)
        encdec = encdec.to(device)
        if(args.stage != 0):
            save_name = os.listdir(args.checkpoint + '/refinement/')[0]
            print("Loading refinement:", save_name)
            refinement = MatteRefinementLayer()
            ckpt = torch.load(args.checkpoint + '/refinement/' + save_name)
            refinement.load_state_dict(ckpt['state_dict'])
            # refinement.load_state_dict(ckpt)
            refinement = refinement.to(device)

    # _ed suffix refers to encoder-decoder part of the model, 
    # _r suffix refers to refinement part
    crit_ed = AlphaCompLoss_u()
    crit_r = AlphaLoss_u()
    optim_ed = optim.Adam(encdec.parameters(), lr=1e-5)
    optim_r = optim.Adam(refinement.parameters(), lr=1e-5)

    # TRAIN

    # Writers for TensorBoard
    train_writer = SummaryWriter('logs/train' + args.save_dir)
    val_writer = SummaryWriter('logs/val' + args.save_dir)

    # best_model_wts_ed = copy.deepcopy(encdec.state_dict())
    # best_model_wts_r = copy.deepcopy(refinement.state_dict())
    # best_loss = 50000 # CHECK THIS
    
    for e in tqdm(range(args.epochs)):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                if(args.stage == 0):
                    encdec.train()
                elif(args.stage == 1):
                    encdec.eval()
                    refinement.train()
                elif(args.stage == 2):
                    encdec.train()
                    refinement.train()
            else:
                encdec.eval()   # Set model to evaluate mode
                if(args.stage != 0):
                    refinement.eval()

            if(args.stage != 1):
                running_loss_ed = []
            if(args.stage != 0):
                running_loss_r = []

            running_sad = []
            running_mse = []

            # Iterate over dataset.
            for i, sample in tqdm(enumerate(dataloaders[phase])):
                # Get inputs and labels, put them on GPU
                inputs_ed, labels, fg, bg = sample['im_map'], sample['mask'], sample['fg'], sample['bg']
                inputs_ed = inputs_ed.to(device)
                labels = labels.to(device)
                fg = fg.to(device)
                bg = bg.to(device)
                trimap = inputs_ed[:,3,:,:]

                # zero the gradients
                optim_ed.zero_grad()
                optim_r.zero_grad()
                
                # Calculate loss
                with torch.set_grad_enabled(phase == 'train'): # Only track grads if in train mode
                    outputs_ed = encdec(inputs_ed) # Output is single channel matte

                    if(args.stage != 0):
                        inputs_r = torch.cat((inputs_ed[:,:3,:,:], outputs_ed), 1)
                        outputs_r = refinement(inputs_r)

                    # _, preds = torch.max(outputs, 1)

                    if(args.stage != 1):
                        loss_ed = crit_ed(outputs_ed, labels, fg, bg, trimap)
                        running_loss_ed.append(loss_ed.item())
                    if(args.stage != 0):
                        loss_r = crit_r(outputs_r, labels, trimap)
                        running_loss_r.append(loss_r.item())

                    # backward + optimize only if in training phase
                    if(phase == 'train'):
                        if(args.stage == 0):
                            loss_ed.backward()
                            optim_ed.step()
                        if(args.stage == 1):
                            loss_r.backward()
                            optim_r.step()
                        if(args.stage != 2):
                            loss_r.backward()
                            optim_ed.step()
                            optim_r.step()

                    if(phase == 'val'):
                        if(args.stage == 0):
                            running_sad.append(sum_absolute_differences(outputs_ed, labels, trimap).item())
                            running_mse.append(mean_squared_error(outputs_ed, labels, trimap).item())
                        else:
                            running_sad.append(sum_absolute_differences(outputs_r, labels, trimap).item())
                            running_mse.append(mean_squared_error(outputs_r, labels, trimap).item())


            # Record average epoch loss for TensorBoard
            epoch_loss_ed = np.array(running_loss_ed).mean()
            epoch_loss_r = np.array(running_loss_r).mean()
            if(phase == 'train'):
                if(args.stage != 1):
                    train_writer.add_scalar("Encoder-Decoder_Loss", epoch_loss_ed, epoch + e)
                if(args.stage != 0):
                    train_writer.add_scalar("Refinement_Loss", epoch_loss_r, epoch + e)
                
            if(phase == 'val'):
                val_writer.add_scalar("Mean-Squared-Error", np.array(running_mse).mean(), epoch+e)
                val_writer.add_scalar("Sum-of-Absolute-Differences", np.array(running_sad).mean(), epoch+e)
                if(args.stage != 1):
                    val_writer.add_scalar("Encoder-Decoder_Loss", epoch_loss_ed, epoch+e)
                if(args.stage != 0):
                    val_writer.add_scalar("Refinement_Loss", epoch_loss_r, epoch+e)

            # deep copy the best model
            # if(phase == 'val' and epoch_loss < best_loss):
            #     best_loss = epoch_loss
            #     best_model_wts = copy.deepcopy(model.state_dict())
            #     best_model_wts_r = copy.deepcopy(model_r.state_dict())

    # print('Best val Loss: {:4f}'.format(best_loss))

    train_writer.close()
    val_writer.close()

    print("Saving model at {}".format(args.save_dir))
    checkpoint(args.epochs+epoch-1, args.save_dir, encdec, encdec=True)
    if(args.stage != 0):
        checkpoint(args.epochs+epoch-1, args.save_dir, refinement, encdec=False)

    # load best model weights
    # model.load_state_dict(best_model_wts)
    # model_r.load_state_dict(best_model_wts_r)
    # torch.save(telescope.state_dict(), MODELS/savename)

def composite(fg, bg, alpha):
    foreground = torch.mul(alpha, fg)
    background = torch.mul(1.0 - alpha, bg)
    return torch.add(foreground, background)

def checkpoint(epoch, save_dir, model, encdec=True):
    if(encdec):
        model_out_path = "{}/encdec/ckpt_encdec_e{}.pth".format(save_dir, epoch)
    else:
        model_out_path = "{}/refinement/ckpt_refinement_e{}.pth".format(save_dir, epoch)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(save_dir + '/encdec')
        os.makedirs(save_dir + '/refinement')
    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
    }, model_out_path )
    print("Checkpoint saved to {}".format(model_out_path))

class _Loss(nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction

class AlphaLoss_u(_Loss):
    def __init__(self, eps=1e-6):
        super(AlphaLoss_u, self).__init__()
        self.eps = eps

    def forward(self, p_mask, gt_mask, trimap):
        return alpha_loss_u(p_mask, gt_mask, trimap, self.eps)
    
# class AlphaCompLoss(_Loss):
#     def __init__(self, eps=1e-6):
#         super(AlphaCompLoss, self).__init__()
#         self.eps = eps
        
#     def forward(self, p_mask, gt_mask, fg, bg):
#         return alpha_loss(p_mask, gt_mask, self.eps) + \
#                comp_loss(p_mask, gt_mask, fg, bg, self.eps)
    
class AlphaCompLoss_u(_Loss):
    def __init__(self, eps=1e-6):
        super(AlphaCompLoss_u, self).__init__()
        self.eps = eps
        
    def forward(self, p_mask, gt_mask, fg, bg, trimap):
        return (0.5 * alpha_loss_u(p_mask, gt_mask, trimap, self.eps)) + \
               (0.5 * comp_loss_u(p_mask, gt_mask, fg, bg, trimap, self.eps))

# def alpha_loss(p_mask, gt_mask, eps=1e-6):
#     return torch.sqrt(gt_mask.sub(p_mask).pow(2).mean() + eps)

def alpha_loss_u(p_mask, gt_mask, trimap, eps=1e-6):
    # only counts loss in "unknown" region of trimap

    sqr_diff = gt_mask.sub(p_mask).pow(2)
    unknown = torch.eq(trimap, torch.FloatTensor(np.ones(gt_mask.shape)*(128./255)).to(device)).float()
    loss = torch.mul(sqr_diff, unknown)
    image_loss = torch.sum(torch.sum(torch.sum(loss, dim=1), dim=1), dim=1)
    alpha_loss = torch.sqrt(image_loss.mean() + eps)
    return alpha_loss

# def comp_loss(p_mask, gt_mask, fg, bg, eps=1e-6):
#     gt_comp = composite(fg, bg, gt_mask)
#     p_comp = composite(fg, bg, p_mask)
#     return torch.sqrt(gt_comp.sub(p_comp).pow(2).sum() + eps)

def comp_loss_u(p_mask, gt_mask, fg, bg, trimap, eps=1e-6):
    # only counts loss in "unknown" region of trimap

    gt_comp = composite(fg, bg, gt_mask) * (1./255)
    p_comp = composite(fg, bg, p_mask) * (1./255)
    bs, h, w = trimap.shape
    ones = torch.FloatTensor(np.ones(trimap.shape)*(128./255)).to(device)
    unknown = torch.eq(trimap, ones).float().expand(3, bs, h, w).contiguous().view(bs,3,h,w)
    s_diff = gt_comp.sub(p_comp).pow(2)
    loss = torch.mul(s_diff, unknown)
    image_loss = torch.sum(torch.sum(torch.sum(loss, dim=1), dim=1), dim=1)
    comp_loss = torch.sqrt(image_loss.mean() + eps)
    return comp_loss

def sum_absolute_differences(p_mask, gt_mask, trimap):
    bs, h, w = trimap.shape
    ones = torch.FloatTensor(np.ones(trimap.shape)*(128./255)).to(device)
    unknown = torch.eq(trimap, ones).float().expand(3, bs, h, w).contiguous().view(bs,3,h,w)
    diffs = gt_mask.sub(p_mask).abs()
    u_diffs = torch.mul(diffs, unknown)
    return u_diffs.sum()

def mean_squared_error(p_mask, gt_mask, trimap):
    bs, h, w = trimap.shape
    ones = torch.FloatTensor(np.ones(trimap.shape)*(128./255)).to(device)
    unknown = torch.eq(trimap, ones).float().expand(3, bs, h, w).contiguous().view(bs,3,h,w)
    diffs = gt_mask.sub(p_mask).pow(2)
    u_diffs = torch.mul(diffs, unknown)
    return u_diffs.mean()


if __name__ == "__main__":
    main()