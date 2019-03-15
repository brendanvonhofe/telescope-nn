from pathlib import Path
import os
import argparse
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
from utils.cyclic_lr import CyclicLR

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
    parser.add_argument('--weighted_loss', type=int, default=0, help='flag to only track loss in unknown region of trimap')
    parser.add_argument('--early_cutoff', type=int, default=100, help='number of epochs to train without increase to val loss before early-stopping')
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
    # dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    # Setup Model
    epoch = 0
    best_loss = 50000
    early_stopping = args.early_cutoff
    # encdec = LinkNet34(1)
    encdec = DeepMattingVGG()
    if(args.checkpoint != 'fresh'):
        save_name = os.listdir('models/' + args.checkpoint + '/encdec/')[np.argmax(np.array([int(s[-8:-4]) for s in os.listdir('models/' + args.checkpoint + '/encdec/')]))]
        print("Loading encoder-decoder:", save_name)
        ckpt = torch.load('models/' + args.checkpoint + '/encdec/' + save_name)
        epoch = ckpt['epoch']
        best_loss = ckpt['loss']
        encdec.load_state_dict(ckpt['state_dict'])
        # encdec.load_state_dict(ckpt)
        
        if(args.stage != 0):
            refinement = MatteRefinementLayer()
            if(os.listdir('models/' + args.checkpoint + '/refinement/')):
                save_name = os.listdir('models/' + args.checkpoint + '/refinement/')[np.argmax(np.array([int(s[-8:-4]) for s in os.listdir('models/' + args.checkpoint + '/refinement/')]))]
                print("Loading refinement:", save_name)
                ckpt = torch.load('models/' + args.checkpoint + '/refinement/' + save_name)
                refinement.load_state_dict(ckpt['state_dict'])
                if(args.stage == 1):
                    best_loss = ckpt['loss']
                # refinement.load_state_dict(ckpt)
            refinement = refinement.to(device)
    encdec = encdec.to(device)

    # _ed suffix refers to encoder-decoder part of the model, 
    # _r suffix refers to refinement part
    if(args.weighted_loss):
        crit_ed = AlphaCompLoss_u()
    else:
        crit_ed = AlphaCompLoss()
    optim_ed = optim.Adam(encdec.parameters(), lr=1e-5)
    sched_ed = CyclicLR(optim_ed, 5e-6, 5e-5, 200)
    if(args.stage != 0):
        if(args.weighted_loss):
            crit_r = AlphaLoss_u()
        else:
            crit_r = AlphaLoss()
        optim_r = optim.Adam(refinement.parameters(), lr=1e-5)
        sched_r = CyclicLR(optim_r, 5e-6, 5e-5, 200)

    # TRAIN

    # Writers for TensorBoard
    train_writer = SummaryWriter('logs/train' + args.save_dir)
    val_writer = SummaryWriter('logs/val' + args.save_dir)
    
    if(args.epochs == -1):
        num_epochs = 1000
    else:
        num_epochs = args.epochs
    for e in tqdm(range(num_epochs)):        
        if(not early_stopping):
            break
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
                inputs_ed, labels, fg, bg = sample['im_map'].to(device), sample['mask'].to(device), sample['fg'].to(device), sample['bg'].to(device)
                # inputs_ed = inputs_ed.to(device)
                # labels = labels.to(device)
                # fg = fg.to(device)
                # bg = bg.to(device)
                trimap = inputs_ed[:,3,:,:]

                # zero the gradients
                optim_ed.zero_grad()
                if(args.stage != 0):
                    optim_r.zero_grad()
                
                # Calculate loss
                with torch.set_grad_enabled(phase == 'train'): # Only track grads if in train mode
                    outputs_ed = encdec(inputs_ed) # Output is single channel matte

                    if(args.stage != 0):
                        inputs_r = torch.cat((inputs_ed[:,:3,:,:], outputs_ed), 1)
                        outputs_r = refinement(inputs_r)

                    # _, preds = torch.max(outputs, 1)

                    if(args.stage != 1):
                        if(args.weighted_loss):
                            loss_ed = crit_ed(outputs_ed, labels, fg, bg, trimap)
                        else:
                            loss_ed = crit_ed(outputs_ed, labels, fg, bg)
                        running_loss_ed.append(loss_ed.item())
                    if(args.stage != 0):
                        if(args.weighted_loss):
                            loss_r = crit_r(outputs_r, labels, trimap)
                        else:
                            loss_r = crit_r(outputs_r, labels)
                        running_loss_r.append(loss_r.item())

                    # backward + optimize only if in training phase
                    if(phase == 'train'):
                        if(args.stage == 0):
                            loss_ed.backward()
                            optim_ed.step()
                            sched_ed.batch_step()
                        if(args.stage == 1):
                            loss_r.backward()
                            optim_r.step()
                            sched_r.batch_step()
                        if(args.stage == 2):
                            loss_ed.backward()
                            optim_ed.step()
                            optim_r.step()
                            sched_ed.batch_step()
                            sched_r.batch_step()

                    if(phase == 'val'):
                        if(args.stage == 0):
                            if(args.weighted_loss):
                                running_sad.append(sum_absolute_differences_u(outputs_ed, labels, trimap).item())
                                running_mse.append(mean_squared_error_u(outputs_ed, labels, trimap).item())
                            else:
                                running_sad.append(sum_absolute_differences(outputs_ed, labels).item())
                                running_mse.append(mean_squared_error(outputs_ed, labels).item())
                        else:
                            if(args.weighted_loss):
                                running_sad.append(sum_absolute_differences_u(outputs_r, labels, trimap).item())
                                running_mse.append(mean_squared_error_u(outputs_r, labels, trimap).item())
                            else:
                                running_sad.append(sum_absolute_differences(outputs_r, labels).item())
                                running_mse.append(mean_squared_error(outputs_r, labels).item())


            # Record average epoch loss for TensorBoard
            if(args.stage != 1):
                epoch_loss_ed = np.array(running_loss_ed).mean()
            if(args.stage != 0):
                epoch_loss_r = np.array(running_loss_r).mean()
            if(phase == 'train'):
                if(args.stage != 1):
                    train_writer.add_scalar("Encoder-Decoder Loss", epoch_loss_ed, epoch + e)
                if(args.stage != 0):
                    train_writer.add_scalar("Refinement Loss", epoch_loss_r, epoch + e)
                
            if(phase == 'val'):
                early_stopping -= 1
                val_writer.add_scalar( "Mean Squared Error", np.array(running_mse).mean(), epoch+e)
                val_writer.add_scalar("Sum of Absolute Differences", np.array(running_sad).mean(), epoch+e)
                if(args.stage != 1):
                    if(epoch_loss_ed < best_loss):
                        early_stopping = args.early_cutoff
                        best_loss = epoch_loss_ed
                        checkpoint(epoch+e, args.save_dir, encdec, best_loss, encdec=True)
                        if(args.stage != 0):
                            checkpoint(epoch+e, args.save_dir, refinement, best_loss, encdec=False)
                    val_writer.add_scalar("Encoder-Decoder Loss", epoch_loss_ed, epoch+e)
                if(args.stage != 0):
                    if(args.stage == 1 and epoch_loss_r < best_loss):
                        early_stopping = args.early_cutoff
                        best_loss = epoch_loss_r
                        checkpoint(epoch+e, args.save_dir, refinement, best_loss, encdec=False)
                    val_writer.add_scalar("Refinement Loss", epoch_loss_r, epoch+e)

    train_writer.close()
    val_writer.close()

    # print("Saving model at {}".format(args.save_dir))
    # checkpoint(args.epochs+epoch-1, args.save_dir, encdec, best_loss_ed, encdec=True)
    # if(args.stage != 0):
    #     checkpoint(args.epochs+epoch-1, args.save_dir, refinement, best_loss_r, encdec=False)


def composite(fg, bg, alpha):
    foreground = torch.mul(alpha, fg)
    background = torch.mul(1.0 - alpha, bg)
    return torch.add(foreground, background)

def checkpoint(epoch, save_dir, model, loss, encdec=True):
    if(encdec):
        model_out_path = "{}/encdec/ckpt_encdec_e{:04d}.pth".format(save_dir, epoch)
    else:
        model_out_path = "{}/refinement/ckpt_refinement_e{:04d}.pth".format(save_dir, epoch)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(save_dir + '/encdec')
        os.makedirs(save_dir + '/refinement')
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'loss': loss
    }, model_out_path )
    tqdm.write("Checkpoint saved to {} with loss {}".format(model_out_path, loss))

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

class AlphaLoss(_Loss):
    def __init__(self, eps=1e-6):
        super(AlphaLoss, self).__init__()
        self.eps = eps

    def forward(self, p_mask, gt_mask):
        return alpha_loss(p_mask, gt_mask, self.eps)
    
class AlphaCompLoss(_Loss):
    def __init__(self, eps=1e-6):
        super(AlphaCompLoss, self).__init__()
        self.eps = eps
        
    def forward(self, p_mask, gt_mask, fg, bg):
        return alpha_loss(p_mask, gt_mask, self.eps) + \
               comp_loss(p_mask, gt_mask, fg, bg, self.eps)
    
class AlphaCompLoss_u(_Loss):
    def __init__(self, eps=1e-6):
        super(AlphaCompLoss_u, self).__init__()
        self.eps = eps
        
    def forward(self, p_mask, gt_mask, fg, bg, trimap):
        return (0.5 * alpha_loss_u(p_mask, gt_mask, trimap, self.eps)) + \
               (0.5 * comp_loss_u(p_mask, gt_mask, fg, bg, trimap, self.eps))

def alpha_loss(p_mask, gt_mask, eps=1e-6):
    sqr_diff = gt_mask.sub(p_mask).pow(2)
    image_loss = torch.sum(torch.sum(torch.sum(sqr_diff, dim=1), dim=1), dim=1)
    alpha_loss = torch.sqrt(image_loss.mean() + eps)
    return alpha_loss

def alpha_loss_u(p_mask, gt_mask, trimap, eps=1e-6):
    # only counts loss in "unknown" region of trimap
    sqr_diff = gt_mask.sub(p_mask).pow(2)
    unknown = torch.eq(trimap, torch.FloatTensor(np.ones(gt_mask.shape)*(128./255)).to(device)).float()
    loss = torch.mul(sqr_diff, unknown)
    image_loss = torch.sum(torch.sum(torch.sum(loss, dim=1), dim=1), dim=1)
    alpha_loss = torch.sqrt(image_loss.mean() + eps)
    return alpha_loss

def comp_loss(p_mask, gt_mask, fg, bg, eps=1e-6):
    gt_comp = composite(fg, bg, gt_mask) * (1./255)
    p_comp = composite(fg, bg, p_mask) * (1./255)
    s_diff = gt_comp.sub(p_comp).pow(2)
    image_loss = torch.sum(torch.sum(torch.sum(s_diff, dim=1), dim=1), dim=1)
    comp_loss = torch.sqrt(image_loss.mean() + eps)
    return comp_loss

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

def sum_absolute_differences(p_mask, gt_mask):
    bs, _, _, _ = p_mask.shape
    diffs = gt_mask.sub(p_mask).abs()
    return diffs.sum() / bs

def sum_absolute_differences_u(p_mask, gt_mask, trimap):
    bs, h, w = trimap.shape
    ones = torch.FloatTensor(np.ones(trimap.shape)*(128./255)).to(device)
    unknown = torch.eq(trimap, ones).float().expand(3, bs, h, w).contiguous().view(bs,3,h,w)
    diffs = gt_mask.sub(p_mask).abs()
    u_diffs = torch.mul(diffs, unknown)
    return u_diffs.sum() / bs

def mean_squared_error(p_mask, gt_mask):
    diffs = gt_mask.sub(p_mask).pow(2)
    return diffs.mean()

def mean_squared_error_u(p_mask, gt_mask, trimap):
    bs, h, w = trimap.shape
    ones = torch.FloatTensor(np.ones(trimap.shape)*(128./255)).to(device)
    unknown = torch.eq(trimap, ones).float().expand(3, bs, h, w).contiguous().view(bs,3,h,w)
    diffs = gt_mask.sub(p_mask).pow(2)
    u_diffs = torch.mul(diffs, unknown)
    return u_diffs.mean()


if __name__ == "__main__":
    main()
