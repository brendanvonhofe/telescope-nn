import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def composite(fg, bg, alpha):
    foreground = torch.mul(alpha, fg)
    background = torch.mul(1.0 - alpha, bg)
    return torch.add(foreground, background)

def alpha_loss(p_mask, gt_mask, eps=1e-6):
    sqr_diff = gt_mask.sub(p_mask).pow(2)
    image_loss = torch.sum(torch.sum(torch.sum(sqr_diff, dim=1), dim=1), dim=1)
    alpha_loss = torch.sqrt(image_loss.mean() + eps)
    return alpha_loss

def alpha_loss_u(p_mask, gt_mask, trimap, eps=1e-6):
    # only counts loss in "unknown" region of trimap
    sqr_diff = gt_mask.sub(p_mask).pow(2)
    weighted = torch.zeros(trimap.shape).cuda()
    weighted[trimap == 128/255.] = 1.
    loss = sqr_diff * weighted
    image_loss = torch.sum(torch.sum(torch.sum(loss, dim=1), dim=1), dim=1)
    alpha_loss = torch.sqrt(image_loss.mean() + eps)
    return alpha_loss

def comp_loss(p_mask, gt_mask, fg, bg, eps=1e-6):
    gt_comp = composite(fg, bg, gt_mask)
    p_comp = composite(fg, bg, p_mask)
    s_diff = gt_comp.sub(p_comp).pow(2)
    image_loss = torch.sum(torch.sum(torch.sum(s_diff, dim=1), dim=1), dim=1)
    comp_loss = torch.sqrt(image_loss.mean() + eps)
    return comp_loss

def comp_loss_u(p_mask, gt_mask, fg, bg, trimap, eps=1e-6):
    # only counts loss in "unknown" region of trimap
    gt_comp = composite(fg, bg, gt_mask)
    p_comp = composite(fg, bg, p_mask)
    s_diff = gt_comp.sub(p_comp).pow(2)
    weighted = torch.zeros(trimap.shape).cuda()
    weighted[trimap == 128/255.] = 1.
    weighted = torch.unsqueeze(weighted, 1)
    weighted = torch.cat((weighted, weighted, weighted), 1)
    loss = s_diff * weighted
    image_loss = torch.sum(torch.sum(torch.sum(loss, dim=1), dim=1), dim=1)
    comp_loss = torch.sqrt(image_loss.mean() + eps)
    return comp_loss

def sum_absolute_differences(p_mask, gt_mask):
    bs, c, h, w = p_mask.shape
    n_pix = bs * h * w
    diffs = gt_mask.sub(p_mask).abs()
    sad = diffs.sum() / (n_pix / 1000)
    return sad

def sum_absolute_differences_u(p_mask, gt_mask, trimap):
    diffs = gt_mask.sub(p_mask).abs()
    weighted = torch.zeros(trimap.shape).cuda()
    weighted[trimap == 128/255.] = 1.
    weighted = torch.unsqueeze(weighted, 1)
    n_pix = torch.sum(weighted)
    weighted_diffs = diffs * weighted
    sad = weighted_diffs.sum() / (n_pix / 1000)
    return sad

def mean_squared_error(p_mask, gt_mask):
    diffs = gt_mask.sub(p_mask).pow(2)
    return diffs.mean()

def mean_squared_error_u(p_mask, gt_mask, trimap):
    diffs = gt_mask.sub(p_mask).pow(2)
    weighted = torch.zeros(trimap.shape).cuda()
    weighted[trimap == 128/255.] = 1.
    weighted = torch.unsqueeze(weighted, 1)
    weighted_diffs = diffs * weighted
    return weighted_diffs.mean()

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
        return alpha_loss_u(gt_mask, trimap, p_mask)

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
        return (0.5 * alpha_loss_u(p_mask, gt_mask, trimap)) + \
               (0.5 * comp_loss_u(p_mask, gt_mask, fg, bg, trimap, self.eps))