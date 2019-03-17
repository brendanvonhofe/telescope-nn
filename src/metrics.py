import numpy as np
import torch
import torch.nn as nn

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