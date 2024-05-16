import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch import linalg as LA

class MotionLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean', relative_motion=False):
        super().__init__(size_average, reduce, reduction)
        self.relative_motion = relative_motion

    def forward(self, input, target):
        loss = motion_loss(input, target, reduction=self.reduction)
        if self.relative_motion:
            loss = loss + relative_motion_loss(input, target, reduction=self.reduction)
        return loss

def motion_loss(motion_pred, motion_target, reduction):
    # https://github.com/dulucas/siMLPe/blob/main/exps/baseline_h36m/train.py    
    loss = LA.vector_norm(motion_pred - motion_target, 2, -1)
    if reduction == 'mean':
        loss = torch.mean(loss)
    elif reduction == 'sum':
        loss = torch.sum(loss)        
    return loss

def relative_motion_loss(motion_pred, motion_target, reduction):
    # https://github.com/dulucas/siMLPe/blob/main/exps/baseline_h36m/train.py
    
    dmotion_pred = gen_velocity(motion_pred)
    dmotion_gt = gen_velocity(motion_target)
    loss = LA.vector_norm(dmotion_pred - dmotion_gt, 2, -1)
    if reduction == 'mean':
        loss = torch.mean(loss)
    elif reduction == 'sum':
        loss = torch.sum(loss)        
    return loss

def gen_velocity(m):
    dm = m[:, 1:] - m[:, :-1]
    return dm
