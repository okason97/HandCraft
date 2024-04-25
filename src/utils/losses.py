import torch
import torch.nn as nn
from torch import linalg as LA

def cce():
    return nn.CrossEntropyLoss()

def motion_loss(motion_pred, motion_target):
    # https://github.com/dulucas/siMLPe/blob/main/exps/baseline_h36m/train.py    
    loss = torch.mean(LA.vector_norm(motion_pred - motion_target, 2, -1))
    return loss

def relative_motion_loss(motion_pred, motion_target):
    # https://github.com/dulucas/siMLPe/blob/main/exps/baseline_h36m/train.py
    
    loss = motion_loss(motion_pred, motion_target)

    dmotion_pred = gen_velocity(motion_pred)
    dmotion_gt = gen_velocity(motion_target)
    dloss = torch.mean(LA.vector_norm(dmotion_pred - dmotion_gt, 2, -1))

    return loss + dloss

def gen_velocity(m):
    dm = m[:, 1:] - m[:, :-1]
    return dm
