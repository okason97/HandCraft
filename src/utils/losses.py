import torch
import torch.nn as nn

def cce():
    return nn.CrossEntropyLoss()

def siMLPe(motion_pred, h36m_motion_target, input_size, use_relative_loss):
    # https://github.com/dulucas/siMLPe/blob/main/exps/baseline_h36m/train.py
    
    b,n,c = h36m_motion_target.shape
    motion_pred = motion_pred.reshape(b,n,input_size[1],input_size[2]).reshape(-1,input_size[2])
    h36m_motion_target = h36m_motion_target.cuda().reshape(b,n,input_size[1],input_size[2]).reshape(-1,input_size[2])
    loss = torch.mean(torch.norm(motion_pred - h36m_motion_target, 2, 1))

    if use_relative_loss:
        motion_pred = motion_pred.reshape(b,n,input_size[1],input_size[2])
        dmotion_pred = gen_velocity(motion_pred)
        motion_gt = h36m_motion_target.reshape(b,n,input_size[1],input_size[2])
        dmotion_gt = gen_velocity(motion_gt)
        dloss = torch.mean(torch.norm((dmotion_pred - dmotion_gt).reshape(-1,input_size[2]), 2, 1))
        loss = loss + dloss
    else:
        loss = loss.mean()

def gen_velocity(m):
    dm = m[:, 1:] - m[:, :-1]
    return dm
