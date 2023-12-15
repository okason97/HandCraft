# src/utils/op.py

from torch.nn.utils import spectral_norm
from torch.nn import init
import torch
import torch.nn as nn
import numpy as np

class eca(nn.Module):
    """Constructs a ECA module. by BangguWu
    https://github.com/BangguWu/ECANet/blob/master/models/eca_module.py
    
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = conv1d(1, 1, k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

def init_weights(modules, initialize):
    for module in modules():
        if (isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Linear)):
            if initialize == "ortho":
                init.orthogonal_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0.)
            elif initialize == "N02":
                init.normal_(module.weight, 0, 0.02)
                if module.bias is not None:
                    module.bias.data.fill_(0.)
            elif initialize in ["glorot", "xavier"]:
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0.)
            else:
                pass
        elif isinstance(module, nn.Embedding):
            if initialize == "ortho":
                init.orthogonal_(module.weight)
            elif initialize == "N02":
                init.normal_(module.weight, 0, 0.02)
            elif initialize in ["glorot", "xavier"]:
                init.xavier_uniform_(module.weight)
            else:
                pass
        else:
            pass

def transformer_layer(in_features, nhead, dim_feedforward, dropout):
    return nn.TransformerEncoderLayer(in_features, nhead, dim_feedforward=dim_feedforward, dropout=dropout)

def linear(in_features, out_features, bias=True):
    return nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

def snlinear(in_features, out_features, bias=True):
    return spectral_norm(nn.Linear(in_features=in_features, out_features=out_features, bias=bias), eps=1e-6)

def conv1d(in_channels, out_channels, k_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
    return nn.Conv1d(in_channels, out_channels, kernel_size=k_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode) 

def snconv1d(in_channels, out_channels, k_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
    return spectral_norm(nn.Conv1d(in_channels, out_channels, kernel_size=k_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode))

def batchnorm(in_features, eps=1e-4, momentum=0.1, affine=True):
    return nn.BatchNorm1d(in_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=True)

def layernorm(in_features, eps=1e-4, elementwise_affine=True, bias=True):
    return nn.LayerNorm(in_features, eps=eps, elementwise_affine=elementwise_affine, bias=bias)

def adjust_learning_rate(optimizer, lr_org, epoch, total_epoch, dataset):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if dataset in ["CIFAR10", "CIFAR100"]:
        lr = lr_org * (0.1 ** (epoch // (total_epoch * 0.5))) * (0.1 ** (epoch // (total_epoch * 0.75)))
    elif dataset in ["Tiny_ImageNet", "ImageNet"]:
        if total_epoch == 300:
            lr = lr_org * (0.1 ** (epoch // 75))
        else:
            lr = lr_org * (0.1 ** (epoch // 30))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr