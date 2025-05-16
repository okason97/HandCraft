# src/utils/op.py

from torch.nn.utils import spectral_norm
from torch.nn import init
import torch
import torch.nn as nn
from torchvision.ops.stochastic_depth import StochasticDepth
import numpy as np
from einops.layers.torch import Rearrange
from typing import Callable, Optional, Union, Tuple

class eca(nn.Module):
    """Constructs a ECA module. by BangguWu
    https://github.com/BangguWu/ECANet/blob/master/models/eca_module.py
    
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, k_size=3):
        super(eca, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = conv1dbasic(1, 1, k_size, padding=(k_size - 1) // 2, bias=False)
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

def dropout(p):
    return nn.Dropout(p)

def drop_path(p,mode):
    return StochasticDepth(p,mode)

def conv1dbasic(in_channels, out_channels, k_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
    return nn.Conv1d(in_channels, out_channels, kernel_size=k_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode) 

def snconv1dbasic(in_channels, out_channels, k_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
    return spectral_norm(nn.Conv1d(in_channels, out_channels, kernel_size=k_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode))

class conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, k_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(conv1d, self).__init__()
        self.pad = nn.ConstantPad1d((dilation*(k_size-1),0),0)
        self.conv = conv1dbasic(in_channels, out_channels, k_size, stride, padding, dilation, groups, bias, padding_mode)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        return x

class snconv1d(nn.Module):
    def __init__(self, in_channels, out_channels, k_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(snconv1d, self).__init__()
        self.pad = nn.ConstantPad1d((dilation*(k_size-1),0),0)
        self.conv = snconv1dbasic(in_channels, out_channels, k_size, stride, padding, dilation, groups, bias, padding_mode)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        return x

class dwconv1d(nn.Module):
    def __init__(self, in_channels, out_channels, k_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(dwconv1d, self).__init__()
        groups = in_channels
        self.pad = nn.ConstantPad1d((dilation*(k_size-1),0),0)
        self.conv = conv1dbasic(in_channels, out_channels, k_size, stride, padding, dilation, groups, bias, padding_mode)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        return x

class sndwconv1d(nn.Module):
    def __init__(self, in_channels, out_channels, k_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(sndwconv1d, self).__init__()
        groups = in_channels
        self.pad = nn.ConstantPad1d((dilation*(k_size-1),0),0)
        self.conv = snconv1dbasic(in_channels, out_channels, k_size, stride, padding, dilation, groups, bias, padding_mode)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        return x

class dwsepconv1d(nn.Module):
    def __init__(self, in_channels, out_channels, k_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(dwsepconv1d, self).__init__() 
        self.pad = nn.ConstantPad1d((dilation*(k_size-1),0),0)
        self.depthwise = conv1dbasic(in_channels=in_channels, out_channels=in_channels, k_size=k_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias, padding_mode=padding_mode)
        self.pointwise = conv1dbasic(in_channels=in_channels, out_channels=out_channels, k_size=1, stride=stride, padding=padding, dilation=dilation, bias=bias, padding_mode=padding_mode)
    def forward(self, x): 
        x = self.pad(x)
        x = self.depthwise(x) 
        x = self.pointwise(x) 
        return x
 
def batchnorm(in_features, eps=1e-4, momentum=0.1, affine=True):
    if not isinstance(in_features, int):
        in_features = in_features[0]
    return nn.BatchNorm1d(in_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=True)

def layernorm(in_features, eps=1e-4, elementwise_affine=True, bias=True):
    if not isinstance(in_features, int):
        in_features = in_features[-1]
    return nn.LayerNorm(in_features, eps=eps, elementwise_affine=elementwise_affine, bias=bias)

class SLayerNorm(nn.Module):
    def __init__(self, dims, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, dims[-2], 1]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, dims[-2], 1]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y

class TLayerNorm(nn.Module):
    def __init__(self, dims, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, 1, dims[-1]]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, 1, dims[-1]]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y

class Spatial_FC(nn.Module):
    def __init__(self, dims):
        super(Spatial_FC, self).__init__()
        self.fc = nn.Linear(dims[-2], dims[-2])
        self.arr0 = Rearrange('b n d -> b d n')
        self.arr1 = Rearrange('b d n -> b n d')

    def forward(self, x):
        x = self.arr0(x)
        x = self.fc(x)
        x = self.arr1(x)
        return x

class Temporal_FC(nn.Module):
    def __init__(self, dims):
        super(Temporal_FC, self).__init__()
        self.fc = nn.Linear(dims[-1], dims[-1])

    def forward(self, x):
        x = self.fc(x)
        return x

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

class DropBlock(torch.nn.Module):
    """Incomplete/Untested"""
    def __init__(self, p, block_size=3):
        super().__init__()
        self.p = 1-p
        self.block_size = block_size

    def forward(self, img):
        invalid = (1 - self.p) / (self.block_size ** 2)
        valid = (img.shape[1] ** 2) / ((img.shape[1] - self.block_size + 1) ** 2)        
        gamma = invalid * valid 
        mask = torch.bernoulli(torch.ones((img.shape[0],img.shape[1])) * gamma)
        mask_block = 1 - F.max_pool1d(
            mask,
            kernel_size=self.block_size,
            stride=1,
            padding=self.block_size // 2,
        )
        mask_ = mask_block.unsqueeze(-1).expand(img.size())
        img = mask_ * img * (mask_.numel() / mask_.sum())
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p},block_size={self.block_size})"
    

class Conv1dNormActivation(nn.Sequential):
    """
    Configurable block used for Convolution1d-Normalization-Activation blocks.

    Args:
        in_channels (int): Number of channels in the input signal.
        out_channels (int): Number of channels produced by the Convolution-Normalization-Activation block.
        kernel_size: (int, optional): Size of the convolving kernel. Default: 3.
        stride (int, optional): Stride of the convolution. Default: 1.
        padding (int, tuple or str, optional): Padding added to both sides of the input. Default: None, in which case it will be calculated as ``padding = (kernel_size - 1) // 2 * dilation``.
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1.
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer won't be used. Default: ``torch.nn.BatchNorm1d``.
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer won't be used. Default: ``torch.nn.ReLU``.
        dilation (int): Spacing between kernel elements. Default: 1.
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``.
        bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]] = 3,
        stride: Union[int, Tuple[int]] = 1,
        padding: Optional[Union[int, Tuple[int], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm1d,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        dilation: Union[int, Tuple[int]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        
        if bias is None:
            bias = norm_layer is None

        layers = [nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride, 
            padding, 
            dilation=dilation, 
            groups=groups, 
            bias=bias
        )]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels))

        if activation_layer is not None:
            layers.append(activation_layer(inplace=inplace))

        super().__init__(*layers)
        self.out_channels = out_channels
