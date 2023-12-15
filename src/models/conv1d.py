# models/conv1d.py

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.ops as ops
import utils.misc as misc

class Conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, dropout, MODULES):
        super(Block, self).__init__()
        self.res = (in_channels==out_channels)
        self.dropout = dropout
        expanded_channels = in_channels * expand_ratio

        self.in_linear = MODULES.linear(in_channels=in_channels, out_channels=expanded_channels)
        self.conv1d = MODULES.conv1d(in_channels=expanded_channels, out_channels=expanded_channels, kernel_size=17, stride=1, padding="same")
        self.eca = MODULES.eca(in_channels=expanded_channels)
        self.bn = MODULES.bn(in_features=expanded_channels)
        self.out_linear = MODULES.linear(in_channels=expanded_channels, out_channels=out_channels)

        self.dropout = MODULES.dropout(p=dropout)

        self.activation = MODULES.act_fn

    def forward(self, x):
        if self.res:
            x0 = x

        x = self.activation(self.in_linear(x))
        x = self.conv1d(x)
        x = self.eca(x)
        x = self.bn(x)
        x = self.dropout(self.out_linear(x))

        if self.res:
            return x + x0
        else:
            return x

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, dropout, MODULES):
        super(Block, self).__init__()
        self.dropout = dropout

        self.conv1d0 = Conv1DBlock(in_channels, in_channels, expand_ratio, dropout, MODULES)
        self.conv1d1 = Conv1DBlock(in_channels, in_channels, expand_ratio, dropout, MODULES)
        self.conv1d2 = Conv1DBlock(in_channels, out_channels, expand_ratio, dropout, MODULES)

        self.bn = MODULES.bn(in_features=in_channels)

    def forward(self, x):

        x = self.conv1d0(x)
        x = self.conv1d1(x)
        x = self.conv1d2(x)

        return x

class Classifier(nn.Module):
    def __init__(self, in_dim, conv_dim, apply_attn, expand_ratio, nheads, dropout, num_classes,
                 init_weights, depth, mixed_precision, MODULES, MODEL):
        super(Classifier, self).__init__()
        self.MODEL = MODEL
        self.in_dims = [conv_dim]*depth,
        self.out_dims = [conv_dim]*depth,

        self.mixed_precision = mixed_precision

        self.blocks = []
        for index in range(len(self.in_dims)):
            self.blocks += [[
                Block(in_channels=self.in_dims[index],
                            out_channels=self.out_dims[index],
                            expand_ratio=expand_ratio,
                            dropout=dropout,
                            MODULES=MODULES)
            ]]

            if apply_attn:
                self.blocks += [[MODULES.transformer_layer(in_features=self.out_dims[index], nhead=nheads, dim_feedforward=self.out_dims[index], dropout=dropout)]]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        self.inlinear = MODULES.linear(in_dim, self.in_dims[0], bias=False)
        self.bn = MODULES.feature_norm(in_features=self.in_dims[0])
        self.top_linear = MODULES.linear(self.out_dims[-1], self.out_dims[-1]*2)
        self.pooling = MODULES.pooling()
        self.outlinear = MODULES.linear(self.out_dims[-1], num_classes)

        self.dropout = MODULES.dropout(p=dropout)

        if init_weights:
            ops.init_weights(self.modules, init_weights)

    def forward(self, x, eval=False):
        with torch.cuda.amp.autocast() if self.mixed_precision and not eval else misc.dummy_context_mgr() as mp:
            cls_output = None
            h = x
            h = self.inlinear(h)
            h = self.bn(h)
            for _, blocklist in enumerate(self.blocks):
                for block in blocklist:
                    h = block(h)
            h = self.dropout(self.top_linear(h))
            h = self.pooling(h)
            h = self.outlinear(h)

        return {
            "h": h,
            "cls_output": cls_output,
        }
