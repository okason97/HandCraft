# models/conv1d.py

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.ops as ops
import utils.misc as misc
from torchvision.ops.stochastic_depth import StochasticDepth

class Conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, dropout, drop_path, MODULES):
        super(Conv1DBlock, self).__init__()
        self.res = (in_channels==out_channels)
        self.dropout = dropout
        expanded_channels = out_channels * expand_ratio

        self.in_linear = MODULES.linear(in_features=in_channels, out_features=expanded_channels)
        self.conv1d = MODULES.conv1d(in_channels=expanded_channels, out_channels=expanded_channels, k_size=17, stride=1, padding="valid")
        self.eca = MODULES.eca()
        self.bn = MODULES.feature_norm(in_features=expanded_channels)
        self.out_linear = MODULES.linear(in_features=expanded_channels, out_features=out_channels)

        self.dropout = MODULES.dropout(p=dropout)
        self.drop_path = MODULES.drop_path(p=drop_path,mode="batch")

        self.activation = MODULES.act_fn

    def forward(self, x):
        if self.res:
            x0 = x

        x = self.activation(self.in_linear(x))
        x = self.conv1d(torch.transpose(x,1,2))
        x = self.eca(x)
        x = self.bn(x)
        x = self.dropout(self.out_linear(torch.transpose(x,1,2)))

        if self.res:
            return self.drop_path(x) + x0
        else:
            return x

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, dropout, drop_path, MODULES):
        super(Block, self).__init__()
        self.dropout = dropout

        self.conv1d0 = Conv1DBlock(in_channels, in_channels, expand_ratio, dropout, drop_path, MODULES)
        self.conv1d1 = Conv1DBlock(in_channels, in_channels, expand_ratio, dropout, drop_path, MODULES)
        self.conv1d2 = Conv1DBlock(in_channels, out_channels, expand_ratio, dropout, drop_path, MODULES)

        self.bn = MODULES.feature_norm(in_features=in_channels)

    def forward(self, x):

        x = self.conv1d0(x)
        x = self.conv1d1(x)
        x = self.conv1d2(x)

        return x

class Classifier(nn.Module):
    def __init__(self, input_size, embed_size, conv_dim, apply_attn, expand_ratio, nheads, dropout,
                 drop_path, num_classes, init_weights, depth, mixed_precision, MODULES, MODEL):
        super(Classifier, self).__init__()
        self.MODEL = MODEL
        f_input_size = input_size[1]*input_size[2]
        self.in_dims = [embed_size]+[conv_dim]*(depth-1)
        self.out_dims = [conv_dim]*depth

        self.mixed_precision = mixed_precision

        self.blocks = []
        for index in range(len(self.in_dims)):
            self.blocks += [[
                Block(in_channels=self.in_dims[index],
                            out_channels=self.out_dims[index],
                            expand_ratio=expand_ratio,
                            dropout=dropout,
                            drop_path=drop_path,
                            MODULES=MODULES)
            ]]

            if apply_attn:
                self.blocks += [[MODULES.transformer_layer(in_features=self.out_dims[index], nhead=nheads, dim_feedforward=self.out_dims[index], dropout=dropout)]]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        self.inlinear = MODULES.linear(f_input_size, self.in_dims[0], bias=False)
        # self.inlineart = MODULES.linear(input_size[0], self.in_dims[0], bias=False)
        # self.emblinear = MODULES.linear(f_input_size, self.in_dims[0], bias=False)
        self.bn = MODULES.feature_norm(in_features=input_size[0])
        self.top_linear = MODULES.linear(self.out_dims[-1], self.out_dims[-1]*2)
        #self.pooling = MODULES.pooling(self.out_dims[-1]*2)
        self.outlinear = MODULES.linear(input_size[0], num_classes)

        self.dropout = MODULES.dropout(p=dropout)

        if init_weights:
            ops.init_weights(self.modules, init_weights)

    def forward(self, x, eval=False):
        with torch.cuda.amp.autocast() if self.mixed_precision and not eval else misc.dummy_context_mgr() as mp:
            h = x
            # ht = torch.transpose(self.inlineart(torch.transpose(h,1,2)),1,2)
            h = self.inlinear(h)
            # h = self.emblinear(h.matmul(ht))
            # h = self.bn(h)
            for _, blocklist in enumerate(self.blocks):
                for block in blocklist:
                    h = block(h)
            h = self.dropout(self.top_linear(h))
            #h = self.pooling(h)
            h = torch.mean(h.view(h.size(0), h.size(1), -1), dim=2)
            h = self.outlinear(h)

        return h

    def update_dropout(self, drop_rate):
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = drop_rate

    def update_drop_path(self, drop_rate):
        for module in self.modules():
            if isinstance(module, StochasticDepth):
                module.p = drop_rate