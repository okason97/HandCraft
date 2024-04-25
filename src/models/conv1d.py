# models/conv1d.py

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.ops as ops
import utils.misc as misc
from torchvision.ops.stochastic_depth import StochasticDepth

class Conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, dropout_p, drop_path, MODULES):
        super(Conv1DBlock, self).__init__()
        self.res = (in_channels==out_channels)
        self.dropout_p = dropout_p
        expanded_channels = out_channels * expand_ratio

        self.in_linear = MODULES.linear(in_features=in_channels, out_features=expanded_channels)
        self.conv1d = MODULES.conv1d(in_channels=expanded_channels, out_channels=expanded_channels, k_size=17, stride=1, padding="valid")
        self.eca = MODULES.eca()
        self.bn = MODULES.feature_norm(in_features=expanded_channels)
        self.out_linear = MODULES.linear(in_features=expanded_channels, out_features=out_channels)

        self.dropout = MODULES.dropout(p=dropout_p)
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

class Model(nn.Module):
    def __init__(self, DATA, RUN, MODULES, MODEL):
        super(Model, self).__init__()

        f_input_size = DATA.input_size[1]*DATA.input_size[2]
        self.in_dims = [MODEL.embed_size]+[MODEL.conv_dim]*(MODEL.depth-1)
        self.out_dims = [MODEL.conv_dim]*MODEL.depth

        self.mixed_precision = RUN.mixed_precision

        self.blocks = []
        for index in range(len(self.in_dims)):
            self.blocks += [[
                Block(in_channels=self.in_dims[index],
                            out_channels=self.out_dims[index],
                            expand_ratio=MODEL.expand_ratio,
                            dropout=MODEL.dropout,
                            drop_path=MODEL.drop_path,
                            MODULES=MODULES)
            ]]

            if MODEL.apply_attn:
                self.blocks += [[MODULES.transformer_layer(in_features=self.out_dims[index], nhead=MODEL.nheads, dim_feedforward=self.out_dims[index], dropout=MODEL.dropout)]]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        self.inlinear = MODULES.linear(f_input_size, self.in_dims[0], bias=False)
        # self.inlineart = MODULES.linear(input_size[0], self.in_dims[0], bias=False)
        # self.emblinear = MODULES.linear(f_input_size, self.in_dims[0], bias=False)
        self.bn = MODULES.feature_norm(in_features=DATA.input_size[0])
        self.top_linear = MODULES.linear(self.out_dims[-1], self.out_dims[-1]*2)
        #self.pooling = MODULES.pooling(self.out_dims[-1]*2)
        self.outlinear = MODULES.linear(DATA.input_size[0], DATA.num_classes)

        self.dropout = MODULES.dropout(p=MODEL.dropout)

        if MODEL.init:
            ops.init_weights(self.modules, MODEL.init)

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