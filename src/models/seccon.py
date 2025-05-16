# models/conv1d.py

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.ops as ops
import utils.misc as misc
from torchvision.ops.stochastic_depth import StochasticDepth

class Conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, MODEL, MODULES):
        super(Conv1DBlock, self).__init__()
        self.res = (in_channels==out_channels)
        expanded_channels = out_channels * MODEL.expand_ratio

        self.in_linear = MODULES.linear(in_features=in_channels, out_features=expanded_channels)
        self.conv1d = MODULES.conv1d(in_channels=expanded_channels, out_channels=expanded_channels, k_size=MODEL.k_size, stride=MODEL.stride, padding="valid", bias=False)
        self.eca = MODULES.eca()
        self.bn = MODULES.feature_norm(in_features=expanded_channels)
        self.out_linear = MODULES.linear(in_features=expanded_channels, out_features=out_channels)
        self.dropout = MODULES.dropout(p=MODEL.dropout)
        self.drop_path = MODULES.drop_path(p=MODEL.drop_path,mode="batch")

        self.activation = MODULES.act_fn

    def forward(self, x, masks=None):
        if self.res:
            x0 = x

        x = self.activation(self.in_linear(x))
        # (n, f, embeding) -> (n, c, f)
        x = self.activation(self.conv1d(torch.transpose(x,1,2)))
        x = self.eca(x)
        x = self.bn(x)
        # (n, c, f) -> (n, f, embeding)
        x = torch.transpose(x,1,2)
        if masks is not None:
            x[masks] = 0
        x = self.dropout(self.out_linear(x))

        if self.res:
            return self.drop_path(x) + x0
        else:
            return x

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, MODEL, MODULES):
        super(Block, self).__init__()

        self.conv1d0 = Conv1DBlock(in_channels, in_channels, MODEL, MODULES)
        self.conv1d1 = Conv1DBlock(in_channels, in_channels, MODEL, MODULES)
        self.conv1d2 = Conv1DBlock(in_channels, out_channels, MODEL, MODULES)

    def forward(self, x, masks=None):

        x = self.conv1d0(x, masks)
        x = self.conv1d1(x, masks)
        x = self.conv1d2(x, masks)

        return x

class Model(nn.Module):
    def __init__(self, DATA, RUN, MODULES, MODEL):
        super(Model, self).__init__()

        f_input_size = DATA.input_size[1]*DATA.input_size[2]
        self.in_dims = [MODEL.embed_size]+[MODEL.conv_dim]*(MODEL.depth-1)
        self.out_dims = [MODEL.conv_dim]*MODEL.depth

        self.mixed_precision = RUN.mixed_precision

        self.blocks = []
        self.attns = []
        self.apply_attn = MODEL.apply_attn
        for index in range(len(self.in_dims)):
            self.blocks += [
                Block(in_channels=self.in_dims[index],
                            out_channels=self.out_dims[index],
                            MODEL=MODEL,
                            MODULES=MODULES)
            ]

            if self.apply_attn:
                self.attns += [MODULES.transformer_layer(in_features=self.out_dims[index], nhead=MODEL.nheads, dim_feedforward=self.out_dims[index], dropout=MODEL.dropout)]

        self.blocks = nn.ModuleList(self.blocks)
        self.attns = nn.ModuleList(self.attns)

        self.inlinear = MODULES.linear(f_input_size, self.in_dims[0], bias=False)
        self.bn = MODULES.feature_norm(in_features=DATA.input_size[0])
        self.top_linear = MODULES.linear(self.out_dims[-1], self.out_dims[-1]*2)
        self.outlinear = MODULES.linear(DATA.input_size[0], DATA.num_classes)
        self.class_token = nn.Parameter(torch.zeros(1, 1, f_input_size))
        class_mask = torch.full((1, 1, f_input_size), True)
        self.out_act = nn.Tanh()

        self.dropout = MODULES.dropout(p=MODEL.dropout)

        if MODEL.init:
            nn.init.xavier_uniform(self.class_token)
            ops.init_weights(self.modules, MODEL.init)

        self.register_buffer('class_mask', class_mask, persistent=False)

    def forward(self, x, masks=None, eval=False):
        with torch.autocast("cuda") if self.mixed_precision and not eval else misc.dummy_context_mgr() as mp:
            h = x
            n = h.shape[0]
            if masks is not None:
                batch_class_mask = self.class_mask.expand(n, -1, -1)
                masks = torch.cat([masks, batch_class_mask], dim=1)
                masks = masks.any(dim=-1) 
                ft = misc.keep_first_true(masks)
                x = F.pad(input=x, pad=(0, 0, 0, 1, 0, 0), mode='constant', value=0)
                x[masks] = 0
                x[ft] = self.class_token.to(x.dtype)
                attn_masks = masks.T
            else:
                # Expand the class token to the full batch
                batch_class_token = self.class_token.expand(n, -1, -1)
                x = torch.cat([x, batch_class_token], dim=1)

            # (n, f, (k*c)) -> (n, f, embeding)
            h = self.inlinear(h)

            for i, block in enumerate(self.blocks):
                h = block(h, masks)
                if self.apply_attn:
                    h = self.attns[i](h, src_key_padding_mask=attn_masks)

            if masks is not None:
                h = h[ft]
            else:
                h = h[:, -1]

            h = self.dropout(self.out_act(self.top_linear(h)))
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