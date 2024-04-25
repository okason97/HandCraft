import torch
from torch import nn
from einops.layers.torch import Rearrange
import utils.misc as misc

class MLPblock(nn.Module):

    def __init__(self, dims, MODULES):
        super().__init__()

        self.fc0 = MODULES.glinear(dims)
        self.norm0 = MODULES.feature_norm(dims)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc0.fc.weight, gain=1e-8)

        nn.init.constant_(self.fc0.fc.bias, 0)

    def forward(self, x):

        x_ = self.fc0(x)
        x_ = self.norm0(x_)
        x = x + x_

        return x

class TransMLP(nn.Module):
    def __init__(self, dims, num_layers, MODULES):
        super().__init__()
        self.mlps = nn.Sequential(*[
            MLPblock(dims, MODULES)
            for i in range(num_layers)])

    def forward(self, x):
        x = self.mlps(x)
        return x

class Model(nn.Module):
    '''
    MLP-based network for human motion prediction.
    siMLPe article: https://arxiv.org/abs/2207.01567
    siMLPe code: https://github.com/dulucas/siMLPe/tree/main
    '''
    def __init__(self, DATA, RUN, MODULES, MODEL):
        super(Model, self).__init__()
        self.arr0 = Rearrange('b n d -> b d n')
        self.arr1 = Rearrange('b d n -> b n d')
        self.mixed_precision = RUN.mixed_precision
        seq = DATA.input_size[-2]*DATA.input_size[-1]
        dim = DATA.input_size[-3]

        self.motion_mlp = TransMLP(
            dims=[seq, dim],
            num_layers=MODEL.depth,
            MODULES=MODULES
        )

        self.temporal_fc_in = MODEL.temporal_fc_in
        self.temporal_fc_out = MODEL.temporal_fc_out
        if self.temporal_fc_in:
            self.motion_fc_in = nn.Linear(dim, dim)
        else:
            self.motion_fc_in = nn.Linear(seq, seq)
        if self.temporal_fc_out:
            self.motion_fc_out = nn.Linear(dim, dim)
        else:
            self.motion_fc_out = nn.Linear(seq, seq)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.motion_fc_out.weight, gain=1e-8)
        nn.init.constant_(self.motion_fc_out.bias, 0)

    def forward(self, motion_input):
        with torch.cuda.amp.autocast() if self.mixed_precision and not eval else misc.dummy_context_mgr() as mp:
            if self.temporal_fc_in:
                motion_feats = self.arr0(motion_input)
                motion_feats = self.motion_fc_in(motion_feats)
            else:
                motion_feats = self.motion_fc_in(motion_input)
                motion_feats = self.arr0(motion_feats)

            motion_feats = self.motion_mlp(motion_feats)

            if self.temporal_fc_out:
                motion_feats = self.motion_fc_out(motion_feats)
                motion_feats = self.arr1(motion_feats)
            else:
                motion_feats = self.arr1(motion_feats)
                motion_feats = self.motion_fc_out(motion_feats)

        return motion_feats