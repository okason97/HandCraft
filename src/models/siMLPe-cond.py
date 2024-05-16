import torch
from torch import nn, Tensor
from einops.layers.torch import Rearrange
import utils.misc as misc
import math

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


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
        
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

        self.embedding = nn.Embedding(DATA.num_classes, MODEL.emb_size)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.motion_fc_out.weight, gain=1e-8)
        nn.init.constant_(self.motion_fc_out.bias, 0)

    def forward(self, x, y):
        with torch.cuda.amp.autocast() if self.mixed_precision and not eval else misc.dummy_context_mgr() as mp:
            if self.temporal_fc_in:
                motion_feats = self.arr0(x)
                motion_feats = self.motion_fc_in(motion_feats)
            else:
                motion_feats = self.motion_fc_in(x)
                motion_feats = self.arr0(motion_feats)

            motion_feats = self.motion_mlp(motion_feats)

            if self.temporal_fc_out:
                motion_feats = self.motion_fc_out(motion_feats)
                motion_feats = self.arr1(motion_feats)
            else:
                motion_feats = self.arr1(motion_feats)
                motion_feats = self.motion_fc_out(motion_feats)

        return motion_feats