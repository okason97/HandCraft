import torch
from torch import nn, Tensor
from einops.layers.torch import Rearrange
import utils.misc as misc
import math
import numpy as np

# https://github.com/facebookresearch/DiT/blob/main/models.py

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

class MLPblock(nn.Module):

    def __init__(self, dims, MODULES, MODEL):
        super().__init__()

        self.fc0 = MODULES.glinear(dims)
        self.norm0 = MODULES.feature_norm(dims)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dims[-1], 3 * dims[-1], bias=True)
        )
        self.noise_scale = MODEL.noise_scale

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc0.fc.weight, gain=1e-8)

        nn.init.constant_(self.fc0.fc.bias, 0)

    def forward(self, x_y):

        shift, scale, gate = self.adaLN_modulation(misc.add_noise(x_y[1], self.noise_scale)).chunk(3, dim=1)
        x_ = modulate(self.norm0(x_y[0]), shift, scale)
        x_y[0] += gate.unsqueeze(1) * self.fc0(x_)

        return x_y

class TransMLP(nn.Module):
    def __init__(self, dims, num_layers, MODULES, MODEL):
        super().__init__()
        self.mlps = nn.Sequential(*[
            MLPblock(dims, MODULES, MODEL)
            for i in range(num_layers)])

    def forward(self, x_y):
        x_y = self.mlps(x_y)
        return x_y[0]

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
            MODULES=MODULES,
            MODEL=MODEL
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

        self.y_embedder = LabelEmbedder(DATA.num_classes, dim, MODEL.class_dropout_prob)

        self.reset_parameters()

    def initialize_weights(self):
        # Initialize linear layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.motion_mlp.mlps:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.motion_fc_out.weight, gain=1e-8)
        nn.init.constant_(self.motion_fc_out.bias, 0)

    def forward(self, x, y):
        with torch.cuda.amp.autocast() if self.mixed_precision and not eval else misc.dummy_context_mgr() as mp:
            y = self.y_embedder(y, self.training)

            if self.temporal_fc_in:
                motion_feats = self.arr0(x)
                motion_feats = self.motion_fc_in(motion_feats)
            else:
                motion_feats = self.motion_fc_in(x)
                motion_feats = self.arr0(motion_feats)

            motion_feats = self.motion_mlp([motion_feats, y])

            if self.temporal_fc_out:
                motion_feats = self.motion_fc_out(motion_feats)
                motion_feats = self.arr1(motion_feats)
            else:
                motion_feats = self.arr1(motion_feats)
                motion_feats = self.motion_fc_out(motion_feats)

        return motion_feats