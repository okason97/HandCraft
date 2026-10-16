"""
Generator A, sliding siMLPe (doc/specs/2026-10-16-generator-a-design.md): from the last context_frames frames and the
sign, the drift of the next pred_frames frames from the last one, and the logit that the sign ends within them. The
blocks are those of CsiMLPe.
"""

import torch
from torch import nn

from models.CsiMLPe import LabelEmbedder, TransMLP


class Model(nn.Module):
    def __init__(self, DATA, RUN, MODULES, MODEL):
        super().__init__()
        features = DATA.num_keypoints * DATA.coords
        d, self.window, self.pred_frames = MODEL.hidden_dim, MODEL.context_frames, DATA.pred_frames
        # standard deviation of a Gaussian noise added to the class embedding, set during generation only
        self.generation_noise = 0.0
        self.embed = nn.Linear(features, d)
        # the sign is added to the features of every frame; the blocks also get it, over the frames of the window
        self.class_embed = nn.Embedding(DATA.num_classes, d)
        self.y_embedder = LabelEmbedder(DATA.num_classes, self.window, MODEL.class_dropout_prob)
        self.mlp = TransMLP(dims=[d, self.window], num_layers=MODEL.depth, MODULES=MODULES, MODEL=MODEL)
        self.temporal_out = nn.Linear(self.window, self.pred_frames)
        self.drift_head = nn.Linear(d, features)
        self.end_head = nn.Linear(d, 1)
        # start by predicting no motion
        nn.init.zeros_(self.drift_head.weight)
        nn.init.zeros_(self.drift_head.bias)

    def forward(self, frames, labels, last_only=False):
        B, T, features = frames.shape
        if last_only:
            frames = frames[:, -self.window :]
        # the window of a position near the start is padded with the first frame
        padded = torch.cat([frames[:, :1].expand(B, self.window - 1, features), frames], dim=1)
        windows = padded.unfold(1, self.window, 1)
        if last_only:
            windows = windows[:, -1:]
        L = windows.shape[1]
        windows = windows.permute(0, 1, 3, 2).reshape(B * L, self.window, features)
        labels = labels.repeat_interleave(L)
        cls = self.class_embed(labels)
        if self.generation_noise > 0:
            cls = cls + self.generation_noise * torch.randn_like(cls)
        h = (self.embed(windows) + cls[:, None]).transpose(1, 2).contiguous()
        h = self.mlp([h, self.y_embedder(labels, self.training)])
        h = self.temporal_out(h).transpose(1, 2)
        drift = self.drift_head(h).view(B, L, self.pred_frames, features)
        return drift, self.end_head(h.mean(dim=1)).view(B, L)
