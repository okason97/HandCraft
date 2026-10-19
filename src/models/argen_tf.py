"""
Generator A, causal Transformer (doc/specs/2026-10-16-generator-a-design.md): from the frames so far and the sign, the
drift of the next pred_frames frames from the last one, and the logit that the sign ends within them.
"""

import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, DATA, RUN, MODULES, MODEL):
        super().__init__()
        features = DATA.num_keypoints * DATA.coords
        self.pred_frames = DATA.pred_frames
        # standard deviation of a Gaussian noise added to the class embedding, set during generation only
        self.generation_noise = 0.0
        d = MODEL.hidden_dim
        self.embed = nn.Linear(features, d)
        self.class_embed = nn.Embedding(DATA.num_classes, d)
        # the class token and up to max_gen_len frames
        self.pos_embed = nn.Embedding(DATA.max_gen_len + 1, d)
        layer = nn.TransformerEncoderLayer(d, MODEL.nheads, MODEL.mlp_dim, MODEL.dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, MODEL.depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.drift_head = nn.Linear(d, self.pred_frames * features)
        self.end_head = nn.Linear(d, 1)
        # start by predicting no motion
        nn.init.zeros_(self.drift_head.weight)
        nn.init.zeros_(self.drift_head.bias)

    def forward(self, frames, labels, last_only=False):
        B, T, features = frames.shape
        if T + 1 > self.pos_embed.num_embeddings:
            raise ValueError("{} frames, more than max_gen_len ({})".format(T, self.pos_embed.num_embeddings - 1))
        cls = self.class_embed(labels)
        if self.generation_noise > 0:
            cls = cls + self.generation_noise * torch.randn_like(cls)
        h = torch.cat([cls[:, None], self.embed(frames)], dim=1) + self.pos_embed(torch.arange(T + 1, device=frames.device))
        mask = nn.Transformer.generate_square_subsequent_mask(T + 1, device=frames.device)
        # position t of the output (after the class token) has seen frames 0 to t
        h = self.norm(self.encoder(h, mask=mask, is_causal=True))[:, 1:]
        if last_only:
            h = h[:, -1:]
        drift = self.drift_head(h).unflatten(-1, (self.pred_frames, features))
        return drift, self.end_head(h)[..., 0]
