# Adapted from sl-hwgat: https://github.com/suvajit-patra/sl-hwgat (models/HWGATE.py and models/model_params.py)
# The MIT License (MIT), Copyright (c) 2024 Suvajit Patra
# See https://github.com/suvajit-patra/sl-hwgat/blob/main/LICENSE for details

# src/models/hwgat.py
#
# HWGAT: Hierarchical Windowed Graph Attention Network (Patra et al., 2024, https://arxiv.org/abs/2407.14224).
# The 29 keypoints are copied into 4 overlapping windows of 16 keypoints (head, one arm and one hand), and attention
# runs inside each window over blocks of `temporal_patch_size` frames, restricted to the edges of the skeleton.
# Changes from the original:
# - the windows are built inside the model (HWGAT builds them in its data pipeline, WindowCreate);
# - the adjacency matrices are buffers built here (HWGATEParams), so they move with the model;
# - the random Fourier projection is a buffer instead of a parameter with requires_grad=False, which the training
#   loop of this repository would make trainable;
# - torch's trunc_normal_ replaces timm's.

import math
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from models.stgcn import HWGAT29_POSES

# indexes of the 29 keypoints (HWGAT29_POSES order) in each window, from HWGAT's WindowCreate
PARTS = {
    "head": [0, 1, 2],
    "l_arm": [3, 5, 7],
    "l_hand": list(range(9, 19)),
    "r_arm": [4, 6, 8],
    "r_hand": list(range(19, 29)),
}
WINDOWS = [
    PARTS["head"] + PARTS["l_arm"] + PARTS["l_hand"],
    PARTS["head"] + PARTS["r_arm"] + PARTS["r_hand"],
    PARTS["head"] + PARTS["l_arm"] + PARTS["r_hand"],
    PARTS["head"] + PARTS["r_arm"] + PARTS["l_hand"],
]
WINDOW_SIZE = 16

# edges between the 16 keypoints of a window, the same for the 4 windows (HWGATEParams.edges)
WINDOW_EDGES = [
    [0, 1], [0, 2], [0, 3], [3, 4], [4, 5], [5, 6], [6, 7], [6, 8], [8, 9], [8, 10], [6, 10], [10, 11], [10, 12],
    [6, 12], [12, 13], [12, 14], [14, 15], [6, 14], [7, 9], [9, 11], [11, 13], [13, 15], [7, 15], [7, 11], [7, 13],
]  # fmt: skip


def window_adjacency(temporal_patch_size, num_windows=4, window_size=WINDOW_SIZE):
    """
    (windows, TP*W, TP*W) adjacency of the tokens of a window: the skeleton edges inside a frame (and self loops),
    and every keypoint with itself in the neighbouring frames of the temporal patch (HWGATEParams.get_adj_mat).
    """
    adj = np.eye(window_size)
    for i, j in WINDOW_EDGES:
        adj[i, j] = adj[j, i] = 1
    out = []
    for _ in range(num_windows):
        rows = []
        for i in range(temporal_patch_size):
            row = []
            for j in range(temporal_patch_size):
                if i == j:
                    row.append(adj)
                elif abs(i - j) == 1:
                    row.append(np.eye(window_size))
                else:
                    row.append(np.zeros((window_size, window_size)))
            rows.append(np.concatenate(row, axis=1))
        out.append(np.concatenate(rows))
    return torch.tensor(np.array(out), dtype=torch.float32)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    pe: torch.Tensor

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).unsqueeze(2)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


def window_partition(x, window_size=16, temporal_patch_size=4):
    W, TP = window_size, temporal_patch_size
    B, F, K, ED = x.shape
    f, nW = F // TP, K // W
    x = x.reshape(B, f, TP, nW, W, ED).transpose(2, 3).contiguous()  # B, F//TP, K//W, TP, W, ED
    x = x.view(B * f * nW, TP * W, ED)  # B_f_nW, W_TP, ED
    return x


def window_reverse(x, window_size=16, temporal_patch_size=4, temporal_dim=128, num_kp=64):
    W, TP = window_size, temporal_patch_size
    F, K = temporal_dim, num_kp
    B_f_nW, W_TP, ED = x.shape
    f, nW = F // TP, K // W
    B = int(B_f_nW / (f * nW))
    x = x.reshape(B, f, nW, TP, W, ED).transpose(2, 3).contiguous()  # B, F//TP, TP, K//W, W, ED
    x = x.view(B, F, K, ED)
    return x


class TemporalMerging(nn.Module):
    def __init__(self, dim, temporal_patch_size):
        super().__init__()
        self.dim = dim
        self.temporal_patch_size = temporal_patch_size

    def forward(self, x):
        TP = self.temporal_patch_size
        B, F, K, ED = x.shape
        f = F // TP

        x = x.reshape(B, f, TP, K, ED).transpose(2, 3).contiguous()
        x = x.reshape(B, f, K, -1).contiguous()  # B F//TP ED*TP

        return x


class MSA(nn.Module):
    adj_mat: Optional[torch.Tensor]

    def __init__(self, dim, num_heads, adj_mat=None, attn_drop=0.0, proj_drop=0.0) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        assert dim % num_heads == 0, 'dim and number of heads are incompatible'

        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.register_buffer("adj_mat", adj_mat, persistent=False)
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, B, f, nW, mask=None):
        B_f_nW, W_TP, ED = x.shape
        qkv = self.qkv(x).reshape(B_f_nW, W_TP, 3, self.num_heads, ED // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale

        attn = q @ k.transpose(-2, -1)

        # making some attn 0 to prevent overfitting
        if self.training:
            attn_copy = attn.detach().clone()
            seed_ = torch.rand(1).item()
            attn_copy = self.softmax(attn_copy)
            index_array = attn_copy > seed_
            index_array = torch.where(index_array, 0, 1)
            attn = attn * index_array

        if mask is not None:
            attn = attn.view(B, f * nW, *attn.shape[1:]) * mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(B * f * nW, *attn.shape[2:])

        if self.adj_mat is not None:
            attn = attn.view(B, f * nW, *attn.shape[1:]) * self.adj_mat.unsqueeze(1)
            attn = attn.view(B * f * nW, *attn.shape[2:])

        attn = attn.masked_fill(attn == 0, float(-10000))
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_f_nW, W_TP, ED)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PartAttentionBlock(nn.Module):
    attn_mask: Optional[torch.Tensor]

    def __init__(
        self,
        dim,
        num_kps=64,
        num_heads=4,
        window_size=16,
        temporal_patch_size=4,
        temporal_dim=128,
        shift_size=0,
        adj_mat=None,
        drop=0.0,
        attn_drop=0.0,
        ff_ratio=4.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_kps = num_kps
        self.num_heads = num_heads
        self.window_size = window_size
        self.temporal_patch_size = temporal_patch_size
        self.temporal_dim = temporal_dim
        self.shift_size = shift_size
        self.drop = drop
        self.attn_drop = attn_drop
        self.ff_dim = dim * ff_ratio
        self.act_layer = act_layer

        self.norm1 = norm_layer(dim)
        self.attn = MSA(dim, num_heads=num_heads, adj_mat=adj_mat, attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = norm_layer(dim)
        self.ff = FeedForward(in_features=dim, hidden_features=int(self.ff_dim), act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            F, K = temporal_dim, num_kps
            frame_mask = torch.zeros((1, F, K, 1))  # 1 H W 1
            t_slices = (slice(0, -temporal_patch_size), slice(-temporal_patch_size, -shift_size), slice(-shift_size, None))

            cnt = 0
            for t in t_slices:
                frame_mask[:, t, :] = cnt
                cnt += 1

            mask_windows = window_partition(frame_mask, window_size, temporal_patch_size).squeeze(2)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(0.0)).masked_fill(attn_mask == 0, float(1))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        W, TP = self.window_size, self.temporal_patch_size
        B, F, K, ED = x.shape
        f, nW = F // TP, K // W

        shortcut = x

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=-self.shift_size, dims=1)
        else:
            shifted_x = x
        x = window_partition(shifted_x, W, TP)

        x = self.norm1(x)
        # MSA/TS-MSA
        x = self.attn(x, B, f, nW, mask=self.attn_mask)  # B_f_nW, W_TP, ED

        x = window_reverse(x, W, TP, F, K)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=self.shift_size, dims=1)
        else:
            shifted_x = x

        x = shortcut + shifted_x
        # FFN
        x = x + self.ff(self.norm2(x))

        return x


class PartAttentionLayer(nn.Module):
    def __init__(
        self,
        dim,
        temporal_patch_size,
        temporal_dim,
        num_kps,
        depth,
        num_heads,
        window_size,
        adj_mat,
        drop=0.0,
        attn_drop=0.0,
        ff_ratio=4.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        i_layer=0,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.window_size = window_size
        self.i_layer = i_layer

        # build blocks
        self.blocks = nn.ModuleList(
            [
                PartAttentionBlock(
                    dim=dim,
                    num_kps=num_kps,
                    num_heads=num_heads,
                    window_size=window_size,
                    temporal_patch_size=temporal_patch_size,
                    temporal_dim=temporal_dim,
                    shift_size=0 if (i % 2 == 0) else temporal_patch_size // 2,
                    adj_mat=adj_mat,
                    drop=drop,
                    attn_drop=attn_drop,
                    ff_ratio=ff_ratio,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # temporal merging layer
        if downsample is not None:
            self.downsample = downsample(dim, temporal_patch_size)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class HWGAT(nn.Module):
    """
    The HWGATE model of sl-hwgat. Input: (batch, frames, 64 window keypoints, coordinates).
    """

    B: torch.Tensor

    def __init__(
        self,
        kp_dim=2,
        num_kps=64,
        temporal_dim=64,
        num_classes=1000,
        embed_dim=128,
        temporal_patch_size=2,
        pe=True,
        depths=(2, 2, 4),
        num_heads=(2, 4, 8),
        window_size=16,
        drop_rate=0.1,
        attn_drop_rate=0.0,
        ff_ratio=2.0,
        norm_layer=nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.kp_dim = kp_dim
        self.num_kps = num_kps
        self.temporal_dim = temporal_dim
        self.window_size = window_size
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.pe = pe
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.temporal_out_dim = temporal_dim // temporal_patch_size ** (self.num_layers - 1)

        assert self.num_kps % window_size == 0, "window size and number of kps are incompatible"
        assert self.temporal_dim % temporal_patch_size ** (self.num_layers - 1) == 0, "temporal dimension and temporal patch size are incompatible"

        mapping_size = embed_dim // 2
        scale = 10

        # fourier mapping (fixed random projection)
        self.register_buffer("B", torch.normal(0.0, 1.0, (mapping_size, self.kp_dim)) * scale)

        # position encoding
        if self.pe:
            self.pos_encoder = PositionalEncoding(embed_dim, drop_rate, temporal_dim)

        # build layers
        adj_mat = window_adjacency(temporal_patch_size, num_kps // window_size, window_size)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            adj_mat_t = torch.cat([adj_mat for _ in range(temporal_dim // temporal_patch_size ** (i_layer + 1))])
            layer = PartAttentionLayer(
                dim=int(embed_dim * 2**i_layer),
                temporal_patch_size=temporal_patch_size,
                temporal_dim=temporal_dim // (temporal_patch_size**i_layer),
                num_kps=num_kps,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                adj_mat=adj_mat_t,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                ff_ratio=ff_ratio,
                norm_layer=norm_layer,
                downsample=TemporalMerging if (i_layer < self.num_layers - 1) else None,
                i_layer=i_layer,
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AvgPool1d(self.temporal_out_dim * self.num_kps)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x_proj = (2.0 * torch.pi * x) @ self.B.transpose(1, 0)
        x = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        if self.pe:
            x = self.pos_encoder(x)

        for layer in self.layers:
            x = layer(x)

        B, f, K, d = x.shape
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 3).reshape(B, d, -1)).squeeze(-1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class Model(nn.Module):
    """
    HWGAT backbone. Expects DATA.poses to be the 29 keypoints of HWGAT (HWGAT29_POSES, as stgcn) and raw frames:
    DATA.transform "none" and a sampling that returns no padded frames. HWGAT was designed for 2D keypoints.
    """

    windows: torch.Tensor

    def __init__(self, DATA: Any, RUN: Any, MODULES: Any, MODEL: Any):
        super().__init__()
        assert DATA.poses == HWGAT29_POSES, "hwgat only supports the 29 keypoints of HWGAT: {poses}".format(poses=HWGAT29_POSES)
        self.num_frames, self.num_nodes, self.in_channels = DATA.input_size
        self.register_buffer("windows", torch.tensor([k for window in WINDOWS for k in window]), persistent=False)
        self.net = HWGAT(
            kp_dim=self.in_channels,
            num_kps=len(WINDOWS) * WINDOW_SIZE,
            temporal_dim=self.num_frames,
            num_classes=DATA.num_classes,
            embed_dim=MODEL.hwgat_embed_dim,
            temporal_patch_size=MODEL.hwgat_temporal_patch_size,
            pe=MODEL.hwgat_pe,
            depths=MODEL.hwgat_depths,
            num_heads=MODEL.hwgat_heads,
            window_size=WINDOW_SIZE,
            drop_rate=MODEL.dropout,
            attn_drop_rate=MODEL.hwgat_attn_dropout,
            ff_ratio=MODEL.hwgat_ff_ratio,
        )

    def forward(self, x: torch.Tensor, masks: Optional[torch.Tensor] = None):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (N, T, V*C)
            masks: unused, kept for compatibility with the other backbones
        """
        N, T, _ = x.size()
        x = x.view(N, T, self.num_nodes, self.in_channels)
        return self.net(x[:, :, self.windows])
