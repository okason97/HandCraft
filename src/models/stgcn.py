# Adapted from sl-hwgat: https://github.com/suvajit-patra/sl-hwgat
# The MIT License (MIT), Copyright (c) 2024 Suvajit Patra
# See https://github.com/suvajit-patra/sl-hwgat/blob/main/LICENSE for details

# src/models/stgcn.py

import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.graph_utils import GraphWithPartition


class ConvTemporalGraphical(nn.Module):
    """The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the graph convolving kernel.
        t_kernel_size (int): Size of the temporal convolving kernel.
        t_stride (int, optional): Stride of the temporal convolution. Default: 1.
        t_padding (int, optional): Temporal zero-padding added to both sides
            of the input. Default: 0.
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1.
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``.
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)`
            format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}
            , V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)
            ` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]
                `,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        t_kernel_size=1,
        t_stride=1,
        t_padding=0,
        t_dilation=1,
        bias=True,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias,
        )

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size

        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum("nkctv,kvw->nctw", (x, A))

        return x.contiguous(), A


class STGCN_BLOCK(nn.Module):
    """
    Applies a spatial temporal graph convolution over an input graph
    sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the temporal convolving kernel and
            graph convolving kernel.
        stride (int, optional): Stride of the temporal convolution. Default: 1.
        dropout (int, optional): Dropout rate of the final output. Default: 0.
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``.
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)`
            format.
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out},
            V)` format.
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V,
            V)` format.
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, dropout=0, residual=True
    ):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A

class FC(nn.Module):
    """
    Fully connected layer head
    Args:
        n_features (int): Number of features in the input.
        num_class (int): Number of class for classification.
        dropout_ratio (float): Dropout ratio to use. Default: 0.2.
        batch_norm (bool): Whether to use batch norm or not. Default: ``False``.
    """
    def __init__(self, n_features, num_class, dropout_ratio=0.2, batch_norm=False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.bn = None
        if batch_norm:
            self.bn = nn.BatchNorm1d(n_features)
            self.bn.weight.data.fill_(1)
            self.bn.bias.data.zero_()
        self.classifier = nn.Linear(n_features, num_class)
        nn.init.normal_(self.classifier.weight, 0, math.sqrt(2.0 / num_class))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape: (batch_size, n_features)
        
        returns:
            torch.Tensor: logits for classification.
        """

        x = self.dropout(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.classifier(x)
        return x

# keypoints and skeleton edges of the 29 keypoint graph used by sl-hwgat (MediaPipe indexes):
# nose, eyes, shoulders, elbows and wrists, then 10 keypoints per hand (wrist, fingertips and finger bases)
HWGAT29_POSES = [["pose", [0, 2, 5, 11, 12, 13, 14, 15, 16]],
                 ["left_hand", [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]],
                 ["right_hand", [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]]]
HAND_EDGES = [[0, 1], [0, 2], [2, 3], [2, 4], [4, 5], [0, 4], [4, 6], [0, 6], [6, 7], [6, 8], [0, 8], [8, 9]]
HWGAT29_EDGES = [[2, 0], [1, 0], [0, 3], [0, 4], [3, 5], [4, 6], [5, 7], [6, 8], [7, 9], [8, 19]] + \
                [[i+9, j+9] for i, j in HAND_EDGES] + [[i+19, j+19] for i, j in HAND_EDGES]


class Model(nn.Module):
    """Spatial temporal graph convolutional network backbone

    This module is proposed in
    `Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition
    <https://arxiv.org/pdf/1801.07455.pdf>`_

    Expects DATA.poses to be HWGAT29_POSES. The input is a sequence of raw frames, so it
    should be used with DATA.transform "none" and a sampling that returns no padded frames.
    """
    # buffer registered in __init__; torch types buffer attributes as Tensor | Module
    A: torch.Tensor

    def __init__(self, DATA: Any, RUN: Any, MODULES: Any, MODEL: Any):
        super().__init__()
        self.num_nodes = DATA.input_size[1]
        self.in_channels = DATA.input_size[2]
        assert DATA.poses == HWGAT29_POSES, "stgcn only supports the 29 keypoints graph: {poses}".format(poses=HWGAT29_POSES)

        self.graph = GraphWithPartition(self.num_nodes, 0, HWGAT29_EDGES)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer("A", A)

        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        self.n_out_features = MODEL.representation_size
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(self.in_channels * A.size(1))
        self.st_gcn_networks = nn.ModuleList(
            (
                STGCN_BLOCK(self.in_channels, 64, kernel_size, 1, residual=False,),
                STGCN_BLOCK(64, 64, kernel_size, 1,),
                STGCN_BLOCK(64, 64, kernel_size, 1,),
                STGCN_BLOCK(64, 64, kernel_size, 1,),
                STGCN_BLOCK(64, 128, kernel_size, 2,),
                STGCN_BLOCK(128, 128, kernel_size, 1,),
                STGCN_BLOCK(128, 128, kernel_size, 1,),
                STGCN_BLOCK(128, 256, kernel_size, 2,),
                STGCN_BLOCK(256, 256, kernel_size, 1,),
                STGCN_BLOCK(256, self.n_out_features, kernel_size, 1,),
            )
        )

        self.edge_importance = nn.ParameterList(
            [nn.Parameter(torch.ones(self.A.size())) for i in self.st_gcn_networks]
        )

        self.head = FC(self.n_out_features, DATA.num_classes, MODEL.dropout)

    def forward(self, x: torch.Tensor, masks: Optional[torch.Tensor] = None):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (N, T, V*C)
            masks: unused, kept for compatibility with the other backbones
        """
        N, T, _ = x.size()
        x = x.float().view(N, T, self.num_nodes, self.in_channels)
        x = x.permute(0, 2, 3, 1).contiguous() # NTVC -> NVCT
        x = x.view(N, self.num_nodes * self.in_channels, T)
        x = self.data_bn(x)
        x = x.view(N, self.num_nodes, self.in_channels, T)
        x = x.permute(0, 2, 3, 1).contiguous() # NVCT -> NCTV

        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance, strict=True):
            x, _ = gcn(x, self.A * importance)

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, -1)

        return self.head(x)
