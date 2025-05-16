import math
from collections import OrderedDict
from functools import partial
from typing import Callable

import torch
import torch.nn as nn

from torchvision.ops.misc import  MLP
from torchvision.utils import _log_api_usage_once

import utils.misc as misc

class MLPBlock(MLP):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i+1}.{type}"
                    new_key = f"{prefix}{3*i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input_tuple: tuple):
        input = input_tuple[0]
        masks = input_tuple[1]
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x0 = torch.nan_to_num(x)
        x, _ = self.self_attention(x0, x0, x0, key_padding_mask=masks, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y, masks


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor, masks: torch.Tensor = None):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        return self.ln(self.layers((self.dropout(input), masks))[0])


class Model(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        DATA: object,
        RUN: object,
        MODULES: object,
        MODEL: object,
    ):
        super().__init__()
        _log_api_usage_once(self)
        self.mixed_precision = RUN.mixed_precision

        seq_length = DATA.input_size[0]
        f_input_size = DATA.input_size[1]*DATA.input_size[2]

        self.embeding = MODULES.linear(f_input_size, MODEL.hidden_dim, bias=False)

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, MODEL.hidden_dim))
        class_mask = torch.full((1, 1, f_input_size), False)
        seq_length += 1

        self.encoder = Encoder(
            seq_length,
            MODEL.depth,
            MODEL.nheads,
            MODEL.hidden_dim,
            MODEL.mlp_dim,
            MODEL.dropout,
            MODULES.feature_norm,
        )
        self.seq_length = seq_length

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if MODEL.representation_size is None:
            heads_layers["head"] = nn.Linear(MODEL.hidden_dim, DATA.num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(MODEL.hidden_dim, MODEL.representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(MODEL.representation_size, DATA.num_classes)

        self.heads = nn.Sequential(heads_layers)

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

        self.register_buffer('class_mask', class_mask, persistent=False)

    def forward(self, x: torch.Tensor, masks: torch.Tensor = None):
        with torch.autocast("cuda") if self.mixed_precision and not eval else misc.dummy_context_mgr() as mp:
            n = x.shape[0]

            if masks is not None:
                # sin esta linea funciona
                #x = torch.where(masks, x, 0)

                batch_class_mask = self.class_mask.expand(n, -1, -1)
                masks = torch.cat([batch_class_mask, masks], dim=1)
                masks = masks.any(dim=-1)

            x = self.embeding(x)

            # Expand the class token to the full batch
            batch_class_token = self.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)

            x = self.encoder(x, masks)

            # Classifier "token" as used by standard language architectures
            x = x[:, 0]

            x = self.heads(x)

        return x
    
    def update_dropout(self, drop_rate):
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = drop_rate