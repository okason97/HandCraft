"""
Pure PyTorch version of the parts of mamba-ssm (https://github.com/state-spaces/mamba, Apache 2.0) that mamba.py uses:
MambaConfig, MixerModel with Mamba1 layers and _init_weights. It has the same parameters, names and initialization as
mamba-ssm, so mamba.py builds the same model, but the selective scan is computed in PyTorch, a chunk of frames at a
time, instead of with the fused CUDA kernel: slower, and it needs no CUDA toolkit to build.
Only the configuration mamba.py uses is supported: Mamba1 layers, RMSNorm, no MLP and no attention layers.
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MambaConfig:
    d_model: int = 2560
    d_intermediate: int = 0
    n_layer: int = 64
    vocab_size: int = 50277
    ssm_cfg: dict = field(default_factory=dict)
    attn_layer_idx: list = field(default_factory=list)
    attn_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))

    def forward(self, x):
        x = x.float()
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()


def selective_scan(u, delta, A, B, C, D, z, delta_bias, chunk=16):
    """selective_scan_ref of mamba-ssm with delta_softplus=True, in float32.
    u, delta, z: (batch, d_inner, length); A: (d_inner, d_state); B, C: (batch, d_state, length); D: (d_inner,).
    The recurrence x_t = exp(delta_t A) x_(t-1) + delta_t B_t u_t is solved a chunk of frames at a time: inside a chunk
    x_t = sum_(s<=t) exp(S_t - S_s) delta_s B_s u_s + exp(S_t) x_in, with S the cumulative sum of delta A over the chunk.
    The exponents are never positive, so this is stable, and it needs length / chunk steps instead of length."""
    dtype_in = u.dtype
    with torch.autocast("cuda", enabled=False):
        u, delta, B, C, z = u.float(), delta.float(), B.float(), C.float(), z.float()
        delta = F.softplus(delta + delta_bias.float()[..., None])
        log_decay = torch.einsum("bdl,dn->bdln", delta, A.float())
        delta_B_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
        x = u.new_zeros(u.shape[0], u.shape[1], A.shape[1])
        causal = torch.ones(chunk, chunk, dtype=torch.bool, device=u.device).tril()
        ys = []
        for start in range(0, u.shape[2], chunk):
            S = log_decay[:, :, start : start + chunk].cumsum(2)  # (batch, d_inner, t, d_state)
            t = S.shape[2]
            diff = (S[:, :, :, None] - S[:, :, None]).masked_fill(~causal[:t, :t, None], float("-inf"))  # (b, d, t, s, n)
            xs = torch.einsum("bdtsn,bdsn->bdtn", diff.exp(), delta_B_u[:, :, start : start + t]) + S.exp() * x[:, :, None]
            ys.append(torch.einsum("bdtn,bnt->bdt", xs, C[:, :, start : start + t]))
            x = xs[:, :, -1]
        y = torch.cat(ys, dim=2) + u * D.float()[:, None]
        y = y * F.silu(z)
    return y.to(dtype_in)


class Mamba(nn.Module):
    """Mamba1 layer of mamba-ssm (mamba_simple.Mamba) with its default settings, computed without the fused kernels."""

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1, bias=conv_bias, device=device, dtype=dtype
        )
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # dt_init="random" with dt_scale=1: the projection uniform in +-dt_rank^-0.5, and its bias the inverse softplus
        # of a step size log-uniform in [dt_min, dt_max]
        dt_init_std = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True  # ty: ignore[unresolved-attribute]

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True  # ty: ignore[unresolved-attribute]
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True  # ty: ignore[unresolved-attribute]

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states):
        """hidden_states: (batch, length, d_model)."""
        length = hidden_states.shape[1]
        xz = self.in_proj(hidden_states).transpose(1, 2)  # (batch, 2 * d_inner, length)
        A = -torch.exp(self.A_log.float())
        x, z = xz.chunk(2, dim=1)
        x = self.act(self.conv1d(x)[..., :length])
        x_dbl = self.x_proj(x.transpose(1, 2))  # (batch, length, dt_rank + 2 * d_state)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.linear(dt, self.dt_proj.weight).transpose(1, 2)  # the bias is added in the scan, as in mamba-ssm
        y = selective_scan(x, dt, A, B.transpose(1, 2), C.transpose(1, 2), self.D, z, self.dt_proj.bias)
        return self.out_proj(y.transpose(1, 2))


class Block(nn.Module):
    """Add the residual, normalize, then apply the mixer; returns the mixer output and the new residual."""

    def __init__(self, dim, mixer, norm_epsilon, residual_in_fp32, device=None, dtype=None):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.norm = RMSNorm(dim, eps=norm_epsilon, device=device, dtype=dtype)
        self.mixer = mixer

    def forward(self, hidden_states, residual=None):
        residual = hidden_states if residual is None else hidden_states + residual
        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        return self.mixer(hidden_states), residual


class MixerModel(nn.Module):
    def __init__(
        self,
        d_model,
        n_layer,
        d_intermediate,
        vocab_size,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=True,
        initializer_cfg=None,
        fused_add_norm=True,
        residual_in_fp32=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        ssm_cfg = dict(ssm_cfg or {})
        if d_intermediate or attn_layer_idx or not rms_norm or ssm_cfg.pop("layer", "Mamba1") != "Mamba1":
            raise NotImplementedError("mamba_ref only supports Mamba1 layers with RMSNorm, without MLP or attention layers")
        # fused_add_norm only selects the fused kernel of mamba-ssm; the computation is the same
        self.embedding = nn.Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            Block(d_model, Mamba(d_model, layer_idx=i, **ssm_cfg, **factory_kwargs), norm_epsilon, residual_in_fp32, **factory_kwargs) for i in range(n_layer)
        )
        self.norm_f = RMSNorm(d_model, eps=norm_epsilon, **factory_kwargs)
        self.apply(lambda module: _init_weights(module, n_layer=n_layer, **(initializer_cfg or {})))

    def forward(self, input_ids, inference_params=None, **mixer_kwargs):
        hidden_states = self.embedding(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)
        residual = hidden_states if residual is None else hidden_states + residual
        return self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))


def _init_weights(module, n_layer, initializer_range=0.02, rescale_prenorm_residual=True, n_residuals_per_layer=1):
    """The initialization of mamba-ssm (mixer_seq_simple._init_weights)."""
    if isinstance(module, nn.Linear):
        if module.bias is not None and not getattr(module.bias, "_no_reinit", False):
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # scale the output projection of each residual block by 1/sqrt(number of residual layers) (GPT-2)
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)
