import utils.misc as misc

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.models.mixer_seq_simple import MixerModel, _init_weights
import utils.misc as misc
from collections import OrderedDict

class Model(nn.Module, GenerationMixin):
        
    def __init__(
        self,
        DATA: object,
        RUN: object,
        MODULES: object,
        MODEL: object,
        initializer_cfg=None,
        device=None,
        dtype=None,
    ) -> None:

        super().__init__()
        #self.inference_params = InferenceParams(max_seqlen=DATA.max_len, max_batch_size=DATA.batch_size)
        config = MambaConfig(
            d_model=MODEL.hidden_dim,
            n_layer=MODEL.depth,
            vocab_size=1, # placeholder value, we replace the embedding layer later
            ssm_cfg=dict(layer="Mamba1"),
            rms_norm=True,
            residual_in_fp32=True,
            fused_add_norm=True,
        )
        factory_kwargs = {"device": device, "dtype": dtype}
        self.mixed_precision = RUN.mixed_precision

        self.backbone = MixerModel(
            d_model=config.d_model,
            n_layer=config.n_layer,
            d_intermediate=config.d_intermediate,
            vocab_size=config.vocab_size,
            ssm_cfg=config.ssm_cfg,
            attn_layer_idx=config.attn_layer_idx,
            attn_cfg=config.attn_cfg,
            rms_norm=config.rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=config.fused_add_norm,
            residual_in_fp32=config.residual_in_fp32,
            **factory_kwargs,
        )
        f_input_size = DATA.input_size[1]*DATA.input_size[2]
        self.backbone.embedding = MODULES.linear(f_input_size, config.d_model, bias=True)
        self.lm_head = MODULES.linear(config.d_model, DATA.num_classes, bias=False)
        self.class_token = nn.Parameter(torch.zeros(1, 1, f_input_size))
        class_mask = torch.full((1, 1, f_input_size), True)

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if MODEL.representation_size is None:
            heads_layers["head"] = nn.Linear(config.d_model, DATA.num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(config.d_model, MODEL.representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(MODEL.representation_size, DATA.num_classes)

        self.heads = nn.Sequential(heads_layers)

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=config.n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

        nn.init.xavier_uniform_(self.class_token)

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

        self.register_buffer('class_mask', class_mask, persistent=False)

    def forward(self, x, masks=None, inference_params=None, **mixer_kwargs):
        """
        "masks" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        n = x.shape[0]

        with torch.autocast("cuda") if self.mixed_precision and not eval else misc.dummy_context_mgr() as mp:
            if masks is not None:
                batch_class_mask = self.class_mask.expand(n, -1, -1)
                masks = torch.cat([masks, batch_class_mask], dim=1)
                masks = masks.any(dim=-1)
                ft = misc.keep_first_true(masks)
                x = F.pad(input=x, pad=(0, 0, 0, 1, 0, 0), mode='constant', value=0)
                #x[masks] = 0
                x[ft] = self.class_token.to(x.dtype)

            else:
                # Expand the class token to the full batch
                batch_class_token = self.class_token.expand(n, -1, -1)
                x = torch.cat([x, batch_class_token], dim=1)

            """
            if masks is not None:
                masks = masks.any(dim=-1)
                row_indices, last_zero_positions = misc.pad_index(masks)
            """

            hidden_states = self.backbone(x, inference_params=inference_params, **mixer_kwargs)

            if masks is not None:
                #hidden_states = hidden_states[row_indices, last_zero_positions]
                hidden_states = hidden_states[ft]
            else:
                hidden_states = hidden_states[:, -1]

            lm_logits = self.heads(hidden_states)
        return lm_logits