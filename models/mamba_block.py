# Reference: https://github.com/state-spaces/mamba/blob/9127d1f47f367f5c9cc49c73ad73557089d02cb8/mamba_ssm/models/mixer_seq_simple.py

import math

import torch
import torch.nn as nn
from functools import partial

from mamba_ssm.modules.mamba_simple import Mamba, Block
from mamba_ssm.models.mixer_seq_simple import _init_weights
from mamba_ssm.ops.triton.layernorm import RMSNorm

from models.linoss.block import MambaStyleLinOSS

# github: https://github.com/state-spaces/mamba/blob/9127d1f47f367f5c9cc49c73ad73557089d02cb8/mamba_ssm/models/mixer_seq_simple.py
def create_block(
    d_model, model_cfg, layer_idx=0, rms_norm=True, fused_add_norm=False, residual_in_fp32=False,
    ):
    mixer_name = model_cfg['mixer']
    d_state = model_cfg['d_state']            # 16
    d_conv = model_cfg['d_conv']              # 4
    expand = model_cfg['expand']              # 4
    norm_epsilon = model_cfg['norm_epsilon']  # 0.00001


    if mixer_name == 'mamba':
        mixer_cls = partial(
            Mamba, layer_idx=layer_idx, d_state=d_state, d_conv=d_conv, expand=expand,
        )
    elif mixer_name == 'linoss':
        mixer_cls = partial(
            MambaStyleLinOSS,
            state_dim=d_state,
            expand=expand,
            d_conv=d_conv,
            causal_conv=model_cfg.get('linoss_causal_conv', True),
            discretization=model_cfg.get('linoss_discretization', 'IMEX'),
            damping=model_cfg.get('linoss_damping', True),
            r_min=model_cfg.get('linoss_r_min', 0.9),
            theta_max=model_cfg.get('linoss_theta_max', math.pi),
            use_triton=model_cfg.get('linoss_use_triton', False),
        )
    else:
        raise ValueError(
            f"Unknown mixer {mixer_name!r}; expected 'mamba' or 'linoss'."
        )

    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon
    )
    block = Block(
            d_model,
            mixer_cls,
            norm_cls=norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            )
    block.layer_idx = layer_idx
    return block

class MambaBlock(nn.Module):
    def __init__(self, in_channels, cfg, mixer=None):
        super(MambaBlock, self).__init__()
        n_layer = 1
        self.forward_blocks  = nn.ModuleList( create_block(in_channels, cfg) for _ in range(n_layer) )
        self.backward_blocks = nn.ModuleList( create_block(in_channels, cfg) for _ in range(n_layer) )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
            )
        )

    def forward(self, x):
        x_forward, x_backward = x.clone(), torch.flip(x, [1])
        resi_forward, resi_backward = None, None

        # Forward
        for layer in self.forward_blocks:
            x_forward, resi_forward = layer(x_forward, resi_forward)
        y_forward = (x_forward + resi_forward) if resi_forward is not None else x_forward

        # Backward
        for layer in self.backward_blocks:
            x_backward, resi_backward = layer(x_backward, resi_backward)
        y_backward = torch.flip((x_backward + resi_backward), [1]) if resi_backward is not None else torch.flip(x_backward, [1])

        return torch.cat([y_forward, y_backward], -1)

class TFMambaBlock(nn.Module):
    """
    Temporal-Frequency Mamba block for sequence modeling.
    
    Attributes:
    cfg (Config): Configuration for the block.
    time_mamba (MambaBlock): Mamba block for temporal dimension.
    freq_mamba (MambaBlock): Mamba block for frequency dimension.
    tlinear (ConvTranspose1d): ConvTranspose1d layer for temporal dimension.
    flinear (ConvTranspose1d): ConvTranspose1d layer for frequency dimension.
    """
    def __init__(self, cfg):
        super(TFMambaBlock, self).__init__()
        self.cfg = cfg
        self.hid_feature = cfg['model_cfg']['hid_feature']

        # Initialize Mamba blocks. Per-axis mixer sub-dicts ('time_mixer' /
        # 'freq_mixer') enable a LinOSS/Mamba hybrid; older flat configs fall
        # back to model_cfg itself (same shape create_block expects).
        mcfg = cfg['model_cfg']
        self.time_mamba = MambaBlock(in_channels=self.hid_feature, cfg=mcfg.get('time_mixer', mcfg))
        self.freq_mamba = MambaBlock(in_channels=self.hid_feature, cfg=mcfg.get('freq_mixer', mcfg))
        
        # Initialize ConvTranspose1d layers
        self.tlinear = nn.ConvTranspose1d(self.hid_feature * 2, self.hid_feature, 1, stride=1)
        self.flinear = nn.ConvTranspose1d(self.hid_feature * 2, self.hid_feature, 1, stride=1)
    
    def forward(self, x):
        """
        Forward pass of the TFMamba block.
        
        Parameters:
        x (Tensor): Input tensor with shape (batch, channels, time, freq).
        
        Returns:
        Tensor: Output tensor after applying temporal and frequency Mamba blocks.
        """
        b, c, t, f = x.size()

        x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        x = self.tlinear( self.time_mamba(x).permute(0,2,1) ).permute(0,2,1) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        x = self.flinear( self.freq_mamba(x).permute(0,2,1) ).permute(0,2,1) + x
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x

