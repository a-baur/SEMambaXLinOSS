"""LinOSS wrapped in a Mamba-block-shaped surround.

Mirrors the standard Mamba block:

    in_proj(in_features -> 2 * inner_dim)  -> split into (u, z)
    causal depthwise Conv1d on u (kernel=d_conv, groups=inner_dim)
    silu
    LinOSS on u
    u * silu(z)
    out_proj(inner_dim -> in_features)

LinOSS replaces Mamba's *selective* SSM with a non-selective one — that's the
only structural difference. ``x_proj``/``dt_proj`` have no analog since LinOSS
has fixed ``(A, B, C)``; the rest of the wiring (``in_proj``, depthwise causal
conv, SiLU gating, ``out_proj``) is faithful to the Mamba block. The optional
``selective_b``/``selective_c``/``selective_d`` flags reintroduce a limited form
of selectivity: Mamba-style input-dependent ``B``/``C``/``D`` (see ``LinOSS``),
which make the input->state, state->output, and skip maps content-dependent
while keeping the oscillator dynamics fixed.

Port of ``MambaStyleLinOSSSequenceMixer`` from the JAX/Equinox ``linax`` repo.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaStyleMixer(nn.Module):
    """Mamba-style SSM block with variable backbone.

    Shapes:
        Input:  (batch, time, in_features)
        Output: (batch, time, in_features)

    Args:
        in_features: Input dim.
        ssm: The state-space model backbone to use.
        expand: Inner-channel expansion factor (Mamba's ``expand``). The SSMs
            hidden state must be of size ``expand * in_features``. Default 4.
        d_conv: Depthwise conv kernel size along the sequence axis. Default 4.
        causal_conv: If True, the depthwise conv is causal via left-padding so
            position ``t`` only sees positions ``<= t``. Default True.
    """

    def __init__(
        self,
        in_features: int,
        ssm: torch.nn.Module,
        expand: int = 4,
        d_conv: int = 4,
        causal_conv: bool = True,
    ):
        super().__init__()
        inner_dim = expand * in_features
        self.in_features = in_features
        self.inner_dim = inner_dim
        self.d_conv = d_conv
        self.causal_conv = causal_conv

        self.in_proj = nn.Linear(in_features, 2 * inner_dim, bias=False)

        self.conv = nn.Conv1d(
            in_channels=inner_dim,
            out_channels=inner_dim,
            kernel_size=d_conv,
            groups=inner_dim,
            padding=0,
        )

        self.ssm = ssm

        self.out_proj = nn.Linear(inner_dim, in_features, bias=False)

    def forward(self, x: torch.Tensor, inference_params=None) -> torch.Tensor:
        # `inference_params` accepted for compatibility with ``mamba_ssm.Block``
        # (which always forwards it). LinOSS has no stateful single-step inference
        # path here, so the argument is ignored.
        del inference_params
        uz = self.in_proj(x)                  # (B, T, 2*inner_dim)
        u, z = uz.chunk(2, dim=-1)            # each (B, T, inner_dim)

        if self.causal_conv:
            pad_left, pad_right = self.d_conv - 1, 0
        else:
            pad_left = (self.d_conv - 1) // 2
            pad_right = self.d_conv - 1 - pad_left

        u_ch = u.transpose(1, 2)              # (B, inner_dim, T)
        u_ch = F.pad(u_ch, (pad_left, pad_right))
        u_ch = self.conv(u_ch)                # (B, inner_dim, T)
        u = u_ch.transpose(1, 2)              # (B, T, inner_dim)
        u = F.silu(u)

        y = self.ssm(u)                       # (B, T, inner_dim)
        y = y * F.silu(z)
        return self.out_proj(y)               # (B, T, in_features)



class LinOSSStyleMixer(nn.Module):

    def __init__(
        self,
        in_features: int,
        ssm: torch.nn.Module,
        expand: int = 4,
        iterations: int = 1,
    ):
        super().__init__()
        inner_dim = expand * in_features
        self.in_features = in_features
        self.inner_dim = inner_dim
        self.iterations = iterations

        self.in_proj = nn.Linear(in_features, inner_dim, bias=False)

        self.ssm = ssm

        self.gelu = nn.GELU()
        self.glu_w1 = nn.Linear(inner_dim, inner_dim)
        self.glu_w2 = nn.Linear(inner_dim, inner_dim)

        self.out_proj = nn.Linear(inner_dim, in_features, bias=False)

    def forward(self, x: torch.Tensor, inference_params=None) -> torch.Tensor:
        del inference_params
        x = self.in_proj(x)                  # (B, T, 2*inner_dim)

        skip = x
        x = self.ssm(x)
        x = self.gelu(x)
        x = torch.sigmoid(self.glu_w1(x)) * self.glu_w2(x) + skip

        return self.out_proj(x)               # (B, T, in_features)
