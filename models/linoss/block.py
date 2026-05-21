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
conv, SiLU gating, ``out_proj``) is faithful to the Mamba block.

Port of ``MambaStyleLinOSSSequenceMixer`` from the JAX/Equinox ``linax`` repo.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.linoss import LinOSS


class MambaStyleLinOSS(nn.Module):
    """LinOSS in the Mamba block surround.

    Shapes:
        Input:  (batch, time, in_features)
        Output: (batch, time, in_features)

    Args:
        in_features: Model dim (Mamba's ``d_model``).
        state_dim: Per-SSM state size (Mamba's ``d_state``). Default 16.
        expand: Inner-channel expansion factor (Mamba's ``expand``). The inner
            dim used by the conv, gate, and SSM is ``expand * in_features``.
            Default 4.
        d_conv: Depthwise conv kernel size along the sequence axis. Default 4.
        causal_conv: If True, the depthwise conv is causal via left-padding so
            position ``t`` only sees positions ``<= t``. Default True.
        discretization, damping, r_min, theta_max, use_triton: Forwarded to the
            inner ``LinOSS`` SSM.
    """

    def __init__(
        self,
        in_features: int,
        state_dim: int = 16,
        expand: int = 4,
        d_conv: int = 4,
        causal_conv: bool = True,
        discretization: Literal["IM", "IMEX"] = "IMEX",
        damping: bool = True,
        r_min: float = 0.9,
        theta_max: float = math.pi,
        use_triton: bool = False,
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

        self.ssm = LinOSS(
            in_features=inner_dim,
            state_dim=state_dim,
            discretization=discretization,
            damping=damping,
            r_min=r_min,
            theta_max=theta_max,
            use_triton=use_triton,
        )

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
