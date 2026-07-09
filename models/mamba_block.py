# Reference: https://github.com/state-spaces/mamba/blob/9127d1f47f367f5c9cc49c73ad73557089d02cb8/mamba_ssm/models/mixer_seq_simple.py

import math
from functools import partial

from mamba_ssm.models.mixer_seq_simple import _init_weights
from mamba_ssm.modules.block import Block
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mamba3 import Mamba3

from models.linoss import LinOSS
from models.selective_lru import SelectiveLRU, SelectiveLRUMIMO
from models.s4d.s4d import S4DCore
from models.s5.s5 import S5
from models.mixer import MambaStyleMixer
from typing import Literal

import torch
from torch import nn

from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn


def _apply_layer_norm(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    norm_layer: nn.Module,
):
    return layer_norm_fn(
        x,
        norm_layer.weight,
        norm_layer.bias,
        residual=residual,
        prenorm=True,
        eps=norm_layer.eps,
        is_rms_norm=isinstance(norm_layer, RMSNorm),
    )


class HybridBlock(nn.Module):
    """Hybrid backbone block.

    Apply two backbones in a single time-frequency block.
    - Mode 'sequential' applies them sequentially with skip connections,
        like MambAttention (attention -> mamba).
    - Mode 'parallel' applies them in parallel and combines outputs
        via gated fusion.
    """

    def __init__(
        self,
        dim: int,
        backbone_1: nn.Module,
        backbone_2: nn.Module,
        mode: Literal["sequential", "parallel"] = "sequential",
        gate_hidden: int | None = None,
        norm_cls=None,
    ):

        super().__init__()
        self.backbone_1 = backbone_1
        self.backbone_2 = backbone_2
        self.mode = mode

        norm_cls = norm_cls if norm_cls is not None else partial(nn.LayerNorm, eps=1e-5)
        self.norm1 = norm_cls(dim)
        self.norm2 = norm_cls(dim) if mode == "sequential" else None

        if mode == "parallel":
            if gate_hidden is None:
                self.gate = nn.Linear(2 * dim, dim)
            else:
                self.gate = nn.Sequential(
                    nn.Linear(2 * dim, gate_hidden),
                    nn.SiLU(),
                    nn.Linear(gate_hidden, dim),
                )

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mamba ``Block``-style prenorm contract: returns ``(hidden, residual)``.

        The residual stream is threaded (and its final add deferred) exactly like
        ``mamba_ssm.modules.block.Block`` so this drops into ``MambaBlock``'s loop,
        which does the closing ``hidden + residual`` add.
        """
        if self.mode == "sequential":
            x, residual = _apply_layer_norm(x, residual, self.norm1)
            x = self.backbone_1(x)
            x, residual = _apply_layer_norm(x, residual, self.norm2)
            x = self.backbone_2(x)
        elif self.mode == "parallel":
            x, residual = _apply_layer_norm(x, residual, self.norm1)
            x1 = self.backbone_1(x)
            x2 = self.backbone_2(x)
            g = torch.sigmoid(self.gate(torch.cat([x1, x2], dim=-1)))
            x = g * x1 + (1.0 - g) * x2
        else:
            raise ValueError(f"Unknown mode {self.mode}. Has to be either 'sequential' or 'parallel'.")
        return x, residual


def _build_mixer_cls(d_model, model_cfg, layer_idx=0):
    """Build the mixer_cls partial for a single backbone.

    The returned partial is callable as ``mixer_cls(d_model)`` (the contract
    ``mamba_ssm``'s ``Block`` expects), so it can be handed to ``Block`` for a
    plain block or instantiated directly for a ``HybridBlock`` backbone. All
    per-mixer hyperparameters (``ssm``, ``d_state``, ``d_conv``, ``expand``,
    ``ssm_params``) are read from ``model_cfg``, which for a hybrid backbone is
    that backbone's own sub-config.
    """
    ssm = model_cfg["ssm"]
    ssm_params = model_cfg.get("ssm_params", {})

    d_state = model_cfg["d_state"]  # 16
    d_conv = model_cfg["d_conv"]  # 4
    expand = model_cfg["expand"]  # 4

    if ssm == "mamba":
        mixer_cls = partial(
            Mamba,
            layer_idx=layer_idx,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
    elif ssm == "mamba2":
        headdim = ssm_params.get("headdim", 64)
        d_inner = expand * d_model
        assert d_inner % headdim == 0, (
            f"Mamba2 requires expand * d_model ({expand} * {d_model} = {d_inner}) "
            f"to be divisible by headdim ({headdim}); got remainder "
            f"{d_inner % headdim}. Adjust headdim or hid_feature."
        )
        mixer_cls = partial(
            Mamba2,
            layer_idx=layer_idx,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            d_ssm=ssm_params.get("d_ssm", None),
            ngroups=ssm_params.get("ngroups", 1),
            conv_init=ssm_params.get("conv_init", None),
            A_init_range=tuple(ssm_params.get("A_init_range", (1, 16))),
            D_has_hdim=ssm_params.get("D_has_hdim", False),
            rmsnorm=ssm_params.get("rmsnorm", True),
            norm_before_gate=ssm_params.get("norm_before_gate", False),
            dt_min=ssm_params.get("dt_min", 1e-3),
            dt_max=ssm_params.get("dt_max", 0.1),
            dt_init_floor=ssm_params.get("dt_init_floor", 1e-4),
            dt_limit=tuple(ssm_params.get("dt_limit", (0.0, float("inf")))),
            bias=ssm_params.get("bias", False),
            conv_bias=ssm_params.get("conv_bias", True),
            chunk_size=ssm_params.get("chunk_size", 256),
            use_mem_eff_path=ssm_params.get("use_mem_eff_path", True),
        )
    elif ssm == "mamba3":
        # Mamba3 has no d_conv; it is head-based. Constraint:
        # (expand * d_model) must be divisible by headdim.
        headdim = ssm_params.get("headdim", 64)
        d_inner = expand * d_model
        assert d_inner % headdim == 0, (
            f"Mamba3 requires expand * d_model ({expand} * {d_model} = {d_inner}) "
            f"to be divisible by headdim ({headdim}); got remainder "
            f"{d_inner % headdim}. Adjust headdim or hid_feature."
        )
        mixer_cls = partial(
            Mamba3,
            layer_idx=layer_idx,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
            ngroups=ssm_params.get("ngroups", 1),
            rope_fraction=ssm_params.get("rope_fraction", 0.5),
            dt_min=ssm_params.get("dt_min", 1e-3),
            dt_max=ssm_params.get("dt_max", 0.1),
            dt_init_floor=ssm_params.get("dt_init_floor", 1e-4),
            A_floor=ssm_params.get("A_floor", 1e-4),
            is_outproj_norm=ssm_params.get("is_outproj_norm", False),
            is_mimo=ssm_params.get("is_mimo", False),
            mimo_rank=ssm_params.get("mimo_rank", 4),
            fuse_pregate_headwise_norm=ssm_params.get("fuse_pregate_headwise_norm", True),
            chunk_size=ssm_params.get("chunk_size", 64),
            dropout=ssm_params.get("dropout", 0.0),
        )
    elif ssm == "linoss":
        ssm = LinOSS(
            in_features=expand * d_model,
            state_dim=d_state,
            discretization=ssm_params.get("discretization", "IMEX"),
            damping=ssm_params.get("damping", True),
            r_min=ssm_params.get("r_min", 0.9),
            theta_max=ssm_params.get("theta_max", math.pi),
            a_from_g=ssm_params.get("a_from_g", True),
            use_triton=ssm_params.get("use_triton", False),
        )
        mixer_cls = partial(
            MambaStyleMixer,
            ssm=ssm,
            expand=expand,
            d_conv=d_conv,
            causal_conv=model_cfg.get("causal_conv", True),
        )

    elif ssm == "selective_lru_mimo":
        ssm = SelectiveLRUMIMO(
            in_features=expand * d_model,
            state_dim=d_state,
            **ssm_params,
        )
        mixer_cls = partial(
            MambaStyleMixer,
            ssm=ssm,
            expand=expand,
            d_conv=d_conv,
            causal_conv=model_cfg.get("causal_conv", True),
        )

    elif ssm == "selective_lru":
        ssm = SelectiveLRU(
            d_model=expand * d_model,
            d_state=d_state,
            dt_rank=ssm_params.get("dt_rank", None),
            dt_min=ssm_params.get("dt_min", 1e-3),
            dt_max=ssm_params.get("dt_max", 0.1),
            theta_max=ssm_params.get("theta_max", math.pi),
            selective_init_std=ssm_params.get("init_std", 1e-2),
            input_norm=ssm_params.get("input_norm", "delta_nu"),
            mag_init=ssm_params.get("mag_init", "mamba"),
            r_min=ssm_params.get("r_min", 0.9),
            r_max=ssm_params.get("r_max", 0.999),
            rank=ssm_params.get("rank", 0),
            use_triton=ssm_params.get("use_triton", False),
            chunk_size=ssm_params.get("chunk_size", 16),
            block_c=ssm_params.get("block_c", 32),
        )
        mixer_cls = partial(
            MambaStyleMixer,
            ssm=ssm,
            expand=expand,
            d_conv=d_conv,
            causal_conv=model_cfg.get("causal_conv", True),
        )

    elif ssm == "s5":
        ssm = S5(
            width=expand * d_model,
            state_width=d_state,
            conj_sym=ssm_params.get("conj_sym", True),
            clip_eigs=ssm_params.get("clip_eigs", False),
            block_count=ssm_params.get("block_count", 1),
            dt_min=ssm_params.get("dt_min", 0.001),
            dt_max=ssm_params.get("dt_max", 0.1),
        )
        mixer_cls = partial(
            MambaStyleMixer,
            ssm=ssm,
            expand=expand,
            d_conv=d_conv,
            causal_conv=model_cfg.get("causal_conv", True),
        )

    elif ssm == "s4d":
        ssm = S4DCore(
            d_model=expand * d_model,
            d_state=d_state,
            dt_min=ssm_params.get("dt_min", 1e-3),
            dt_max=ssm_params.get("dt_max", 0.1),
        )
        mixer_cls = partial(
            MambaStyleMixer,
            ssm=ssm,
            expand=expand,
            d_conv=d_conv,
            causal_conv=model_cfg.get("causal_conv", True),
        )

    else:
        raise ValueError(
            f"Unknown mixer {ssm!r}; expected 'mamba', 'mamba2', 'mamba3', "
            "'linoss', 'selective_lru', 'selective_lru_mimo', 's4d', or 's5'."
        )

    return mixer_cls


# github: https://github.com/state-spaces/mamba/blob/9127d1f47f367f5c9cc49c73ad73557089d02cb8/mamba_ssm/models/mixer_seq_simple.py
def create_block(
    d_model,
    model_cfg,
    layer_idx=0,
    rms_norm=True,
    fused_add_norm=False,
    residual_in_fp32=False,
):
    norm_epsilon = model_cfg.get("norm_epsilon", 1e-5)  # 0.00001
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon)

    # Hybrid: two backbones defined by their own sub-configs
    if model_cfg["ssm"] == "hybrid":
        hybrid_cfg = model_cfg.get("hybrid", {})
        cfg_1 = model_cfg["backbone_1"]
        cfg_2 = model_cfg["backbone_2"]
        backbone_1 = _build_mixer_cls(d_model, cfg_1, layer_idx)(d_model)
        backbone_2 = _build_mixer_cls(d_model, cfg_2, layer_idx)(d_model)
        block = HybridBlock(
            dim=d_model,
            backbone_1=backbone_1,
            backbone_2=backbone_2,
            mode=hybrid_cfg.get("mode", "sequential"),
            gate_hidden=hybrid_cfg.get("gate_hidden", None),
            norm_cls=norm_cls,
        )
        block.layer_idx = layer_idx
        return block

    mixer_cls = _build_mixer_cls(d_model, model_cfg, layer_idx)
    # New mamba_ssm Block requires mlp_cls as a positional arg; nn.Identity keeps
    # these blocks mixer-only (no MLP), matching the previous behavior.
    block = Block(
        d_model,
        mixer_cls,
        nn.Identity,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


class MambaBlock(nn.Module):
    def __init__(self, in_channels, cfg):
        super(MambaBlock, self).__init__()
        n_layer = 1
        self.forward_blocks = nn.ModuleList(create_block(in_channels, cfg) for _ in range(n_layer))
        self.backward_blocks = nn.ModuleList(
            create_block(in_channels, cfg) for _ in range(n_layer)
        )

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
        y_backward = (
            torch.flip((x_backward + resi_backward), [1])
            if resi_backward is not None
            else torch.flip(x_backward, [1])
        )

        return torch.cat([y_forward, y_backward], -1)


def _layer_spec_matches(spec, layer_idx, num_layers):
    """Whether ``layer_idx`` is selected by a ``layers`` override spec.

    Each element of ``spec`` is either an int (negative counts from the end, so
    ``-1`` is the deepest layer) or an inclusive range string ``"a-b"`` with
    non-negative bounds.
    """
    for item in spec:
        if isinstance(item, str) and "-" in item.strip().lstrip("-"):
            lo_s, hi_s = item.split("-")
            if int(lo_s) <= layer_idx <= int(hi_s):
                return True
        else:
            idx = int(item)
            if idx < 0:
                idx += num_layers
            if idx == layer_idx:
                return True
    return False


class TFMambaBlock(nn.Module):
    """Temporal-Frequency Mamba block for sequence modeling.

    Attributes:
    cfg (Config): Configuration for the block.
    time_mamba (MambaBlock): Mamba block for temporal dimension.
    freq_mamba (MambaBlock): Mamba block for frequency dimension.
    tlinear (ConvTranspose1d): ConvTranspose1d layer for temporal dimension.
    flinear (ConvTranspose1d): ConvTranspose1d layer for frequency dimension.
    """

    def __init__(self, cfg, layer_idx=0, num_layers=1):
        super(TFMambaBlock, self).__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.hid_feature = cfg["model_cfg"]["hid_feature"]

        mcfg = cfg["model_cfg"]
        time_cfg = mcfg.get("time_mixer", mcfg)
        freq_cfg = mcfg.get("freq_mixer", mcfg)

        for override in mcfg.get("layer_overrides") or []:
            if _layer_spec_matches(override.get("layers", []), layer_idx, num_layers):
                if "time_mixer" in override:
                    time_cfg = override["time_mixer"]
                if "freq_mixer" in override:
                    freq_cfg = override["freq_mixer"]

        self.time_mamba = MambaBlock(in_channels=self.hid_feature, cfg=time_cfg)
        self.freq_mamba = MambaBlock(in_channels=self.hid_feature, cfg=freq_cfg)

        # Initialize ConvTranspose1d layers
        self.tlinear = nn.ConvTranspose1d(self.hid_feature * 2, self.hid_feature, 1, stride=1)
        self.flinear = nn.ConvTranspose1d(self.hid_feature * 2, self.hid_feature, 1, stride=1)

    def forward(self, x):
        """Forward pass of the TFMamba block.

        Parameters:
        x (Tensor): Input tensor with shape (batch, channels, time, freq).

        Returns:
        Tensor: Output tensor after applying temporal and frequency Mamba blocks.
        """
        b, c, t, f = x.size()

        x = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        x = self.tlinear(self.time_mamba(x).permute(0, 2, 1)).permute(0, 2, 1) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
        x = self.flinear(self.freq_mamba(x).permute(0, 2, 1)).permute(0, 2, 1) + x
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x
