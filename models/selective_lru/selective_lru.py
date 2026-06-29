from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _uniform_init(shape, std: float = 1.0) -> torch.Tensor:
    return torch.empty(*shape).uniform_(-std, std)


def _inv_softplus(x: torch.Tensor) -> torch.Tensor:
    # Inverse of softplus: returns y such that softplus(y) = x, for x > 0. Stable form
    # used by Mamba's dt init (x + log(-expm1(-x)) == x + log1p(-exp(-x))).
    return x + torch.log(-torch.expm1(-x))


def _project_input(B_complex: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # Real-times-complex projection: (N, F) x (B, T, F) -> (B, T, N) complex.
    Bu_real = torch.einsum("nf,btf->btn", B_complex.real, x)
    Bu_imag = torch.einsum("nf,btf->btn", B_complex.imag, x)
    return torch.complex(Bu_real, Bu_imag)


class SelectiveLRU(nn.Module):
    """SISO selective SSM low-rank shared projections.

    Shapes:
        Input:  (batch, time, in_features)
        Output: (batch, time, in_features)

    Args:
        d_model: Number of input channels
        d_state: Number of complex oscillator modes *per channel*. Default 16.
        dt_rank: Bottleneck rank of the two selective time-step paths. ``None`` defaults
            to ``ceil(in_features / 64)``, which — with ``expand=4`` so
            ``in_features = 4 * d_model`` — reproduces Mamba's ``ceil(d_model / 16)``.
        dt_min: Lower bound of the baseline damping time-step band (Mamba's ``dt_min``).
            Default 1e-3.
        dt_max: Upper bound of the baseline damping time-step band. Default 0.1.
        theta_max: Upper bound for the baseline per-mode frequency spread
            ``omega in [0, theta_max)``. Default pi.
        selective_init_std: Std of the low-rank selective projection weights at init.
            Small, so selectivity is learned as a perturbation off a working oscillator
            bank rather than from scratch. Default 1e-2.
        use_triton: If True, run the fused ``mamboss6_triton`` kernel — which keeps the
            per-channel state off HBM (forward streams it in registers; backward recomputes
            it in chunks) — instead of the materialized Python reference path. Requires CUDA.
        chunk_size: Backward recompute chunk width for the fused kernel (peak extra HBM
            scales with it). Default 16. Ignored on the reference path.
        block_c: Per-program channel-slab width for the fused kernel (throughput knob —
            bigger shares more B/C loads and shrinks atomic traffic, but spills registers
            past ~64). Default 32. Ignored on the reference path.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        dt_rank: int | None = None,
        dt_min: float = 1e-3,
        dt_max: float = 0.1,
        theta_max: float = math.pi,
        selective_init_std: float = 1e-2,
        use_triton: bool = False,
        chunk_size: int = 16,
        block_c: int = 32,
        device=None,
        dtype=None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.use_triton = use_triton
        self.chunk_size = chunk_size
        self.block_c = block_c
        if dt_rank is None:
            dt_rank = max(1, math.ceil(d_model / 64))
        self.dt_rank = dt_rank


        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, 2 * d_state, bias=False)

        # data-dependent magnitude and phase, bottlenecked via dt_rank
        self.dt_nu_down = nn.Linear(d_model, dt_rank, bias=False)
        self.dt_nu_up = nn.Linear(dt_rank, d_model, bias=True)
        self.dt_theta_down = nn.Linear(d_model, dt_rank, bias=False)
        self.dt_theta_up = nn.Linear(dt_rank, d_model, bias=True)

        # Per-channel static dynamics.
        # - Decay rate a = exp(A_log) > 0 (init reproduces Mamba's S4D-real A = 1..state_dim)
        # - Base frequency omega spread over the full angular range.
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.omega = nn.Parameter(torch.rand(d_model, d_state) * theta_max)

        self.D = nn.Parameter(torch.randn(d_model))

        with torch.no_grad():
            # Magnitude init: softplus(dt_nu_up.bias) uniform in [dt_min, dt_max]
            # (like Mamba's dt init), so baseline nu = exp(-dt * a) lands near the unit disk.
            dt = torch.exp(
                torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            )
            self.dt_nu_up.bias.copy_(_inv_softplus(dt))

            # Phase init: delta_theta = 0, so theta = omega at init (the full
            # angular range, matching D-LinOSS init).
            self.dt_theta_up.bias.zero_()

            # Selective weights start small (perturbation off a static oscillator bank).
            for proj in (self.dt_nu_down, self.dt_nu_up, self.dt_theta_down, self.dt_theta_up):
                proj.weight.normal_(0.0, selective_init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_, T, C = x.shape
        N = self.d_state

        # Per-channel selective time-steps (low-rank). Delta_nu > 0 via softplus.
        delta_nu = F.softplus(self.dt_nu_up(self.dt_nu_down(x)))      # (B, T, C) > 0
        delta_theta = self.dt_theta_up(self.dt_theta_down(x))         # (B, T, C)

        a = torch.exp(self.A_log)                                     # (C, N) > 0
        B_sel = self.B_proj(x)                                        # (B, T, N) real, shared
        C_sel = self.C_proj(x)                                        # (B, T, 2N)
        C_complex = torch.complex(C_sel[..., :N], C_sel[..., N:])     # (B, T, N) shared

        if self.use_triton:
            # Fused scan + readout: the state never loads into HBM.
            from selective_lru.selective_lru_triton import fused_selective_lru
            return fused_selective_lru(
                delta_nu, delta_theta, a, self.omega, B_sel, C_complex, x, self.D,
                block_t=self.chunk_size, block_c=self.block_c,
            )
        else:  # Materialized path: per-channel complex selective scan.
            nu = torch.exp(-delta_nu.unsqueeze(-1) * a)                  # (B, T, C, N) in (0,1)
            theta = self.omega + delta_theta.unsqueeze(-1)               # (B, T, C, N)
            lam = torch.polar(nu, theta)                                 # (B, T, C, N) complex

            Bu = (delta_nu.unsqueeze(-1) * B_sel.unsqueeze(-2)) * x.unsqueeze(-1)  # (B, T, C, N)
            Bu = Bu.to(lam.dtype)

            h = _selective_recurrence(lam.reshape(B_, T, C * N), Bu.reshape(B_, T, C * N))
            h = h.reshape(B_, T, C, N)

            Cy = torch.einsum("btn,btcn->btc", C_complex, h).real        # (B, T, C)
            return Cy + x * self.D


class SelectiveLRUMIMO(nn.Module):

    def __init__(
        self,
        in_features: int,
        state_dim: int = 16,
        r_min: float = 0.9,
        r_max: float = 0.999,
        theta_max: float = math.pi,
        selective_init_std: float = 1e-2,
        normalize_input: bool = True,
        use_triton: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.state_dim = state_dim
        self.normalize_input = normalize_input
        self.use_triton = use_triton

        self.nu_proj = nn.Linear(in_features, state_dim)
        self.theta_proj = nn.Linear(in_features, state_dim)

        with torch.no_grad():
            mags = torch.sqrt(torch.rand(state_dim) * (r_max ** 2 - r_min ** 2) + r_min ** 2)
            self.nu_proj.bias.copy_(torch.logit(mags))

            self.theta_proj.bias.copy_(torch.rand(state_dim) * theta_max)

            self.nu_proj.weight.normal_(0.0, selective_init_std)
            self.theta_proj.weight.normal_(0.0, selective_init_std)

        B_std = 1.0 / math.sqrt(in_features)
        self.B = nn.Parameter(_uniform_init((state_dim, in_features, 2), std=B_std))
        C_std = 1.0 / math.sqrt(state_dim)
        self.C = nn.Parameter(_uniform_init((in_features, state_dim, 2), std=C_std))
        self.D = nn.Parameter(torch.randn(in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nu = torch.sigmoid(self.nu_proj(x))
        theta = self.theta_proj(x)
        lam = torch.polar(nu, theta)

        B_complex = torch.complex(self.B[..., 0], self.B[..., 1])
        C_complex = torch.complex(self.C[..., 0], self.C[..., 1])

        Bu = _project_input(B_complex, x)            # (B, T, N) complex
        if self.normalize_input:
            gamma = torch.sqrt((1.0 - nu ** 2).clamp_min(1e-6))  # (B, T, N) real
            Bu = Bu * gamma

        if self.use_triton:
            from models.selective_lru.selective_lru_mimo_triton import selective_scan_triton
            h = selective_scan_triton(lam, Bu)       # (B, T, N) complex
        else:
            h = _selective_recurrence(lam, Bu)       # (B, T, N) complex

        Cy = torch.einsum("fn,btn->btf", C_complex, h).real
        Du = x * self.D
        return Cy + Du


def _selective_recurrence(lam: torch.Tensor, Bu: torch.Tensor) -> torch.Tensor:
    B, T, N = Bu.shape
    h = Bu.new_zeros(B, N)
    out = Bu.new_empty(B, T, N)
    for t in range(T):
        h = lam[:, t] * h + Bu[:, t]
        out[:, t] = h
    return out
