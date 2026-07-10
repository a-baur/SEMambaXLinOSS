from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _uniform_init(shape, std: float = 1.0) -> torch.Tensor:
    return torch.empty(*shape).uniform_(-std, std)


def _inv_softplus(x: torch.Tensor) -> torch.Tensor:
    # Inverse of softplus (x > 0), in the stable form used by Mamba's dt init.
    return x + torch.log(-torch.expm1(-x))


def _project_input(B_complex: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # Real-times-complex projection: (N, F) x (B, T, F) -> (B, T, N) complex.
    Bu_real = torch.einsum("nf,btf->btn", B_complex.real, x)
    Bu_imag = torch.einsum("nf,btf->btn", B_complex.imag, x)
    return torch.complex(Bu_real, Bu_imag)


class SelectiveLRU(nn.Module):
    """SISO selective SSM with a bank of complex oscillators per channel.

    Shapes: (batch, time, d_model) -> (batch, time, d_model).

    Per channel c and mode n, with selective step ``delta_nu > 0`` and detuning
    ``delta_theta`` (both per token and channel, shared over modes):

        lam   = exp(delta_nu * (-a + j * (omega + delta_theta)))     # exact ZOH
        Bu    = sqrt(1 - |lam|^2) * (B_re + j B_im) * x              # LRU input norm
        h_t   = lam * h_{t-1} + Bu_t
        y_t   = Re(<C_t, h_t>) + D * x_t

    Properties by construction:
      - ``delta_nu -> 0``  =>  ``lam -> 1`` and ``Bu -> 0``: exact selective hold/skip
        for every mode (magnitude and phase are discretized consistently).
      - ``sqrt(1 - |lam|^2)`` bounds state energy as poles approach the unit circle.
      - ``B_t`` (complex, token-dependent) chooses the drive phase per oscillator;
        ``C_t`` the readout phase. Both are shared across channels; ``a``/``omega``
        are per (channel, mode) and static.

    All token-dependent quantities come from one fused projection
    ``x_proj: d_model -> 2*dt_rank + 4*d_state`` laid out as
    ``[dt_nu (r) | dt_theta (r) | B_re (N) | B_im (N) | C_re (N) | C_im (N)]``,
    followed by the two low-rank "up" maps for the selective steps. The up weights
    are zero-initialized (LoRA-style): at init ``delta_nu = softplus(bias)`` is
    exactly Mamba's dt init and ``delta_theta = 0``, with full-scale gradients.

    Init of the static bank: decay rates ``a = 1..d_state`` (S4D-real); phases tile
    a deterministic grid — ``n_real_modes`` real poles (theta = 0; pure decays for
    envelope / stationary-floor tracking) followed by oscillators evenly spaced on
    ``(0, theta_max]``. The grid is the realized *phase-per-step* at init: omega
    stores ``theta_0 / dt`` per channel. The same grid in every channel keeps mode
    ``n`` semantically consistent under the channel-shared ``B_t`` / ``C_t``.

    For latent-STFT speech enhancement, sensible per-branch settings (via the
    ``time_mixer`` / ``freq_mixer`` sub-configs): time branch ``theta_max`` in the
    speech modulation band (~0.8 rad at 160 frames/s) with a few real modes;
    frequency branch full ``pi`` with a shorter ``dt`` band (e.g. 5e-3..0.3).

    Args:
        d_model: Number of channels.
        d_state: Complex modes per channel. Default 16.
        dt_rank: Bottleneck rank of the selective-step paths. ``None`` =>
            ``ceil(d_model / 64)`` (Mamba's ``ceil(d/16)`` at ``expand=4``).
        dt_min, dt_max: Baseline timestep band (Mamba dt init). Defaults 1e-3, 0.1.
        theta_max: Top of the oscillator phase grid at init. Default pi.
        n_real_modes: Leading modes initialized as real poles. Default 0.
        use_triton: Run the fused CUDA kernel (state stays in registers; backward
            recomputes in chunks) instead of the materialized reference path.
        chunk_size: Backward recompute chunk width (peak extra HBM scales with it).
        block_c: Backward per-program channel-slab width. (The forward kernel
            autotunes its own block size.)

    ``dt_nu_up.bias`` / ``dt_theta_up.bias`` carry ``_no_reinit`` (mamba_ssm's
    ``_init_weights`` zeroes untagged Linear biases); ``A_log`` / ``omega`` / ``D``
    carry ``_no_weight_decay``.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        dt_rank: int | None = None,
        dt_min: float = 1e-3,
        dt_max: float = 0.1,
        theta_max: float = math.pi,
        n_real_modes: int = 0,
        use_triton: bool = False,
        chunk_size: int = 16,
        block_c: int = 32,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if not 0 <= n_real_modes <= d_state:
            raise ValueError(f"n_real_modes must be in [0, d_state={d_state}], got {n_real_modes}")
        self.d_model = d_model
        self.d_state = d_state
        self.n_real_modes = n_real_modes
        self.use_triton = use_triton
        self.chunk_size = chunk_size
        self.block_c = block_c
        self.dt_rank = dt_rank if dt_rank is not None else max(1, math.ceil(d_model / 64))
        r, N = self.dt_rank, d_state

        # Fused token-dependent projection: [dt_nu | dt_theta | B_re B_im | C_re C_im].
        self.x_proj = nn.Linear(d_model, 2 * r + 4 * N, bias=False)
        self.dt_nu_up = nn.Linear(r, d_model, bias=True)
        self.dt_theta_up = nn.Linear(r, d_model, bias=True)

        # Static per-(channel, mode) dynamics.
        A = torch.arange(1, N + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))

        dt = torch.exp(
            torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        n_osc = N - n_real_modes
        theta0 = torch.cat([
            torch.zeros(n_real_modes),
            torch.linspace(theta_max / max(n_osc, 1), theta_max, n_osc),
        ])
        self.omega = nn.Parameter(theta0.expand(d_model, N) / dt.unsqueeze(-1))

        self.D = nn.Parameter(torch.ones(d_model))

        with torch.no_grad():
            # LoRA-style selective init: up weights zero => delta_theta = 0 and
            # delta_nu = softplus(bias) = dt exactly, with full-scale gradients.
            self.dt_nu_up.weight.zero_()
            self.dt_theta_up.weight.zero_()
            self.dt_nu_up.bias.copy_(_inv_softplus(dt))
            self.dt_theta_up.bias.zero_()

        self.dt_nu_up.bias._no_reinit = True
        self.dt_theta_up.bias._no_reinit = True
        self.A_log._no_weight_decay = True
        self.omega._no_weight_decay = True
        self.D._no_weight_decay = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_, T, C = x.shape
        r, N = self.dt_rank, self.d_state

        dnu_r, dth_r, B_sel, C_sel = torch.split(self.x_proj(x), [r, r, 2 * N, 2 * N], dim=-1)
        delta_nu = F.softplus(self.dt_nu_up(dnu_r))                   # (B, T, C) > 0
        delta_theta = self.dt_theta_up(dth_r)                         # (B, T, C)
        a = torch.exp(self.A_log)                                     # (C, N) > 0

        if self.use_triton:
            from models.selective_lru.selective_lru_triton import fused_selective_lru
            return fused_selective_lru(
                delta_nu, delta_theta, a, self.omega,
                B_sel[..., :N], B_sel[..., N:], C_sel[..., :N], C_sel[..., N:],
                x, self.D, block_t=self.chunk_size, block_c=self.block_c,
            )

        # Materialized reference path (CPU / debugging).
        nu = torch.exp(-delta_nu.unsqueeze(-1) * a)                   # (B, T, C, N)
        theta = delta_nu.unsqueeze(-1) * (self.omega + delta_theta.unsqueeze(-1))
        lam = torch.polar(nu, theta)
        gain = torch.sqrt((1.0 - nu * nu).clamp_min(1e-6))
        Bmat = torch.complex(B_sel[..., :N], B_sel[..., N:]).unsqueeze(-2)  # (B, T, 1, N)
        Bu = (gain * x.unsqueeze(-1)) * Bmat                          # (B, T, C, N) complex

        h = _selective_recurrence(lam.reshape(B_, T, C * N), Bu.reshape(B_, T, C * N))
        h = h.reshape(B_, T, C, N)

        C_complex = torch.complex(C_sel[..., :N], C_sel[..., N:])
        Cy = torch.einsum("btn,btcn->btc", C_complex, h).real
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

        # These biases are the r/theta init; protect them from _init_weights.
        self.nu_proj.bias._no_reinit = True
        self.theta_proj.bias._no_reinit = True

        B_std = 1.0 / math.sqrt(in_features)
        self.B = nn.Parameter(_uniform_init((state_dim, in_features, 2), std=B_std))
        C_std = 1.0 / math.sqrt(state_dim)
        self.C = nn.Parameter(_uniform_init((in_features, state_dim, 2), std=C_std))
        self.D = nn.Parameter(torch.randn(in_features))
        self.D._no_weight_decay = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nu = torch.sigmoid(self.nu_proj(x))
        theta = self.theta_proj(x)
        lam = torch.polar(nu, theta)

        B_complex = torch.complex(self.B[..., 0], self.B[..., 1])
        C_complex = torch.complex(self.C[..., 0], self.C[..., 1])

        Bu = _project_input(B_complex, x)
        if self.normalize_input:
            gamma = torch.sqrt((1.0 - nu ** 2).clamp_min(1e-6))
            Bu = Bu * gamma

        if self.use_triton:
            from models.selective_lru.selective_lru_mimo_triton import selective_scan_triton
            h = selective_scan_triton(lam, Bu)
        else:
            h = _selective_recurrence(lam, Bu)

        Cy = torch.einsum("fn,btn->btf", C_complex, h).real
        return Cy + x * self.D


def _selective_recurrence(lam: torch.Tensor, Bu: torch.Tensor) -> torch.Tensor:
    B, T, N = Bu.shape
    h = Bu.new_zeros(B, N)
    out = Bu.new_empty(B, T, N)
    for t in range(T):
        h = lam[:, t] * h + Bu[:, t]
        out[:, t] = h
    return out