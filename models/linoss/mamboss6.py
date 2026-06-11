"""MambOSS6: a per-channel (SISO) oscillatory selective SSM with Mamba's S6 factorization.

Where ``MambOSS`` (``selective_linoss.py``) is a single *MIMO* bank of ``state_dim``
complex modes shared across all feature channels — paying a dense complex
``(state_dim, in_features)`` ``B``/``C`` plus full-rank selective ``W_nu``/``W_theta`` —
``MambOSS6`` adopts Mamba's S6 parameter factorization and swaps Mamba's real,
decay-only transition for a complex (oscillatory) one:

  * **Per-channel SISO.** Each of the ``in_features`` channels carries its own bank of
    ``state_dim`` complex oscillator modes (effective state ``in_features * state_dim``
    — ~two orders of magnitude larger than MambOSS's single shared bank). Cross-channel
    mixing happens in the surrounding ``in_proj``/``out_proj`` (``MambaStyleMixer``),
    exactly as in Mamba; the SSM itself is diagonal per channel.
  * **Shared, selective B/C.** ``B`` (real) and ``C`` (complex) are ``state_dim``-wide
    vectors produced from the input token and *reused across all channels* — Mamba's
    trick — instead of dense per-channel matrices.
  * **Low-rank, decoupled selectivity.** Damping ``nu`` and frequency ``theta`` are each
    driven by a *separate* Mamba-style low-rank time-step path (a ``dt_rank`` bottleneck),
    preserving D-LinOSS's magnitude/frequency decoupling while keeping the selective
    machinery cheap.

Transition. For channel ``c``, mode ``n``, step ``k`` the complex eigenvalue is

    lambda = nu * exp(i theta),
    nu     = exp(-Delta_nu_{c,k} * a_{c,n}),    a = exp(A_log) > 0   (selective damping)
    theta  = omega_{c,n} + delta_theta_{c,k},                       (selective frequency)

with per-channel, input-dependent ``Delta_nu = softplus(low_rank_nu(x)) > 0`` and
``delta_theta = low_rank_theta(x)``. Because ``a > 0`` and ``Delta_nu > 0``, ``|lambda| =
nu < 1`` for *any* input schedule: the diagonal complex transition is normal, so the
schedule-free stability guarantee of ``selective_linoss.md`` (sections 6, 10) carries over
per channel. In one phrase: *Mamba's S6 with oscillatory eigenvalues*, with the magnitude
and frequency knobs kept decoupled.

The recurrence ``h_k = lambda_k h_{k-1} + (Delta_nu * B)_k x_k`` is the same complex
selective-LRU scan as ``MambOSS`` — the ``(channel, mode)`` axes are folded into one — so
it reuses ``_selective_recurrence`` / ``selective_scan_triton`` from the MambOSS path
unchanged.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.linoss.selective_linoss import _selective_recurrence


def _inv_softplus(x: torch.Tensor) -> torch.Tensor:
    # Inverse of softplus: returns y such that softplus(y) = x, for x > 0. Stable form
    # used by Mamba's dt init (x + log(-expm1(-x)) == x + log1p(-exp(-x))).
    return x + torch.log(-torch.expm1(-x))


class MambOSS6(nn.Module):
    """Per-channel oscillatory selective SSM with Mamba's low-rank shared projections.

    Shapes:
        Input:  (batch, time, in_features)
        Output: (batch, time, in_features)

    Args:
        in_features: Model dim fed to the mixer (the Mamba-style block's inner dim).
            This is the *channel* axis — each channel is an independent oscillator bank.
        state_dim: Number of complex oscillator modes *per channel*. Default 16.
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
        in_features: int,
        state_dim: int = 16,
        dt_rank: int | None = None,
        dt_min: float = 1e-3,
        dt_max: float = 0.1,
        theta_max: float = math.pi,
        selective_init_std: float = 1e-2,
        use_triton: bool = False,
        chunk_size: int = 16,
        block_c: int = 32,
    ):
        super().__init__()
        self.in_features = in_features
        self.state_dim = state_dim
        self.use_triton = use_triton
        self.chunk_size = chunk_size
        self.block_c = block_c
        if dt_rank is None:
            dt_rank = max(1, math.ceil(in_features / 64))
        self.dt_rank = dt_rank

        # Shared, input-dependent B (real) and C (complex): state_dim-wide vectors reused
        # across every channel (Mamba-style), not dense per-channel matrices.
        self.B_proj = nn.Linear(in_features, state_dim, bias=False)
        self.C_proj = nn.Linear(in_features, 2 * state_dim, bias=False)

        # Decoupled, low-rank selective time-steps: the nu-path (damping) and theta-path
        # (frequency) each get their own dt_rank bottleneck, broadcast per channel.
        self.dt_nu_down = nn.Linear(in_features, dt_rank, bias=False)
        self.dt_nu_up = nn.Linear(dt_rank, in_features, bias=True)
        self.dt_theta_down = nn.Linear(in_features, dt_rank, bias=False)
        self.dt_theta_up = nn.Linear(dt_rank, in_features, bias=True)

        # Per-(channel, mode) static dynamics. Decay rate a = exp(A_log) > 0 (init
        # reproduces Mamba's S4D-real A = 1..state_dim); base frequency omega spread over
        # the full angular range.
        A = torch.arange(1, state_dim + 1, dtype=torch.float32).repeat(in_features, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.omega = nn.Parameter(torch.rand(in_features, state_dim) * theta_max)

        self.D = nn.Parameter(torch.randn(in_features))

        with torch.no_grad():
            # nu-path baseline: softplus(dt_nu_up.bias) uniform in [dt_min, dt_max]
            # (Mamba's dt init), so baseline nu = exp(-dt * a) lands near the unit disk.
            dt = torch.exp(
                torch.rand(in_features) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            )
            self.dt_nu_up.bias.copy_(_inv_softplus(dt))
            # theta-path baseline: delta_theta = 0, so theta = omega at init (the full
            # angular range, matching MambOSS/D-LinOSS init).
            self.dt_theta_up.bias.zero_()
            # Selective weights start small (perturbation off a working oscillator bank).
            for proj in (self.dt_nu_down, self.dt_nu_up, self.dt_theta_down, self.dt_theta_up):
                proj.weight.normal_(0.0, selective_init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_, T, C = x.shape
        N = self.state_dim

        # Per-channel selective time-steps (low-rank). Delta_nu > 0 via softplus.
        delta_nu = F.softplus(self.dt_nu_up(self.dt_nu_down(x)))      # (B, T, C) > 0
        delta_theta = self.dt_theta_up(self.dt_theta_down(x))         # (B, T, C)

        a = torch.exp(self.A_log)                                     # (C, N) > 0
        Bsel = self.B_proj(x)                                        # (B, T, N) real, shared
        C_sel = self.C_proj(x)                                       # (B, T, 2N)
        C_complex = torch.complex(C_sel[..., :N], C_sel[..., N:])    # (B, T, N) shared

        if self.use_triton:
            # Fused scan + readout: the (B, T, C, N) state never hits HBM.
            from models.linoss.mamboss6_triton import fused_mamboss6
            return fused_mamboss6(
                delta_nu, delta_theta, a, self.omega, Bsel, C_complex, x, self.D,
                block_t=self.chunk_size, block_c=self.block_c,
            )

        # Reference (materialized) path: per-(channel, mode) complex selective scan.
        nu = torch.exp(-delta_nu.unsqueeze(-1) * a)                   # (B, T, C, N) in (0,1)
        theta = self.omega + delta_theta.unsqueeze(-1)               # (B, T, C, N)
        lam = torch.polar(nu, theta)                                 # (B, T, C, N) complex

        # Discretized drive (Delta_nu * B) * x: B is shared across channels, x is the
        # per-channel input — the Mamba B-bar = Delta * B convention, oscillatory here.
        Bu = (delta_nu.unsqueeze(-1) * Bsel.unsqueeze(-2)) * x.unsqueeze(-1)  # (B, T, C, N)
        Bu = Bu.to(lam.dtype)

        # Fold (channel, mode) into one independent-recurrence axis for the complex scan.
        h = _selective_recurrence(lam.reshape(B_, T, C * N), Bu.reshape(B_, T, C * N))
        h = h.reshape(B_, T, C, N)

        # Readout: y_c = Re( sum_n C_n * h_{c,n} ), with C shared across channels.
        Cy = torch.einsum("btn,btcn->btc", C_complex, h).real        # (B, T, C)
        return Cy + x * self.D
