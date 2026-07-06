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
            from models.selective_lru.selective_lru_triton import fused_selective_lru
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
    """Multi-head selective LRU with optional rank-r additive selective B/C.

    ``n_heads=1, rank=0`` reproduces the original shared-state MIMO exactly
    (static-B/C parameters carry a leading head axis of size 1, but the
    computation is identical; port old checkpoints with ``.unsqueeze(0)``).

    Head axis (block-diagonal state sharing):
        Channels and modes are split into ``n_heads`` groups; head ``h`` owns
        ``state_dim / n_heads`` modes fed by (and read out to) only its
        ``in_features / n_heads`` channels, i.e. static B and C are
        block-diagonal. This interpolates between the shared bank (H=1, LRU/S5
        style) and SISO-like private banks (H -> in_features, Mamba style).
        NOTE at fixed ``state_dim``, static B/C params and FLOPs shrink by a
        factor of H; scale ``state_dim`` proportionally to ``n_heads`` for an
        iso-capacity comparison.

    Rank-r additive selective B/C (per head, S6-faithful translation):
        B_t = B_static + sum_i  b_t^(i) (v^(i))^T      acting as
            Bu_t += sum_i  s^B_t,i * b_t^(i),          s^B_t,i = <v^(i), u_t>
        C_t = C_static + sum_i  a^(i) (c_t^(i))^T      acting as
            y_t  += sum_i  s^C_t,i * Re(<c_t^(i), h_t>) * a^(i)
        with input-dependent write directions b_t (real, mirroring the SISO
        ``B_proj``) and read vectors c_t (complex, mirroring the SISO
        ``C_proj``), input-dependent scalar coefficients s^B, s^C, and static
        learnable output directions a. The coefficient paths are initialized
        with ``selective_init_std`` so the model starts as the validated
        static multi-head MIMO and learns the corrections as perturbations.
        The gamma normalization is applied to the TOTAL input map
        (static + selective), keeping the LRU variance argument intact.
        The scan itself is untouched (all corrections are feedthrough), so
        stability (|lambda| < 1 by construction) and the Triton kernel are
        unaffected.
    """

    def __init__(
        self,
        in_features: int,
        state_dim: int = 16,
        n_heads: int = 1,
        rank: int = 0,
        r_min: float = 0.9,
        r_max: float = 0.999,
        theta_max: float = math.pi,
        selective_init_std: float = 1e-2,
        normalize_input: bool = True,
        use_triton: bool = False,
    ):
        super().__init__()
        if in_features % n_heads != 0:
            raise ValueError(f"in_features={in_features} not divisible by n_heads={n_heads}")
        if state_dim % n_heads != 0:
            raise ValueError(f"state_dim={state_dim} not divisible by n_heads={n_heads}")
        self.in_features = in_features
        self.state_dim = state_dim
        self.n_heads = n_heads
        self.rank = rank
        self.normalize_input = normalize_input
        self.use_triton = use_triton
        self.f_head = in_features // n_heads   # channels per head (F_h)
        self.n_head = state_dim // n_heads     # modes per head (N_h)

        H, Nh, Fh = n_heads, self.n_head, self.f_head

        # Pole selectivity: unchanged, over all state_dim = H * N_h modes. The
        # nu/theta projections stay full-input (dense over F) so H=1 is exactly
        # the old module and pole selectivity can use cross-head context.
        self.nu_proj = nn.Linear(in_features, state_dim)
        self.theta_proj = nn.Linear(in_features, state_dim)

        # Static block-diagonal B/C: head h maps its F_h channels to its N_h modes.
        B_std = 1.0 / math.sqrt(Fh)
        self.B = nn.Parameter(_uniform_init((H, Nh, Fh, 2), std=B_std))
        C_std = 1.0 / math.sqrt(Nh)
        self.C = nn.Parameter(_uniform_init((H, Fh, Nh, 2), std=C_std))
        self.D = nn.Parameter(torch.randn(in_features))

        # Rank-r additive selective B/C (skipped entirely when rank == 0).
        if rank > 0:
            # Write directions b_t (real) and read vectors c_t (complex): full
            # scale. Coefficients s^B, s^C: small init -> starts as static model.
            self.b_dir = nn.Linear(in_features, H * rank * Nh, bias=False)
            self.b_coef = nn.Linear(in_features, H * rank, bias=False)
            self.c_read = nn.Linear(in_features, 2 * H * rank * Nh, bias=False)
            self.c_coef = nn.Linear(in_features, H * rank, bias=False)
            # Static per-rank output directions a^(h,i) in R^{F_h}.
            self.c_out = nn.Parameter(_uniform_init((H, rank, Fh), std=1.0 / math.sqrt(Fh)))

        with torch.no_grad():
            # Ring init of pole magnitudes in [r_min, r_max] (LRU), full phase spread.
            mags = torch.sqrt(torch.rand(state_dim) * (r_max ** 2 - r_min ** 2) + r_min ** 2)
            self.nu_proj.bias.copy_(torch.logit(mags))
            self.theta_proj.bias.copy_(torch.rand(state_dim) * theta_max)
            self.nu_proj.weight.normal_(0.0, selective_init_std)
            self.theta_proj.weight.normal_(0.0, selective_init_std)

            if rank > 0:
                self.b_dir.weight.normal_(0.0, 1.0 / math.sqrt(in_features))
                self.c_read.weight.normal_(0.0, 1.0 / math.sqrt(in_features))
                # Width-independent perturbation scale: coefficient std at init
                # is ~selective_init_std regardless of in_features.
                coef_std = selective_init_std / math.sqrt(in_features)
                self.b_coef.weight.normal_(0.0, coef_std)
                self.c_coef.weight.normal_(0.0, coef_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Bsz, T, Fdim = x.shape
        H, Nh, Fh, r = self.n_heads, self.n_head, self.f_head, self.rank

        # Selective pole (magnitude strictly inside the unit disk).
        nu = torch.sigmoid(self.nu_proj(x))                    # (B, T, N)
        theta = self.theta_proj(x)                             # (B, T, N)
        lam = torch.polar(nu, theta)                           # (B, T, N) complex

        xh = x.view(Bsz, T, H, Fh)

        # Static block-diagonal input map (complex weights x real input).
        B_c = torch.complex(self.B[..., 0], self.B[..., 1])    # (H, Nh, Fh)
        Bu = torch.complex(
            torch.einsum("hnf,bthf->bthn", B_c.real, xh),
            torch.einsum("hnf,bthf->bthn", B_c.imag, xh),
        )                                                      # (B, T, H, Nh)

        # Rank-r selective write: Bu += sum_i s^B_i * b_t^(i)  (real directions;
        # phase behavior stays in the pole).
        if r > 0:
            s_b = self.b_coef(x).view(Bsz, T, H, r)            # ~0 at init
            b_t = self.b_dir(x).view(Bsz, T, H, r, Nh)
            Bu = Bu + torch.einsum("bthr,bthrn->bthn", s_b, b_t).to(Bu.dtype)

        Bu = Bu.reshape(Bsz, T, self.state_dim)
        if self.normalize_input:
            # Applied to the *total* input map (static + selective) so the
            # LRU variance normalization stays valid with the correction on.
            gamma = torch.sqrt((1.0 - nu ** 2).clamp_min(1e-6))  # (B, T, N)
            Bu = Bu * gamma

        if self.use_triton:
            # Scan is unchanged by heads/rank: same kernel.
            from models.selective_lru.selective_lru_mimo_triton import selective_scan_triton
            h = selective_scan_triton(lam, Bu)                 # (B, T, N) complex
        else:
            h = _selective_recurrence(lam, Bu)                 # (B, T, N) complex

        hh = h.view(Bsz, T, H, Nh)

        # Static block-diagonal readout.
        C_c = torch.complex(self.C[..., 0], self.C[..., 1])    # (H, Fh, Nh)
        y = torch.einsum("hfn,bthn->bthf", C_c, hh).real       # (B, T, H, Fh)

        # Rank-r selective read: y += sum_i s^C_i * Re(<c_t^(i), h>) * a^(i).
        if r > 0:
            s_c = self.c_coef(x).view(Bsz, T, H, r)            # ~0 at init
            cr = self.c_read(x).view(Bsz, T, H, r, 2 * Nh)
            c_t = torch.complex(cr[..., :Nh], cr[..., Nh:])    # (B, T, H, r, Nh)
            read = torch.einsum("bthrn,bthn->bthr", c_t, hh).real
            y = y + torch.einsum("bthr,hrf->bthf", s_c * read, self.c_out)

        return y.reshape(Bsz, T, Fdim) + x * self.D


def _selective_recurrence(lam: torch.Tensor, Bu: torch.Tensor) -> torch.Tensor:
    B, T, N = Bu.shape
    h = Bu.new_zeros(B, N)
    out = Bu.new_empty(B, T, N)
    for t in range(T):
        h = lam[:, t] * h + Bu[:, t]
        out[:, t] = h
    return out