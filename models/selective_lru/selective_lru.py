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
        envelope_readout: Add the quadratic band-power readout ``y += <g, |h|^2>``
            (learned static ``g``, zero-init). Rotation-invariant envelope feature a
            linear readout cannot express; well-scaled because of the gamma input
            normalization. Default False.
        mode_detune: Per-mode gains on the selective detuning,
            ``theta = delta_nu * (omega + s * delta_theta)`` (learned static ``s``,
            ones-init), lifting the rigid-rotation constraint on each channel's mode
            bank. Default False.
        rate_scale: Frame-rate conditioning factor multiplying the selective step
            (``fps_ref / fps``); makes the continuous-time semantics transferable
            across frame rates. Overridable per forward call. Default 1.0.
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
        envelope_readout: bool = False,
        mode_detune: bool = False,
        rate_scale: float = 1.0,
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
        self.envelope_readout = envelope_readout
        self.mode_detune = mode_detune
        self.rate_scale = float(rate_scale)
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

        # Quadratic envelope readout y += <g, |h|^2>: per-(channel, mode) band
        # power, i.e. the demodulated envelope of that modulation band. Zero-init:
        # exact no-op at start, learned from zero.
        if envelope_readout:
            self.env_gain = nn.Parameter(torch.zeros(d_model, N))
            self.env_gain._no_weight_decay = True
        # Per-mode detune gains: theta = delta_nu * (omega + s * delta_theta), so a
        # token can move modes non-rigidly. Ones-init: identical to the shared
        # detuning at start.
        if mode_detune:
            self.detune_gain = nn.Parameter(torch.ones(d_model, N))
            self.detune_gain._no_weight_decay = True

        self.dt_nu_up.bias._no_reinit = True
        self.dt_theta_up.bias._no_reinit = True
        self.A_log._no_weight_decay = True
        self.omega._no_weight_decay = True
        self.D._no_weight_decay = True

    def forward(
        self,
        x: torch.Tensor,
        h_init: torch.Tensor | None = None,
        rate_scale: float | None = None,
        return_final_state: bool = False,
    ):
        """Apply the SSM. ``h_init``: optional initial state, complex ``(B, C, N)``
        (e.g. the detached final state of the previous chunk for TBPTT / streaming).
        ``rate_scale``: per-call override of the frame-rate conditioning; the
        selective step is ``rate_scale * softplus(...)``, so a model trained at a
        reference frame rate transfers to rate ``fps`` via ``fps_ref / fps``.
        ``return_final_state``: also return ``h_{T-1}`` as complex ``(B, C, N)``;
        gradients flow through it unless the caller detaches.
        """
        B_, T, C = x.shape
        r, N = self.dt_rank, self.d_state

        dnu_r, dth_r, B_sel, C_sel = torch.split(self.x_proj(x), [r, r, 2 * N, 2 * N], dim=-1)
        delta_nu = F.softplus(self.dt_nu_up(dnu_r))                   # (B, T, C) > 0
        rs = self.rate_scale if rate_scale is None else float(rate_scale)
        if rs != 1.0:
            delta_nu = delta_nu * rs
        delta_theta = self.dt_theta_up(dth_r)                         # (B, T, C)
        a = torch.exp(self.A_log)                                     # (C, N) > 0

        g = self.env_gain if self.envelope_readout else None
        s = self.detune_gain if self.mode_detune else None
        h0_re = h_init.real.contiguous() if h_init is not None else None
        h0_im = h_init.imag.contiguous() if h_init is not None else None

        if self.use_triton:
            from models.selective_lru.selective_lru_triton import fused_selective_lru
            y, hf_re, hf_im = fused_selective_lru(
                delta_nu, delta_theta, a, self.omega,
                B_sel[..., :N], B_sel[..., N:], C_sel[..., :N], C_sel[..., N:],
                x, self.D, g, s, h0_re, h0_im,
                block_t=self.chunk_size, block_c=self.block_c,
                want_final_state=return_final_state,
            )
            if return_final_state:
                return y, torch.complex(hf_re, hf_im)
            return y

        # Materialized reference path (CPU / debugging).
        nu = torch.exp(-delta_nu.unsqueeze(-1) * a)                   # (B, T, C, N)
        dth_n = delta_theta.unsqueeze(-1)
        base = self.omega + (s * dth_n if s is not None else dth_n)
        theta = delta_nu.unsqueeze(-1) * base
        lam = torch.polar(nu, theta)
        gain = torch.sqrt((1.0 - nu * nu).clamp_min(1e-6))
        Bmat = torch.complex(B_sel[..., :N], B_sel[..., N:]).unsqueeze(-2)  # (B, T, 1, N)
        Bu = (gain * x.unsqueeze(-1)) * Bmat                          # (B, T, C, N) complex

        h0 = torch.complex(h0_re, h0_im).reshape(B_, C * N) if h_init is not None else None
        h = _selective_recurrence(lam.reshape(B_, T, C * N), Bu.reshape(B_, T, C * N), h0)
        h = h.reshape(B_, T, C, N)

        C_complex = torch.complex(C_sel[..., :N], C_sel[..., N:])
        y = torch.einsum("btn,btcn->btc", C_complex, h).real + x * self.D
        if g is not None:
            y = y + torch.einsum("cn,btcn->btc", g, h.real ** 2 + h.imag ** 2)
        if return_final_state:
            return y, h[:, -1]
        return y


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

    Phase parameterization (``phase_mode``):
        The pole phase ``theta`` (``lam = nu * e^{i*theta}``) can be:
        - ``"selective"`` (default): input-dependent ``theta = theta_proj(x)``
          — the original behavior.
        - ``"learnable"``: a per-mode ``nn.Parameter`` of shape ``(state_dim,)``,
          shared across time/batch (non-selective, still trained). Seeded with
          the same ``[0, theta_max)`` ring spread as the selective bias.
        - ``"fixed"``: a per-mode constant buffer (no grad) at ``phase_value``
          radians (e.g. ``0`` -> pure decay, no oscillation).
        Magnitude selectivity (``nu_proj``) is unaffected in all modes. Only the
        construction of ``theta`` changes; the scan/kernel consume the finished
        ``lam`` and need no changes.
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
        phase_mode: str = "selective",
        phase_value: float = 0.0,
        compile_surround: bool = False,
    ):
        super().__init__()
        if in_features % n_heads != 0:
            raise ValueError(f"in_features={in_features} not divisible by n_heads={n_heads}")
        if state_dim % n_heads != 0:
            raise ValueError(f"state_dim={state_dim} not divisible by n_heads={n_heads}")
        if phase_mode not in ("selective", "learnable", "fixed"):
            raise ValueError(
                f"phase_mode={phase_mode!r} must be 'selective', 'learnable', or 'fixed'"
            )
        self.in_features = in_features
        self.state_dim = state_dim
        self.n_heads = n_heads
        self.rank = rank
        self.normalize_input = normalize_input
        self.use_triton = use_triton
        self.phase_mode = phase_mode
        self.compile_surround = compile_surround
        self.f_head = in_features // n_heads   # channels per head (F_h)
        self.n_head = state_dim // n_heads     # modes per head (N_h)

        H, Nh, Fh = n_heads, self.n_head, self.f_head

        self.nu_proj = nn.Linear(in_features, state_dim)
        if phase_mode == "selective":
            self.theta_proj = nn.Linear(in_features, state_dim)
        elif phase_mode == "learnable":
            self.theta = nn.Parameter(torch.empty(state_dim))
        else:  # "fixed"
            self.register_buffer("theta", torch.full((state_dim,), float(phase_value)))

        # Static block-diagonal B/C: head h maps its F_h channels to its N_h modes.
        B_std = 1.0 / math.sqrt(Fh)
        self.B = nn.Parameter(_uniform_init((H, Nh, Fh, 2), std=B_std))
        C_std = 1.0 / math.sqrt(Nh)
        self.C = nn.Parameter(_uniform_init((H, Fh, Nh, 2), std=C_std))
        self.D = nn.Parameter(torch.randn(in_features))

        # Rank-r additive selective B/C (skipped entirely when rank == 0).
        if rank > 0:
            self.b_dir = nn.Linear(in_features, H * rank * Nh, bias=False)
            self.b_coef = nn.Linear(in_features, H * rank, bias=False)
            self.c_read = nn.Linear(in_features, 2 * H * rank * Nh, bias=False)
            self.c_coef = nn.Linear(in_features, H * rank, bias=False)
            # Stored 2D as (H*rank, Fh) rather than (H, rank, Fh): the size-1
            # rank axis makes the 3D grad's strides ambiguous, which trips DDP's
            # gradient-layout check. Viewed back to (H, rank, Fh) in _rank_read.
            self.c_out = nn.Parameter(_uniform_init((H * rank, Fh), std=1.0 / math.sqrt(Fh)))

        with torch.no_grad():
            # Ring init of pole magnitudes in [r_min, r_max] (LRU), full phase spread.
            mags = torch.sqrt(torch.rand(state_dim) * (r_max ** 2 - r_min ** 2) + r_min ** 2)
            self.nu_proj.bias.copy_(torch.logit(mags))
            self.nu_proj.weight.normal_(0.0, selective_init_std)
            if phase_mode == "selective":
                self.theta_proj.bias.copy_(torch.rand(state_dim) * theta_max)
                self.theta_proj.weight.normal_(0.0, selective_init_std)
            elif phase_mode == "learnable":
                self.theta.copy_(torch.rand(state_dim) * theta_max)

            if rank > 0:
                self.b_dir.weight.normal_(0.0, 1.0 / math.sqrt(in_features))
                self.c_read.weight.normal_(0.0, 1.0 / math.sqrt(in_features))
                coef_std = selective_init_std / math.sqrt(in_features)
                self.b_coef.weight.normal_(0.0, coef_std)
                self.c_coef.weight.normal_(0.0, coef_std)

        # Compile only the (fully real) pre-scan surround. _rank_read stays eager:
        # its complex ops (torch.complex, complex einsum) can't be lowered by
        # TorchInductor — it warns and falls back anyway — and compiling it emitted
        # a non-contiguous c_out grad that trips DDP's gradient-layout check.
        self._pre = torch.compile(self._pre_scan) if compile_surround else self._pre_scan
        self._rank = self._rank_read

    def _pre_scan(self, x: torch.Tensor):
        """Everything before the scan: pole + input map, as split real/imag.

        Returns ``(lam_re, lam_im, Bu_re, Bu_im)`` — each ``(B, T, N)`` real —
        plus never materializing a complex tensor (``lam = nu*e^{i theta}`` is
        written directly as ``nu*cos``/``nu*sin``).
        """
        Bsz, T, _ = x.shape
        H, Nh, Fh, r = self.n_heads, self.n_head, self.f_head, self.rank

        # Selective pole (magnitude strictly inside the unit disk).
        nu = torch.sigmoid(self.nu_proj(x))                    # (B, T, N)
        if self.phase_mode == "selective":
            theta = self.theta_proj(x)                         # (B, T, N)
        else:
            theta = self.theta                                 # (N,), broadcasts over (B, T)
        lam_re = nu * torch.cos(theta)                         # (B, T, N)
        lam_im = nu * torch.sin(theta)

        xh = x.view(Bsz, T, H, Fh)

        # Static block-diagonal input map (real/imag weights x real input).
        Bu_re = torch.einsum("hnf,bthf->bthn", self.B[..., 0], xh)   # (B, T, H, Nh)
        Bu_im = torch.einsum("hnf,bthf->bthn", self.B[..., 1], xh)

        if r > 0:
            s_b = self.b_coef(x).view(Bsz, T, H, r)            # ~0 at init
            b_t = self.b_dir(x).view(Bsz, T, H, r, Nh)
            Bu_re = Bu_re + torch.einsum("bthr,bthrn->bthn", s_b, b_t)

        Bu_re = Bu_re.reshape(Bsz, T, self.state_dim)
        Bu_im = Bu_im.reshape(Bsz, T, self.state_dim)
        if self.normalize_input:
            gamma = torch.sqrt((1.0 - nu ** 2).clamp_min(1e-6))  # (B, T, N)
            Bu_re = Bu_re * gamma
            Bu_im = Bu_im * gamma

        return lam_re, lam_im, Bu_re, Bu_im

    def _rank_read(self, x, hh):
        """Rank-r selective read correction ``sum_i s^C_i Re(<c_t^(i), h>) a^(i)``.

        ``hh`` is the complex state viewed as ``(B, T, H, Nh)``. Returns the
        additive correction shaped ``(B, T, Fd)``.
        """
        Bsz, T, Fdim = x.shape
        H, Nh, r = self.n_heads, self.n_head, self.rank

        s_c = self.c_coef(x).view(Bsz, T, H, r)                # ~0 at init
        cr = self.c_read(x).view(Bsz, T, H, r, 2 * Nh)
        c_t = torch.complex(cr[..., :Nh], cr[..., Nh:])        # (B, T, H, r, Nh)
        read = torch.einsum("bthrn,bthn->bthr", c_t, hh).real
        c_out = self.c_out.view(H, r, self.f_head)             # (H, rank, Fh)
        corr = torch.einsum("bthr,hrf->bthf", s_c * read, c_out)
        return corr.reshape(Bsz, T, Fdim)

    def _post_scan(self, x, h_re, h_im):
        """Reference (non-Triton) readout: ``Re(C h) + D x`` plus the rank read.

        The Triton path folds the static ``Re(C h) + D x`` into the fused kernel;
        this pure-PyTorch version is used for CPU / correctness.
        """
        Bsz, T, Fdim = x.shape
        H, Nh = self.n_heads, self.n_head

        hh = torch.complex(h_re, h_im).view(Bsz, T, H, Nh)

        C_c = torch.complex(self.C[..., 0], self.C[..., 1])    # (H, Fh, Nh)
        y = torch.einsum("hfn,bthn->bthf", C_c, hh).real.reshape(Bsz, T, Fdim)
        y = y + x * self.D

        if self.rank > 0:
            y = y + self._rank_read(x, hh)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lam_re, lam_im, Bu_re, Bu_im = self._pre(x)

        if self.use_triton:
            from models.selective_lru.selective_lru_mimo_triton import (
                fused_selective_lru_mimo,
            )
            y, h_re, h_im = fused_selective_lru_mimo(
                lam_re.contiguous(), lam_im.contiguous(),
                Bu_re.contiguous(), Bu_im.contiguous(),
                self.C[..., 0], self.C[..., 1], x, self.D,
                self.n_heads, self.n_head, self.f_head,
            )
            if self.rank > 0:
                hh = torch.complex(h_re, h_im).view(
                    x.shape[0], x.shape[1], self.n_heads, self.n_head
                )
                y = y + self._rank(x, hh)
            return y

        h = _selective_recurrence(
            torch.complex(lam_re, lam_im), torch.complex(Bu_re, Bu_im)
        )
        return self._post_scan(x, h.real, h.imag)


def _selective_recurrence(
    lam: torch.Tensor, Bu: torch.Tensor, h0: torch.Tensor | None = None
) -> torch.Tensor:
    B, T, N = Bu.shape
    h = Bu.new_zeros(B, N) if h0 is None else h0
    out = Bu.new_empty(B, T, N)
    for t in range(T):
        h = lam[:, t] * h + Bu[:, t]
        out[:, t] = h
    return out


def _subsample(t: torch.Tensor, k: int) -> torch.Tensor:
    """Flatten, randomly subsample to <=k elements, detach to CPU float32."""
    t = t.reshape(-1)
    if t.numel() > k:
        idx = torch.randint(0, t.numel(), (k,), device=t.device)
        t = t[idx]
    return t.detach().float().cpu()


def _collect_cell_stats(delta_nu, theta, nu, gain, h, lin, env, k: int = 50_000) -> dict:
    """Compact, detached cell-internal statistics from one instrumented forward.

    Returns subsampled activation distributions (histogram-ready) plus scalar
    reductions that must be exact over the whole tensor (energy tails, the
    envelope-vs-readout contribution ratio, the buzz-band escape fraction). All
    quantities are the *on-data* realized values, not the static baseline.
    """
    h2 = h.real ** 2 + h.imag ** 2                       # (B, T, C, N) = |h|^2
    theta_wrapped = torch.remainder(theta + math.pi, 2.0 * math.pi) - math.pi
    stats = {
        "delta_nu": _subsample(delta_nu, k),             # (B, T, C)
        "theta": _subsample(theta_wrapped, k),           # (B, T, C, N) functional
        "abs_lam": _subsample(nu, k),                    # (B, T, C, N) = |lam|
        "gamma": _subsample(gain, k),                    # (B, T, C, N) input gain
        "h2": _subsample(h2, k),                         # (B, T, C, N)
        # Exact scalar reductions (no subsampling).
        "h2_max": float(h2.max()),
        "h2_mean": float(h2.mean()),
        "abs_lam_p9999": float(torch.quantile(_subsample(nu, k).double(), 0.9999)),
        "frac_theta_escape": float((theta_wrapped.abs() > 1.6).float().mean()),
        "frac_lam_pileup": float((nu > 0.9999).float().mean()),
        "lin_norm": float(lin.norm()),                   # ||Re<C, h>||
        "env_norm": float(env.norm()) if env is not None else None,  # ||<g, |h|^2>||
    }
    return stats
