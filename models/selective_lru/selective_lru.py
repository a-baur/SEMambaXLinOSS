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
        input_gain: Input normalization of ``Bu = gain * B_t * x``. One of:
            - ``"gamma"`` (default): LRU energy norm ``sqrt(1 - |lam|^2)``. Unit
              steady-state energy at every pole radius; gating sharpness is
              square-root-compressed (~ sqrt(2*delta_nu*a) for small steps).
            - ``"mamba"``: Euler / S6 gain ``delta_nu`` (real, shared over modes).
              Linear (sharp) gating; state energy unnormalized near |lam| -> 1.
            - ``"zoh"``: exact ZOH ``(lam - 1) / A_c`` with ``A_c = -a + j*omega~``
              (complex, per mode). Asymptotically ``delta_nu`` for small steps
              (sharp) AND bounded by ``1/|A_c|`` (safe); also rotates the drive
              phase per (channel, mode), which "gamma"/"mamba" cannot.
            All three preserve exact skip semantics (``delta_nu -> 0 => Bu -> 0``).
            No parameters are added: state dicts are identical across modes, but
            checkpoints are NOT semantically interchangeable, and the modes are
            NOT equivalent at init (this is a mutually exclusive ablation arm,
            not a no-op feature flag). The envelope readout's scale calibration
            assumes "gamma"; with other gains ``|h|^2`` is unnormalized.
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
        input_gain: str = "gamma",
        use_triton: bool = False,
        chunk_size: int = 16,
        block_c: int = 32,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if not 0 <= n_real_modes <= d_state:
            raise ValueError(f"n_real_modes must be in [0, d_state={d_state}], got {n_real_modes}")
        if input_gain not in ("gamma", "mamba", "zoh"):
            raise ValueError(f"input_gain must be 'gamma', 'mamba', or 'zoh', got {input_gain!r}")
        self.input_gain = input_gain
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
                input_gain=self.input_gain,
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
        Bmat = torch.complex(B_sel[..., :N], B_sel[..., N:]).unsqueeze(-2)  # (B, T, 1, N)
        if self.input_gain == "gamma":
            gain = torch.sqrt((1.0 - nu * nu).clamp_min(1e-6))
            Bu = (gain * x.unsqueeze(-1)) * Bmat                      # (B, T, C, N) complex
        elif self.input_gain == "mamba":
            # Euler / S6 gain: delta_nu, real, shared over modes.
            Bu = (delta_nu * x).unsqueeze(-1) * Bmat
        else:  # "zoh": (lam - 1) / A_c, complex per (channel, mode).
            # lam - 1 via expm1 to avoid cancellation at small delta_nu * |A_c|:
            #   Re(lam) - 1 = expm1(-dnu*a) * cos(theta) - 2 sin^2(theta/2)
            dn = delta_nu.unsqueeze(-1)
            lm1_re = torch.expm1(-dn * a) * torch.cos(theta) - 2.0 * torch.sin(0.5 * theta).square()
            lm1_im = nu * torch.sin(theta)
            den = (a * a + base * base).clamp_min(1e-8)               # |A_c|^2
            g_re = (base * lm1_im - a * lm1_re) / den                 # (lam-1) * conj(A_c) / |A_c|^2
            g_im = -(a * lm1_im + base * lm1_re) / den
            Bu = (torch.complex(g_re, g_im) * x.unsqueeze(-1)) * Bmat

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
    """Multi-head selective MIMO SSM with exact-ZOH dynamics (SISO-consistent).

    The pole parameterization mirrors ``SelectiveLRU`` exactly; the state
    layout stays MIMO: one shared bank of ``state_dim`` complex modes, split
    block-diagonally over ``n_heads`` heads. Head ``h`` owns
    ``state_dim / n_heads`` modes fed by (and read out to) only its
    ``in_features / n_heads`` channels through static dense complex B/C.

    The head is the MIMO analog of the SISO channel: the selective step
    ``delta_nu`` and detuning ``delta_theta`` are per (token, head), shared
    over the head's modes, exactly as the SISO shares them over each channel's
    mode bank. Per head h and mode n:

        delta_nu    = rate_scale * softplus(dt_nu_proj(x))        # (B,T,H) > 0
        delta_theta = dt_theta_proj(x)                            # (B,T,H)
        lam  = exp(delta_nu * (-a + j*(omega + s * delta_theta))) # exact ZOH
        Bu   = input_gain * (B_static x  [+ rank-r correction])
        h_t  = lam * h_{t-1} + Bu_t
        y    = Re(C_static h) [+ <g, |h|^2>] + D * x  [+ rank-r read]

    ``input_gain`` selects the drive scaling, exactly as in ``SelectiveLRU``:
    ``"gamma"`` (default) the LRU energy norm ``sqrt(1 - |lam|^2)``, ``"mamba"``
    the Euler/S6 gain ``delta_nu``, ``"zoh"`` the exact ZOH gain
    ``(lam - 1) / A_c`` (complex, ``A_c = -a + j*base``), or ``"none"`` (raw
    ``B x``). The gain is applied to the TOTAL input map (static + rank
    correction). The legacy ``normalize_input`` bool maps to
    ``gamma`` / ``none``.

    Properties by construction (as in the SISO cell):
      - ``delta_nu -> 0`` => ``lam -> 1`` and ``Bu -> 0``: exact selective
        hold/skip; magnitude and phase are discretized consistently. (Holds for
        ``gamma`` / ``mamba`` / ``zoh``; ``none`` does not preserve the skip.)
      - ``gamma`` bounds state energy as poles approach the unit circle,
        keeping the LRU variance argument intact; ``zoh`` is bounded by
        ``1/|A_c|`` and additionally rotates the drive phase per (head, mode).

    Init of the static bank (SISO conventions): decay rates ``a = 1..N_h``
    per head (S4D-real); phases tile the deterministic grid —
    ``n_real_modes`` real poles followed by oscillators evenly spaced on
    ``(0, theta_max]`` — identically in every head; ``omega`` stores
    ``theta_0 / dt`` with ``dt`` per head, log-uniform in
    ``[dt_min, dt_max]``. The ``dt_*_proj`` weights are zero-initialized
    (LoRA-style): at init ``delta_theta = 0`` and
    ``delta_nu = softplus(bias) = dt`` exactly, with full-scale gradients.
    Unlike the SISO there is no low-rank bottleneck on the selective paths:
    the target dimension is ``n_heads`` (small), so direct
    ``Linear(in_features, n_heads)`` maps are used.

    Rank-r additive selective B/C (per head, S6-faithful):
    ``Bu_t += sum_i <v_i, u_t> b_t^(i)`` with input-dependent real write
    directions, and ``y_t += sum_i s^C_{t,i} Re(<c_t^(i), h_t>) a^(i)`` with
    input-dependent complex read vectors; coefficient paths are
    ``selective_init_std``-small at init. All corrections and the optional
    envelope readout are feedthrough, computed from the state sequence
    outside the scan, so the fused Triton kernel is consumed unchanged (it
    receives the finished ``lam`` / ``Bu`` and returns ``y`` plus the state).

    Flags (SISO extras, all exact no-ops at init):
        envelope_readout: quadratic band-power readout ``y += <g, |h|^2>``
            with block-diagonal ``g`` of shape ``(H, F_h, N_h)`` — each
            channel reads its head's mode powers, mirroring static C.
            Zero-init. Default False. Its scale calibration assumes the
            ``gamma`` gain; with other gains ``|h|^2`` is unnormalized.
        input_gain: Drive scaling ``Bu = gain * (B_static x [+ rank-r])``. One
            of ``"gamma"`` (default), ``"mamba"``, ``"zoh"``, ``"none"``; see
            the class summary. Mutually exclusive ablation arm (adds no
            parameters, but checkpoints are not semantically interchangeable).
        mode_detune: per-mode gains ``s`` of shape ``(H, N_h)`` on the shared
            detuning, ``theta = delta_nu * (omega + s * delta_theta)``.
            Ones-init. Default False.
        rate_scale: frame-rate conditioning factor on the selective step
            (``fps_ref / fps``); constructor default, overridable per forward
            call. Default 1.0.

    Removed relative to previous revisions: ``phase_mode`` / ``phase_value``
    (phase is always ZOH-selective), ``r_min`` / ``r_max`` (the ring init is
    replaced by the dt-band init), and ``compile_surround`` together with the
    pre-/post-scan split (single monolithic ``forward``; wrap externally if
    compilation is wanted, noting TorchInductor falls back on the complex
    ops). CHECKPOINT BREAK: state dicts of the sigmoid/ring parameterization
    (``nu_proj`` / ``theta_proj``) do not load into this class. The
    state-passing API (``h_init`` / ``return_final_state``) is deliberately
    NOT provided here.

    ``dt_nu_proj.bias`` / ``dt_theta_proj.bias`` carry ``_no_reinit``
    (mamba_ssm's ``_init_weights`` zeroes untagged Linear biases — the
    critical dt-init bugfix); ``A_log`` / ``omega`` / ``D`` / ``env_gain`` /
    ``detune_gain`` carry ``_no_weight_decay``.
    """

    def __init__(
        self,
        in_features: int,
        state_dim: int = 16,
        n_heads: int = 1,
        rank: int = 0,
        dt_min: float = 1e-3,
        dt_max: float = 0.1,
        theta_max: float = math.pi,
        n_real_modes: int = 0,
        envelope_readout: bool = False,
        mode_detune: bool = False,
        rate_scale: float = 1.0,
        selective_init_std: float = 1e-2,
        input_gain: str = "gamma",
        normalize_input: bool | None = None,
        use_triton: bool = False,
    ):
        super().__init__()
        if in_features % n_heads != 0:
            raise ValueError(f"in_features={in_features} not divisible by n_heads={n_heads}")
        if state_dim % n_heads != 0:
            raise ValueError(f"state_dim={state_dim} not divisible by n_heads={n_heads}")
        if not 0 <= n_real_modes <= state_dim // n_heads:
            raise ValueError(
                f"n_real_modes must be in [0, state_dim/n_heads={state_dim // n_heads}], "
                f"got {n_real_modes}"
            )

        if input_gain not in ("gamma", "mamba", "zoh", "none"):
            raise ValueError(
                f"input_gain must be 'gamma', 'mamba', 'zoh', or 'none', got {input_gain!r}"
            )
        self.in_features = in_features
        self.state_dim = state_dim
        self.n_heads = n_heads
        self.rank = rank
        self.n_real_modes = n_real_modes
        self.envelope_readout = envelope_readout
        self.mode_detune = mode_detune
        self.rate_scale = float(rate_scale)
        self.input_gain = input_gain
        self.use_triton = use_triton
        self.f_head = in_features // n_heads  # channels per head (F_h)
        self.n_head = state_dim // n_heads  # modes per head (N_h)

        H, Nh, Fh = n_heads, self.n_head, self.f_head

        # Per-(token, head) selective step and detuning, shared over the
        # head's modes (SISO: per channel, shared over the channel's modes).
        self.dt_nu_proj = nn.Linear(in_features, H, bias=True)
        self.dt_theta_proj = nn.Linear(in_features, H, bias=True)

        # Static per-(head, mode) dynamics: S4D-real decays, deterministic
        # phase grid realized at the per-head baseline step dt.
        A = torch.arange(1, Nh + 1, dtype=torch.float32).repeat(H, 1)
        self.A_log = nn.Parameter(torch.log(A))  # (H, Nh)

        dt = torch.exp(torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        n_osc = Nh - n_real_modes
        theta0 = torch.cat(
            [
                torch.zeros(n_real_modes),
                torch.linspace(theta_max / max(n_osc, 1), theta_max, n_osc),
            ]
        )
        self.omega = nn.Parameter(theta0.expand(H, Nh) / dt.unsqueeze(-1))  # (H, Nh)

        # Static block-diagonal B/C: head h maps its F_h channels to its N_h modes.
        B_std = 1.0 / math.sqrt(Fh)
        self.B = nn.Parameter(_uniform_init((H, Nh, Fh, 2), std=B_std))
        C_std = 1.0 / math.sqrt(Nh)
        self.C = nn.Parameter(_uniform_init((H, Fh, Nh, 2), std=C_std))
        self.D = nn.Parameter(torch.ones(in_features))

        # Rank-r additive selective B/C (skipped entirely when rank == 0).
        if rank > 0:
            self.b_dir = nn.Linear(in_features, H * rank * Nh, bias=False)
            self.b_coef = nn.Linear(in_features, H * rank, bias=False)
            self.c_read = nn.Linear(in_features, 2 * H * rank * Nh, bias=False)
            self.c_coef = nn.Linear(in_features, H * rank, bias=False)
            # Stored 2D as (H*rank, Fh) rather than (H, rank, Fh): the size-1
            # rank axis makes the 3D grad's strides ambiguous, which trips DDP's
            # gradient-layout check. Viewed back to (H, rank, Fh) at use.
            self.c_out = nn.Parameter(_uniform_init((H * rank, Fh), std=1.0 / math.sqrt(Fh)))

        with torch.no_grad():
            # LoRA-style selective init: proj weights zero => delta_theta = 0
            # and delta_nu = softplus(bias) = dt exactly, full-scale gradients.
            self.dt_nu_proj.weight.zero_()
            self.dt_theta_proj.weight.zero_()
            self.dt_nu_proj.bias.copy_(_inv_softplus(dt))
            self.dt_theta_proj.bias.zero_()

            if rank > 0:
                self.b_dir.weight.normal_(0.0, 1.0 / math.sqrt(in_features))
                self.c_read.weight.normal_(0.0, 1.0 / math.sqrt(in_features))
                coef_std = selective_init_std / math.sqrt(in_features)
                self.b_coef.weight.normal_(0.0, coef_std)
                self.c_coef.weight.normal_(0.0, coef_std)

        # Quadratic envelope readout y += <g, |h|^2>, block-diagonal like C:
        # channel f of head h reads the head's mode powers. Zero-init no-op.
        if envelope_readout:
            self.env_gain = nn.Parameter(torch.zeros(H, Fh, Nh))
            self.env_gain._no_weight_decay = True
        # Per-mode detune gains: theta = delta_nu * (omega + s * delta_theta).
        # Ones-init: identical to the rigid shared detuning at start.
        if mode_detune:
            self.detune_gain = nn.Parameter(torch.ones(H, Nh))
            self.detune_gain._no_weight_decay = True

        self.dt_nu_proj.bias._no_reinit = True
        self.dt_theta_proj.bias._no_reinit = True
        self.A_log._no_weight_decay = True
        self.omega._no_weight_decay = True
        self.D._no_weight_decay = True

    def forward(self, x: torch.Tensor, rate_scale: float | None = None) -> torch.Tensor:
        """Apply the SSM. ``rate_scale``: per-call override of the frame-rate
        conditioning; the selective step is ``rate_scale * softplus(...)``.
        """
        Bsz, T, Fdim = x.shape
        H, Nh, N = self.n_heads, self.n_head, self.state_dim

        # Selective step / detuning per (token, head), shared over modes.
        delta_nu = F.softplus(self.dt_nu_proj(x))  # (B, T, H) > 0
        rs = self.rate_scale if rate_scale is None else float(rate_scale)
        if rs != 1.0:
            delta_nu = delta_nu * rs
        delta_theta = self.dt_theta_proj(x)  # (B, T, H)

        # Exact-ZOH pole, written as nu*cos / nu*sin (no complex tensors yet).
        a = torch.exp(self.A_log)  # (H, Nh) > 0
        dnu_n = delta_nu.unsqueeze(-1)  # (B, T, H, 1)
        dth_n = delta_theta.unsqueeze(-1)  # (B, T, H, 1)
        nu = torch.exp(-dnu_n * a)  # (B, T, H, Nh)
        base = self.omega + (self.detune_gain * dth_n if self.mode_detune else dth_n)
        theta = dnu_n * base  # (B, T, H, Nh)
        lam_re = (nu * torch.cos(theta)).reshape(Bsz, T, N)
        lam_im = (nu * torch.sin(theta)).reshape(Bsz, T, N)

        # Static block-diagonal input map (+ rank-r selective write), then the
        # input gain on the TOTAL input map (static + rank correction). Every
        # gain vanishes as delta_nu -> 0, so the selective hold/skip is exact.
        xh = x.view(Bsz, T, H, self.f_head)
        Bu_re = torch.einsum("hnf,bthf->bthn", self.B[..., 0], xh)  # (B, T, H, Nh)
        Bu_im = torch.einsum("hnf,bthf->bthn", self.B[..., 1], xh)
        if self.rank > 0:
            s_b = self.b_coef(x).view(Bsz, T, H, self.rank)  # ~0 at init
            b_t = self.b_dir(x).view(Bsz, T, H, self.rank, Nh)
            Bu_re = Bu_re + torch.einsum("bthr,bthrn->bthn", s_b, b_t)

        if self.input_gain == "gamma":
            # LRU energy norm sqrt(1 - |lam|^2): real, per (head, mode).
            gain = torch.sqrt((1.0 - nu * nu).clamp_min(1e-6))
            Bu_re = Bu_re * gain
            Bu_im = Bu_im * gain
        elif self.input_gain == "mamba":
            # Euler / S6 gain delta_nu: real, shared over the head's modes.
            Bu_re = Bu_re * dnu_n
            Bu_im = Bu_im * dnu_n
        elif self.input_gain == "zoh":
            # Exact ZOH gain (lam - 1) / A_c, complex per (head, mode), with
            # A_c = -a + j*base. expm1 avoids cancellation at small delta_nu*|A_c|:
            #   Re(lam) - 1 = expm1(-dnu*a) * cos(theta) - 2 sin^2(theta/2).
            lm1_re = torch.expm1(-dnu_n * a) * torch.cos(theta)
            lm1_re = lm1_re - 2.0 * torch.sin(0.5 * theta).square()
            lm1_im = nu * torch.sin(theta)
            den = (a * a + base * base).clamp_min(1e-8)  # |A_c|^2
            g_re = (base * lm1_im - a * lm1_re) / den  # Re[(lam-1)*conj(A_c)/|A_c|^2]
            g_im = -(a * lm1_im + base * lm1_re) / den
            Bu_re, Bu_im = g_re * Bu_re - g_im * Bu_im, g_re * Bu_im + g_im * Bu_re
        # else "none": raw B*x drive, no gain (skip semantics not preserved).

        Bu_re = Bu_re.reshape(Bsz, T, N)
        Bu_im = Bu_im.reshape(Bsz, T, N)

        # Scan + static readout: fused kernel (returns y and the state
        # sequence) or the materialized reference path.
        if self.use_triton:
            from models.selective_lru.selective_lru_mimo_triton import (
                fused_selective_lru_mimo,
            )

            y, h_re, h_im = fused_selective_lru_mimo(
                lam_re.contiguous(),
                lam_im.contiguous(),
                Bu_re.contiguous(),
                Bu_im.contiguous(),
                self.C[..., 0],
                self.C[..., 1],
                x,
                self.D,
                H,
                Nh,
                self.f_head,
            )
            hh = torch.complex(h_re, h_im).view(Bsz, T, H, Nh)
        else:
            h = _selective_recurrence(torch.complex(lam_re, lam_im), torch.complex(Bu_re, Bu_im))
            hh = h.view(Bsz, T, H, Nh)
            C_c = torch.complex(self.C[..., 0], self.C[..., 1])  # (H, Fh, Nh)
            y = torch.einsum("hfn,bthn->bthf", C_c, hh).real.reshape(Bsz, T, Fdim)
            y = y + x * self.D

        # Feedthrough extras from the state sequence (identical either path).
        if self.envelope_readout:
            p = hh.real**2 + hh.imag**2  # (B, T, H, Nh)
            y = y + torch.einsum("hfn,bthn->bthf", self.env_gain, p).reshape(Bsz, T, Fdim)
        if self.rank > 0:
            s_c = self.c_coef(x).view(Bsz, T, H, self.rank)  # ~0 at init
            cr = self.c_read(x).view(Bsz, T, H, self.rank, 2 * Nh)
            c_t = torch.complex(cr[..., :Nh], cr[..., Nh:])  # (B, T, H, r, Nh)
            read = torch.einsum("bthrn,bthn->bthr", c_t, hh).real
            c_out = self.c_out.view(H, self.rank, self.f_head)
            y = y + torch.einsum("bthr,hrf->bthf", s_c * read, c_out).reshape(Bsz, T, Fdim)
        return y


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
