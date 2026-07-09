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
        input_norm: How the (real) input map ``Bu`` is scaled before the scan.
            - ``"delta_nu"`` (default, current behavior): ``Bu = delta_nu * B_sel * x`` —
              Mamba's ZOH-style input gain (small time-step -> small drive).
            - ``"gamma"``: ``Bu = sqrt(1 - nu**2) * B_sel * x`` — the LRU
              variance-preserving normalization (drive shrinks as the pole nears the unit
              circle). The natural partner of ``mag_init="ring"``.
            Only the input gain changes; the pole ``lam`` and the scan are untouched.
        mag_init: Initialization of the baseline pole magnitude ``nu = exp(-delta_nu * a)``.
            - ``"mamba"`` (default, current behavior): ``a = 1..d_state`` (S4D-real) with
              baseline ``delta_nu = softplus(bias)`` drawn from ``[dt_min, dt_max]``
              (Mamba's dt init), so baseline ``nu`` sits just inside the unit disk.
            - ``"ring"``: baseline magnitudes drawn from the LRU ring ``[r_min, r_max]``,
              realized within the same ``nu = exp(-delta_nu * a)`` parameterization by
              fixing baseline ``delta_nu = 1`` and setting ``a = -log(nu_target)``.
              ``dt_min``/``dt_max`` are ignored in this mode.
        r_min: Lower bound of the ``mag_init="ring"`` pole-magnitude band. Default 0.9.
        r_max: Upper bound of the ``mag_init="ring"`` pole-magnitude band. Default 0.999.
        rank: Rank of an optional cross-channel readout correction added on top of the
            (channel-diagonal) base readout. ``0`` (default) disables it entirely — no
            extra parameters, forward identical to the base model. ``rank > 0`` adds a
            low-rank input-dependent cross-channel coupling on the read side, initialized
            to ~0 so training starts from the base model. Reference path only: ``rank > 0``
            forces the materialized path even when ``use_triton`` is set. See
            ``_rank_read``.
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
        input_norm: str = "delta_nu",
        mag_init: str = "mamba",
        r_min: float = 0.9,
        r_max: float = 0.999,
        rank: int = 0,
        use_triton: bool = False,
        chunk_size: int = 16,
        block_c: int = 32,
        device=None,
        dtype=None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if input_norm not in ("delta_nu", "gamma"):
            raise ValueError(f"input_norm={input_norm!r} must be 'delta_nu' or 'gamma'")
        if mag_init not in ("mamba", "ring"):
            raise ValueError(f"mag_init={mag_init!r} must be 'mamba' or 'ring'")
        if rank < 0:
            raise ValueError(f"rank={rank} must be >= 0")
        self.d_model = d_model
        self.d_state = d_state
        self.input_norm = input_norm
        self.mag_init = mag_init
        self.rank = rank
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

        # Optional rank-r cross-channel readout correction (skipped entirely when
        # rank == 0, so the default is byte-identical to the base model). The base
        # readout is channel-diagonal (channel c reads only its own bank); this adds a
        # low-rank cross-channel coupling on the read side: pool the per-channel states
        # into `rank` aggregate banks (c_pool), read each with an input-dependent complex
        # vector (c_read) + scalar (c_coef), and scatter to output channels (c_out).
        if rank > 0:
            self.c_read = nn.Linear(d_model, 2 * rank * d_state, bias=False)
            self.c_coef = nn.Linear(d_model, rank, bias=False)
            self.c_pool = nn.Parameter(torch.empty(rank, d_model))
            self.c_out = nn.Parameter(torch.empty(rank, d_model))

        # Per-channel static dynamics.
        # - Decay rate a = exp(A_log) > 0, baseline pole magnitude nu = exp(-delta_nu * a).
        # - Base frequency omega spread over the full angular range.
        self.omega = nn.Parameter(torch.rand(d_model, d_state) * theta_max)
        if mag_init == "mamba":
            # a = 1..state_dim (Mamba's S4D-real A).
            A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
            self.A_log = nn.Parameter(torch.log(A))
        else:  # "ring": baseline nu drawn from the LRU ring; a = -log(nu_target) with
            # baseline delta_nu = 1 (set below), so baseline nu = exp(-a) = nu_target.
            mags = torch.sqrt(
                torch.rand(d_model, d_state) * (r_max**2 - r_min**2) + r_min**2
            )
            self.A_log = nn.Parameter(torch.log(-torch.log(mags)))

        self.D = nn.Parameter(torch.randn(d_model))

        with torch.no_grad():
            # Magnitude init via the baseline delta_nu = softplus(dt_nu_up.bias).
            if mag_init == "mamba":
                # softplus(bias) uniform in [dt_min, dt_max] (Mamba's dt init), so
                # baseline nu = exp(-dt * a) lands near the unit disk.
                dt = torch.exp(
                    torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min))
                    + math.log(dt_min)
                )
            else:  # "ring": baseline delta_nu = 1 so baseline nu = exp(-a) = nu_target.
                dt = torch.ones(d_model)
            self.dt_nu_up.bias.copy_(_inv_softplus(dt))

            # Phase init: delta_theta = 0, so theta = omega at init (the full
            # angular range, matching D-LinOSS init).
            self.dt_theta_up.bias.zero_()

            # Selective weights start small (perturbation off a static oscillator bank).
            for proj in (self.dt_nu_down, self.dt_nu_up, self.dt_theta_down, self.dt_theta_up):
                proj.weight.normal_(0.0, selective_init_std)

            # Rank-r read correction: c_coef starts tiny so the scalar gate s^C ~ 0, i.e.
            # the correction vanishes at init and the model begins as the base readout.
            if rank > 0:
                self.c_read.weight.normal_(0.0, 1.0 / math.sqrt(d_model))
                self.c_coef.weight.normal_(0.0, selective_init_std / math.sqrt(d_model))
                self.c_pool.normal_(0.0, 1.0 / math.sqrt(d_model))
                self.c_out.copy_(_uniform_init((rank, d_model), std=1.0 / math.sqrt(d_model)))

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

        if self.use_triton and self.rank == 0:
            # Fused scan + readout: the state never loads into HBM. (The rank-r read
            # correction needs the full state, which the kernel does not return, so
            # rank > 0 falls through to the materialized path below.)
            from models.selective_lru.selective_lru_triton import fused_selective_lru
            return fused_selective_lru(
                delta_nu, delta_theta, a, self.omega, B_sel, C_complex, x, self.D,
                block_t=self.chunk_size, block_c=self.block_c,
                use_gamma=self.input_norm == "gamma",
            )
        else:  # Materialized path: per-channel complex selective scan.
            nu = torch.exp(-delta_nu.unsqueeze(-1) * a)                  # (B, T, C, N) in (0,1)
            theta = self.omega + delta_theta.unsqueeze(-1)               # (B, T, C, N)
            lam = torch.polar(nu, theta)                                 # (B, T, C, N) complex

            if self.input_norm == "gamma":
                gain = torch.sqrt((1.0 - nu**2).clamp_min(1e-6))         # (B, T, C, N) LRU norm
            else:
                gain = delta_nu.unsqueeze(-1)                            # (B, T, C, 1) ZOH gain
            Bu = (gain * B_sel.unsqueeze(-2)) * x.unsqueeze(-1)          # (B, T, C, N)
            Bu = Bu.to(lam.dtype)

            h = _selective_recurrence(lam.reshape(B_, T, C * N), Bu.reshape(B_, T, C * N))
            h = h.reshape(B_, T, C, N)

            Cy = torch.einsum("btn,btcn->btc", C_complex, h).real        # (B, T, C)
            if self.rank > 0:
                Cy = Cy + self._rank_read(x, h)
            return Cy + x * self.D

    def _rank_read(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Rank-r cross-channel readout correction (reference path only).

        The base readout is channel-diagonal — channel ``c`` reads only its own bank
        ``h[:, :, c, :]`` with the shared ``C``. This adds a rank-r cross-channel
        coupling: pool the per-channel states into ``rank`` aggregate banks with static
        weights ``c_pool``, read each with an input-dependent complex vector ``c_t`` and
        scalar gate ``s^C``, then scatter to the output channels via static directions
        ``c_out``. The channel-mixing matrix ``sum_i c_out[i] c_pool[i]^T`` has rank
        ``<= rank``. Because ``c_coef`` starts tiny, the correction is ~0 at init and is
        causal (``h_t`` and ``c_t``/``s^C`` depend only on inputs up to ``t``).

        Args:
            x: Input ``(B, T, C)``.
            h: Complex per-channel state ``(B, T, C, N)`` from the scan.

        Returns:
            Additive readout correction ``(B, T, C)``.
        """
        Bsz, T, _ = x.shape
        r, N = self.rank, self.d_state
        cr = self.c_read(x).view(Bsz, T, r, 2 * N)
        c_t = torch.complex(cr[..., :N], cr[..., N:])                 # (B, T, r, N) read vecs
        hbar = torch.einsum("rc,btcn->btrn", self.c_pool.to(h.dtype), h)  # (B, T, r, N) pooled
        read = torch.einsum("btrn,btrn->btr", c_t, hbar).real        # (B, T, r)
        s_c = self.c_coef(x)                                          # (B, T, r), ~0 at init
        return torch.einsum("btr,rc->btc", s_c * read, self.c_out)   # (B, T, C)


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


def _selective_recurrence(lam: torch.Tensor, Bu: torch.Tensor) -> torch.Tensor:
    B, T, N = Bu.shape
    h = Bu.new_zeros(B, N)
    out = Bu.new_empty(B, T, N)
    for t in range(T):
        h = lam[:, t] * h + Bu[:, t]
        out[:, t] = h
    return out