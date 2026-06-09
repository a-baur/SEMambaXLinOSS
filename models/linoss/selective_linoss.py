"""Selective LinOSS (S-LinOSS): an input-dependent oscillatory SSM via scaled rotations.

Implements the model described in ``selective_linoss.md``. The base ``LinOSS`` is
linear time-invariant: its transition is a fixed matrix derived from ``(A, G, dt)``
through IM/IMEX integration. S-LinOSS instead parameterizes the per-mode transition
directly in spectral coordinates as a *scaled rotation* and makes its two degrees of
freedom — magnitude ``nu`` (damping, ``|lambda|``) and phase ``theta`` (frequency,
``arg lambda``) — input-dependent.

In the equivalent complex/diagonal form (spec section 5) the recurrence collapses to a
selective Linear Recurrent Unit mode::

    h_k = lambda_k * h_{k-1} + (B u)_k,    lambda_k = nu_k * exp(i theta_k),

with selective spectral coordinates produced from the input token ``u_k``::

    nu_k    = sigmoid(W_nu    u_k + c_nu)     in (0, 1)   (selective damping)
    theta_k =          W_theta u_k + c_theta              (selective frequency)

Because the scaled rotation is *normal* (``|lambda_k| = nu_k < 1`` exactly), the
unforced state norm contracts monotonically and ``|prod_k lambda_k| <= 1`` for *any*
input-driven schedule. That gives a structural, schedule-free stability guarantee
(spec section 6) that selective ``(A_k, G_k, dt_k)`` fed through the non-normal IMEX
block cannot provide.

Compared with ``LinOSS`` this trades the genuine discretized-forced-oscillator identity
of the transition for orthogonal eigenvectors, keeping the full unit-disk eigenvalue
spectrum that D-LinOSS targets while adding Mamba/S6-style selectivity. See the spec's
section 12 for the honest caveat: it is a principled normal surrogate, not a drop-in
equivalent to a (hypothetical) selective D-LinOSS.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from models.linoss.linoss import _project_input, _uniform_init


class MambOSS(nn.Module):
    """Selective oscillatory state-space mixer (scaled-rotation form).

    Shapes:
        Input:  (batch, time, in_features)
        Output: (batch, time, in_features)

    Args:
        in_features: Model dim fed to the mixer (the Mamba-style block's inner dim).
        state_dim: Number of independent complex oscillator modes. Default 16.
        r_min: Lower bound of the baseline eigenvalue magnitude band (``nu`` at zero
            selective contribution lands in ``[r_min, r_max]``). Default 0.9.
        r_max: Upper bound of the baseline magnitude band. Kept strictly below 1 so the
            magnitude bias ``logit(nu)`` is finite. Default 0.999.
        theta_max: Upper bound for the baseline phase spread ``theta in [0, theta_max)``.
            Default pi (a conjugate pair then sweeps the full angular range).
        selective_init_std: Std of the selective projection weights ``W_nu``/``W_theta``
            at init. Small, so selectivity is learned as a perturbation off a working
            oscillator bank rather than from scratch. Default 1e-2.
        normalize_input: If True, scale the input drive by ``sqrt(1 - nu^2)`` (LRU-style)
            so each mode's output variance is independent of its damping. Without it,
            near-unit modes accumulate input with gain ``1/(1 - nu^2)`` and dominate the
            readout by 1-2 orders of magnitude at init. Default True.
        use_triton: If True, run the complex recurrence with the Triton scan kernel
            (``selective_triton.py``) instead of the sequential Python loop. Requires
            CUDA inputs. Default False.
    """

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

        # Selective spectral coordinates: nu_k = sigmoid(W_nu u + c_nu) and
        # theta_k = W_theta u + c_theta. The biases are the per-mode baseline; the
        # weights inject input dependence.
        self.nu_proj = nn.Linear(in_features, state_dim)
        self.theta_proj = nn.Linear(in_features, state_dim)

        with torch.no_grad():
            # Magnitude bias: at zero input contribution nu = sigmoid(c_nu) is radially
            # uniform in [r_min, r_max], reproducing D-LinOSS's near-unit-disk init band.
            mags = torch.sqrt(torch.rand(state_dim) * (r_max ** 2 - r_min ** 2) + r_min ** 2)
            self.nu_proj.bias.copy_(torch.logit(mags))
            # Phase bias: spread baseline theta across the full angular range.
            self.theta_proj.bias.copy_(torch.rand(state_dim) * theta_max)
            # Selective weights start small (perturbation off the working bank).
            self.nu_proj.weight.normal_(0.0, selective_init_std)
            self.theta_proj.weight.normal_(0.0, selective_init_std)

        # Fixed complex input/output projections and a real skip, matching the base
        # LinOSS port. The readout uses the full complex state (spec's position-only
        # readout y = C*Im(h) is the special case Re(C) = 0).
        B_std = 1.0 / math.sqrt(in_features)
        self.B = nn.Parameter(_uniform_init((state_dim, in_features, 2), std=B_std))
        C_std = 1.0 / math.sqrt(state_dim)
        self.C = nn.Parameter(_uniform_init((in_features, state_dim, 2), std=C_std))
        self.D = nn.Parameter(torch.randn(in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input-dependent magnitude and phase -> per-step complex eigenvalue lambda_k.
        nu = torch.sigmoid(self.nu_proj(x))          # (B, T, N) in (0, 1)
        theta = self.theta_proj(x)                   # (B, T, N), periodic via cos/sin
        lam = torch.polar(nu, theta)                 # (B, T, N) complex, |lam| = nu < 1

        B_complex = torch.complex(self.B[..., 0], self.B[..., 1])
        C_complex = torch.complex(self.C[..., 0], self.C[..., 1])

        Bu = _project_input(B_complex, x)            # (B, T, N) complex
        if self.normalize_input:
            # LRU-style normalization: with |lambda_k| = nu_k, an unforced mode
            # accumulates input with white-noise variance gain 1/(1 - nu_k^2). Scaling
            # the drive by sqrt(1 - nu_k^2) makes every mode contribute equal variance
            # regardless of its damping, instead of letting near-unit modes dominate.
            gamma = torch.sqrt((1.0 - nu ** 2).clamp_min(1e-6))  # (B, T, N) real
            Bu = Bu * gamma

        if self.use_triton:
            from models.linoss.selective_triton import selective_scan_triton
            h = selective_scan_triton(lam, Bu)       # (B, T, N) complex
        else:
            h = _selective_recurrence(lam, Bu)       # (B, T, N) complex

        Cy = torch.einsum("fn,btn->btf", C_complex, h).real
        Du = x * self.D
        return Cy + Du


def _selective_recurrence(lam: torch.Tensor, Bu: torch.Tensor) -> torch.Tensor:
    # Scalar complex recurrence h_k = lam_k * h_{k-1} + Bu_k, with h_0 = 0.
    # lam and Bu are (batch, time, state). Returns the complex state trajectory.
    #
    # This is a first-order linear recurrence with input-dependent coefficients, the
    # same structure used by S5/LRU/Mamba, so it is associative and admits an O(log T)
    # parallel/Triton scan (spec section 8). PyTorch has no built-in associative scan,
    # so we run it sequentially — matching the base LinOSS port's recurrence.
    B, T, N = Bu.shape
    h = Bu.new_zeros(B, N)
    out = Bu.new_empty(B, T, N)
    for t in range(T):
        h = lam[:, t] * h + Bu[:, t]
        out[:, t] = h
    return out
