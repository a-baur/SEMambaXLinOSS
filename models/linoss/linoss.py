"""
LinOSS based on https://github.com/camail-official/discretax/blob/main/src/discretax/sequence_mixers/linoss.py

PyTorch port of the LinOSS (Linear Oscillatory State-Space) sequence mixer
from "Oscillatory State-Space Models" (https://openreview.net/pdf?id=GRMfXcAAFh).

Supports LinOSS-IM, LinOSS-IMEX, and Damped LinOSS-IMEX variants. The JAX
original uses a parallel associative scan; this port uses a sequential
recurrence (PyTorch has no built-in associative scan).
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn


class LinOSS(nn.Module):
    """LinOSS sequence mixer.

    Shapes:
        Input:  (batch, time, in_features)
        Output: (batch, time, in_features)
    """

    def __init__(
        self,
        in_features: int,
        state_dim: int = 64,
        discretization: Literal["IM", "IMEX"] = "IMEX",
        damping: bool = True,
        r_min: float = 0.9,
        theta_max: float = math.pi,
        use_triton: bool = False,
    ):
        super().__init__()
        if discretization == "IM" and damping:
            raise NotImplementedError(
                "Discretization=IM with damping=True is not implemented."
            )
        if discretization not in ("IM", "IMEX"):
            raise NotImplementedError(f"Discretization {discretization} not implemented")

        self.in_features = in_features
        self.state_dim = state_dim
        self.discretization = discretization
        self.damping = damping
        self.use_triton = use_triton

        self.steps = nn.Parameter(torch.randn(state_dim) * 0.5)

        with torch.no_grad():
            steps_init = torch.sigmoid(self.steps)

        if discretization == "IMEX" and damping:
            r_max = 1.0
            mags = torch.sqrt(
                torch.rand(state_dim) * (r_max ** 2 - r_min ** 2) + r_min ** 2
            )
            G_init = (1.0 - mags ** 2) / (steps_init * mags ** 2)
            self.G_diag = nn.Parameter(G_init)

            theta = torch.rand(state_dim) * theta_max
            A_init = _map_theta_to_A(theta, torch.relu(G_init), steps_init)
            self.A_diag = nn.Parameter(A_init)
        else:
            self.register_parameter("G_diag", None)
            self.A_diag = nn.Parameter(torch.rand(state_dim))

        B_std = 1.0 / math.sqrt(in_features)
        self.B = nn.Parameter(_uniform_init((state_dim, in_features, 2), std=B_std))
        C_std = 1.0 / math.sqrt(state_dim)
        self.C = nn.Parameter(_uniform_init((in_features, state_dim, 2), std=C_std))
        self.D = nn.Parameter(torch.randn(in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        steps = torch.sigmoid(self.steps)

        B_complex = torch.complex(self.B[..., 0], self.B[..., 1])
        C_complex = torch.complex(self.C[..., 0], self.C[..., 1])

        if self.discretization == "IM":
            A_diag = torch.relu(self.A_diag)
            ys = _apply_linoss_im(A_diag, B_complex, x, steps, use_triton=self.use_triton)
        else:  # IMEX
            if self.damping:
                G_diag = torch.relu(self.G_diag)
                sqrt_term = torch.sqrt(1.0 + steps * G_diag)
                A_low = (2.0 + steps * G_diag - 2.0 * sqrt_term) / steps ** 2
                A_high = (2.0 + steps * G_diag + 2.0 * sqrt_term) / steps ** 2
                A_diag = (
                    A_low
                    + torch.relu(self.A_diag - A_low)
                    - torch.relu(self.A_diag - A_high)
                )
                ys = _apply_damped_linoss_imex(
                    A_diag, G_diag, B_complex, x, steps, use_triton=self.use_triton
                )
            else:
                A_diag = torch.relu(self.A_diag)
                ys = _apply_linoss_imex(
                    A_diag, B_complex, x, steps, use_triton=self.use_triton
                )

        # Cy + Du
        Cy = torch.einsum("fn,btn->btf", C_complex, ys).real
        Du = x * self.D
        return Cy + Du


def _uniform_init(shape, std: float = 1.0) -> torch.Tensor:
    return torch.empty(*shape).uniform_(-std, std)


def _map_theta_to_A(
    thetas: torch.Tensor, G_diag: torch.Tensor, steps: torch.Tensor
) -> torch.Tensor:
    cos_inv2 = 1.0 / torch.cos(thetas) ** 2
    tan2 = torch.tan(thetas) ** 2
    sqrt_term = torch.sqrt(steps ** 4 * cos_inv2 + steps ** 5 * G_diag * cos_inv2)
    common = -(steps ** 2) * (
        -4.0 - 2.0 * steps * G_diag - 4.0 * tan2 - 2.0 * steps * G_diag * tan2
    )
    denom = 2.0 * steps ** 4 * (1.0 + tan2)
    A_plus = (4.0 * sqrt_term + common) / denom
    A_minus = (-4.0 * sqrt_term + common) / denom
    return torch.where(thetas > math.pi / 2, A_plus, A_minus)


def _project_input(B_complex: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # Real-times-complex projection: (N, F) x (B, T, F) -> (B, T, N) complex.
    Bu_real = torch.einsum("nf,btf->btn", B_complex.real, x)
    Bu_imag = torch.einsum("nf,btf->btn", B_complex.imag, x)
    return torch.complex(Bu_real, Bu_imag)


def _linoss_recurrence(
    M_11: torch.Tensor,
    M_12: torch.Tensor,
    M_21: torch.Tensor,
    M_22: torch.Tensor,
    F1: torch.Tensor,
    F2: torch.Tensor,
    use_triton: bool = False,
) -> torch.Tensor:
    # State evolves as
    #   [y1; y2]_t = [[M_11, M_12], [M_21, M_22]] [y1; y2]_{t-1} + [F1; F2]_t
    # with y_0 = 0. Returns the y2 trajectory.
    if use_triton:
        from models.linoss.triton import linoss_scan_triton
        return linoss_scan_triton(M_11, M_12, M_21, M_22, F1, F2)

    B, T, N = F1.shape
    y1 = F1.new_zeros(B, N)
    y2 = F2.new_zeros(B, N)
    out = F2.new_empty(B, T, N)
    for t in range(T):
        new_y1 = M_11 * y1 + M_12 * y2 + F1[:, t]
        new_y2 = M_21 * y1 + M_22 * y2 + F2[:, t]
        y1, y2 = new_y1, new_y2
        out[:, t] = y2
    return out


def _apply_linoss_im(
    A_diag: torch.Tensor,
    B_complex: torch.Tensor,
    x: torch.Tensor,
    step: torch.Tensor,
    use_triton: bool = False,
) -> torch.Tensor:
    Bu = _project_input(B_complex, x)

    schur = 1.0 / (1.0 + step ** 2 * A_diag)
    M_11 = 1.0 - step ** 2 * A_diag * schur
    M_12 = -step * A_diag * schur
    M_21 = step * schur
    M_22 = schur

    F1 = M_11 * Bu * step
    F2 = M_21 * Bu * step
    return _linoss_recurrence(M_11, M_12, M_21, M_22, F1, F2, use_triton=use_triton)


def _apply_linoss_imex(
    A_diag: torch.Tensor,
    B_complex: torch.Tensor,
    x: torch.Tensor,
    step: torch.Tensor,
    use_triton: bool = False,
) -> torch.Tensor:
    Bu = _project_input(B_complex, x)

    M_11 = torch.ones_like(A_diag)
    M_12 = -step * A_diag
    M_21 = step
    M_22 = 1.0 - step ** 2 * A_diag

    F1 = Bu * step
    F2 = Bu * step ** 2
    return _linoss_recurrence(M_11, M_12, M_21, M_22, F1, F2, use_triton=use_triton)


def _apply_damped_linoss_imex(
    A_diag: torch.Tensor,
    G_diag: torch.Tensor,
    B_complex: torch.Tensor,
    x: torch.Tensor,
    step: torch.Tensor,
    use_triton: bool = False,
) -> torch.Tensor:
    Bu = _project_input(B_complex, x)

    S = 1.0 + step * G_diag
    inv_S = 1.0 / S
    M_11 = inv_S
    M_12 = -step * inv_S * A_diag
    M_21 = step * inv_S
    M_22 = 1.0 - step ** 2 * inv_S * A_diag

    F1 = (step * inv_S) * Bu
    F2 = (step ** 2 * inv_S) * Bu
    return _linoss_recurrence(M_11, M_12, M_21, M_22, F1, F2, use_triton=use_triton)
