"""Triton-accelerated LinOSS scan with custom autograd.

Replaces the sequential Python `for t in range(T)` loop in
`models.linoss._linoss_recurrence` with a fused Triton kernel. Forward and
backward each launch one kernel; the backward derives the adjoint from
    s_t = M s_{t-1} + F_t,  y2_t = s_t[1]
giving
    g_t = (0, dY2_t) + M^T g_{t+1},    g_T = 0
    dF1_t = g_t[0],  dF2_t = g_t[1]
    dM_ij += g_t[i] * s_{t-1}[j]       (s_{-1} = 0)

Triton has no native complex64. The transition M is real, so the real and
imaginary halves of (y1, y2, F1, F2) evolve under identical dynamics
independently — we stack them along the batch axis (2B sequences) and
run a single real-valued kernel.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _linoss_scan_fwd_kernel(
    M_11_ptr, M_12_ptr, M_21_ptr, M_22_ptr,
    F1_ptr, F2_ptr,
    Y1_ptr, Y2_ptr,
    T, N,
    stride_b, stride_t, stride_n,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    M_11 = tl.load(M_11_ptr + offs_n, mask=mask_n, other=0.0)
    M_12 = tl.load(M_12_ptr + offs_n, mask=mask_n, other=0.0)
    M_21 = tl.load(M_21_ptr + offs_n, mask=mask_n, other=0.0)
    M_22 = tl.load(M_22_ptr + offs_n, mask=mask_n, other=0.0)

    y1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    y2 = tl.zeros((BLOCK_N,), dtype=tl.float32)

    base = pid_b * stride_b + offs_n * stride_n

    for t in range(T):
        offs_t = base + t * stride_t
        f1 = tl.load(F1_ptr + offs_t, mask=mask_n, other=0.0)
        f2 = tl.load(F2_ptr + offs_t, mask=mask_n, other=0.0)
        new_y1 = M_11 * y1 + M_12 * y2 + f1
        new_y2 = M_21 * y1 + M_22 * y2 + f2
        y1 = new_y1
        y2 = new_y2
        tl.store(Y1_ptr + offs_t, y1, mask=mask_n)
        tl.store(Y2_ptr + offs_t, y2, mask=mask_n)


@triton.jit
def _linoss_scan_bwd_kernel(
    M_11_ptr, M_12_ptr, M_21_ptr, M_22_ptr,
    Y1_ptr, Y2_ptr, dY2_ptr,
    dF1_ptr, dF2_ptr,
    dM_11_ptr, dM_12_ptr, dM_21_ptr, dM_22_ptr,
    T, N,
    stride_b, stride_t, stride_n,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    M_11 = tl.load(M_11_ptr + offs_n, mask=mask_n, other=0.0)
    M_12 = tl.load(M_12_ptr + offs_n, mask=mask_n, other=0.0)
    M_21 = tl.load(M_21_ptr + offs_n, mask=mask_n, other=0.0)
    M_22 = tl.load(M_22_ptr + offs_n, mask=mask_n, other=0.0)

    a1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    a2 = tl.zeros((BLOCK_N,), dtype=tl.float32)

    dM_11_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    dM_12_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    dM_21_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    dM_22_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    base = pid_b * stride_b + offs_n * stride_n

    # Reverse scan for t = T-1 down to 1: full update + dM accumulation
    # (we need s_{t-1} = Y[t-1] for dM; safe only when t >= 1).
    for k in range(T - 1):
        t = T - 1 - k
        offs_t = base + t * stride_t
        offs_tm1 = base + (t - 1) * stride_t

        dy2 = tl.load(dY2_ptr + offs_t, mask=mask_n, other=0.0)
        new_a1 = M_11 * a1 + M_21 * a2
        new_a2 = dy2 + M_12 * a1 + M_22 * a2

        tl.store(dF1_ptr + offs_t, new_a1, mask=mask_n)
        tl.store(dF2_ptr + offs_t, new_a2, mask=mask_n)

        y1_prev = tl.load(Y1_ptr + offs_tm1, mask=mask_n, other=0.0)
        y2_prev = tl.load(Y2_ptr + offs_tm1, mask=mask_n, other=0.0)
        dM_11_acc += new_a1 * y1_prev
        dM_12_acc += new_a1 * y2_prev
        dM_21_acc += new_a2 * y1_prev
        dM_22_acc += new_a2 * y2_prev

        a1 = new_a1
        a2 = new_a2

    # t = 0: s_{-1} = 0 contributes nothing to dM; just propagate dF.
    dy2 = tl.load(dY2_ptr + base, mask=mask_n, other=0.0)
    new_a1 = M_11 * a1 + M_21 * a2
    new_a2 = dy2 + M_12 * a1 + M_22 * a2
    tl.store(dF1_ptr + base, new_a1, mask=mask_n)
    tl.store(dF2_ptr + base, new_a2, mask=mask_n)

    tl.atomic_add(dM_11_ptr + offs_n, dM_11_acc, mask=mask_n)
    tl.atomic_add(dM_12_ptr + offs_n, dM_12_acc, mask=mask_n)
    tl.atomic_add(dM_21_ptr + offs_n, dM_21_acc, mask=mask_n)
    tl.atomic_add(dM_22_ptr + offs_n, dM_22_acc, mask=mask_n)


def _block_n_for(state_dim: int) -> int:
    # Power-of-2 block size that contains the full state dim when feasible;
    # otherwise tile along N. Capped at 64 so the kernel stays register-light.
    pow2 = 1 << (max(state_dim, 1) - 1).bit_length()
    return max(16, min(pow2, 64))


def _complex_to_stacked(z: torch.Tensor) -> torch.Tensor:
    # (B, T, N) complex -> (2B, T, N) real, [real-half ; imag-half] along batch.
    return torch.cat([z.real.contiguous(), z.imag.contiguous()], dim=0)


def _stacked_to_complex(z: torch.Tensor, B: int) -> torch.Tensor:
    return torch.complex(z[:B], z[B:])


class LinOSSScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M_11, M_12, M_21, M_22, F1, F2):
        if not F1.is_cuda:
            raise RuntimeError("LinOSSScanFunction requires CUDA tensors.")
        if F1.shape != F2.shape:
            raise ValueError(f"F1 {F1.shape} != F2 {F2.shape}")

        # Contiguize once: the kernel reads M_ij with implicit unit stride
        # (`M_ij_ptr + offs_n`), so the same tensor must back both the forward
        # launch and the saved-for-backward copy.
        M_11 = M_11.contiguous()
        M_12 = M_12.contiguous()
        M_21 = M_21.contiguous()
        M_22 = M_22.contiguous()

        B, T, N = F1.shape

        F1_s = _complex_to_stacked(F1)
        F2_s = _complex_to_stacked(F2)
        Y1_s = torch.empty_like(F1_s)
        Y2_s = torch.empty_like(F2_s)

        BLOCK_N = _block_n_for(N)
        grid = (2 * B, triton.cdiv(N, BLOCK_N))

        _linoss_scan_fwd_kernel[grid](
            M_11, M_12, M_21, M_22,
            F1_s, F2_s, Y1_s, Y2_s,
            T, N,
            F1_s.stride(0), F1_s.stride(1), F1_s.stride(2),
            BLOCK_N=BLOCK_N,
        )

        ctx.save_for_backward(M_11, M_12, M_21, M_22, Y1_s, Y2_s)
        ctx.B = B
        ctx.T = T
        ctx.N = N
        ctx.BLOCK_N = BLOCK_N

        return _stacked_to_complex(Y2_s, B)

    @staticmethod
    def backward(ctx, dY2):
        M_11, M_12, M_21, M_22, Y1_s, Y2_s = ctx.saved_tensors
        B, T, N, BLOCK_N = ctx.B, ctx.T, ctx.N, ctx.BLOCK_N

        dY2_s = _complex_to_stacked(dY2.resolve_conj())
        dF1_s = torch.empty_like(dY2_s)
        dF2_s = torch.empty_like(dY2_s)

        dM_11 = torch.zeros_like(M_11)
        dM_12 = torch.zeros_like(M_12)
        dM_21 = torch.zeros_like(M_21)
        dM_22 = torch.zeros_like(M_22)

        grid = (2 * B, triton.cdiv(N, BLOCK_N))

        _linoss_scan_bwd_kernel[grid](
            M_11, M_12, M_21, M_22,
            Y1_s, Y2_s, dY2_s,
            dF1_s, dF2_s,
            dM_11, dM_12, dM_21, dM_22,
            T, N,
            dY2_s.stride(0), dY2_s.stride(1), dY2_s.stride(2),
            BLOCK_N=BLOCK_N,
        )

        dF1 = _stacked_to_complex(dF1_s, B)
        dF2 = _stacked_to_complex(dF2_s, B)
        return dM_11, dM_12, dM_21, dM_22, dF1, dF2


def linoss_scan_triton(
    M_11: torch.Tensor,
    M_12: torch.Tensor,
    M_21: torch.Tensor,
    M_22: torch.Tensor,
    F1: torch.Tensor,
    F2: torch.Tensor,
) -> torch.Tensor:
    """Triton-accelerated LinOSS scan.

    Args:
        M_11, M_12, M_21, M_22: (N,) real float32. Transition matrix entries
            (block-diagonal with N independent 2x2 blocks).
        F1, F2: (B, T, N) complex64. Per-step inputs.

    Returns:
        Y2: (B, T, N) complex64. The y2 component of the state trajectory.
    """
    return LinOSSScanFunction.apply(M_11, M_12, M_21, M_22, F1, F2)
