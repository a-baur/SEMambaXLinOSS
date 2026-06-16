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
independently. Rather than materializing a stacked (2B, T, N) real tensor, we
pass the complex inputs as their `view_as_real` (B, T, N, 2) views and let each
program carry *both* halves through the scan in registers — one program per
sequence instead of two. This avoids the cat/complex host-side copies, halves
the launch grid, and halves the reduction traffic into dM.

The per-mode dM reduction is the shared-parameter gradient
`dM_ij = Σ_b Σ_t g_t[i]·s_{t-1}[j]`. Rather than `tl.atomic_add` into a shared
`(N,)` buffer, each program writes its own `(B, N)` partial via a pure `tl.store`
(every slot written exactly once) and the partials are reduced with an external
`.sum(0)`. The deterministic tree-sum is slightly *more* accurate than the
nondeterministic atomic on large folded batches, and is CUDA-graph-safe (no
buffer is accumulated-into across replays).
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
    stride_b, stride_t, stride_n, stride_c,
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

    y1_re = tl.zeros((BLOCK_N,), dtype=tl.float32)
    y1_im = tl.zeros((BLOCK_N,), dtype=tl.float32)
    y2_re = tl.zeros((BLOCK_N,), dtype=tl.float32)
    y2_im = tl.zeros((BLOCK_N,), dtype=tl.float32)

    base = pid_b * stride_b + offs_n * stride_n

    for t in range(T):
        offs_t = base + t * stride_t
        f1_re = tl.load(F1_ptr + offs_t, mask=mask_n, other=0.0)
        f1_im = tl.load(F1_ptr + offs_t + stride_c, mask=mask_n, other=0.0)
        f2_re = tl.load(F2_ptr + offs_t, mask=mask_n, other=0.0)
        f2_im = tl.load(F2_ptr + offs_t + stride_c, mask=mask_n, other=0.0)

        new_y1_re = M_11 * y1_re + M_12 * y2_re + f1_re
        new_y1_im = M_11 * y1_im + M_12 * y2_im + f1_im
        new_y2_re = M_21 * y1_re + M_22 * y2_re + f2_re
        new_y2_im = M_21 * y1_im + M_22 * y2_im + f2_im

        y1_re = new_y1_re
        y1_im = new_y1_im
        y2_re = new_y2_re
        y2_im = new_y2_im

        tl.store(Y1_ptr + offs_t, y1_re, mask=mask_n)
        tl.store(Y1_ptr + offs_t + stride_c, y1_im, mask=mask_n)
        tl.store(Y2_ptr + offs_t, y2_re, mask=mask_n)
        tl.store(Y2_ptr + offs_t + stride_c, y2_im, mask=mask_n)


@triton.jit
def _linoss_scan_bwd_kernel(
    M_11_ptr, M_12_ptr, M_21_ptr, M_22_ptr,
    Y1_ptr, Y2_ptr, dY2_ptr,
    dF1_ptr, dF2_ptr,
    dM_11_ptr, dM_12_ptr, dM_21_ptr, dM_22_ptr,  # (B, N) per-batch-row partials
    T, N,
    stride_b, stride_t, stride_n, stride_c,
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

    a1_re = tl.zeros((BLOCK_N,), dtype=tl.float32)
    a1_im = tl.zeros((BLOCK_N,), dtype=tl.float32)
    a2_re = tl.zeros((BLOCK_N,), dtype=tl.float32)
    a2_im = tl.zeros((BLOCK_N,), dtype=tl.float32)

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

        dy2_re = tl.load(dY2_ptr + offs_t, mask=mask_n, other=0.0)
        dy2_im = tl.load(dY2_ptr + offs_t + stride_c, mask=mask_n, other=0.0)

        new_a1_re = M_11 * a1_re + M_21 * a2_re
        new_a1_im = M_11 * a1_im + M_21 * a2_im
        new_a2_re = dy2_re + M_12 * a1_re + M_22 * a2_re
        new_a2_im = dy2_im + M_12 * a1_im + M_22 * a2_im

        tl.store(dF1_ptr + offs_t, new_a1_re, mask=mask_n)
        tl.store(dF1_ptr + offs_t + stride_c, new_a1_im, mask=mask_n)
        tl.store(dF2_ptr + offs_t, new_a2_re, mask=mask_n)
        tl.store(dF2_ptr + offs_t + stride_c, new_a2_im, mask=mask_n)

        y1_prev_re = tl.load(Y1_ptr + offs_tm1, mask=mask_n, other=0.0)
        y1_prev_im = tl.load(Y1_ptr + offs_tm1 + stride_c, mask=mask_n, other=0.0)
        y2_prev_re = tl.load(Y2_ptr + offs_tm1, mask=mask_n, other=0.0)
        y2_prev_im = tl.load(Y2_ptr + offs_tm1 + stride_c, mask=mask_n, other=0.0)

        # Fold real + imag contributions into the per-program accumulator.
        dM_11_acc += new_a1_re * y1_prev_re + new_a1_im * y1_prev_im
        dM_12_acc += new_a1_re * y2_prev_re + new_a1_im * y2_prev_im
        dM_21_acc += new_a2_re * y1_prev_re + new_a2_im * y1_prev_im
        dM_22_acc += new_a2_re * y2_prev_re + new_a2_im * y2_prev_im

        a1_re = new_a1_re
        a1_im = new_a1_im
        a2_re = new_a2_re
        a2_im = new_a2_im

    # t = 0: s_{-1} = 0 contributes nothing to dM; just propagate dF.
    dy2_re = tl.load(dY2_ptr + base, mask=mask_n, other=0.0)
    dy2_im = tl.load(dY2_ptr + base + stride_c, mask=mask_n, other=0.0)
    new_a1_re = M_11 * a1_re + M_21 * a2_re
    new_a1_im = M_11 * a1_im + M_21 * a2_im
    new_a2_re = dy2_re + M_12 * a1_re + M_22 * a2_re
    new_a2_im = dy2_im + M_12 * a1_im + M_22 * a2_im
    tl.store(dF1_ptr + base, new_a1_re, mask=mask_n)
    tl.store(dF1_ptr + base + stride_c, new_a1_im, mask=mask_n)
    tl.store(dF2_ptr + base, new_a2_re, mask=mask_n)
    tl.store(dF2_ptr + base + stride_c, new_a2_im, mask=mask_n)

    # Pure store of per-batch-row partials: slot (pid_b, offs_n), written once.
    # No atomic -> CUDA-graph-safe; reduced by an external .sum(0).
    part_off = pid_b * N + offs_n
    tl.store(dM_11_ptr + part_off, dM_11_acc, mask=mask_n)
    tl.store(dM_12_ptr + part_off, dM_12_acc, mask=mask_n)
    tl.store(dM_21_ptr + part_off, dM_21_acc, mask=mask_n)
    tl.store(dM_22_ptr + part_off, dM_22_acc, mask=mask_n)


def _block_n_for(state_dim: int) -> int:
    # Power-of-2 block size that contains the full state dim when feasible;
    # otherwise tile along N. Capped at 64 so the kernel stays register-light.
    pow2 = 1 << (max(state_dim, 1) - 1).bit_length()
    return max(16, min(pow2, 64))


def _num_warps_for(block_n: int) -> int:
    # One lane per state element; with block_n <= 64 a single warp covers 32
    # lanes, so 1-2 warps suffice. Keeping programs small lets more of them stay
    # resident to hide the sequential scan's memory latency.
    return max(1, block_n // 32)


def _real_strides(T: int, N: int) -> tuple[int, int, int, int]:
    # Element strides of a contiguous (B, T, N, 2) real view of (B, T, N) complex.
    return (T * N * 2, N * 2, 2, 1)


class LinOSSScanFunction(torch.autograd.Function):
    """autograd.Function wrapping the partial-reduction Triton scan.

    The backward computes the per-mode transition gradient
    `dM_ij = Σ_b Σ_t g_t[i]·s_{t-1}[j]` (a reduction over the folded batch) by
    having each program write its own `(B, N)` partial via a pure `tl.store`
    (every slot written once) and reducing with an external `.sum(0)`, rather than
    `tl.atomic_add` into a shared `(N,)` buffer. The deterministic tree-sum is
    slightly more accurate than the nondeterministic atomic on large folded
    batches, and — unlike an atomic into a replayed buffer — is CUDA-graph-safe.

    The scan is exposed as an opaque autograd.Function (a `torch.compile` graph
    break) rather than an in-graph `torch.library.triton_op`. The break is
    load-bearing: tracing the scan in-graph also pulls LinOSS's surrounding
    complex einsum readout into Inductor, which mis-codegens complex ops. (Note:
    Inductor mis-compiles this model's complex *forward* even with the scan kept
    as a graph break — `torch.compile` is currently unreliable for the LinOSS
    generator regardless of this kernel; see the partial-reduction backward,
    validated standalone against a float64 reference.)
    """

    @staticmethod
    def forward(ctx, M_11, M_12, M_21, M_22, F1, F2):
        if not F1.is_cuda:
            raise RuntimeError("linoss_scan_triton requires CUDA tensors.")
        if F1.shape != F2.shape:
            raise ValueError(f"F1 {F1.shape} != F2 {F2.shape}")

        M_11 = M_11.contiguous()
        M_12 = M_12.contiguous()
        M_21 = M_21.contiguous()
        M_22 = M_22.contiguous()
        F1 = F1.contiguous()
        F2 = F2.contiguous()

        B, T, N = F1.shape
        Y1 = torch.empty_like(F1)
        Y2 = torch.empty_like(F2)
        BLOCK_N = _block_n_for(N)
        grid = (B, triton.cdiv(N, BLOCK_N))
        sb, st, sn, sc = _real_strides(T, N)
        _linoss_scan_fwd_kernel[grid](
            M_11, M_12, M_21, M_22,
            torch.view_as_real(F1), torch.view_as_real(F2),
            torch.view_as_real(Y1), torch.view_as_real(Y2),
            T, N, sb, st, sn, sc,
            BLOCK_N=BLOCK_N, num_warps=_num_warps_for(BLOCK_N),
        )

        ctx.save_for_backward(M_11, M_12, M_21, M_22, Y1, Y2)
        return Y2

    @staticmethod
    def backward(ctx, dY2):
        M_11, M_12, M_21, M_22, Y1, Y2 = ctx.saved_tensors
        dY2 = dY2.resolve_conj().contiguous()
        B, T, N = dY2.shape
        dF1 = torch.empty_like(dY2)
        dF2 = torch.empty_like(dY2)
        opts = dict(device=dY2.device, dtype=M_11.dtype)
        dM_11_p = torch.empty((B, N), **opts)
        dM_12_p = torch.empty((B, N), **opts)
        dM_21_p = torch.empty((B, N), **opts)
        dM_22_p = torch.empty((B, N), **opts)
        BLOCK_N = _block_n_for(N)
        grid = (B, triton.cdiv(N, BLOCK_N))
        sb, st, sn, sc = _real_strides(T, N)
        _linoss_scan_bwd_kernel[grid](
            M_11, M_12, M_21, M_22,
            torch.view_as_real(Y1), torch.view_as_real(Y2), torch.view_as_real(dY2),
            torch.view_as_real(dF1), torch.view_as_real(dF2),
            dM_11_p, dM_12_p, dM_21_p, dM_22_p,
            T, N, sb, st, sn, sc,
            BLOCK_N=BLOCK_N, num_warps=_num_warps_for(BLOCK_N),
        )
        return (
            dM_11_p.sum(0), dM_12_p.sum(0), dM_21_p.sum(0), dM_22_p.sum(0),
            dF1, dF2,
        )


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
