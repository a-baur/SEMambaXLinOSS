from __future__ import annotations

import torch
import triton
import triton.language as tl

from models.linoss.triton import _block_n_for


@triton.jit
def _selective_scan_fwd_kernel(
    lam_re_ptr, lam_im_ptr,
    Bu_re_ptr, Bu_im_ptr,
    h_re_ptr, h_im_ptr,
    T, N,
    stride_b, stride_t, stride_n,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    hp = tl.zeros((BLOCK_N,), dtype=tl.float32)
    hq = tl.zeros((BLOCK_N,), dtype=tl.float32)

    base = pid_b * stride_b + offs_n * stride_n

    for t in range(T):
        off = base + t * stride_t
        a = tl.load(lam_re_ptr + off, mask=mask_n, other=0.0)
        b = tl.load(lam_im_ptr + off, mask=mask_n, other=0.0)
        u = tl.load(Bu_re_ptr + off, mask=mask_n, other=0.0)
        v = tl.load(Bu_im_ptr + off, mask=mask_n, other=0.0)

        # h_t = lambda_t * h_{t-1} + Bu_t  (complex multiply, written out).
        new_hp = a * hp - b * hq + u
        new_hq = b * hp + a * hq + v
        hp = new_hp
        hq = new_hq

        tl.store(h_re_ptr + off, hp, mask=mask_n)
        tl.store(h_im_ptr + off, hq, mask=mask_n)


@triton.jit
def _selective_scan_bwd_kernel(
    lam_re_ptr, lam_im_ptr,
    h_re_ptr, h_im_ptr,
    g_re_ptr, g_im_ptr,
    dlam_re_ptr, dlam_im_ptr,
    dBu_re_ptr, dBu_im_ptr,
    T, N,
    stride_b, stride_t, stride_n,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    # Running adjoint G_{t+1} (starts at G_T = 0).
    Gp = tl.zeros((BLOCK_N,), dtype=tl.float32)
    Gq = tl.zeros((BLOCK_N,), dtype=tl.float32)

    base = pid_b * stride_b + offs_n * stride_n

    for k in range(T):
        t = T - 1 - k
        off = base + t * stride_t

        gp = tl.load(g_re_ptr + off, mask=mask_n, other=0.0)
        gq = tl.load(g_im_ptr + off, mask=mask_n, other=0.0)

        # lambda_{t+1} (the coefficient that multiplied h_t). Masked off at t = T-1,
        # where the future adjoint G_{t+1} is zero anyway.
        next_mask = mask_n & (t < T - 1)
        off_next = base + (t + 1) * stride_t
        a_next = tl.load(lam_re_ptr + off_next, mask=next_mask, other=0.0)
        b_next = tl.load(lam_im_ptr + off_next, mask=next_mask, other=0.0)

        # G_t = g_t + conj(lambda_{t+1}) * G_{t+1}  (conj => -b_next on the imag mix).
        new_Gp = gp + a_next * Gp + b_next * Gq
        new_Gq = gq - b_next * Gp + a_next * Gq
        Gp = new_Gp
        Gq = new_Gq

        # dBu_t = G_t.
        tl.store(dBu_re_ptr + off, Gp, mask=mask_n)
        tl.store(dBu_im_ptr + off, Gq, mask=mask_n)

        # dlam_t = G_t * conj(h_{t-1}); h_{-1} = 0 so t = 0 contributes nothing.
        prev_mask = mask_n & (t > 0)
        off_prev = base + (t - 1) * stride_t
        hp_prev = tl.load(h_re_ptr + off_prev, mask=prev_mask, other=0.0)
        hq_prev = tl.load(h_im_ptr + off_prev, mask=prev_mask, other=0.0)
        dA = Gp * hp_prev + Gq * hq_prev
        dB = -Gp * hq_prev + Gq * hp_prev
        tl.store(dlam_re_ptr + off, dA, mask=mask_n)
        tl.store(dlam_im_ptr + off, dB, mask=mask_n)


class SelectiveScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lam, Bu):
        if not lam.is_cuda:
            raise RuntimeError("SelectiveScanFunction requires CUDA tensors.")
        if lam.shape != Bu.shape:
            raise ValueError(f"lam {lam.shape} != Bu {Bu.shape}")

        B, T, N = Bu.shape

        lam_re = lam.real.contiguous()
        lam_im = lam.imag.contiguous()
        Bu_re = Bu.real.contiguous()
        Bu_im = Bu.imag.contiguous()
        h_re = torch.empty_like(lam_re)
        h_im = torch.empty_like(lam_re)

        BLOCK_N = _block_n_for(N)
        grid = (B, triton.cdiv(N, BLOCK_N))

        _selective_scan_fwd_kernel[grid](
            lam_re, lam_im,
            Bu_re, Bu_im,
            h_re, h_im,
            T, N,
            lam_re.stride(0), lam_re.stride(1), lam_re.stride(2),
            BLOCK_N=BLOCK_N,
        )

        ctx.save_for_backward(lam_re, lam_im, h_re, h_im)
        ctx.B = B
        ctx.T = T
        ctx.N = N
        ctx.BLOCK_N = BLOCK_N

        return torch.complex(h_re, h_im)

    @staticmethod
    def backward(ctx, dh):
        lam_re, lam_im, h_re, h_im = ctx.saved_tensors
        B, T, N, BLOCK_N = ctx.B, ctx.T, ctx.N, ctx.BLOCK_N

        # resolve_conj() so .real/.imag read the actual stored values even if the
        # upstream grad arrived as a lazy conjugate view.
        dh = dh.resolve_conj()
        g_re = dh.real.contiguous()
        g_im = dh.imag.contiguous()

        dlam_re = torch.empty_like(lam_re)
        dlam_im = torch.empty_like(lam_re)
        dBu_re = torch.empty_like(lam_re)
        dBu_im = torch.empty_like(lam_re)

        grid = (B, triton.cdiv(N, BLOCK_N))

        _selective_scan_bwd_kernel[grid](
            lam_re, lam_im,
            h_re, h_im,
            g_re, g_im,
            dlam_re, dlam_im,
            dBu_re, dBu_im,
            T, N,
            lam_re.stride(0), lam_re.stride(1), lam_re.stride(2),
            BLOCK_N=BLOCK_N,
        )

        return torch.complex(dlam_re, dlam_im), torch.complex(dBu_re, dBu_im)


def selective_scan_triton(lam: torch.Tensor, Bu: torch.Tensor) -> torch.Tensor:
    """Triton-accelerated S-LinOSS selective scan.

    Computes ``h_k = lam_k * h_{k-1} + Bu_k`` with ``h_{-1} = 0``.

    Args:
        lam: (B, T, N) complex64. Per-step transition eigenvalues ``lambda_k``.
        Bu: (B, T, N) complex64. Per-step inputs ``(B u)_k``.

    Returns:
        h: (B, T, N) complex64. The complex state trajectory.
    """
    return SelectiveScanFunction.apply(lam, Bu)
