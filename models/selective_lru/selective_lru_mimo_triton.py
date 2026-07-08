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


class SelectiveScanRealFunction(torch.autograd.Function):
    """Real-in/real-out selective scan.

    Takes and returns the real/imaginary halves as separate real tensors so the
    caller never pays for ``.real.contiguous()`` splits or ``torch.complex``
    recombines around the kernel (which measured ~7x the kernel cost itself).
    The kernels are unchanged — they already operate on split re/im pointers.
    """

    @staticmethod
    def forward(ctx, lam_re, lam_im, Bu_re, Bu_im):
        if not lam_re.is_cuda:
            raise RuntimeError("SelectiveScanRealFunction requires CUDA tensors.")
        if lam_re.shape != Bu_re.shape:
            raise ValueError(f"lam {lam_re.shape} != Bu {Bu_re.shape}")

        B, T, N = Bu_re.shape

        # No-ops when the caller already passed contiguous tensors (the fast path).
        lam_re = lam_re.contiguous()
        lam_im = lam_im.contiguous()
        Bu_re = Bu_re.contiguous()
        Bu_im = Bu_im.contiguous()
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
        ctx.BLOCK_N = BLOCK_N

        return h_re, h_im

    @staticmethod
    def backward(ctx, dh_re, dh_im):
        lam_re, lam_im, h_re, h_im = ctx.saved_tensors
        B, T, N = lam_re.shape
        BLOCK_N = ctx.BLOCK_N

        g_re = dh_re.contiguous()
        g_im = dh_im.contiguous()

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

        return dlam_re, dlam_im, dBu_re, dBu_im


def selective_scan_triton_real(
    lam_re: torch.Tensor, lam_im: torch.Tensor,
    Bu_re: torch.Tensor, Bu_im: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton selective scan on split real/imag tensors (no complex repacking).

    Computes ``h_k = lam_k * h_{k-1} + Bu_k`` with ``h_{-1} = 0`` where
    ``lam = lam_re + i*lam_im`` and ``Bu = Bu_re + i*Bu_im``.

    Returns:
        (h_re, h_im): the real and imaginary halves of the complex state, each
        ``(B, T, N)`` real.
    """
    return SelectiveScanRealFunction.apply(lam_re, lam_im, Bu_re, Bu_im)


def selective_scan_triton(lam: torch.Tensor, Bu: torch.Tensor) -> torch.Tensor:
    """Complex-tensor convenience wrapper around :func:`selective_scan_triton_real`.

    Kept for tests / callers that hold complex tensors. The module fast path uses
    the split-real entry point directly to avoid the repacking.

    Args:
        lam: (B, T, N) complex64. Per-step transition eigenvalues ``lambda_k``.
        Bu: (B, T, N) complex64. Per-step inputs ``(B u)_k``.

    Returns:
        h: (B, T, N) complex64. The complex state trajectory.
    """
    h_re, h_im = selective_scan_triton_real(
        lam.real.contiguous(), lam.imag.contiguous(),
        Bu.real.contiguous(), Bu.imag.contiguous(),
    )
    return torch.complex(h_re, h_im)


# ---------------------------------------------------------------------------
# Fused MIMO scan + block-diagonal readout + D skip.
#
# One program owns one (batch row, head): it streams the head's Nh complex modes
# in registers over time, and at each step emits the head's Fh output channels
# directly — Re(C_h h_t) + D x_t — so the (B,T,N) state never round-trips through
# a separate readout GEMM (the measured bottleneck). The state is still written
# out (h_re/h_im), both as backward scratch and so the rank-r selective read can
# run in PyTorch on the complex state. Everything is the *real decomposition* of
# the complex recurrence, matching the validated selective-scan backward.
# ---------------------------------------------------------------------------


@triton.jit
def _fused_mimo_fwd_kernel(
    lam_re_ptr, lam_im_ptr, Bu_re_ptr, Bu_im_ptr,   # (B, T, N)
    Cre_ptr, Cim_ptr,                               # (H, Fh, Nh)
    x_ptr, D_ptr,                                   # (B, T, Fd), (Fd,)
    y_ptr, h_re_ptr, h_im_ptr,                      # (B, T, Fd), (B, T, N)
    T, H, Nh, Fh, N, Fd,
    BLOCK_N: tl.constexpr, BLOCK_F: tl.constexpr,
):
    pb = tl.program_id(0)
    ph = tl.program_id(1)

    on = tl.arange(0, BLOCK_N)
    of = tl.arange(0, BLOCK_F)
    mn = on < Nh
    mf = of < Fh
    m2 = mf[:, None] & mn[None, :]

    cbase = ph * Fh * Nh + of[:, None] * Nh + on[None, :]        # (BLOCK_F, BLOCK_N)
    Cre = tl.load(Cre_ptr + cbase, mask=m2, other=0.0)
    Cim = tl.load(Cim_ptr + cbase, mask=m2, other=0.0)
    Dv = tl.load(D_ptr + ph * Fh + of, mask=mf, other=0.0)       # (BLOCK_F,)

    hre = tl.zeros((BLOCK_N,), dtype=tl.float32)
    him = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for t in range(T):
        n_off = pb * T * N + t * N + ph * Nh + on
        f_off = pb * T * Fd + t * Fd + ph * Fh + of

        lre = tl.load(lam_re_ptr + n_off, mask=mn, other=0.0)
        lim = tl.load(lam_im_ptr + n_off, mask=mn, other=0.0)
        bre = tl.load(Bu_re_ptr + n_off, mask=mn, other=0.0)
        bim = tl.load(Bu_im_ptr + n_off, mask=mn, other=0.0)

        # h_t = lambda_t * h_{t-1} + Bu_t  (complex).
        new_hre = lre * hre - lim * him + bre
        new_him = lre * him + lim * hre + bim
        hre = new_hre
        him = new_him

        tl.store(h_re_ptr + n_off, hre, mask=mn)
        tl.store(h_im_ptr + n_off, him, mask=mn)

        # y[f] = sum_n (Cre[f,n] hre[n] - Cim[f,n] him[n]) + D[f] x[f].
        contrib = tl.where(m2, Cre * hre[None, :] - Cim * him[None, :], 0.0)
        xv = tl.load(x_ptr + f_off, mask=mf, other=0.0)
        yv = tl.sum(contrib, axis=1) + Dv * xv                  # (BLOCK_F,)
        tl.store(y_ptr + f_off, yv, mask=mf)


@triton.jit
def _fused_mimo_bwd_kernel(
    lam_re_ptr, lam_im_ptr,                         # (B, T, N)
    h_re_ptr, h_im_ptr,                             # (B, T, N)  saved states
    Cre_ptr, Cim_ptr,                               # (H, Fh, Nh)
    x_ptr, D_ptr,                                   # (B, T, Fd), (Fd,)
    dy_ptr,                                         # (B, T, Fd)  upstream grad of y
    dh_re_ptr, dh_im_ptr,                           # (B, T, N)   upstream grad of h (rank read)
    dlam_re_ptr, dlam_im_ptr, dBu_re_ptr, dBu_im_ptr,  # (B, T, N)
    dx_ptr,                                         # (B, T, Fd)
    dCre_ptr, dCim_ptr,                             # (H, Fh, Nh)  atomic (reduce over batch)
    dD_ptr,                                         # (Fd,)        atomic (reduce over batch)
    T, H, Nh, Fh, N, Fd,
    BLOCK_N: tl.constexpr, BLOCK_F: tl.constexpr, USE_DH: tl.constexpr,
):
    pb = tl.program_id(0)
    ph = tl.program_id(1)

    on = tl.arange(0, BLOCK_N)
    of = tl.arange(0, BLOCK_F)
    mn = on < Nh
    mf = of < Fh
    m2 = mf[:, None] & mn[None, :]

    cbase = ph * Fh * Nh + of[:, None] * Nh + on[None, :]
    Cre = tl.load(Cre_ptr + cbase, mask=m2, other=0.0)
    Cim = tl.load(Cim_ptr + cbase, mask=m2, other=0.0)
    Dv = tl.load(D_ptr + ph * Fh + of, mask=mf, other=0.0)

    dCre_acc = tl.zeros((BLOCK_F, BLOCK_N), dtype=tl.float32)
    dCim_acc = tl.zeros((BLOCK_F, BLOCK_N), dtype=tl.float32)
    dD_acc = tl.zeros((BLOCK_F,), dtype=tl.float32)

    # Adjoint carry G_{t+1} and lambda_{t+1} carry, both zero at the sequence end.
    Gre = tl.zeros((BLOCK_N,), dtype=tl.float32)
    Gim = tl.zeros((BLOCK_N,), dtype=tl.float32)
    lnr = tl.zeros((BLOCK_N,), dtype=tl.float32)
    lni = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k in range(T):
        t = T - 1 - k
        n_off = pb * T * N + t * N + ph * Nh + on
        f_off = pb * T * Fd + t * Fd + ph * Fh + of

        h_re_t = tl.load(h_re_ptr + n_off, mask=mn, other=0.0)
        h_im_t = tl.load(h_im_ptr + n_off, mask=mn, other=0.0)

        prev_mask = mn & (t > 0)
        n_off_prev = pb * T * N + (t - 1) * N + ph * Nh + on
        hm_re = tl.load(h_re_ptr + n_off_prev, mask=prev_mask, other=0.0)
        hm_im = tl.load(h_im_ptr + n_off_prev, mask=prev_mask, other=0.0)

        dyv = tl.load(dy_ptr + f_off, mask=mf, other=0.0)       # (BLOCK_F,)
        xv = tl.load(x_ptr + f_off, mask=mf, other=0.0)

        # Local adjoint injected at t from the static readout (real decomposition):
        #   dL/dhre[n] =  sum_f dy[f] Cre[f,n];  dL/dhim[n] = -sum_f dy[f] Cim[f,n].
        g_re = tl.sum(tl.where(m2, Cre * dyv[:, None], 0.0), axis=0)   # (BLOCK_N,)
        g_im = -tl.sum(tl.where(m2, Cim * dyv[:, None], 0.0), axis=0)
        if USE_DH:
            g_re = g_re + tl.load(dh_re_ptr + n_off, mask=mn, other=0.0)
            g_im = g_im + tl.load(dh_im_ptr + n_off, mask=mn, other=0.0)

        # G_t = g_t + conj(lambda_{t+1}) G_{t+1}.
        Gtr = g_re + lnr * Gre + lni * Gim
        Gti = g_im + lnr * Gim - lni * Gre

        # dBu_t = G_t.
        tl.store(dBu_re_ptr + n_off, Gtr, mask=mn)
        tl.store(dBu_im_ptr + n_off, Gti, mask=mn)
        # dlam_t = G_t * conj(h_{t-1}).
        dlre = Gtr * hm_re + Gti * hm_im
        dlim = Gti * hm_re - Gtr * hm_im
        tl.store(dlam_re_ptr + n_off, dlre, mask=mn)
        tl.store(dlam_im_ptr + n_off, dlim, mask=mn)

        # Readout parameter grads (need h_t): dC reduces over batch (atomic at end).
        dCre_acc += tl.where(m2, dyv[:, None] * h_re_t[None, :], 0.0)
        dCim_acc += tl.where(m2, -dyv[:, None] * h_im_t[None, :], 0.0)
        dD_acc += dyv * xv
        tl.store(dx_ptr + f_off, dyv * Dv, mask=mf)

        Gre = Gtr
        Gim = Gti
        lnr = tl.load(lam_re_ptr + n_off, mask=mn, other=0.0)
        lni = tl.load(lam_im_ptr + n_off, mask=mn, other=0.0)

    tl.atomic_add(dCre_ptr + cbase, dCre_acc, mask=m2)
    tl.atomic_add(dCim_ptr + cbase, dCim_acc, mask=m2)
    tl.atomic_add(dD_ptr + ph * Fh + of, dD_acc, mask=mf)


class _FusedSelectiveLRUMIMO(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lam_re, lam_im, Bu_re, Bu_im, Cre, Cim, x, D, H, Nh, Fh):
        """Fused scan + readout; returns ``(y, h_re, h_im)``."""
        if not lam_re.is_cuda:
            raise RuntimeError("_FusedSelectiveLRUMIMO requires CUDA tensors.")
        B, T, N = lam_re.shape
        Fd = x.shape[2]

        lam_re = lam_re.contiguous()
        lam_im = lam_im.contiguous()
        Bu_re = Bu_re.contiguous()
        Bu_im = Bu_im.contiguous()
        Cre = Cre.contiguous()
        Cim = Cim.contiguous()
        x = x.contiguous()
        D = D.contiguous()

        y = torch.empty((B, T, Fd), device=lam_re.device, dtype=torch.float32)
        h_re = torch.empty_like(lam_re)
        h_im = torch.empty_like(lam_re)

        BLOCK_N = triton.next_power_of_2(Nh)
        BLOCK_F = triton.next_power_of_2(Fh)
        grid = (B, H)
        _fused_mimo_fwd_kernel[grid](
            lam_re, lam_im, Bu_re, Bu_im, Cre, Cim, x, D, y, h_re, h_im,
            T, H, Nh, Fh, N, Fd, BLOCK_N=BLOCK_N, BLOCK_F=BLOCK_F,
        )

        ctx.save_for_backward(lam_re, lam_im, h_re, h_im, Cre, Cim, x, D)
        ctx.dims = (B, T, H, Nh, Fh, N, Fd)
        ctx.blocks = (BLOCK_N, BLOCK_F)
        return y, h_re, h_im

    @staticmethod
    def backward(ctx, dy, dh_re, dh_im):
        """Grads for the scan inputs + readout params (``dh`` from the rank read)."""
        lam_re, lam_im, h_re, h_im, Cre, Cim, x, D = ctx.saved_tensors
        B, T, H, Nh, Fh, N, Fd = ctx.dims
        BLOCK_N, BLOCK_F = ctx.blocks

        use_dh = dh_re is not None
        if use_dh:
            dh_re = dh_re.contiguous()
            dh_im = dh_im.contiguous()
        else:  # h outputs unused (rank == 0): pass dummy pointers, kernel skips them.
            dh_re = lam_re.new_empty(1)
            dh_im = lam_re.new_empty(1)

        dlam_re = torch.empty_like(lam_re)
        dlam_im = torch.empty_like(lam_re)
        dBu_re = torch.empty_like(lam_re)
        dBu_im = torch.empty_like(lam_re)
        dx = torch.empty((B, T, Fd), device=lam_re.device, dtype=torch.float32)
        dCre = torch.zeros_like(Cre)
        dCim = torch.zeros_like(Cim)
        dD = torch.zeros_like(D)

        grid = (B, H)
        _fused_mimo_bwd_kernel[grid](
            lam_re, lam_im, h_re, h_im, Cre, Cim, x, D, dy.contiguous(),
            dh_re, dh_im,
            dlam_re, dlam_im, dBu_re, dBu_im, dx, dCre, dCim, dD,
            T, H, Nh, Fh, N, Fd, BLOCK_N=BLOCK_N, BLOCK_F=BLOCK_F, USE_DH=use_dh,
        )

        # grads for (lam_re, lam_im, Bu_re, Bu_im, Cre, Cim, x, D, H, Nh, Fh)
        return dlam_re, dlam_im, dBu_re, dBu_im, dCre, dCim, dx, dD, None, None, None


def fused_selective_lru_mimo(
    lam_re, lam_im, Bu_re, Bu_im, C_re, C_im, x, D, n_heads, n_head, f_head,
):
    """Fused selective MIMO scan + block-diagonal readout + D skip.

    ``lam_re``/``lam_im`` and ``Bu_re``/``Bu_im`` are the ``(B, T, N)`` real
    halves of the per-mode transition and input (``N = n_heads * n_head``).
    ``C_re``/``C_im`` are the ``(n_heads, f_head, n_head)`` halves of the
    block-diagonal readout. ``x`` is the ``(B, T, Fd)`` module input (fed only
    through the D skip here) and ``D`` the ``(Fd,)`` skip weights
    (``Fd = n_heads * f_head``). ``n_heads``/``n_head``/``f_head`` are the head
    counts/widths.

    Returns:
        (y, h_re, h_im): ``y`` is ``(B, T, Fd) = Re(C h) + D x``; ``h_re``/
        ``h_im`` are the ``(B, T, N)`` state halves (used by the rank-r
        selective read, if any).
    """
    return _FusedSelectiveLRUMIMO.apply(
        lam_re, lam_im, Bu_re, Bu_im,
        C_re.contiguous(), C_im.contiguous(), x, D, n_heads, n_head, f_head,
    )
