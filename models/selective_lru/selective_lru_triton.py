from __future__ import annotations

import torch
import triton
import triton.language as tl


def _block_n(n: int) -> int:
    return max(16, triton.next_power_of_2(n))


_FWD_CONFIGS = [
    triton.Config({"BLOCK_C": bc}, num_warps=w)
    for bc in (16, 32, 64)
    for w in (2, 4, 8)
]


@triton.autotune(configs=_FWD_CONFIGS, key=["T", "C", "N"])
@triton.jit
def _selective_lru_fwd_kernel(
    dnu_ptr, dth_ptr, x_ptr,          # (B, T, C)
    a_ptr, om_ptr, D_ptr,             # (C, N), (C, N), (C,)
    Bsr_ptr, Bsi_ptr,                 # (B, T, N)
    Cre_ptr, Cim_ptr,                 # (B, T, N)
    y_ptr,                            # (B, T, C)
    ckpt_re_ptr, ckpt_im_ptr,         # (B, NC, C, N)
    B, T, C, N, NC,
    BLOCK_C: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_T: tl.constexpr,
    SAVE_CKPT: tl.constexpr,
):
    pb = tl.program_id(0)
    pcb = tl.program_id(1)

    oc = pcb * BLOCK_C + tl.arange(0, BLOCK_C)       # (BLOCK_C,)
    on = tl.arange(0, BLOCK_N)                        # (BLOCK_N,)
    mc = oc < C
    mn = on < N
    m2 = mc[:, None] & mn[None, :]

    cn = oc[:, None] * N + on[None, :]               # index into (C, N)
    a = tl.load(a_ptr + cn, mask=m2, other=0.0)
    om = tl.load(om_ptr + cn, mask=m2, other=0.0)
    Dc = tl.load(D_ptr + oc, mask=mc, other=0.0)

    c_base = pb * T * C + oc
    n_base = pb * T * N + on

    hre = tl.zeros((BLOCK_C, BLOCK_N), dtype=tl.float32)
    him = tl.zeros((BLOCK_C, BLOCK_N), dtype=tl.float32)

    for t in range(T):
        if SAVE_CKPT:
            if t % BLOCK_T == 0:
                jc = t // BLOCK_T
                ck = pb * NC * C * N + jc * C * N + cn
                tl.store(ckpt_re_ptr + ck, hre, mask=m2)
                tl.store(ckpt_im_ptr + ck, him, mask=m2)

        c_off = c_base + t * C
        n_off = n_base + t * N
        dnu = tl.load(dnu_ptr + c_off, mask=mc, other=0.0)
        dth = tl.load(dth_ptr + c_off, mask=mc, other=0.0)
        xv = tl.load(x_ptr + c_off, mask=mc, other=0.0)
        Bsr = tl.load(Bsr_ptr + n_off, mask=mn, other=0.0)
        Bsi = tl.load(Bsi_ptr + n_off, mask=mn, other=0.0)
        Cr = tl.load(Cre_ptr + n_off, mask=mn, other=0.0)
        Ci = tl.load(Cim_ptr + n_off, mask=mn, other=0.0)

        # lam = exp(dnu * (-a + j(om + dth)));  Bu = sqrt(1 - |lam|^2) * (Bsr + jBsi) * x
        nu = tl.exp(-dnu[:, None] * a)
        th = dnu[:, None] * (om + dth[:, None])
        lre = nu * tl.cos(th)
        lim = nu * tl.sin(th)
        gx = tl.sqrt(tl.maximum(1.0 - nu * nu, 1e-6)) * xv[:, None]

        new_hre = lre * hre - lim * him + gx * Bsr[None, :]
        new_him = lre * him + lim * hre + gx * Bsi[None, :]
        hre = new_hre
        him = new_him

        contrib = tl.where(m2, Cr[None, :] * hre - Ci[None, :] * him, 0.0)
        yv = tl.sum(contrib, axis=1) + Dc * xv          # (BLOCK_C,)
        tl.store(y_ptr + c_off, yv, mask=mc)


@triton.jit
def _selective_lru_bwd_kernel(
    dnu_ptr, dth_ptr, x_ptr,          # (B, T, C)
    a_ptr, om_ptr, D_ptr,             # (C, N), (C, N), (C,)
    Bsr_ptr, Bsi_ptr,                 # (B, T, N)
    Cre_ptr, Cim_ptr,                 # (B, T, N)
    dy_ptr,                           # (B, T, C)  upstream grad
    ckpt_re_ptr, ckpt_im_ptr,         # (B, NC, C, N)  chunk-boundary states (from forward)
    ht_re_ptr, ht_im_ptr,             # (B, C, BLOCK_T, N)  within-chunk state scratch
    d_dnu_ptr, d_dth_ptr, dx_ptr,     # (B, T, C)  per-element grads (direct write)
    da_ptr, dom_ptr, dD_ptr,          # (C, N), (C, N), (C,)  batch-reduced grads (atomic)
    dBsr_ptr, dBsi_ptr, dCre_ptr, dCim_ptr,   # (B, T, N)  channel-reduced grads (atomic)
    B, T, C, N, NC,
    BLOCK_C: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pb = tl.program_id(0)
    pcb = tl.program_id(1)

    oc = pcb * BLOCK_C + tl.arange(0, BLOCK_C)
    on = tl.arange(0, BLOCK_N)
    mc = oc < C
    mn = on < N
    m2 = mc[:, None] & mn[None, :]

    cn = oc[:, None] * N + on[None, :]
    a = tl.load(a_ptr + cn, mask=m2, other=0.0)
    om = tl.load(om_ptr + cn, mask=m2, other=0.0)
    Dc = tl.load(D_ptr + oc, mask=mc, other=0.0)

    # Within-chunk h scratch is indexed by the global channel id.
    ht_cbase = pb * C * BLOCK_T * N + oc[:, None] * BLOCK_T * N + on[None, :]   # + li*N

    # Register accumulators for the batch-reduced grads (one atomic flush at the end).
    da_acc = tl.zeros((BLOCK_C, BLOCK_N), dtype=tl.float32)
    dom_acc = tl.zeros((BLOCK_C, BLOCK_N), dtype=tl.float32)
    dD_acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    # Adjoint carry and lambda_{t+1} carry (both 0 at the sequence end).
    Gre = tl.zeros((BLOCK_C, BLOCK_N), dtype=tl.float32)
    Gim = tl.zeros((BLOCK_C, BLOCK_N), dtype=tl.float32)
    lnr = tl.zeros((BLOCK_C, BLOCK_N), dtype=tl.float32)
    lni = tl.zeros((BLOCK_C, BLOCK_N), dtype=tl.float32)

    for jr in range(NC):
        jc = NC - 1 - jr
        t0 = jc * BLOCK_T
        ck = pb * NC * C * N + jc * C * N + cn
        bnd_re = tl.load(ckpt_re_ptr + ck, mask=m2, other=0.0)
        bnd_im = tl.load(ckpt_im_ptr + ck, mask=m2, other=0.0)

        # Refill h_t for t in [t0, t0+BLOCK_T) from the chunk-entering state.
        bre = bnd_re
        bim = bnd_im
        for li in range(BLOCK_T):
            t = t0 + li
            if t < T:
                c_off = pb * T * C + t * C + oc
                n_off = pb * T * N + t * N + on
                dnu = tl.load(dnu_ptr + c_off, mask=mc, other=0.0)
                dth = tl.load(dth_ptr + c_off, mask=mc, other=0.0)
                xv = tl.load(x_ptr + c_off, mask=mc, other=0.0)
                Bsr = tl.load(Bsr_ptr + n_off, mask=mn, other=0.0)
                Bsi = tl.load(Bsi_ptr + n_off, mask=mn, other=0.0)
                nu = tl.exp(-dnu[:, None] * a)
                th = dnu[:, None] * (om + dth[:, None])
                lre = nu * tl.cos(th)
                lim = nu * tl.sin(th)
                gx = tl.sqrt(tl.maximum(1.0 - nu * nu, 1e-6)) * xv[:, None]
                n_re = lre * bre - lim * bim + gx * Bsr[None, :]
                n_im = lre * bim + lim * bre + gx * Bsi[None, :]
                bre = n_re
                bim = n_im
                tl.store(ht_re_ptr + ht_cbase + li * N, bre, mask=m2)
                tl.store(ht_im_ptr + ht_cbase + li * N, bim, mask=m2)

        # Reverse over the chunk, high t to low.
        for lr in range(BLOCK_T):
            li = BLOCK_T - 1 - lr
            t = t0 + li
            if t < T:
                c_off = pb * T * C + t * C + oc
                n_off = pb * T * N + t * N + on
                dnu = tl.load(dnu_ptr + c_off, mask=mc, other=0.0)
                dth = tl.load(dth_ptr + c_off, mask=mc, other=0.0)
                xv = tl.load(x_ptr + c_off, mask=mc, other=0.0)
                Bsr = tl.load(Bsr_ptr + n_off, mask=mn, other=0.0)
                Bsi = tl.load(Bsi_ptr + n_off, mask=mn, other=0.0)
                Cr = tl.load(Cre_ptr + n_off, mask=mn, other=0.0)
                Ci = tl.load(Cim_ptr + n_off, mask=mn, other=0.0)
                dyv = tl.load(dy_ptr + c_off, mask=mc, other=0.0)

                ht_re = tl.load(ht_re_ptr + ht_cbase + li * N, mask=m2, other=0.0)
                ht_im = tl.load(ht_im_ptr + ht_cbase + li * N, mask=m2, other=0.0)
                if li > 0:
                    hm_re = tl.load(ht_re_ptr + ht_cbase + (li - 1) * N, mask=m2, other=0.0)
                    hm_im = tl.load(ht_im_ptr + ht_cbase + (li - 1) * N, mask=m2, other=0.0)
                else:
                    hm_re = bnd_re
                    hm_im = bnd_im

                nu = tl.exp(-dnu[:, None] * a)
                base = om + dth[:, None]
                th = dnu[:, None] * base
                costh = tl.cos(th)
                sinth = tl.sin(th)
                gain = tl.sqrt(tl.maximum(1.0 - nu * nu, 1e-6))

                # g_t = dy * conj(C); G_t = g_t + conj(lambda_{t+1}) G_{t+1}.
                gre = dyv[:, None] * Cr[None, :]
                gim = -dyv[:, None] * Ci[None, :]
                Gtr = gre + lnr * Gre + lni * Gim
                Gti = gim + lnr * Gim - lni * Gre

                # Input map Bu = gain * (Bsr + jBsi) * x;  dBu = G_t (complex).
                s = Gtr * Bsr[None, :] + Gti * Bsi[None, :]
                dBsr_c = tl.sum(tl.where(m2, Gtr * gain * xv[:, None], 0.0), axis=0)
                dBsi_c = tl.sum(tl.where(m2, Gti * gain * xv[:, None], 0.0), axis=0)
                tl.atomic_add(dBsr_ptr + n_off, dBsr_c, mask=mn)
                tl.atomic_add(dBsi_ptr + n_off, dBsi_c, mask=mn)
                dx_bu = tl.sum(tl.where(m2, s * gain, 0.0), axis=1)
                # gain = sqrt(1 - nu^2): the input map feeds gradient into nu too.
                dnu_bu = (s * xv[:, None]) * (-nu / gain)
                d_dnu_bu = tl.sum(tl.where(m2, dnu_bu * (-a) * nu, 0.0), axis=1)
                da_bu = dnu_bu * (-dnu[:, None]) * nu

                # dlam = G_t conj(h_{t-1}); nu and theta = dnu * (om + dth) chains.
                dlre = Gtr * hm_re + Gti * hm_im
                dlim = Gti * hm_re - Gtr * hm_im
                dnu_n = dlre * costh + dlim * sinth
                dth_n = nu * (dlim * costh - dlre * sinth)
                d_dnu_lam = tl.sum(
                    tl.where(m2, dnu_n * (-a) * nu + dth_n * base, 0.0), axis=1
                )
                d_dth = tl.sum(tl.where(m2, dth_n * dnu[:, None], 0.0), axis=1)
                dom_acc += dth_n * dnu[:, None]
                da_acc += dnu_n * (-dnu[:, None]) * nu + da_bu

                # Readout grads at t (need h_t): dC reduces over channels (atomic).
                dCr_c = tl.sum(tl.where(m2, dyv[:, None] * ht_re, 0.0), axis=0)
                dCi_c = tl.sum(tl.where(m2, -dyv[:, None] * ht_im, 0.0), axis=0)
                tl.atomic_add(dCre_ptr + n_off, dCr_c, mask=mn)
                tl.atomic_add(dCim_ptr + n_off, dCi_c, mask=mn)
                dD_acc += dyv * xv

                tl.store(d_dnu_ptr + c_off, d_dnu_bu + d_dnu_lam, mask=mc)
                tl.store(d_dth_ptr + c_off, d_dth, mask=mc)
                tl.store(dx_ptr + c_off, dx_bu + dyv * Dc, mask=mc)

                Gre = Gtr
                Gim = Gti
                lnr = nu * costh
                lni = nu * sinth

    tl.atomic_add(da_ptr + cn, da_acc, mask=m2)
    tl.atomic_add(dom_ptr + cn, dom_acc, mask=m2)
    tl.atomic_add(dD_ptr + oc, dD_acc, mask=mc)


class _SelectiveLRUScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, delta_nu, delta_theta, a, omega, Bsr, Bsi, Cre, Cim, x, D,
                block_t, block_c):
        if not delta_nu.is_cuda:
            raise RuntimeError("fused selective-LRU scan requires CUDA tensors.")
        B, T, C = delta_nu.shape
        N = a.shape[1]

        dnu = delta_nu.contiguous()
        dth = delta_theta.contiguous()
        a = a.contiguous()
        omega = omega.contiguous()
        Bsr = Bsr.contiguous()
        Bsi = Bsi.contiguous()
        Cre = Cre.contiguous()
        Cim = Cim.contiguous()
        x = x.contiguous()
        D = D.contiguous()

        NC = triton.cdiv(T, block_t)

        y = torch.empty((B, T, C), device=dnu.device, dtype=torch.float32)
        # Grad mode is disabled inside Function.forward, so query ctx.needs_input_grad
        # (set by autograd before the call) to decide whether to keep backward state.
        save = any(ctx.needs_input_grad)
        if save:
            ckpt_re = torch.empty((B, NC, C, N), device=dnu.device, dtype=torch.float32)
            ckpt_im = torch.empty((B, NC, C, N), device=dnu.device, dtype=torch.float32)
        else:
            ckpt_re = dnu.new_empty(1)
            ckpt_im = dnu.new_empty(1)

        grid = lambda META: (B, triton.cdiv(C, META["BLOCK_C"]))
        _selective_lru_fwd_kernel[grid](
            dnu, dth, x, a, omega, D, Bsr, Bsi, Cre, Cim, y, ckpt_re, ckpt_im,
            B, T, C, N, NC,
            BLOCK_N=_block_n(N), BLOCK_T=block_t, SAVE_CKPT=save,
        )

        if save:
            ctx.save_for_backward(dnu, dth, a, omega, Bsr, Bsi, Cre, Cim, x, D,
                                  ckpt_re, ckpt_im)
            ctx.shapes = (B, T, C, N, NC)
            ctx.block_t = block_t
            ctx.block_c = block_c
        return y

    @staticmethod
    def backward(ctx, dy):
        dnu, dth, a, omega, Bsr, Bsi, Cre, Cim, x, D, ckpt_re, ckpt_im = ctx.saved_tensors
        B, T, C, N, NC = ctx.shapes
        BLOCK_T = ctx.block_t
        BLOCK_C = min(ctx.block_c, triton.next_power_of_2(C))
        dev = dnu.device
        grid = (B, triton.cdiv(C, BLOCK_C))

        ht_re = torch.empty((B, C, BLOCK_T, N), device=dev, dtype=torch.float32)
        ht_im = torch.empty((B, C, BLOCK_T, N), device=dev, dtype=torch.float32)

        d_dnu = torch.empty((B, T, C), device=dev, dtype=torch.float32)
        d_dth = torch.empty((B, T, C), device=dev, dtype=torch.float32)
        dx = torch.empty((B, T, C), device=dev, dtype=torch.float32)
        da = torch.zeros((C, N), device=dev, dtype=torch.float32)
        dom = torch.zeros((C, N), device=dev, dtype=torch.float32)
        dD = torch.zeros((C,), device=dev, dtype=torch.float32)
        dBsr = torch.zeros((B, T, N), device=dev, dtype=torch.float32)
        dBsi = torch.zeros((B, T, N), device=dev, dtype=torch.float32)
        dCre = torch.zeros((B, T, N), device=dev, dtype=torch.float32)
        dCim = torch.zeros((B, T, N), device=dev, dtype=torch.float32)

        _selective_lru_bwd_kernel[grid](
            dnu, dth, x, a, omega, D, Bsr, Bsi, Cre, Cim, dy.contiguous(),
            ckpt_re, ckpt_im, ht_re, ht_im,
            d_dnu, d_dth, dx, da, dom, dD, dBsr, dBsi, dCre, dCim,
            B, T, C, N, NC,
            BLOCK_C=BLOCK_C, BLOCK_N=_block_n(N), BLOCK_T=BLOCK_T,
            num_warps=4,
        )

        return (d_dnu, d_dth, da, dom, dBsr, dBsi, dCre, dCim, dx, dD, None, None)


def fused_selective_lru(
    delta_nu: torch.Tensor,
    delta_theta: torch.Tensor,
    a: torch.Tensor,
    omega: torch.Tensor,
    Bsel_re: torch.Tensor,
    Bsel_im: torch.Tensor,
    Csel_re: torch.Tensor,
    Csel_im: torch.Tensor,
    x: torch.Tensor,
    D: torch.Tensor,
    block_t: int = 16,
    block_c: int = 32,
) -> torch.Tensor:
    """Fused selective-LRU scan + readout. Returns ``y`` of shape ``(B, T, C)``.

    Computes, per channel/mode:
        lam = exp(delta_nu * (-a + j*(omega + delta_theta)))
        Bu  = sqrt(1 - |lam|^2) * (Bsel_re + j*Bsel_im) * x
        h_t = lam_t * h_{t-1} + Bu_t;   y = Re(<C, h>) + D * x

    The forward kernel autotunes its channel-block size and warp count.
    ``block_t`` is the backward recompute chunk width (peak extra HBM scales with
    it); ``block_c`` the backward channel-slab width.
    """
    return _SelectiveLRUScan.apply(
        delta_nu, delta_theta, a, omega,
        Bsel_re.contiguous(), Bsel_im.contiguous(),
        Csel_re.contiguous(), Csel_im.contiguous(),
        x, D, block_t, block_c,
    )