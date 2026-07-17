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
    g_ptr, s_ptr,                     # (C, N) each; dummies unless ENVELOPE / DETUNE
    h0_re_ptr, h0_im_ptr,             # (B, C, N); dummies unless HAS_H0
    y_ptr,                            # (B, T, C)
    hf_re_ptr, hf_im_ptr,             # (B, C, N); dummies unless SAVE_HF
    ckpt_re_ptr, ckpt_im_ptr,         # (B, NC, C, N)
    B, T, C, N, NC,
    BLOCK_C: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_T: tl.constexpr,
    SAVE_CKPT: tl.constexpr,
    ENVELOPE: tl.constexpr,
    DETUNE: tl.constexpr,
    HAS_H0: tl.constexpr,
    SAVE_HF: tl.constexpr,
    GAIN: tl.constexpr,          # 0 = gamma (LRU), 1 = mamba (Euler), 2 = exact ZOH
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
    if ENVELOPE:
        gq = tl.load(g_ptr + cn, mask=m2, other=0.0)
    if DETUNE:
        sd = tl.load(s_ptr + cn, mask=m2, other=0.0)

    c_base = pb * T * C + oc
    n_base = pb * T * N + on
    bcn = pb * C * N + cn                            # index into (B, C, N)

    if HAS_H0:
        hre = tl.load(h0_re_ptr + bcn, mask=m2, other=0.0)
        him = tl.load(h0_im_ptr + bcn, mask=m2, other=0.0)
    else:
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

        # lam = exp(dnu * (-a + j(om + s*dth)));  Bu = gain * (Bsr + jBsi) * x with
        # gain in {sqrt(1-|lam|^2), dnu, (lam-1)/A_c} selected by GAIN.
        nu = tl.exp(-dnu[:, None] * a)
        if DETUNE:
            base = om + sd * dth[:, None]
        else:
            base = om + dth[:, None]
        th = dnu[:, None] * base
        lre = nu * tl.cos(th)
        lim = nu * tl.sin(th)
        if GAIN == 0:
            gx = tl.sqrt(tl.maximum(1.0 - nu * nu, 1e-6)) * xv[:, None]
            bur = gx * Bsr[None, :]
            bui = gx * Bsi[None, :]
        elif GAIN == 1:
            gx = (dnu * xv)[:, None]
            bur = gx * Bsr[None, :]
            bui = gx * Bsi[None, :]
        else:
            # (lam - 1)/A_c with A_c = -a + j*base; expm1 form avoids cancellation.
            sh = tl.sin(0.5 * th)
            lm1r = tl.math.expm1(-dnu[:, None] * a) * tl.cos(th) - 2.0 * sh * sh
            den = tl.maximum(a * a + base * base, 1e-8)
            gr = (base * lim - a * lm1r) / den
            gi = -(a * lim + base * lm1r) / den
            bur = (gr * Bsr[None, :] - gi * Bsi[None, :]) * xv[:, None]
            bui = (gr * Bsi[None, :] + gi * Bsr[None, :]) * xv[:, None]

        new_hre = lre * hre - lim * him + bur
        new_him = lre * him + lim * hre + bui
        hre = new_hre
        him = new_him

        contrib = Cr[None, :] * hre - Ci[None, :] * him
        if ENVELOPE:
            contrib += gq * (hre * hre + him * him)
        yv = tl.sum(tl.where(m2, contrib, 0.0), axis=1) + Dc * xv    # (BLOCK_C,)
        tl.store(y_ptr + c_off, yv, mask=mc)

    if SAVE_HF:
        tl.store(hf_re_ptr + bcn, hre, mask=m2)
        tl.store(hf_im_ptr + bcn, him, mask=m2)


@triton.jit
def _selective_lru_bwd_kernel(
    dnu_ptr, dth_ptr, x_ptr,          # (B, T, C)
    a_ptr, om_ptr, D_ptr,             # (C, N), (C, N), (C,)
    Bsr_ptr, Bsi_ptr,                 # (B, T, N)
    Cre_ptr, Cim_ptr,                 # (B, T, N)
    g_ptr, s_ptr,                     # (C, N); dummies unless ENVELOPE / DETUNE
    dy_ptr,                           # (B, T, C)  upstream grad
    dhf_re_ptr, dhf_im_ptr,           # (B, C, N)  upstream grad of h_final; dummies unless HAS_DHF
    ckpt_re_ptr, ckpt_im_ptr,         # (B, NC, C, N)  chunk-boundary states (from forward)
    ht_re_ptr, ht_im_ptr,             # (B, C, BLOCK_T, N)  within-chunk state scratch
    d_dnu_ptr, d_dth_ptr, dx_ptr,     # (B, T, C)  per-element grads (direct write)
    da_ptr, dom_ptr, dD_ptr,          # (C, N), (C, N), (C,)  batch-reduced grads (atomic)
    dg_ptr, ds_ptr,                   # (C, N)  batch-reduced grads (atomic); dummies if off
    dBsr_ptr, dBsi_ptr, dCre_ptr, dCim_ptr,   # (B, T, N)  channel-reduced grads (atomic)
    dh0_re_ptr, dh0_im_ptr,           # (B, C, N)  grad of h_init (direct write); dummies unless NEED_DH0
    B, T, C, N, NC,
    BLOCK_C: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_T: tl.constexpr,
    ENVELOPE: tl.constexpr,
    DETUNE: tl.constexpr,
    HAS_DHF: tl.constexpr,
    NEED_DH0: tl.constexpr,
    GAIN: tl.constexpr,          # 0 = gamma (LRU), 1 = mamba (Euler), 2 = exact ZOH
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
    if ENVELOPE:
        gq = tl.load(g_ptr + cn, mask=m2, other=0.0)
    if DETUNE:
        sd = tl.load(s_ptr + cn, mask=m2, other=0.0)
    bcn = pb * C * N + cn

    # Within-chunk h scratch is indexed by the global channel id.
    ht_cbase = pb * C * BLOCK_T * N + oc[:, None] * BLOCK_T * N + on[None, :]   # + li*N

    # Register accumulators for the batch-reduced grads (one atomic flush at the end).
    da_acc = tl.zeros((BLOCK_C, BLOCK_N), dtype=tl.float32)
    dom_acc = tl.zeros((BLOCK_C, BLOCK_N), dtype=tl.float32)
    dD_acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    dg_acc = tl.zeros((BLOCK_C, BLOCK_N), dtype=tl.float32)
    ds_acc = tl.zeros((BLOCK_C, BLOCK_N), dtype=tl.float32)

    # Adjoint carry and lambda_{t+1} carry. A gradient on h_final enters as the
    # initial carry with lambda_{T} := 1 (h_final = h_{T-1} is used directly).
    if HAS_DHF:
        Gre = tl.load(dhf_re_ptr + bcn, mask=m2, other=0.0)
        Gim = tl.load(dhf_im_ptr + bcn, mask=m2, other=0.0)
        lnr = tl.full((BLOCK_C, BLOCK_N), 1.0, dtype=tl.float32)
    else:
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
                if DETUNE:
                    base = om + sd * dth[:, None]
                else:
                    base = om + dth[:, None]
                th = dnu[:, None] * base
                lre = nu * tl.cos(th)
                lim = nu * tl.sin(th)
                if GAIN == 0:
                    gx = tl.sqrt(tl.maximum(1.0 - nu * nu, 1e-6)) * xv[:, None]
                    bur = gx * Bsr[None, :]
                    bui = gx * Bsi[None, :]
                elif GAIN == 1:
                    gx = (dnu * xv)[:, None]
                    bur = gx * Bsr[None, :]
                    bui = gx * Bsi[None, :]
                else:
                    sh = tl.sin(0.5 * th)
                    lm1r = tl.math.expm1(-dnu[:, None] * a) * tl.cos(th) - 2.0 * sh * sh
                    den = tl.maximum(a * a + base * base, 1e-8)
                    gr = (base * lim - a * lm1r) / den
                    gi = -(a * lim + base * lm1r) / den
                    bur = (gr * Bsr[None, :] - gi * Bsi[None, :]) * xv[:, None]
                    bui = (gr * Bsi[None, :] + gi * Bsr[None, :]) * xv[:, None]
                n_re = lre * bre - lim * bim + bur
                n_im = lre * bim + lim * bre + bui
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
                if DETUNE:
                    base = om + sd * dth[:, None]
                else:
                    base = om + dth[:, None]
                th = dnu[:, None] * base
                costh = tl.cos(th)
                sinth = tl.sin(th)

                # g_t = dy * conj(C) [+ 2*dy*g*h_t from the envelope term];
                # G_t = g_t + conj(lambda_{t+1}) G_{t+1}.
                gre = dyv[:, None] * Cr[None, :]
                gim = -dyv[:, None] * Ci[None, :]
                if ENVELOPE:
                    gre += 2.0 * dyv[:, None] * gq * ht_re
                    gim += 2.0 * dyv[:, None] * gq * ht_im
                    dg_acc += dyv[:, None] * (ht_re * ht_re + ht_im * ht_im)
                Gtr = gre + lnr * Gre + lni * Gim
                Gti = gim + lnr * Gim - lni * Gre

                # Input map Bu = gain * (Bsr + jBsi) * x;  dBu = G_t (complex).
                if GAIN == 0:
                    gain = tl.sqrt(tl.maximum(1.0 - nu * nu, 1e-6))
                    sBG = Gtr * Bsr[None, :] + Gti * Bsi[None, :]
                    dBsr_c = tl.sum(tl.where(m2, Gtr * gain * xv[:, None], 0.0), axis=0)
                    dBsi_c = tl.sum(tl.where(m2, Gti * gain * xv[:, None], 0.0), axis=0)
                    dx_bu = tl.sum(tl.where(m2, sBG * gain, 0.0), axis=1)
                    # gain = sqrt(1 - nu^2): the input map feeds gradient into nu too.
                    dnu_bu = (sBG * xv[:, None]) * (-nu / gain)
                    d_dnu_bu = tl.sum(tl.where(m2, dnu_bu * (-a) * nu, 0.0), axis=1)
                    da_bu = dnu_bu * (-dnu[:, None]) * nu
                elif GAIN == 1:
                    sBG = Gtr * Bsr[None, :] + Gti * Bsi[None, :]
                    gx = dnu * xv
                    dBsr_c = tl.sum(tl.where(m2, Gtr * gx[:, None], 0.0), axis=0)
                    dBsi_c = tl.sum(tl.where(m2, Gti * gx[:, None], 0.0), axis=0)
                    dx_bu = tl.sum(tl.where(m2, sBG * dnu[:, None], 0.0), axis=1)
                    d_dnu_bu = tl.sum(tl.where(m2, sBG * xv[:, None], 0.0), axis=1)
                    da_bu = tl.zeros((BLOCK_C, BLOCK_N), dtype=tl.float32)
                else:
                    # gain = (lam - 1)/A_c (complex), A_c = -a + j*base.
                    lre = nu * costh
                    lim = nu * sinth
                    sh = tl.sin(0.5 * th)
                    lm1r = tl.math.expm1(-dnu[:, None] * a) * costh - 2.0 * sh * sh
                    den = tl.maximum(a * a + base * base, 1e-8)
                    gr = (base * lim - a * lm1r) / den
                    gi = -(a * lim + base * lm1r) / den
                    # dB = x * conj(gain) * G, reduced over channels.
                    dBsr_c = tl.sum(tl.where(m2, (gr * Gtr + gi * Gti) * xv[:, None], 0.0), axis=0)
                    dBsi_c = tl.sum(tl.where(m2, (gr * Gti - gi * Gtr) * xv[:, None], 0.0), axis=0)
                    # dx = sum_n Re(conj(gain * B) * G).
                    wre = gr * Bsr[None, :] - gi * Bsi[None, :]
                    wim = gr * Bsi[None, :] + gi * Bsr[None, :]
                    dx_bu = tl.sum(tl.where(m2, wre * Gtr + wim * Gti, 0.0), axis=1)
                    # d(gain) as a complex adjoint: dgain = x * conj(B) * G.
                    dgr = xv[:, None] * (Bsr[None, :] * Gtr + Bsi[None, :] * Gti)
                    dgi = xv[:, None] * (Bsr[None, :] * Gti - Bsi[None, :] * Gtr)
                    # d gain / d dnu = lam (exactly).
                    d_dnu_bu = tl.sum(tl.where(m2, dgr * lre + dgi * lim, 0.0), axis=1)
                    # d gain / d A_c = (dnu*lam - gain)/A_c =: q;
                    # d/da = -q, d/d base = j*q  (A_c = -a + j*base).
                    pr = dnu[:, None] * lre - gr
                    pi = dnu[:, None] * lim - gi
                    q_re = (base * pi - a * pr) / den
                    q_im = -(a * pi + base * pr) / den
                    da_bu = -(dgr * q_re + dgi * q_im)
                    dwt_g = dgi * q_re - dgr * q_im
                tl.atomic_add(dBsr_ptr + n_off, dBsr_c, mask=mn)
                tl.atomic_add(dBsi_ptr + n_off, dBsi_c, mask=mn)

                # dlam = G_t conj(h_{t-1}); nu and theta = dnu * (om + dth) chains.
                dlre = Gtr * hm_re + Gti * hm_im
                dlim = Gti * hm_re - Gtr * hm_im
                dnu_n = dlre * costh + dlim * sinth
                dth_n = nu * (dlim * costh - dlre * sinth)
                d_dnu_lam = tl.sum(
                    tl.where(m2, dnu_n * (-a) * nu + dth_n * base, 0.0), axis=1
                )
                # Total gradient into omega~ = base: theta chain + (zoh) gain chain.
                if GAIN == 2:
                    dwt = dth_n * dnu[:, None] + dwt_g
                else:
                    dwt = dth_n * dnu[:, None]
                if DETUNE:
                    d_dth = tl.sum(tl.where(m2, dwt * sd, 0.0), axis=1)
                    ds_acc += dwt * dth[:, None]
                else:
                    d_dth = tl.sum(tl.where(m2, dwt, 0.0), axis=1)
                dom_acc += dwt
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

    if NEED_DH0:
        # Adjoint carried past t=0: dh0 = conj(lambda_0) * G_0.
        tl.store(dh0_re_ptr + bcn, lnr * Gre + lni * Gim, mask=m2)
        tl.store(dh0_im_ptr + bcn, lnr * Gim - lni * Gre, mask=m2)

    tl.atomic_add(da_ptr + cn, da_acc, mask=m2)
    tl.atomic_add(dom_ptr + cn, dom_acc, mask=m2)
    tl.atomic_add(dD_ptr + oc, dD_acc, mask=mc)
    if ENVELOPE:
        tl.atomic_add(dg_ptr + cn, dg_acc, mask=m2)
    if DETUNE:
        tl.atomic_add(ds_ptr + cn, ds_acc, mask=m2)


class _SelectiveLRUScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, delta_nu, delta_theta, a, omega, Bsr, Bsi, Cre, Cim, x, D,
                g, s, h0_re, h0_im, block_t, block_c, want_final_state, gain_mode):
        # Unused-output grads arrive as None (not materialized zeros), so the
        # backward can branch on plain Python bools without a GPU sync.
        ctx.set_materialize_grads(False)
        if not delta_nu.is_cuda:
            raise RuntimeError("fused selective-LRU scan requires CUDA tensors.")
        B, T, C = delta_nu.shape
        N = a.shape[1]
        envelope = g is not None
        detune = s is not None
        has_h0 = h0_re is not None

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
        dummy = dnu.new_zeros(1)
        g_t = g.contiguous() if envelope else dummy
        s_t = s.contiguous() if detune else dummy
        h0r = h0_re.contiguous() if has_h0 else dummy
        h0i = h0_im.contiguous() if has_h0 else dummy

        NC = triton.cdiv(T, block_t)
        save = any(ctx.needs_input_grad)
        # The final state is needed as an output, and (cheaply) whenever grads are
        # wanted, so the backward can support an incoming grad on it.
        save_hf = want_final_state or save

        y = torch.empty((B, T, C), device=dnu.device, dtype=torch.float32)
        if save_hf:
            hf_re = torch.empty((B, C, N), device=dnu.device, dtype=torch.float32)
            hf_im = torch.empty((B, C, N), device=dnu.device, dtype=torch.float32)
        else:
            hf_re, hf_im = dummy, dummy
        if save:
            ckpt_re = torch.empty((B, NC, C, N), device=dnu.device, dtype=torch.float32)
            ckpt_im = torch.empty((B, NC, C, N), device=dnu.device, dtype=torch.float32)
        else:
            ckpt_re, ckpt_im = dummy, dummy

        grid = lambda META: (B, triton.cdiv(C, META["BLOCK_C"]))
        _selective_lru_fwd_kernel[grid](
            dnu, dth, x, a, omega, D, Bsr, Bsi, Cre, Cim, g_t, s_t, h0r, h0i,
            y, hf_re, hf_im, ckpt_re, ckpt_im,
            B, T, C, N, NC,
            BLOCK_N=_block_n(N), BLOCK_T=block_t, SAVE_CKPT=save,
            ENVELOPE=envelope, DETUNE=detune, HAS_H0=has_h0, SAVE_HF=save_hf,
            GAIN=gain_mode,
        )

        if save:
            ctx.save_for_backward(dnu, dth, a, omega, Bsr, Bsi, Cre, Cim, x, D,
                                  g_t, s_t, ckpt_re, ckpt_im)
            ctx.shapes = (B, T, C, N, NC)
            ctx.block_t = block_t
            ctx.block_c = block_c
            ctx.flags = (envelope, detune, has_h0)
            ctx.gain_mode = gain_mode
        return y, hf_re, hf_im

    @staticmethod
    def backward(ctx, dy, dhf_re, dhf_im):
        (dnu, dth, a, omega, Bsr, Bsi, Cre, Cim, x, D,
         g_t, s_t, ckpt_re, ckpt_im) = ctx.saved_tensors
        B, T, C, N, NC = ctx.shapes
        envelope, detune, has_h0 = ctx.flags
        need_dh0 = has_h0 and (ctx.needs_input_grad[12] or ctx.needs_input_grad[13])
        # With materialize_grads(False) these are None iff the output was unused,
        # so the flags below are plain Python bools (Triton constexprs).
        if dhf_re is None and dhf_im is not None:
            dhf_re = torch.zeros_like(dhf_im)
        if dhf_im is None and dhf_re is not None:
            dhf_im = torch.zeros_like(dhf_re)
        has_dhf = dhf_re is not None
        if dy is None:  # possible when only h_final feeds the loss
            dy = torch.zeros((B, T, C), device=dnu.device, dtype=torch.float32)
        BLOCK_T = ctx.block_t
        BLOCK_C = min(ctx.block_c, triton.next_power_of_2(C))
        dev = dnu.device
        dummy = dnu.new_zeros(1)
        grid = (B, triton.cdiv(C, BLOCK_C))

        ht_re = torch.empty((B, C, BLOCK_T, N), device=dev, dtype=torch.float32)
        ht_im = torch.empty((B, C, BLOCK_T, N), device=dev, dtype=torch.float32)

        d_dnu = torch.empty((B, T, C), device=dev, dtype=torch.float32)
        d_dth = torch.empty((B, T, C), device=dev, dtype=torch.float32)
        dx = torch.empty((B, T, C), device=dev, dtype=torch.float32)
        da = torch.zeros((C, N), device=dev, dtype=torch.float32)
        dom = torch.zeros((C, N), device=dev, dtype=torch.float32)
        dD = torch.zeros((C,), device=dev, dtype=torch.float32)
        dg = torch.zeros((C, N), device=dev, dtype=torch.float32) if envelope else dummy
        ds = torch.zeros((C, N), device=dev, dtype=torch.float32) if detune else dummy
        dBsr = torch.zeros((B, T, N), device=dev, dtype=torch.float32)
        dBsi = torch.zeros((B, T, N), device=dev, dtype=torch.float32)
        dCre = torch.zeros((B, T, N), device=dev, dtype=torch.float32)
        dCim = torch.zeros((B, T, N), device=dev, dtype=torch.float32)
        dh0r = torch.empty((B, C, N), device=dev, dtype=torch.float32) if need_dh0 else dummy
        dh0i = torch.empty((B, C, N), device=dev, dtype=torch.float32) if need_dh0 else dummy

        _selective_lru_bwd_kernel[grid](
            dnu, dth, x, a, omega, D, Bsr, Bsi, Cre, Cim, g_t, s_t,
            dy.contiguous(),
            dhf_re.contiguous() if has_dhf else dummy,
            dhf_im.contiguous() if has_dhf else dummy,
            ckpt_re, ckpt_im, ht_re, ht_im,
            d_dnu, d_dth, dx, da, dom, dD, dg, ds,
            dBsr, dBsi, dCre, dCim, dh0r, dh0i,
            B, T, C, N, NC,
            BLOCK_C=BLOCK_C, BLOCK_N=_block_n(N), BLOCK_T=BLOCK_T,
            ENVELOPE=envelope, DETUNE=detune, HAS_DHF=has_dhf, NEED_DH0=need_dh0,
            GAIN=ctx.gain_mode,
            num_warps=4,
        )

        return (d_dnu, d_dth, da, dom, dBsr, dBsi, dCre, dCim, dx, dD,
                dg if envelope else None, ds if detune else None,
                dh0r if need_dh0 else None, dh0i if need_dh0 else None,
                None, None, None, None)


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
    g: torch.Tensor | None = None,
    s: torch.Tensor | None = None,
    h0_re: torch.Tensor | None = None,
    h0_im: torch.Tensor | None = None,
    block_t: int = 16,
    block_c: int = 32,
    want_final_state: bool = False,
    input_gain: str = "gamma",
):
    """Fused selective-LRU scan + readout.

    Returns ``(y, hf_re, hf_im)`` with ``y`` of shape ``(B, T, C)`` and the final
    state parts of shape ``(B, C, N)`` (dummy 1-element tensors unless
    ``want_final_state`` or gradients are required).

        lam  = exp(delta_nu * (-a + j*(omega [+ s*] delta_theta)))
        Bu   = gain * (Bsel_re + j*Bsel_im) * x
        h_t  = lam_t * h_{t-1} + Bu_t      (h_{-1} = h0, default 0)
        y    = Re(<C, h>) [+ <g, |h|^2>] + D * x

    ``input_gain`` selects the input normalization (compile-time specialized):
    ``"gamma"``: sqrt(1 - |lam|^2) (LRU energy norm); ``"mamba"``: delta_nu
    (Euler / S6, real, shared over modes); ``"zoh"``: (lam - 1)/A_c with
    A_c = -a + j*(omega [+ s*] delta_theta) (exact ZOH, complex per mode). The
    gain is computed in registers from quantities the kernel already holds; it
    is never materialized in HBM.

    Optional pieces (compile-time specialized; absent tensors cost nothing):
    ``g`` enables the quadratic envelope readout, ``s`` the per-mode detune gains,
    ``h0_*`` the initial state (gradient supported), and a gradient arriving on the
    final state is injected into the adjoint, so chunked/TBPTT training composes.
    """
    gain_mode = {"gamma": 0, "mamba": 1, "zoh": 2}[input_gain]
    return _SelectiveLRUScan.apply(
        delta_nu, delta_theta, a, omega,
        Bsel_re.contiguous(), Bsel_im.contiguous(),
        Csel_re.contiguous(), Csel_im.contiguous(),
        x, D, g, s, h0_re, h0_im, block_t, block_c, want_final_state, gain_mode,
    )