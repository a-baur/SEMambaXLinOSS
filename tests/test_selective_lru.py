"""Tests for models.linoss.SelectiveLRU.SelectiveLRU (per-channel oscillatory selective SSM)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from selective_lru.selective_lru import SelectiveLRU


def _make(d_model=8, d_state=4, seed=0, **kw):
    torch.manual_seed(seed)
    return SelectiveLRU(d_model=d_model, d_state=d_state, **kw)


def test_forward_shape_and_dtype():
    model = _make()
    x = torch.randn(2, 7, 8)
    y = model(x)
    assert y.shape == (2, 7, 8)
    assert y.dtype == torch.float32
    assert torch.isfinite(y).all()


def test_backward_flows_to_params_and_input():
    model = _make()
    x = torch.randn(2, 7, 8, requires_grad=True)
    model(x).sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"


def test_causality():
    # Output at time t depends only on inputs at times <= t. Exercises the selective
    # transition (lambda_k and the drive both depend on the current token x_k).
    model = _make().eval()
    x = torch.randn(1, 10, 8)
    t_perturb = 5

    with torch.no_grad():
        y_ref = model(x)
        x_pert = x.clone()
        x_pert[:, t_perturb:] += torch.randn_like(x_pert[:, t_perturb:])
        y_pert = model(x_pert)

    assert torch.allclose(y_ref[:, :t_perturb], y_pert[:, :t_perturb], atol=1e-6)
    assert not torch.allclose(y_ref[:, t_perturb:], y_pert[:, t_perturb:])


def test_batch_independence():
    model = _make().eval()
    x0 = torch.randn(1, 6, 8)
    x1 = torch.randn(1, 6, 8)

    with torch.no_grad():
        y_batched = model(torch.cat([x0, x1], dim=0))
        y_split = torch.cat([model(x0), model(x1)], dim=0)

    assert torch.allclose(y_batched, y_split, atol=1e-6)


def test_zero_input_zero_output():
    # B@0 = 0 and D*0 = 0, so a zero input produces a zero output (state starts at 0).
    model = _make().eval()
    x = torch.zeros(2, 5, 8)
    with torch.no_grad():
        y = model(x)
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-7)


def test_deterministic_forward():
    model = _make().eval()
    x = torch.randn(2, 5, 8)
    with torch.no_grad():
        assert torch.equal(model(x), model(x))


def test_parameter_shapes_and_effective_state():
    d_model, d_state = 8, 4
    m = SelectiveLRU(d_model=d_model, d_state=d_state)
    # Shared selective B/C (d_state wide), not dense per-channel matrices.
    assert m.B_proj.weight.shape == (d_state, d_model)
    assert m.C_proj.weight.shape == (2 * d_state, d_model)
    # Per-(channel, mode) static dynamics -> effective state d_model * d_state.
    assert m.A_log.shape == (d_model, d_state)
    assert m.omega.shape == (d_model, d_state)
    assert m.D.shape == (d_model,)
    # dt_rank defaults to ceil(d_model / 64) -> 1 here.
    assert m.dt_rank == 1
    assert m.dt_nu_down.weight.shape == (1, d_model)
    assert m.dt_nu_up.weight.shape == (d_model, 1)


def test_transition_is_a_contraction():
    # nu = exp(-Delta_nu * a) with Delta_nu = softplus(...) > 0 and a = exp(A_log) > 0,
    # so every eigenvalue magnitude is strictly < 1 for any input — the structural,
    # schedule-free stability guarantee (selective_linoss.md sections 6, 10) per channel.
    torch.manual_seed(0)
    m = _make(d_model=8, d_state=4)
    x = torch.randn(3, 12, 8)
    delta_nu = F.softplus(m.dt_nu_up(m.dt_nu_down(x)))    # (B, T, C) > 0
    a = torch.exp(m.A_log)                                 # (C, N) > 0
    nu = torch.exp(-delta_nu.unsqueeze(-1) * a)            # (B, T, C, N)
    assert (nu < 1.0).all() and (nu > 0.0).all()


def test_baseline_nu_in_unit_disk_band():
    # At zero selective contribution, baseline Delta_nu = softplus(dt_nu_up.bias) lies in
    # [dt_min, dt_max], so baseline |lambda| = exp(-Delta_nu * a) is near (but below) 1.
    m = SelectiveLRU(d_model=16, d_state=8, dt_min=1e-3, dt_max=0.1)
    dt = F.softplus(m.dt_nu_up.bias)
    assert (dt >= 1e-3 - 1e-6).all() and (dt <= 0.1 + 1e-6).all()


def test_option_defaults_are_current_behavior():
    # The new knobs must default to the pre-existing behavior.
    m = _make()
    assert m.input_norm == "delta_nu"
    assert m.mag_init == "mamba"


def test_invalid_options_raise():
    with pytest.raises(ValueError):
        _make(input_norm="bogus")
    with pytest.raises(ValueError):
        _make(mag_init="bogus")


def test_ring_init_baseline_nu_in_band():
    # mag_init="ring": at zero selective contribution baseline |lambda| = exp(-delta_nu * a)
    # must land in [r_min, r_max]. Baseline delta_nu = softplus(dt_nu_up.bias) (== 1 here).
    r_min, r_max = 0.85, 0.99
    m = SelectiveLRU(d_model=16, d_state=8, mag_init="ring", r_min=r_min, r_max=r_max)
    delta_nu = F.softplus(m.dt_nu_up.bias)                 # (C,)
    nu = torch.exp(-delta_nu.unsqueeze(-1) * torch.exp(m.A_log))   # (C, N)
    assert (nu >= r_min - 1e-4).all() and (nu <= r_max + 1e-4).all()


def test_gamma_norm_forward_is_finite_and_stable():
    # input_norm="gamma": LRU variance-preserving drive; still causal-stable and finite.
    m = _make(input_norm="gamma").eval()
    x = torch.randn(2, 12, 8)
    with torch.no_grad():
        y = m(x)
    assert y.shape == x.shape and torch.isfinite(y).all()


def test_gamma_norm_backward_flows():
    m = _make(input_norm="gamma")
    x = torch.randn(2, 7, 8, requires_grad=True)
    m(x).sum().backward()
    assert torch.isfinite(x.grad).all()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_rank0_is_base_model():
    # Default rank=0: no correction params exist, so the base model is untouched.
    m = _make()
    assert m.rank == 0
    assert not hasattr(m, "c_read") and not hasattr(m, "c_coef")


def test_rank_correction_vanishes_when_gated_off():
    # With the scalar gate c_coef zeroed, the rank-r read correction is exactly 0, so the
    # output must equal the base (rank=0) model given identical shared parameters. This is
    # the "base behavior is preserved" guarantee: rank>0 starts as the base model.
    base = _make(d_model=8, d_state=4, rank=0, seed=1).eval()
    ranked = _make(d_model=8, d_state=4, rank=2, seed=2).eval()
    with torch.no_grad():
        base_params = dict(base.named_parameters())
        for name, p in ranked.named_parameters():
            if name in base_params:
                p.copy_(base_params[name])
        ranked.c_coef.weight.zero_()
    x = torch.randn(2, 9, 8)
    with torch.no_grad():
        assert torch.allclose(ranked(x), base(x), atol=1e-6)


def test_rank_correction_flows_and_is_active():
    m = _make(d_model=8, d_state=4, rank=2)
    # A non-trivial gate makes the correction actually change the output.
    with torch.no_grad():
        m.c_coef.weight.normal_(0.0, 0.5)
    base_readout = _make(d_model=8, d_state=4, rank=2).eval()
    with torch.no_grad():
        for name, p in base_readout.named_parameters():
            p.copy_(dict(m.named_parameters())[name])
        base_readout.c_coef.weight.zero_()

    x = torch.randn(2, 9, 8, requires_grad=True)
    y = m(x)
    assert y.shape == (2, 9, 8) and torch.isfinite(y).all()
    with torch.no_grad():
        assert not torch.allclose(y, base_readout(x.detach()), atol=1e-5)  # correction active

    y.sum().backward()
    for name in ("c_read", "c_coef", "c_pool", "c_out"):
        p = getattr(m, name)
        g = (p.weight if isinstance(p, torch.nn.Linear) else p).grad
        assert g is not None and torch.isfinite(g).all(), name


def test_rank_correction_is_causal():
    # The correction reads the causal state h_t with input-dependent c_t/s^C, so output
    # at t still depends only on inputs <= t.
    m = _make(d_model=8, d_state=4, rank=2).eval()
    with torch.no_grad():
        m.c_coef.weight.normal_(0.0, 0.5)
    x = torch.randn(1, 10, 8)
    tp = 5
    with torch.no_grad():
        y_ref = m(x)
        x_pert = x.clone()
        x_pert[:, tp:] += torch.randn_like(x_pert[:, tp:])
        y_pert = m(x_pert)
    assert torch.allclose(y_ref[:, :tp], y_pert[:, :tp], atol=1e-6)
    assert not torch.allclose(y_ref[:, tp:], y_pert[:, tp:])


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _copy_model_to(model: SelectiveLRU, device, use_triton: bool) -> SelectiveLRU:
    twin = SelectiveLRU(
        d_model=model.d_model,
        d_state=model.d_state,
        dt_rank=model.dt_rank,
        input_norm=model.input_norm,
        mag_init=model.mag_init,
        use_triton=use_triton,
    ).to(device)
    with torch.no_grad():
        for (_, p_src), (_, p_dst) in zip(model.named_parameters(), twin.named_parameters()):
            p_dst.copy_(p_src.to(device))
    return twin


def _fused_reference(dnu, dth, a, om, Bs, Cre, Cim, x, D, use_gamma=False):
    # Materialized equivalent of the fused op, for grad checking.
    nu = torch.exp(-dnu.unsqueeze(-1) * a)
    theta = om + dth.unsqueeze(-1)
    lam = torch.polar(nu, theta)
    if use_gamma:
        gain = torch.sqrt((1.0 - nu**2).clamp_min(1e-6))
    else:
        gain = dnu.unsqueeze(-1)
    Bu = ((gain * Bs.unsqueeze(-2)) * x.unsqueeze(-1)).to(lam.dtype)
    B, T, C, N = lam.shape
    h = torch.zeros(B, C, N, dtype=lam.dtype, device=lam.device)
    outs = []
    for t in range(T):
        h = lam[:, t] * h + Bu[:, t]
        outs.append(h)
    h = torch.stack(outs, 1)
    Cy = torch.einsum("btn,btcn->btc", torch.complex(Cre, Cim), h).real
    return Cy + x * D


@cuda_only
@pytest.mark.parametrize("use_gamma", [False, True])
@pytest.mark.parametrize("B,T,C,N,chunk", [(2, 19, 3, 4, 16), (2, 40, 5, 4, 16),
                                           (3, 16, 4, 8, 16), (1, 33, 2, 4, 8)])
def test_fused_op_matches_reference_direct(B, T, C, N, chunk, use_gamma):
    # Fused scan+readout vs the materialized reference: forward and all 9 input grads.
    # Shapes cover T that is / isn't a multiple of the chunk width (single & multi chunk).
    # Both input-gain modes: delta_nu (Mamba ZOH) and gamma (LRU normalization).
    from selective_lru.selective_lru_triton import fused_selective_lru

    torch.manual_seed(B + T + C + N)
    raw = dict(
        dnu=torch.nn.functional.softplus(torch.randn(B, T, C, device="cuda")) * 0.3,
        dth=torch.randn(B, T, C, device="cuda"),
        a=torch.exp(torch.randn(C, N, device="cuda") * 0.5),
        om=torch.randn(C, N, device="cuda"),
        Bs=torch.randn(B, T, N, device="cuda"),
        Cre=torch.randn(B, T, N, device="cuda"),
        Cim=torch.randn(B, T, N, device="cuda"),
        x=torch.randn(B, T, C, device="cuda"),
        D=torch.randn(C, device="cuda"),
    )
    ref_in = {k: v.clone().requires_grad_(True) for k, v in raw.items()}
    tri_in = {k: v.clone().requires_grad_(True) for k, v in raw.items()}

    y_ref = _fused_reference(**ref_in, use_gamma=use_gamma)
    y_tri = fused_selective_lru(
        tri_in["dnu"], tri_in["dth"], tri_in["a"], tri_in["om"], tri_in["Bs"],
        torch.complex(tri_in["Cre"], tri_in["Cim"]), tri_in["x"], tri_in["D"], block_t=chunk,
        use_gamma=use_gamma,
    )
    assert torch.allclose(y_tri, y_ref, atol=1e-5, rtol=1e-5)

    go = torch.randn_like(y_ref)
    y_ref.backward(go)
    y_tri.backward(go)
    for k in raw:
        assert torch.allclose(tri_in[k].grad, ref_in[k].grad, atol=1e-4, rtol=1e-4), (
            f"{k} grad mismatch: {(tri_in[k].grad - ref_in[k].grad).abs().max().item()}"
        )


@cuda_only
@pytest.mark.parametrize("input_norm", ["delta_nu", "gamma"])
def test_triton_forward_matches_reference(input_norm):
    cpu_model = _make(d_model=8, d_state=4, input_norm=input_norm)
    triton_model = _copy_model_to(cpu_model, torch.device("cuda"), use_triton=True)
    ref_model = _copy_model_to(cpu_model, torch.device("cuda"), use_triton=False)

    torch.manual_seed(123)
    x = torch.randn(3, 19, 8, device="cuda")
    with torch.no_grad():
        y_ref = ref_model(x)
        y_triton = triton_model(x)

    assert torch.allclose(y_triton, y_ref, atol=1e-5, rtol=1e-5), (
        f"max abs diff: {(y_triton - y_ref).abs().max().item()}"
    )


@cuda_only
@pytest.mark.parametrize("input_norm", ["delta_nu", "gamma"])
def test_triton_backward_matches_reference(input_norm):
    cpu_model = _make(d_model=8, d_state=4, input_norm=input_norm)
    triton_model = _copy_model_to(cpu_model, torch.device("cuda"), use_triton=True)
    ref_model = _copy_model_to(cpu_model, torch.device("cuda"), use_triton=False)

    torch.manual_seed(7)
    x = torch.randn(2, 15, 8, device="cuda")
    x_ref = x.detach().clone().requires_grad_(True)
    x_tri = x.detach().clone().requires_grad_(True)

    ref_model(x_ref).sum().backward()
    triton_model(x_tri).sum().backward()

    assert torch.allclose(x_tri.grad, x_ref.grad, atol=1e-4, rtol=1e-4), (
        f"x.grad max abs diff: {(x_tri.grad - x_ref.grad).abs().max().item()}"
    )

    ref_params = dict(ref_model.named_parameters())
    for name, p in triton_model.named_parameters():
        ref_grad = ref_params[name].grad
        assert p.grad is not None, f"no grad for {name}"
        max_diff = (p.grad - ref_grad).abs().max().item()
        assert torch.allclose(p.grad, ref_grad, atol=1e-4, rtol=1e-4), (
            f"{name} grad mismatch, max abs diff: {max_diff}"
        )
