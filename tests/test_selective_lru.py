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


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _copy_model_to(model: SelectiveLRU, device, use_triton: bool) -> SelectiveLRU:
    twin = SelectiveLRU(
        d_model=model.d_model,
        d_state=model.d_state,
        dt_rank=model.dt_rank,
        use_triton=use_triton,
    ).to(device)
    with torch.no_grad():
        for (_, p_src), (_, p_dst) in zip(model.named_parameters(), twin.named_parameters()):
            p_dst.copy_(p_src.to(device))
    return twin


def _fused_reference(dnu, dth, a, om, Bs, Cre, Cim, x, D):
    # Materialized equivalent of the fused op, for grad checking.
    nu = torch.exp(-dnu.unsqueeze(-1) * a)
    theta = om + dth.unsqueeze(-1)
    lam = torch.polar(nu, theta)
    Bu = ((dnu.unsqueeze(-1) * Bs.unsqueeze(-2)) * x.unsqueeze(-1)).to(lam.dtype)
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
@pytest.mark.parametrize("B,T,C,N,chunk", [(2, 19, 3, 4, 16), (2, 40, 5, 4, 16),
                                           (3, 16, 4, 8, 16), (1, 33, 2, 4, 8)])
def test_fused_op_matches_reference_direct(B, T, C, N, chunk):
    # Fused scan+readout vs the materialized reference: forward and all 9 input grads.
    # Shapes cover T that is / isn't a multiple of the chunk width (single & multi chunk).
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

    y_ref = _fused_reference(**ref_in)
    y_tri = fused_selective_lru(
        tri_in["dnu"], tri_in["dth"], tri_in["a"], tri_in["om"], tri_in["Bs"],
        torch.complex(tri_in["Cre"], tri_in["Cim"]), tri_in["x"], tri_in["D"], block_t=chunk,
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
def test_triton_forward_matches_reference():
    cpu_model = _make(d_model=8, d_state=4)
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
def test_triton_backward_matches_reference():
    cpu_model = _make(d_model=8, d_state=4)
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
