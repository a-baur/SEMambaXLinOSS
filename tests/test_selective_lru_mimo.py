"""Tests for models.linoss.selective_linoss.SelectiveLinOSS (S-LinOSS)."""

from __future__ import annotations

import pytest
import torch

from selective_lru.selective_lru import _selective_recurrence
from selective_lru import SelectiveLRUMIMO


def _make(in_features=8, state_dim=16, seed=0):
    torch.manual_seed(seed)
    return SelectiveLRUMIMO(in_features=in_features, state_dim=state_dim)


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
    # Output at time t must depend only on inputs at times <= t. This exercises the
    # selective transition too: lambda_k depends on the current token x_k.
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


def test_parameter_shapes():
    in_features, state_dim = 8, 16
    m = SelectiveLRUMIMO(in_features=in_features, state_dim=state_dim)
    assert m.nu_proj.weight.shape == (state_dim, in_features)
    assert m.theta_proj.weight.shape == (state_dim, in_features)
    assert m.B.shape == (state_dim, in_features, 2)
    assert m.C.shape == (in_features, state_dim, 2)
    assert m.D.shape == (in_features,)


def test_eigenvalue_magnitude_init_in_band():
    # At zero selective contribution the baseline magnitude nu = sigmoid(c_nu) must
    # fall in the [r_min, r_max] band the init targets (spec section 9).
    m = SelectiveLRUMIMO(in_features=8, state_dim=64, r_min=0.9, r_max=0.999)
    nu0 = torch.sigmoid(m.nu_proj.bias)
    assert (nu0 >= 0.9).all() and (nu0 <= 0.999).all()


def test_transition_is_a_contraction():
    # |lambda_k| = nu_k < 1 strictly, so the unforced state norm must shrink each step
    # for any input-driven schedule (the structural stability guarantee, spec section 6).
    torch.manual_seed(0)
    lam = torch.polar(torch.rand(2, 20, 16), torch.randn(2, 20, 16))  # |lam| < 1
    h0 = torch.randn(2, 16, dtype=torch.cfloat)
    Bu = torch.zeros(2, 20, 16, dtype=torch.cfloat)
    Bu[:, 0] = h0  # inject the initial state, then let it evolve unforced

    h = _selective_recurrence(lam, Bu)
    norms = h.abs().sum(dim=-1)  # (B, T)
    # Strictly decreasing after the initial injection.
    assert (norms[:, 1:] <= norms[:, :-1] + 1e-6).all()


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _copy_model_to(model: SelectiveLRUMIMO, device, use_triton: bool) -> SelectiveLRUMIMO:
    """Fresh SelectiveLinOSS with the same config on `device`, params copied over."""
    twin = SelectiveLRUMIMO(
        in_features=model.in_features,
        state_dim=model.state_dim,
        use_triton=use_triton,
    ).to(device)
    with torch.no_grad():
        for (_, p_src), (_, p_dst) in zip(model.named_parameters(), twin.named_parameters()):
            p_dst.copy_(p_src.to(device))
    return twin


@cuda_only
def test_triton_scan_matches_reference_direct():
    # Raw selective scan vs the sequential reference, forward + grads w.r.t. lam, Bu.
    from selective_lru.selective_lru_mimo_triton import selective_scan_triton

    torch.manual_seed(0)
    B, T, N = 3, 17, 16
    nu = torch.rand(B, T, N, device="cuda") * 0.99  # |lam| < 1
    theta = torch.randn(B, T, N, device="cuda")
    Bu = torch.randn(B, T, N, dtype=torch.cfloat, device="cuda")

    def run(fn, nu, theta, Bu):
        nu = nu.detach().clone().requires_grad_(True)
        theta = theta.detach().clone().requires_grad_(True)
        Bu = Bu.detach().clone().requires_grad_(True)
        lam = torch.polar(nu, theta)
        h = fn(lam, Bu)
        # Mix real + imag so grads flow through both components of h.
        (h.real.sum() + 2.0 * h.imag.sum()).backward()
        return h, nu.grad, theta.grad, Bu.grad

    h_ref, dnu_ref, dth_ref, dBu_ref = run(_selective_recurrence, nu, theta, Bu)
    h_tri, dnu_tri, dth_tri, dBu_tri = run(selective_scan_triton, nu, theta, Bu)

    assert torch.allclose(h_tri, h_ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(dnu_tri, dnu_ref, atol=1e-4, rtol=1e-4)
    assert torch.allclose(dth_tri, dth_ref, atol=1e-4, rtol=1e-4)
    assert torch.allclose(dBu_tri, dBu_ref, atol=1e-4, rtol=1e-4)


@cuda_only
def test_triton_forward_matches_reference():
    cpu_model = _make(in_features=8, state_dim=16)
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
    cpu_model = _make(in_features=8, state_dim=16)
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
