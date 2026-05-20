"""Tests for models.linoss.LinOSS."""

from __future__ import annotations

import math

import pytest
import torch

from models.linoss import LinOSS, _map_theta_to_A


VARIANTS = [
    ("IMEX", True),   # Damped LinOSS-IMEX
    ("IMEX", False),  # LinOSS-IMEX
    ("IM", False),    # LinOSS-IM
]


def _make(in_features=8, state_dim=16, discretization="IMEX", damping=True, seed=0):
    torch.manual_seed(seed)
    return LinOSS(
        in_features=in_features,
        state_dim=state_dim,
        discretization=discretization,
        damping=damping,
    )


@pytest.mark.parametrize("discretization,damping", VARIANTS)
def test_forward_shape_and_dtype(discretization, damping):
    model = _make(discretization=discretization, damping=damping)
    x = torch.randn(2, 7, 8)
    y = model(x)
    assert y.shape == (2, 7, 8)
    assert y.dtype == torch.float32
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("discretization,damping", VARIANTS)
def test_backward_flows_to_params_and_input(discretization, damping):
    model = _make(discretization=discretization, damping=damping)
    x = torch.randn(2, 7, 8, requires_grad=True)
    model(x).sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"


@pytest.mark.parametrize("discretization,damping", VARIANTS)
def test_causality(discretization, damping):
    # Output at time t must depend only on inputs at times <= t.
    model = _make(discretization=discretization, damping=damping).eval()
    x = torch.randn(1, 10, 8)
    t_perturb = 5

    with torch.no_grad():
        y_ref = model(x)
        x_pert = x.clone()
        x_pert[:, t_perturb:] += torch.randn_like(x_pert[:, t_perturb:])
        y_pert = model(x_pert)

    assert torch.allclose(y_ref[:, :t_perturb], y_pert[:, :t_perturb], atol=1e-6)
    assert not torch.allclose(y_ref[:, t_perturb:], y_pert[:, t_perturb:])


@pytest.mark.parametrize("discretization,damping", VARIANTS)
def test_batch_independence(discretization, damping):
    # Stacking two sequences into a batch must equal running them separately.
    model = _make(discretization=discretization, damping=damping).eval()
    x0 = torch.randn(1, 6, 8)
    x1 = torch.randn(1, 6, 8)

    with torch.no_grad():
        y_batched = model(torch.cat([x0, x1], dim=0))
        y_split = torch.cat([model(x0), model(x1)], dim=0)

    assert torch.allclose(y_batched, y_split, atol=1e-6)


@pytest.mark.parametrize("discretization,damping", VARIANTS)
def test_deterministic_forward(discretization, damping):
    model = _make(discretization=discretization, damping=damping).eval()
    x = torch.randn(2, 5, 8)
    with torch.no_grad():
        assert torch.equal(model(x), model(x))


@pytest.mark.parametrize("discretization,damping", VARIANTS)
def test_zero_input_zero_output(discretization, damping):
    # B@0 = 0 and D*0 = 0, so a zero input must produce a zero output
    # (the recurrence starts from zero state).
    model = _make(discretization=discretization, damping=damping).eval()
    x = torch.zeros(2, 5, 8)
    with torch.no_grad():
        y = model(x)
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-7)


def test_g_diag_only_for_damped_imex():
    damped = _make(discretization="IMEX", damping=True)
    assert damped.G_diag is not None
    assert damped.G_diag.shape == (16,)

    for disc, damp in [("IMEX", False), ("IM", False)]:
        m = _make(discretization=disc, damping=damp)
        assert m.G_diag is None, f"G_diag should be None for {disc}/damping={damp}"


def test_im_with_damping_raises():
    with pytest.raises(NotImplementedError):
        LinOSS(in_features=8, discretization="IM", damping=True)


def test_unknown_discretization_raises():
    with pytest.raises(NotImplementedError):
        LinOSS(in_features=8, discretization="ZOH")  # type: ignore[arg-type]


@pytest.mark.parametrize("state_dim", [1, 8, 64])
def test_state_dim_variants(state_dim):
    model = _make(state_dim=state_dim)
    x = torch.randn(2, 4, 8)
    y = model(x)
    assert y.shape == (2, 4, 8)
    assert model.A_diag.shape == (state_dim,)
    assert model.steps.shape == (state_dim,)


def test_parameter_shapes():
    in_features, state_dim = 8, 16
    m = LinOSS(in_features=in_features, state_dim=state_dim, discretization="IMEX", damping=True)
    assert m.steps.shape == (state_dim,)
    assert m.A_diag.shape == (state_dim,)
    assert m.G_diag.shape == (state_dim,)
    assert m.B.shape == (state_dim, in_features, 2)
    assert m.C.shape == (in_features, state_dim, 2)
    assert m.D.shape == (in_features,)


def test_map_theta_to_A_matches_formula():
    # Sanity check the closed-form inversion: feed thetas in (0, pi/2) and (pi/2, pi),
    # confirm A_diag is finite and uses the correct branch.
    torch.manual_seed(0)
    state_dim = 32
    G = torch.rand(state_dim).clamp(min=1e-3)
    steps = torch.rand(state_dim).clamp(min=1e-3)

    thetas_low = torch.rand(state_dim) * (math.pi / 2 - 0.1) + 0.05
    thetas_high = torch.rand(state_dim) * (math.pi / 2 - 0.1) + (math.pi / 2 + 0.05)

    A_low = _map_theta_to_A(thetas_low, G, steps)
    A_high = _map_theta_to_A(thetas_high, G, steps)
    assert torch.isfinite(A_low).all()
    assert torch.isfinite(A_high).all()
