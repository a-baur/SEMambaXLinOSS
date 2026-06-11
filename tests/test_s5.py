"""Tests for the PyTorch S5 port (models.s5.s5).

Covers the two components added to bring the port in line with the original
lindermanlab/S5 and tk-rusch/linoss implementations: conjugate symmetry
(``conj_sym``) and eigenvalue clipping (``clip_eigs``).
"""

from __future__ import annotations

import pytest
import torch
from models.s5.s5 import S5, S5SSM


def _make(width=8, state_width=16, conj_sym=True, clip_eigs=False, seed=0):
    torch.manual_seed(seed)
    return S5(
        width=width,
        state_width=state_width,
        conj_sym=conj_sym,
        clip_eigs=clip_eigs,
    )


@pytest.mark.parametrize("conj_sym", [True, False])
def test_forward_shape_and_dtype(conj_sym):
    model = _make(conj_sym=conj_sym)
    x = torch.randn(2, 7, 8)
    y = model(x)
    assert y.shape == (2, 7, 8)
    assert y.dtype == torch.float32
    assert torch.isfinite(y).all()


def test_conj_sym_halves_latent_state():
    # With conjugate symmetry the latent state holds only one of each conjugate
    # eigen-pair, so the number of stored eigenvalues is halved.
    full = _make(state_width=16, conj_sym=False)
    half = _make(state_width=16, conj_sym=True)
    assert full.seq.Lambda_re.shape[-1] == 16
    assert half.seq.Lambda_re.shape[-1] == 8


def test_clip_eigs_constrains_real_part():
    model = _make(clip_eigs=True)
    with torch.no_grad():
        model.seq.Lambda_re.fill_(1.0)  # push eigenvalues into the unstable half-plane
    Lambda = model.seq.get_Lambda()
    assert (Lambda.real <= -1e-4 + 1e-9).all()


def test_clip_eigs_off_leaves_real_part_untouched():
    model = _make(clip_eigs=False)
    with torch.no_grad():
        model.seq.Lambda_re.fill_(1.0)
    Lambda = model.seq.get_Lambda()
    assert torch.allclose(Lambda.real, torch.ones_like(Lambda.real))


@pytest.mark.parametrize("conj_sym", [True, False])
def test_backward_flows_to_dynamics_params(conj_sym):
    model = _make(conj_sym=conj_sym)
    x = torch.randn(2, 7, 8, requires_grad=True)
    model(x).sum().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    for attr in ["Lambda_re", "Lambda_im", "log_step"]:
        p = getattr(model.seq, attr)
        assert p.grad is not None, f"no grad for {attr}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {attr}"


def test_isinstance_s5ssm_for_optimizer_partitioning():
    # train.create_partitioned_optimizer relies on the inner module being an S5SSM.
    model = _make()
    assert any(isinstance(m, S5SSM) for m in model.modules())


@pytest.mark.parametrize("conj_sym", [True, False])
def test_block_recurrence_matches_rnn(conj_sym):
    # Mirrors the block / block-recurrent / rnn comparison in s5.py's __main__:
    # the same sequence processed (1) in one shot, (2) in chunks while carrying
    # the latent state across the boundary, and (3) one timestep at a time via
    # forward_rnn must all produce the same outputs and final state. S5SSM acts on
    # a single (T, H) sequence -- the batch vmap lives in the outer S5 wrapper.
    model = _make(conj_sym=conj_sym)
    ssm = model.seq
    H = model.width
    T = 16

    torch.manual_seed(1)
    x = torch.randn(T, H)

    with torch.no_grad():
        # (1) whole sequence at once
        y_full, state_full = ssm.forward(x, return_state=True)

        # (2) two chunks, carrying the latent state across the split
        y_a, state_a = ssm.forward(x[: T // 2], return_state=True)
        y_b, state_b = ssm.forward(x[T // 2 :], state=state_a, return_state=True)
        y_chunked = torch.cat([y_a, y_b], dim=0)

        # (3) one timestep at a time, seeded from the zero initial state
        state = ssm.initial_state(None)
        ys = []
        for t in range(T):
            y_t, state = ssm.forward_rnn(x[t], state)
            ys.append(y_t)
        y_rnn = torch.stack(ys, dim=0)

    assert y_full.shape == (T, H)
    # block-recurrent == block
    assert torch.allclose(y_full, y_chunked, atol=1e-5)
    assert torch.allclose(state_full, state_b, atol=1e-5)
    # rnn == block
    assert torch.allclose(y_full, y_rnn, atol=1e-5)
    assert torch.allclose(state_full, state, atol=1e-5)
