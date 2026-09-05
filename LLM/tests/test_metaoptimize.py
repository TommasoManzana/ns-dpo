"""Unit tests for src/metaoptimize.py (MetaOptimize step-size learning, ported
from the upstream HF.py). CPU only, tiny models; run with `pytest LLM/tests`."""
import math

import pytest
import torch

from src.metaoptimize import MetaOptimizeHF


def base_cfg(**overrides):
    cfg = dict(
        alg_base="RMSProp", normalizer_param_base=0.99, weight_decay_base=0.0,
        alg_meta="fixed",
        alpha0=1e-2, discount=1.0, stepsize_groups="scalar",
    )
    cfg.update(overrides)
    return cfg


def quadratic_step(params, target):
    """loss = 0.5 * ||w - target||^2 ; populates p.grad like the trainer does."""
    loss = sum(0.5 * ((p - t) ** 2).sum() for p, t in zip(params, target))
    loss.backward()
    return loss.item()


def test_fixed_meta_keeps_alpha_at_alpha0():
    w = torch.nn.Parameter(torch.ones(3))
    opt = MetaOptimizeHF([w], base_cfg(alg_meta="fixed", alpha0=1e-2))
    for _ in range(5):
        quadratic_step([w], [torch.zeros(3)])
        metrics = opt.step()
        opt.zero_grad()
        assert abs(metrics["meta/alpha"] - 1e-2) < 1e-9


def test_rmsprop_base_first_step_is_alpha_times_sign_of_grad():
    # first RMSProp step: trace = g^2 and the bias correction makes mu = 1,
    # so the normalized gradient is sign(g) and w <- w - alpha * sign(g)
    w = torch.nn.Parameter(torch.tensor([1.0, -2.0, 3.0]))
    opt = MetaOptimizeHF([w], base_cfg(alg_meta="fixed", alpha0=1e-2))
    quadratic_step([w], [torch.zeros(3)])  # grad = w
    opt.step()
    expected = torch.tensor([1.0, -2.0, 3.0]) - 1e-2 * torch.tensor([1.0, -1.0, 1.0])
    assert torch.allclose(w.detach(), expected, atol=1e-6)


def test_lion_meta_increases_beta_when_alpha_too_small():
    # far from the optimum with a tiny alpha, every step's update -delta_w
    # accumulates into h with h.g < 0, so the meta step should push beta up
    w = torch.nn.Parameter(torch.full((4,), 10.0))
    cfg = base_cfg(alg_meta="Lion", meta_stepsize=1e-2, momentum_param_meta=0.9,
                   Lion_beta2_meta=0.9, weight_decay_meta=0.0, alpha0=1e-4)
    opt = MetaOptimizeHF([w], cfg)
    betas = []
    for _ in range(10):
        quadratic_step([w], [torch.zeros(4)])
        betas.append(opt.step()["meta/beta"])
        opt.zero_grad()
    assert math.isclose(betas[0], math.log(1e-4), rel_tol=1e-6)  # step 1: h == 0, no meta signal
    assert all(b2 > b1 for b1, b2 in zip(betas[1:], betas[2:]))
    assert betas[-1] > betas[0]


def test_step_raises_when_a_param_has_no_grad():
    w = torch.nn.Parameter(torch.ones(2))
    opt = MetaOptimizeHF([w], base_cfg())
    with pytest.raises(ValueError, match="no gradient"):
        opt.step()


def test_state_dict_round_trip_reproduces_next_step():
    def make():
        torch.manual_seed(0)
        w = torch.nn.Parameter(torch.randn(5))
        cfg = base_cfg(alg_meta="Lion", meta_stepsize=1e-2, momentum_param_meta=0.9,
                       Lion_beta2_meta=0.9, weight_decay_meta=0.0, alpha0=1e-3)
        return w, MetaOptimizeHF([w], cfg)

    w1, opt1 = make()
    for _ in range(3):
        quadratic_step([w1], [torch.zeros(5)]); opt1.step(); opt1.zero_grad()

    w2, opt2 = make()
    with torch.no_grad():
        w2.copy_(w1)
    opt2.load_state_dict(opt1.state_dict())

    quadratic_step([w1], [torch.zeros(5)]); m1 = opt1.step()
    quadratic_step([w2], [torch.zeros(5)]); m2 = opt2.step()
    assert m1 == m2
    assert torch.equal(w1.detach(), w2.detach())


def test_from_model_uses_only_trainable_params():
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1))
    for p in model[0].parameters():
        p.requires_grad_(False)          # "frozen base" layer
    opt = MetaOptimizeHF.from_model(model, base_cfg())
    assert len(opt.params) == 2          # weight + bias of model[1] only
    assert all(p.requires_grad for p in opt.params)
    model(torch.ones(4, 3)).sum().backward()
    opt.step()                           # frozen params have no grad; must not raise
