# Copyright (c) 2026 Adrian R. Minut
# SPDX-License-Identifier: GPL-3.0

"""Tests for vlbench.image_classification.train — optimizer factory and batch fns."""

import types
import pytest
import torch
import torch.nn as nn

from vlbench.image_classification.train import (
    get_optimizer,
    do_trainbatch_ivon,
    do_trainbatch_adahessian,
    do_evalbatch,
)
from vloptimizers.ivon import IVON
from vloptimizers.adahessian import AdaHessian


# ---------------------------------------------------------------------------
# Tiny model fixture
# ---------------------------------------------------------------------------
class TinyModel(nn.Module):
    def __init__(self, nin=4, nout=3):
        super().__init__()
        self.fc = nn.Linear(nin, nout)

    def forward(self, x):
        return self.fc(x)


def make_args(**overrides):
    """Build a minimal argparse-like namespace for get_optimizer."""
    defaults = dict(
        optimizer="ivon",
        learning_rate=0.01,
        momentum=0.9,
        momentum_hess=0.99,
        weight_decay=1e-4,
        hess_init=1.0,
        ess=50.0,
        clip_radius=float("inf"),
        mc_samples=1,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# get_optimizer
# ---------------------------------------------------------------------------
class TestGetOptimizer:
    @pytest.fixture(autouse=True)
    def model(self):
        return TinyModel()

    def test_ivon_returns_ivon(self, model):
        args = make_args(optimizer="ivon")
        opt = get_optimizer(args, model)
        assert isinstance(opt, IVON)

    def test_sgd_returns_sgd(self, model):
        args = make_args(optimizer="sgd")
        opt = get_optimizer(args, model)
        assert isinstance(opt, torch.optim.SGD)

    def test_adamw_returns_adamw(self, model):
        args = make_args(optimizer="adamw")
        opt = get_optimizer(args, model)
        assert isinstance(opt, torch.optim.AdamW)

    def test_adahessian_returns_adahessian(self, model):
        args = make_args(optimizer="adahessian")
        opt = get_optimizer(args, model)
        assert isinstance(opt, AdaHessian)

    def test_unknown_optimizer_raises(self, model):
        args = make_args(optimizer="notanoptimizer")
        with pytest.raises(ValueError):
            get_optimizer(args, model)


# ---------------------------------------------------------------------------
# do_trainbatch_ivon (smoke test only)
# ---------------------------------------------------------------------------
class TestDoTrainbatchIvon:
    def test_returns_prob_gt_loss(self):
        model = TinyModel(nin=4, nout=3)
        opt = IVON(model.parameters(), lr=0.01, ess=50.0)
        imgs = torch.randn(8, 4)
        gts = torch.randint(0, 3, (8,))
        prob, gt, loss = do_trainbatch_ivon((imgs, gts), model, opt, mc_samples=1)
        assert prob.shape == (8, 3)
        assert isinstance(loss, float)

    def test_probabilities_sum_to_one(self):
        model = TinyModel(nin=4, nout=3)
        opt = IVON(model.parameters(), lr=0.01, ess=50.0)
        imgs = torch.randn(6, 4)
        gts = torch.zeros(6, dtype=torch.long)
        prob, _, _ = do_trainbatch_ivon((imgs, gts), model, opt)
        assert torch.allclose(prob.sum(dim=1), torch.ones(6), atol=1e-5)


# ---------------------------------------------------------------------------
# do_trainbatch_adahessian (smoke test only)
# ---------------------------------------------------------------------------
class TestDoTrainbatchAdahessian:
    def test_output_shape(self):
        model = TinyModel(nin=4, nout=3)
        opt = AdaHessian(model.parameters(), lr=0.01)
        imgs = torch.randn(4, 4)
        gts = torch.randint(0, 3, (4,))
        prob, gt, loss = do_trainbatch_adahessian((imgs, gts), model, opt)
        assert prob.shape == (4, 3)
        assert isinstance(loss, float)


# ---------------------------------------------------------------------------
# do_evalbatch
# ---------------------------------------------------------------------------
class TestDoEvalbatch:
    def test_output_shape_and_loss(self):
        model = TinyModel(nin=4, nout=3)
        imgs = torch.randn(5, 4)
        gts = torch.randint(0, 3, (5,))
        with torch.no_grad():
            prob, gt, loss = do_evalbatch((imgs, gts), model)
        assert prob.shape == (5, 3)
        assert torch.allclose(prob.sum(dim=1), torch.ones(5), atol=1e-5)
        assert isinstance(loss, float) and loss >= 0.0
