# Copyright (c) 2026 Adrian R. Minut
# SPDX-License-Identifier: GPL-3.0

import torch
import torch.nn as nn
from vloptimizers.ivon import IVON


def test_ivon_initialization():
    model = nn.Linear(10, 2)
    optimizer = IVON(model.parameters(), lr=1e-3, ess=1000)
    assert isinstance(optimizer, IVON)
    assert optimizer.defaults["ess"] == 1000


def test_ivon_sampling():
    """Verify IVON posterior sampling."""
    model = nn.Linear(10, 2)
    optimizer = IVON(model.parameters(), lr=0.1, ess=100)

    x = torch.randn(4, 10)

    # Initialize h_mom with sampled params
    with optimizer.sampled_params(train=True):
        optimizer.zero_grad()
        model(x).sum().backward()
    optimizer.step()

    # Capture mean
    mean_val = list(model.parameters())[0].clone()

    with optimizer.sampled_params(train=True):
        sampled_val = list(model.parameters())[0].clone()
        assert not torch.equal(mean_val, sampled_val)

    # Should restore mean
    restored_val = list(model.parameters())[0].clone()
    assert torch.equal(mean_val, restored_val)


def test_ivon_approx_methods():
    """Verify both price and gradsq approximations."""
    for approx in ["price", "gradsq"]:
        model = nn.Linear(10, 2)
        optimizer = IVON(model.parameters(), lr=0.1, ess=100, hess_approx=approx)

        x = torch.randn(4, 10)
        with optimizer.sampled_params(train=True):
            optimizer.zero_grad()
            model(x).sum().backward()
        optimizer.step()

        # Check that we have momentum in groups (from _init_buffers)
        assert len(optimizer.param_groups[0]["momentum"]) > 0
