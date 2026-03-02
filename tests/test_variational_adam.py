# Copyright (c) 2026 Adrian R. Minut
# SPDX-License-Identifier: GPL-3.0

import torch
import torch.nn as nn
from vloptimizers.variational_adam import VariationalAdam


def test_variational_adam_initialization():
    model = nn.Linear(10, 2)
    optimizer = VariationalAdam(model.parameters(), lr=1e-3, prior_variance=1e-2)
    assert isinstance(optimizer, VariationalAdam)
    assert optimizer.prior_variance == 1e-2


def test_variational_adam_step():
    """Verify that VariationalAdam updates parameters and performs sampling."""
    model = nn.Linear(10, 2)
    optimizer = VariationalAdam(model.parameters(), lr=1e-3)

    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))
    criterion = nn.CrossEntropyLoss()

    # Step 1: Initialize state
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    state = optimizer.state[next(model.parameters())]
    assert "mean" in state
    assert "variance" in state

    # Step 2: Check sampling (params should change on every call if noise is added)

    # After step, parameters are sampled: p = mean + noise
    p_val1 = next(model.parameters()).clone()

    # If we call step again, it should resample
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    p_val2 = next(model.parameters()).clone()
    assert not torch.equal(p_val1, p_val2)
