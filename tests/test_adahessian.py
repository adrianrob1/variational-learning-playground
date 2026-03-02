# Copyright (c) 2026 Adrian R. Minut
# SPDX-License-Identifier: GPL-3.0

import torch
import torch.nn as nn
from vloptimizers.adahessian import AdaHessian
import pytest


def test_adahessian_initialization():
    model = nn.Linear(10, 2)
    optimizer = AdaHessian(model.parameters(), lr=0.1)
    assert isinstance(optimizer, AdaHessian)
    assert optimizer.defaults["lr"] == 0.1


def test_adahessian_step():
    """Verify that AdaHessian updates parameters and requires create_graph=True."""
    model = nn.Linear(10, 2)
    optimizer = AdaHessian(model.parameters(), lr=1.0)

    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))
    criterion = nn.CrossEntropyLoss()

    # Forward pass
    output = model(x)
    loss = criterion(output, y)

    # Backward pass WITHOUT create_graph should fail during set_hessian if it tries
    # to compute second order grads, but actually Hutchinson uses grads.
    # In AdaHessian.py: h_zs = torch.autograd.grad(grads, params, grad_outputs=zs, ...)
    # This requires grads to have grad_fn, which requires create_graph=True in the previous backward.

    # Use autograd.grad to avoid memory leak warning
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, create_graph=True)
    for p, g in zip(params, grads):
        p.grad = g

    initial_params = [p.clone() for p in model.parameters()]
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    for p_init, p_new in zip(initial_params, model.parameters()):
        assert not torch.equal(p_init, p_new)


def test_adahessian_no_create_graph_fails():
    """Verify that missing create_graph=True causes failure."""
    model = nn.Linear(10, 2)
    optimizer = AdaHessian(model.parameters(), lr=1.0)

    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))
    criterion = nn.CrossEntropyLoss()

    output = model(x)
    loss = criterion(output, y)
    loss.backward()  # No create_graph

    with pytest.raises(
        RuntimeError,
        match="element 0 of tensors does not require grad and does not have a grad_fn",
    ):
        optimizer.step()
