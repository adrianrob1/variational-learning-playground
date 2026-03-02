# Copyright (c) 2026 Adrian R. Minut
# SPDX-License-Identifier: GPL-3.0

import torch
import torch.nn as nn
from vloptimizers.vogn import VOGN, VOGNClosure


def test_vogn_initialization():
    model = nn.Linear(10, 2)
    optimizer = VOGN(model.parameters(), lr=1e-3, data_size=100)
    assert isinstance(optimizer, VOGN)


def test_vogn_step():
    """Verify VOGN step with closure."""
    model = nn.Linear(10, 2)
    optimizer = VOGN(model.parameters(), lr=0.1, data_size=4)

    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))
    criterion = nn.CrossEntropyLoss()

    closure = VOGNClosure(model, criterion)
    closure.update_minibatch(x, y)

    initial_params = [p.clone() for p in model.parameters()]
    optimizer.step(closure)

    for p_init, p_new in zip(initial_params, model.parameters()):
        assert not torch.equal(p_init, p_new)
