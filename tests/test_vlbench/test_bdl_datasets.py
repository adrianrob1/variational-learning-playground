# Copyright (c) 2026 Adrian R. Minut
# SPDX-License-Identifier: GPL-3.0

import os
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from vldatasets.standard.npz_dataset import NPZDataset
from vldatasets.standard.bdl_competition import get_bdl_loaders


def test_npz_dataset():
    """Verify that NPZDataset correctly returns samples from arrays."""
    x = np.random.randn(10, 5)
    y = np.random.randint(0, 2, size=(10,))
    dataset = NPZDataset(x, y)
    assert len(dataset) == 10
    xi, yi = dataset[0]
    assert xi.shape == (5,)
    assert yi.item() in [0, 1]


@pytest.mark.skipif(
    not os.path.exists("data/energy_anon.npz"), reason="UCI data not found"
)
def test_uci_loader():
    """Verify that UCI loader returns the expected batch structure."""
    loaders = get_bdl_loaders(
        "uci", "data", tbatch=10, vbatch=10, workers=0, device="cpu"
    )
    assert len(loaders) == 2
    train_loader, test_loader = loaders
    assert isinstance(train_loader, DataLoader)
    batch = next(iter(train_loader))
    assert len(batch) == 2
    x, y = batch
    assert x.shape[1] == 8  # Energy dataset features (confirmed by test failure)
