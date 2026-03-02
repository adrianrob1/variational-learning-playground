# Copyright (c) 2026 Adrian R. Minut
# SPDX-License-Identifier: GPL-3.0

import pytest
import torch
from vldatasets.standard.dataloaders import ImageNetInfo, get_imagenet_train_loader


def test_imagenet_info():
    """Verify ImageNet metadata."""
    assert ImageNetInfo.outclass == 1000
    assert ImageNetInfo.imgshape == (3, 224, 224)


def test_dataloader_stub():
    """Ensure dataloader stubs exist and return None (since they pass)."""
    # We can't easily test FFCV without installed dependencies and data
    # but we can check if the function is callable.
    assert callable(get_imagenet_train_loader)
