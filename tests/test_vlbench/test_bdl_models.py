# Copyright (c) 2026 Adrian R. Minut
# SPDX-License-Identifier: GPL-3.0

import torch
import pytest
from vlbench.models.bdl_competition import (
    make_uci_mlp,
    make_cifar_alexnet,
    make_medmnist_cnn,
)


def test_make_uci_mlp():
    """Verify that the UCI MLP has the correct input and output shapes."""
    num_features = 14
    data_info = {"num_features": num_features}
    model = make_uci_mlp(data_info)
    x = torch.randn(1, num_features)
    y = model(x)
    assert y.shape == (1, 2)  # Mean and log-variance


def test_make_cifar_alexnet():
    """Verify that the CIFAR AlexNet has the correct output shape."""
    num_classes = 10
    data_info = {"num_classes": num_classes}
    model = make_cifar_alexnet(data_info)
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    assert y.shape == (1, num_classes)


def test_make_medmnist_cnn():
    """Verify that the MedMNIST CNN has the correct output shape."""
    num_classes = 8
    data_info = {"num_classes": num_classes}
    model = make_medmnist_cnn(data_info)
    x = torch.randn(1, 3, 28, 28)
    y = model(x)
    assert y.shape == (1, num_classes)
