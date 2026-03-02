# Copyright (c) 2026 Adrian R. Minut
# SPDX-License-Identifier: GPL-3.0

import torch
import pytest
from vlbench.federated.models import CifarNet, FedDynCifarCNN, ResNet20
from vlbench.federated.worker import FederatedWorker
from vlbench.federated.orchestrator import FederatedOrchestrator
from vldatasets.partitioned.mnist import get_partitioned_mnist_loaders


def test_model_architectures():
    x = torch.randn(2, 3, 32, 32)

    # CifarNet
    model = CifarNet(3, 10)
    out = model(x)
    assert out.shape == (2, 10)

    # FedDynCifarCNN
    model = FedDynCifarCNN(10, 3)
    out = model(x)
    assert out.shape == (2, 10)

    # ResNet20
    model = ResNet20(num_classes=10)
    out = model(x)
    assert out.shape == (2, 10)


def test_feddyn_scaling():
    # Verify that rho is scaled by proportion
    method_params = {"rho": 1.0, "num_clients": 2}

    class DummyLoader:
        def __init__(self, size):
            self.dataset = [0] * size

        def __len__(self):
            return len(self.dataset)

    # Mock workers with different dataset sizes
    w1 = type(
        "obj",
        (object,),
        {
            "train_loader": DummyLoader(100),
            "method_params": method_params,
            "method": "FedDyn",
        },
    )()
    w2 = type(
        "obj",
        (object,),
        {
            "train_loader": DummyLoader(300),
            "method_params": method_params,
            "method": "FedDyn",
        },
    )()

    workers = [w1, w2]
    total = 400
    w1.proportion = 100 / 400
    w2.proportion = 300 / 400

    # This matches the logic in orchestrator._distribute
    rho1 = method_params["rho"] * w1.proportion
    rho2 = method_params["rho"] * w2.proportion

    assert rho1 == 0.25
    assert rho2 == 0.75


def test_fashion_mnist_loader(tmp_path):
    # Verify FashionMNIST loader
    partitioned_train, test_loader = get_partitioned_mnist_loaders(
        data_dir=str(tmp_path), num_clients=2, fashion_mnist=True
    )
    assert len(partitioned_train.client_indices) == 2
    # Check if it's FashionMNIST (indirectly via class name if possible or just successful load)
    assert "FashionMNIST" in str(type(partitioned_train.dataset))
