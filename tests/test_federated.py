# Copyright (c) 2026 Adrian R. Minut
# SPDX-License-Identifier: GPL-3.0

import torch
import torch.nn as nn
import pytest
from vlbench.federated.worker import FederatedWorker
from vlbench.federated.orchestrator import FederatedOrchestrator
from vldatasets.partitioned.toy import ToyDataGenerator
from vloptimizers.federated.ivon_federated import IVONFederated
from vlbench.utils.federated_utils import get_parameters_vector


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def toy_data():
    return ToyDataGenerator(num_clients=2, num_samples=100)


def test_toy_data_generator(toy_data):
    assert toy_data.state["num_clients"] == 2
    (X, y), _ = toy_data.data_split(0)
    assert X.shape[0] == 200
    assert y.shape[0] == 200


def test_federated_flow_fedavg(toy_data):
    device = torch.device("cpu")
    model = SimpleModel().to(device)

    workers = []
    for i in range(2):
        local_model = SimpleModel().to(device)
        (X, y), _ = toy_data.data_split(i)
        train_dataset = torch.utils.data.TensorDataset(X, y)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=200)

        optimizer = torch.optim.SGD(local_model.parameters(), lr=0.1)
        worker = FederatedWorker(
            model=local_model,
            optimizer=optimizer,
            method="FedAvg",
            client_idx=i,
            device=device,
            train_loader=train_loader,
            test_loader=None,
            method_params={"num_clients": 2},
        )
        workers.append(worker)

    orchestrator = FederatedOrchestrator(
        global_model=model,
        workers=workers,
        device=device,
        method="FedAvg",
        method_params={},
    )

    initial_params = get_parameters_vector(model).clone()
    orchestrator.run_round()
    final_params = get_parameters_vector(model)

    assert not torch.equal(initial_params, final_params)


def test_federated_flow_fedadmm(toy_data):
    """
    Test the federated learning flow using the FedADMM method.

    This function verifies the expected input/output side-effects of employing
    the FedADMM method locally, specifically that after one training round, the
    model parameters have changed significantly, and the duals are updated.
    """
    device = torch.device("cpu")
    model = SimpleModel().to(device)

    workers = []
    for i in range(2):
        local_model = SimpleModel().to(device)
        (X, y), _ = toy_data.data_split(i)
        train_dataset = torch.utils.data.TensorDataset(X, y)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=200)

        optimizer = torch.optim.SGD(local_model.parameters(), lr=0.1)
        worker = FederatedWorker(
            model=local_model,
            optimizer=optimizer,
            method="FedADMM",
            client_idx=i,
            device=device,
            train_loader=train_loader,
            test_loader=None,
            method_params={"num_clients": 2, "rho": 0.01},
        )
        workers.append(worker)

    orchestrator = FederatedOrchestrator(
        global_model=model,
        workers=workers,
        device=device,
        method="FedADMM",
        method_params={"rho": 0.01},
    )

    initial_params = get_parameters_vector(model).clone()
    orchestrator.run_round()
    final_params = get_parameters_vector(model)

    assert not torch.equal(initial_params, final_params)
    for worker in workers:
        assert torch.max(torch.abs(worker.dual_weights)) > 0


def test_federated_flow_feddyn(toy_data):
    """
    Test the federated learning flow using the FedDyn method.

    This function verifies the expected input/output side-effects of employing
    the FedDyn method locally, verifying parameter progression.
    """
    device = torch.device("cpu")
    model = SimpleModel().to(device)

    workers = []
    for i in range(2):
        local_model = SimpleModel().to(device)
        (X, y), _ = toy_data.data_split(i)
        train_dataset = torch.utils.data.TensorDataset(X, y)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=200)

        optimizer = torch.optim.SGD(local_model.parameters(), lr=0.1)
        worker = FederatedWorker(
            model=local_model,
            optimizer=optimizer,
            method="FedDyn",
            client_idx=i,
            device=device,
            train_loader=train_loader,
            test_loader=None,
            method_params={"num_clients": 2, "rho": 0.01},
        )
        workers.append(worker)

    orchestrator = FederatedOrchestrator(
        global_model=model,
        workers=workers,
        device=device,
        method="FedDyn",
        method_params={"num_clients": 2, "rho": 0.01},
    )

    initial_params = get_parameters_vector(model).clone()
    orchestrator.run_round()
    final_params = get_parameters_vector(model)

    assert not torch.equal(initial_params, final_params)
    for worker in workers:
        assert torch.max(torch.abs(worker.dual_weights)) > 0


def test_federated_flow_fedprox(toy_data):
    """
    Test the federated learning flow using the FedProx method.

    This function verifies the expected input/output side-effects of employing
    the FedProx method locally.
    """
    device = torch.device("cpu")
    model = SimpleModel().to(device)

    workers = []
    for i in range(2):
        local_model = SimpleModel().to(device)
        (X, y), _ = toy_data.data_split(i)
        train_dataset = torch.utils.data.TensorDataset(X, y)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=200)

        optimizer = torch.optim.SGD(local_model.parameters(), lr=0.1)
        worker = FederatedWorker(
            model=local_model,
            optimizer=optimizer,
            method="FedProx",
            client_idx=i,
            device=device,
            train_loader=train_loader,
            test_loader=None,
            method_params={"num_clients": 2, "mu": 0.01},
        )
        workers.append(worker)

    orchestrator = FederatedOrchestrator(
        global_model=model,
        workers=workers,
        device=device,
        method="FedProx",
        method_params={"num_clients": 2, "mu": 0.01},
    )

    initial_params = get_parameters_vector(model).clone()
    orchestrator.run_round()
    final_params = get_parameters_vector(model)

    assert not torch.equal(initial_params, final_params)
    for worker in workers:
        assert worker.dual_weights is None  # FedProx doesn't use dual_weights


def test_ivon_federated_optimizer():
    model = SimpleModel()
    num_params = get_parameters_vector(model).numel()
    prior_mean = torch.zeros(num_params)
    prior_prec = torch.ones(num_params)

    dual_mean = torch.zeros(num_params)
    dual_prec = torch.zeros(num_params)

    optimizer = IVONFederated(
        model.parameters(),
        lr=0.1,
        ess=100,
        prior_mean=prior_mean,
        prior_prec=prior_prec,
        dual_mean=dual_mean,
        dual_prec=dual_prec,
    )

    def closure():
        optimizer.zero_grad()
        x = torch.randn(10, 2)
        y = torch.randint(0, 2, (10,))
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        return loss

    initial_params = get_parameters_vector(model).clone()
    optimizer.step(closure)
    final_params = get_parameters_vector(model)

    assert not torch.equal(initial_params, final_params)
