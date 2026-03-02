# Copyright (c) 2026 Adrian R. Minut
# Copyright (c) 2024 ABI Team
# 
# SPDX-License-Identifier: GPL-3.0

import torch
import torch.nn as nn
from typing import Dict, Any
from ..utils.federated_utils import get_parameters_vector, set_parameters_vector


class FederatedWorker:
    """Handles local training for a federated client.

    Ported and refactored from bayes-admm/local_client_worker.py.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        method: str,
        client_idx: int,
        device: torch.device,
        train_loader: Any,
        test_loader: Any,
        method_params: Dict[str, Any],
    ):
        self.model = model
        self.optimizer = optimizer
        self.method = method
        self.client_idx = client_idx
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.method_params = method_params

        self.criterion = nn.CrossEntropyLoss()

        # State variables
        self.received_global_weights = None
        self.dual_weights = None
        self.dual_mean = None
        self.dual_prec = None
        self.proportion = 1.0 / self.method_params.get("num_clients", 1)  # Default
        self.last_train_loss = 0.0

        # Initialization based on method
        if self.method in ["FedADMM", "FedDyn", "FedLap"]:
            num_params = get_parameters_vector(self.model).numel()
            self.dual_weights = torch.zeros(num_params, device=self.device)
            self.received_global_weights = torch.zeros(num_params, device=self.device)
        elif self.method in ["FedLapCov", "FedIVON"]:
            num_params = get_parameters_vector(self.model).numel()
            self.dual_mean = torch.zeros(num_params, device=self.device)
            self.dual_prec = torch.zeros(num_params, device=self.device)

        # For FedIVON, the optimizer (IVONFederated) handles duals internally via its constructor params
        # but we might need to update them between rounds.

    def train(self, epochs: int = 1):
        self.model.train()
        for _ in range(epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                if isinstance(data, torch.Tensor):
                    data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()

                if self.method == "FedIVON":
                    # IVON uses closures for MC sampling
                    def closure():
                        self.optimizer.zero_grad()
                        output = self.model(data)
                        loss = self.criterion(output, target)
                        # The IVONFederated optimizer handles its own regularization (prior/dual)
                        loss.backward()
                        return loss

                    self.optimizer.step(closure)
                else:
                    output = self.model(data)
                    loss = self._compute_loss(output, target)
                    loss.backward()
                    self.optimizer.step()

                self.last_train_loss = loss.item()

    def _compute_loss(self, output, target):
        loss = self.criterion(output, target)

        if self.method == "FedProx":
            if self.received_global_weights is not None:
                mu = self.method_params.get("mu", 0.01)
                curr_weights = get_parameters_vector(self.model)
                loss += (mu / 2) * torch.norm(
                    curr_weights - self.received_global_weights
                ) ** 2

        elif self.method in ["FedADMM", "FedDyn"]:
            rho = self.method_params.get("rho", 0.01)
            # For FedDyn, rho is scaled by proportion Nk/N
            if self.method == "FedDyn":
                rho *= self.proportion

            curr_weights = get_parameters_vector(self.model)
            # Loss = f(w) + <y, w - z> + (rho/2) ||w - z||^2
            loss += torch.dot(
                self.dual_weights, curr_weights - self.received_global_weights
            )
            loss += (rho / 2) * torch.norm(
                curr_weights - self.received_global_weights
            ) ** 2

        return loss

    def update_dual(self):
        if self.method in ["FedADMM", "FedDyn"]:
            rho = self.method_params.get("rho", 0.01)
            if self.method == "FedDyn":
                rho *= self.proportion
            curr_weights = get_parameters_vector(self.model)
            self.dual_weights += rho * (curr_weights - self.received_global_weights)
        # FedIVON dual update is typically handled by the server sending new duals
        # or by the client updating them based on local posterior.
        # In bayes-admm it was: self.dual_mean += rho * (curr_mean - global_mean) etc.

    def set_global_weights(self, weights: torch.Tensor):
        self.received_global_weights = weights.to(self.device)
        if self.method != "FedIVON":
            set_parameters_vector(self.model, self.received_global_weights)

    def test(self, loader=None):
        if loader is None:
            loader = self.test_loader
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in loader:
                if isinstance(data, torch.Tensor):
                    data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(loader.dataset)
        accuracy = correct / len(loader.dataset)
        return test_loss, accuracy
