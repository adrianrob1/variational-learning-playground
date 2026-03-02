# Copyright (c) 2026 Adrian R. Minut
# Copyright (c) 2024 ABI Team
# 
# SPDX-License-Identifier: GPL-3.0

import torch
import torch.nn as nn
from typing import List, Dict, Any
from .worker import FederatedWorker
from ..utils.federated_utils import get_parameters_vector, set_parameters_vector


class FederatedOrchestrator:
    """Orchestrates the federated learning process across multiple workers.

    Ported and refactored from bayes-admm/train.py.
    """

    def __init__(
        self,
        global_model: nn.Module,
        workers: List[FederatedWorker],
        device: torch.device,
        method: str,
        method_params: Dict[str, Any],
    ):
        self.global_model = global_model
        self.workers = workers
        self.device = device
        self.method = method
        self.method_params = method_params

        # Calculate total samples and proportions
        self.total_samples = sum(len(w.train_loader.dataset) for w in self.workers)
        for w in self.workers:
            w.proportion = len(w.train_loader.dataset) / self.total_samples

        # Initial global weights
        self.global_weights = get_parameters_vector(self.global_model).to(self.device)

        # Method specific server state
        if self.method == "FedIVON":
            self.global_mean = self.global_weights.clone()
            # precision init from weight decay
            wd = self.method_params.get("weight_decay", 0.01)
            self.global_prec = wd * torch.ones_like(self.global_weights)

    def run_round(self):
        """Execute one communication round."""
        # 1. Distribute global model to workers
        self._distribute()

        # 2. Local training
        for worker in self.workers:
            worker.train(epochs=self.method_params.get("local_epochs", 1))

        # 3. Aggregate results
        self._aggregate()

        # 4. Local dual updates (if applicable)
        for worker in self.workers:
            worker.update_dual()

    def _distribute(self):
        if self.method == "FedIVON":
            # Distribution for FedIVON involves sending both mean and prec
            # In simple ADMM it's just sending global_mean
            for worker in self.workers:
                worker.received_global_weights = self.global_mean
                # Also update worker's internal optimizer if it supports dual updates
                if hasattr(worker.optimizer, "_set_dual"):
                    # rho acts as a multiplier for dual variables in ADMM
                    rho = self.method_params.get("rho", 0.1)
                    # For FedDyn/FedLap, rho is often scaled by proportion
                    if self.method in ["FedDyn", "FedLap"]:
                        rho *= worker.proportion
                    worker.optimizer._set_dual(worker.dual_mean, worker.dual_prec)
        else:
            for worker in self.workers:
                worker.set_global_weights(self.global_weights)

    def _aggregate(self):
        if self.method == "FedIVON":
            new_global_mean = torch.zeros_like(self.global_mean)
            new_global_prec = torch.zeros_like(self.global_prec)

            # Simple average for now, could be weighted
            total_workers = len(self.workers)
            for worker in self.workers:
                # Ask worker for its local posterior (mean/prec)
                if hasattr(worker.optimizer, "_get_posterior"):
                    m_list, s_list = worker.optimizer._get_posterior()
                    m = torch.cat(m_list) if isinstance(m_list, list) else m_list
                    s = torch.cat(s_list) if isinstance(s_list, list) else s_list
                    new_global_mean += m / total_workers
                    new_global_prec += s / total_workers

            self.global_mean = new_global_mean
            self.global_prec = new_global_prec
            set_parameters_vector(self.global_model, self.global_mean)
        else:
            # Standard FedAvg aggregation
            new_weights = torch.zeros_like(self.global_weights)
            total_workers = len(self.workers)
            for worker in self.workers:
                new_weights += get_parameters_vector(worker.model) / total_workers

            self.global_weights = new_weights
            set_parameters_vector(self.global_model, self.global_weights)

    def evaluate(self, loader=None):
        """Evaluate the global model."""
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        test_loss = 0
        correct = 0
        total = 0

        # Use first worker's test loader if none provided
        if loader is None and self.workers:
            loader = self.workers[0].test_loader

        if loader is None:
            return 0, 0

        with torch.no_grad():
            for data, target in loader:
                if isinstance(data, torch.Tensor):
                    data, target = data.to(self.device), target.to(self.device)

                output = self.global_model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(target)

        test_loss /= len(loader)
        accuracy = correct / total
        return test_loss, accuracy

    def compute_variational_objective(self):
        """Compute the total variational objective (ELBO) across all clients."""
        if self.method != "FedIVON":
            return None

        total_vo = 0
        for worker in self.workers:
            # Local expected NLL
            expected_nll = worker.last_train_loss
            # KL term
            m_list, s_list = worker.optimizer._get_posterior()
            m = torch.cat(m_list) if isinstance(m_list, list) else m_list
            s = torch.cat(s_list) if isinstance(s_list, list) else s_list

            from ..utils.federated_utils import kldivergence

            prior_mean = worker.optimizer.prior_mean
            if prior_mean is None:
                # Fallback to zero if not set (though it should be for IVON)
                num_params = get_parameters_vector(worker.model).numel()
                prior_mean = torch.zeros(num_params, device=self.device)

            prior_prec = worker.optimizer.prior_prec
            if prior_prec is None:
                wd = self.method_params.get("weight_decay", 0.01)
                prior_prec = wd * torch.ones_like(prior_mean)

            kl = kldivergence(
                m,
                1.0 / s,
                prior_mean,
                1.0 / prior_prec,
            )

            tau = self.method_params.get("temperature", 1.0)
            total_vo += (len(worker.train_loader.dataset) / tau) * expected_nll + kl

        return total_vo.item()
