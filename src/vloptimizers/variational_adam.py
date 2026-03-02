# Copyright (c) 2026 Adrian R. Minut
# 
# SPDX-License-Identifier: GPL-3.0

import torch
from torch.optim import AdamW
from contextlib import contextmanager


class VariationalAdam(AdamW):
    """
    Variational Adam optimizer that wraps PyTorch's AdamW implementation.

    Adds variational inference on top of AdamW: maintains posterior mean and variance for each parameter, and samples from the posterior to set parameter values.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        prior_variance=1e-2,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    ):
        if prior_variance <= 0.0:
            raise ValueError("Prior variance must be positive")

        self.prior_variance = prior_variance
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self._sampling_enabled = True

    @contextmanager
    def sampled_params(self, train: bool = False):
        """
        Context manager that samples parameters from the posterior for the forward pass,
        then restores the posterior mean afterward.
        """
        self._sample_params()
        try:
            yield
        finally:
            self._restore_param_means()

    @torch.no_grad()
    def _sample_params(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "mean" in state:
                    mean, variance = state["mean"], state["variance"]
                    noise = torch.randn_like(p.data) * variance.sqrt()
                    p.data.copy_(mean + noise)

    @torch.no_grad()
    def _restore_param_means(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "mean" in state:
                    p.data.copy_(state["mean"])

    def step(self, closure=None):
        """
        Performs a single optimization step with variational inference.
        """
        # First, let AdamW do its standard update
        loss = super().step(closure)

        # Now apply variational updates on top
        for group in self.param_groups:
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    continue

                # Initialize variational state if needed
                if "mean" not in state:
                    state["mean"] = torch.clone(p.data).detach()
                    state["variance"] = torch.full_like(p.data, self.prior_variance)

                mean, variance = state["mean"], state["variance"]
                exp_avg_sq = state["exp_avg_sq"]
                step = state["step"]

                # Compute bias correction for second moment
                bias_correction2 = 1 - beta2**step

                # Update posterior variance using AdamW's second moment estimate
                variance.copy_(
                    self.prior_variance
                    / (self.prior_variance + exp_avg_sq / bias_correction2)
                )

                # Update posterior mean to track the AdamW-updated parameters
                mean.copy_(p.data)

        return loss

    def get_posterior_means(self):
        """
        Returns the posterior means for all parameters.
        """
        posterior_means = []
        for group in self.param_groups:
            for p in group["params"]:
                if p in self.state:
                    posterior_means.append(self.state[p]["mean"])
        return posterior_means
