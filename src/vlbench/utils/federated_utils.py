# Copyright (c) 2026 Adrian R. Minut
# Copyright (c) 2026 ABI Team
# 
# SPDX-License-Identifier: GPL-3.0
# SPDX-License-Identifier: Apache 2.0

import torch
import copy
from typing import Dict


def get_parameters_vector(model: torch.nn.Module) -> torch.Tensor:
    """Return model parameters as a flattened vector."""
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach()


def set_parameters_vector(model: torch.nn.Module, vector: torch.Tensor):
    """Set model parameters from a flattened vector."""
    torch.nn.utils.vector_to_parameters(copy.deepcopy(vector), model.parameters())


def get_gradients_vector(model: torch.nn.Module) -> torch.Tensor:
    """Return model gradients as a flattened vector."""
    grads = []
    for p in model.parameters():
        if p.grad is None:
            grads.append(torch.zeros_like(p).flatten())
        else:
            grads.append(p.grad.detach().flatten())
    return torch.cat(grads)


def kldivergence(mp, sigp, mq, sigq):
    """KL(p||q) for diagonal Gaussians.

    Ported from bayes-admm/utils.py.
    """
    kl = ((mp - mq) ** 2.0) / sigq
    kl += torch.log(sigq) - torch.log(sigp) - 1.0
    kl += sigp / sigq
    return 0.5 * kl.sum()
