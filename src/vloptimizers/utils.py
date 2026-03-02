# Copyright (c) 2026 Adrian R. Minut
# Copyright (c) 2026 ABI Team
#
# SPDX-License-Identifier: GPL-3.0

"""
Utility functions for optimizer analysis.

This module provides utilities for extracting covariance matrices and
computing KL divergence terms for variational optimizers (IVON).
"""

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from vloptimizers import IVON


# =============================================================================
# Covariance Extraction
# =============================================================================


def get_ivon_covariance(
    optimizer: "IVON", ess: float, weight_decay: float
) -> np.ndarray:
    """
    Extract diagonal covariance from IVON optimizer.

    IVON maintains a diagonal Hessian approximation. The posterior covariance
    is computed as: Σ = 1 / (ess * (H + weight_decay))

    Args:
        optimizer: IVON optimizer instance
        ess: Effective sample size (scales the precision)
        weight_decay: L2 regularization strength

    Returns:
        2x2 diagonal covariance matrix as numpy array

    Notes:
        If the Hessian is not yet initialized, returns an identity matrix
        scaled by the prior variance 1 / (ess * weight_decay).
    """
    group = optimizer.param_groups[0]
    hess = group.get("hess")

    if hess is None:
        # Return identity scaled by prior variance if not initialized
        p = group["params"][0]
        hess = torch.zeros_like(p)

    variance = 1.0 / (ess * (hess + weight_decay))
    return np.diag(variance.cpu().numpy())


# =============================================================================
# KL Divergence Computation
# =============================================================================


def compute_kl_term(
    h: torch.Tensor,
    m: torch.Tensor,
    ess: float,
    wd: float,
    omit_constants: bool = False,
) -> float:
    """
    Compute KL divergence KL(q||p) for Gaussian posterior q vs isotropic Gaussian prior p.

    This function computes the KL divergence between:
    - q: Gaussian posterior with mean m and diagonal covariance Σ = diag(1 / (ess * (H + wd)))
    - p: Isotropic Gaussian prior N(0, 1 / (ess * wd * I))

    Args:
        h: Hessian diagonal or eigenvalues (precision parameters)
        m: Mean of the posterior distribution
        ess: Effective sample size
        wd: Weight decay (L2 regularization strength)
        omit_constants: If True, uses simplified formula from demo2d.py that omits additive constants. If False, uses exact KL derivation.

    Returns:
        KL divergence value (scalar)

    Notes:
        When omit_constants=True, uses:
            KL = 0.5 * delta * sum(m^2) + 0.5 * delta * sum(σ^2) - 0.5 * sum(log(σ^2))
            where delta = wd * ess and σ^2 = 1 / (ess * (h + wd))

        When omit_constants=False, uses exact KL:
            KL = 0.5 * [sum(log((h+wd)/wd) + wd/(h+wd) - 1) + ess * wd * sum(m^2)]
    """
    if omit_constants:
        delta = wd * ess

        # sigma^2 = 1 / (ess * (h + wd))
        precision = ess * (h + wd)
        sigma2 = 1.0 / precision

        # Term 1: 0.5 * delta * sum(m^2)
        term1 = 0.5 * delta * (m**2).sum().item()

        # Term 2: 0.5 * delta * sum(sigma^2)
        term2 = 0.5 * delta * sigma2.sum().item()

        # Term 3: -0.5 * sum(log(sigma^2))
        term3 = -0.5 * torch.log(sigma2).sum().item()

        return term1 + term2 + term3

    else:
        # Exact KL derivation
        h_plus_wd = h + wd
        curvature_term = (torch.log(h_plus_wd / wd) + wd / h_plus_wd - 1.0).sum().item()

        # Mean term
        mean_term = ess * wd * (m**2).sum().item()

        return 0.5 * (curvature_term + mean_term)


def get_ivon_kl(
    optimizer: "IVON", ess: float, wd: float, omit_constants: bool = False
) -> float:
    """
    Compute total KL divergence for IVON optimizer.

    Sums KL divergence across all parameters in the optimizer.

    Args:
        optimizer: IVON optimizer instance
        ess: Effective sample size
        wd: Weight decay
        omit_constants: Whether to omit additive constants in KL computation

    Returns:
        Total KL divergence (scalar)
    """
    total_kl = 0.0
    for group in optimizer.param_groups:
        hess = group.get("hess")
        hess_init = group.get("hess_init", 1.0)

        for p in group["params"]:
            if p.requires_grad:
                h = hess if hess is not None else torch.full_like(p.data, hess_init)
                total_kl += compute_kl_term(h, p.data, ess, wd, omit_constants)
    return total_kl
