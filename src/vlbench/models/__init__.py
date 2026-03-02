# Copyright (c) 2026 Adrian R. Minut
# SPDX-License-Identifier: GPL-3.0

"""Model architectures: ResNets, DenseNets, SWAG wrapper, MC-Dropout."""

from ._registry import (
    STANDARDMODELS,
    MCDROPMODELS,
    BBBMODELS,
    SWAGMODELS,
    MODELS,
    savemodel,
    loadmodel,
)
from .swag import SWAG
from .mcdropout import MCDropout

__all__ = [
    "STANDARDMODELS",
    "MCDROPMODELS",
    "BBBMODELS",
    "SWAGMODELS",
    "MODELS",
    "savemodel",
    "loadmodel",
    "SWAG",
    "MCDropout",
]
