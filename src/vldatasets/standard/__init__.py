# Copyright (c) 2026 Adrian R. Minut
# SPDX-License-Identifier: GPL-3.0

"""Dataset loaders for vlbench (CIFAR-10/100, TinyImageNet, SVHN)."""

from .dataloaders import (
    TRAINDATALOADERS,
    TESTDATALOADER,
    NTRAIN,
    NTEST,
    INSIZE,
    OUTCLASS,
    dup_collate_fn,
    DatasetConfig,
)
from .mnist import get_mnist_train_loaders, get_mnist_test_loader, MNISTInfo
from .ood_datasets import (
    get_svhn_loader,
    get_flowers102_loader,
    SVHNInfo,
    Flowers102Info,
)
from .cifar10c import get_cifar10c_loader, CIFAR10C
from .tinyimagenet import TinyImageNet
from .bdl_competition import get_bdl_loaders

__all__ = [
    "TRAINDATALOADERS",
    "TESTDATALOADER",
    "NTRAIN",
    "NTEST",
    "INSIZE",
    "OUTCLASS",
    "dup_collate_fn",
    "TinyImageNet",
    "get_svhn_loader",
    "get_flowers102_loader",
    "SVHNInfo",
    "Flowers102Info",
    "get_bdl_loaders",
    "get_cifar10c_loader",
    "CIFAR10C",
    "DatasetConfig",
    "get_mnist_train_loaders",
    "get_mnist_test_loader",
    "MNISTInfo",
]
