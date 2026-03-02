# Copyright (c) 2026 Adrian R. Minut
# Copyright (c) 2024 ABI Team
# 
# SPDX-License-Identifier: GPL-3.0

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from .core import PartitionedDataset
from os.path import join as pjoin
from torch.utils.data import DataLoader
from typing import Tuple


def get_partitioned_cifar10_loaders(
    data_dir: str,
    num_clients: int,
    alpha1: float = 1e6,
    alpha2: float = 0.5,
    dataset_proportion: float = 1.0,
    seed: int = 42,
    batch_size: int = 32,
    workers: int = 2,
    pin_memory: bool = True,
) -> Tuple[PartitionedDataset, DataLoader]:
    cifar10_dir = pjoin(data_dir, "cifar10")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=cifar10_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root=cifar10_dir, train=False, download=True, transform=transform
    )

    partitioned_train = PartitionedDataset(
        train_dataset,
        num_clients,
        num_classes=10,
        alpha1=alpha1,
        alpha2=alpha2,
        dataset_proportion=dataset_proportion,
        seed=seed,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )

    return partitioned_train, test_loader


def get_partitioned_cifar100_loaders(
    data_dir: str,
    num_clients: int,
    alpha1: float = 1e6,
    alpha2: float = 0.5,
    dataset_proportion: float = 1.0,
    seed: int = 42,
    batch_size: int = 32,
    workers: int = 2,
    pin_memory: bool = True,
) -> Tuple[PartitionedDataset, DataLoader]:
    cifar100_dir = pjoin(data_dir, "cifar100")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )

    train_dataset = datasets.CIFAR100(
        root=cifar100_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR100(
        root=cifar100_dir, train=False, download=True, transform=transform
    )

    partitioned_train = PartitionedDataset(
        train_dataset,
        num_clients,
        num_classes=100,
        alpha1=alpha1,
        alpha2=alpha2,
        dataset_proportion=dataset_proportion,
        seed=seed,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )

    return partitioned_train, test_loader
