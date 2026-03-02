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


def get_partitioned_mnist_loaders(
    data_dir: str,
    num_clients: int,
    alpha1: float = 1e6,
    alpha2: float = 0.5,
    dataset_proportion: float = 1.0,
    seed: int = 42,
    batch_size: int = 32,
    workers: int = 2,
    pin_memory: bool = True,
    fashion_mnist: bool = False,
) -> Tuple[PartitionedDataset, DataLoader]:
    dataset_name = "fashion_mnist" if fashion_mnist else "mnist"
    mnist_dir = pjoin(data_dir, dataset_name)

    if fashion_mnist:
        dataset_class = datasets.FashionMNIST
        normalize = transforms.Normalize((0.2860,), (0.3530,))
    else:
        dataset_class = datasets.MNIST
        normalize = transforms.Normalize((0.1307,), (0.3081,))

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_dataset = dataset_class(
        root=mnist_dir, train=True, download=True, transform=transform
    )
    test_dataset = dataset_class(
        root=mnist_dir, train=False, download=True, transform=transform
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
