# Copyright (c) 2026 Adrian R. Minut
# Copyright (c) 2024 ABI Team
# 
# SPDX-License-Identifier: GPL-3.0

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from typing import Tuple, Optional
from os.path import join as pjoin


class MNISTInfo:
    outclass = 10
    imgshape = (1, 28, 28)
    counts = {"train": 60000, "test": 10000}
    mean = (0.1307,)
    std = (0.3081,)


def get_mnist_train_loaders(
    data_dir: str,
    train_val_split: float,
    workers: int,
    pin_memory: bool,
    tbatch: int,
    vbatch: int,
) -> Tuple[DataLoader, DataLoader]:
    mnist_dir = pjoin(data_dir, "mnist")
    normalize = transforms.Normalize(mean=MNISTInfo.mean, std=MNISTInfo.std)

    train_data = datasets.MNIST(
        root=mnist_dir,
        train=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        ),
        download=True,
    )

    nb_train = int(len(train_data) * train_val_split)
    train_indices = list(range(nb_train))
    val_indices = list(range(nb_train, len(train_data)))

    train_loader = DataLoader(
        Subset(train_data, train_indices),
        batch_size=tbatch,
        num_workers=workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_loader = DataLoader(
        Subset(train_data, val_indices),
        batch_size=vbatch,
        num_workers=workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def get_mnist_test_loader(
    data_dir: str, workers: int, pin_memory: bool, batch: int
) -> DataLoader:
    mnist_dir = pjoin(data_dir, "mnist")
    normalize = transforms.Normalize(mean=MNISTInfo.mean, std=MNISTInfo.std)

    test_data = datasets.MNIST(
        root=mnist_dir,
        train=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        ),
        download=True,
    )

    return DataLoader(
        test_data,
        batch_size=batch,
        num_workers=workers,
        shuffle=False,
        pin_memory=pin_memory,
    )
