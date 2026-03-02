# Copyright (c) 2026 Adrian R. Minut
# Copyright (c) 2024 ABI Team
#
# SPDX-License-Identifier: GPL-3.0 OR Apache-2.0

import os
import numpy as np
import einops
from torch.utils.data import DataLoader
from .npz_dataset import NPZDataset


def get_bdl_loaders(dataset_name, data_dir, tbatch, vbatch, workers, device):
    """
    Returns train and test data loaders for BDL competition datasets (uci, cifar10, medmnist).

    Args:
        dataset_name (str): Name of the dataset ('uci', 'cifar10', 'medmnist').
        data_dir (str): Path to data directory.
        tbatch (int): Train batch size.
        vbatch (int): Eval batch size.
        workers (int): Number of workers for data loading.
        device (str): Device to use.

    Returns:
        tuple: (train_loader, test_loader)
    """
    data_names = {
        "uci": "energy_anon.npz",
        "cifar10": "cifar_anon.npz",
        "medmnist": "dermamnist_anon.npz",
    }

    data_path = os.path.join(data_dir, data_names[dataset_name])
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found at {data_path}. Please run download_data.sh in tasks/bdl_competition."
        )

    data = np.load(data_path)
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]

    if dataset_name in ("cifar10", "medmnist"):
        # Shape is (N, H, W, C) in legacy, needs (N, C, H, W)
        x_train = einops.rearrange(x_train, "n h w c -> n c h w")
        x_test = einops.rearrange(x_test, "n h w c -> n c h w")

        if dataset_name == "medmnist":
            x_train = x_train.astype(np.float32) / 255.0
            x_test = x_test.astype(np.float32) / 255.0
            y_train = y_train.astype(np.int64)
            y_test = y_test.astype(np.int64)
        else:  # cifar10
            x_train = x_train.astype(np.float32)
            x_test = x_test.astype(np.float32)
            y_train = y_train.astype(np.int64)
            y_test = y_test.astype(np.int64)
    else:  # uci
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        y_train = y_train.astype(np.float32)
        y_test = y_test.astype(np.float32)

    train_dataset = NPZDataset(x_train, y_train)
    test_dataset = NPZDataset(x_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=tbatch,
        shuffle=True,
        num_workers=workers,
        pin_memory=(device == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=vbatch,
        shuffle=False,
        num_workers=workers,
        pin_memory=(device == "cuda"),
    )

    return train_loader, test_loader
