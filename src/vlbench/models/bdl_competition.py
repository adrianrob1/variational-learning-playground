# Copyright (c) 2026 Adrian R. Minut
# Copyright (c) 2024 ABI Team
#
# SPDX-License-Identifier: GPL-3.0

import torch.nn as nn


def make_uci_mlp(data_info, hidden_dims=[50], activation="ReLU"):
    """
    Creates an MLP for the UCI regression dataset.

    Args:
        data_info (dict): Dictionary containing 'num_features' (int).
        hidden_dims (list): List of integers for hidden layer sizes.
        activation (str): Activation function name from nn module.

    Returns:
        nn.Sequential: The UCI MLP model.
    """
    num_features = data_info["num_features"]
    act = getattr(nn, activation)

    layers = []
    curr_dim = num_features
    for h in hidden_dims:
        layers.append(nn.Linear(curr_dim, h))
        layers.append(act())
        curr_dim = h

    layers.append(nn.Linear(curr_dim, 2))
    return nn.Sequential(*layers)


def make_cifar_alexnet(data_info, width_mult=1, activation="SiLU"):
    """
    Creates an AlexNet-like CNN for CIFAR-10 classification.

    Args:
        data_info (dict): Dictionary containing 'num_classes' (int).
        width_mult (float): Multiplier for channel widths.
        activation (str): Activation function name from nn module.

    Returns:
        nn.Sequential: The CIFAR AlexNet model.
    """
    num_classes = data_info["num_classes"]
    act = getattr(nn, activation)

    c1, c2, c3, c4, c5 = [int(x * width_mult) for x in [64, 128, 256, 128, 64]]
    d1, d2 = [int(x * width_mult) for x in [256, 256]]

    return nn.Sequential(
        nn.Conv2d(3, c1, kernel_size=3, padding=1),
        act(),
        nn.MaxPool2d(2, stride=2, padding=0),
        nn.Conv2d(c1, c2, kernel_size=3, padding=1),
        act(),
        nn.MaxPool2d(2, stride=2, padding=0),
        nn.ZeroPad2d((0, 1, 0, 1)),
        nn.Conv2d(c2, c3, kernel_size=2, padding=0),
        act(),
        nn.ZeroPad2d((0, 1, 0, 1)),
        nn.Conv2d(c3, c4, kernel_size=2, padding=0),
        act(),
        nn.ZeroPad2d((0, 1, 0, 1)),
        nn.Conv2d(c4, c5, kernel_size=2, padding=0),
        act(),
        nn.Flatten(),
        # Fixed spatial size after 2 maxpools on 32x32 = 8x8.
        # Actually in legacy it was 64 * 8 * 8 = 4096.
        nn.Linear(c5 * 8 * 8, d1),
        act(),
        nn.Linear(d1, d2),
        act(),
        nn.Linear(d2, num_classes),
    )


def make_medmnist_cnn(data_info, width_mult=1, activation="ReLU"):
    """
    Creates a LeNet-like CNN for MedMNIST classification.

    Args:
        data_info (dict): Dictionary containing 'num_classes' (int).
        width_mult (float): Multiplier for channel widths.
        activation (str): Activation function name from nn module.

    Returns:
        nn.Sequential: The MedMNIST CNN model.
    """
    num_classes = data_info["num_classes"]
    act = getattr(nn, activation)

    c1, c2 = [int(x * width_mult) for x in [6, 16]]
    d1, d2 = [int(x * width_mult) for x in [120, 84]]

    return nn.Sequential(
        nn.Conv2d(3, c1, kernel_size=5, padding=2),
        act(),
        nn.AvgPool2d(2, stride=2, padding=0),
        nn.Conv2d(c1, c2, kernel_size=5, padding=0),
        act(),
        nn.AvgPool2d(2, stride=2, padding=0),
        nn.Flatten(),
        # MedMNIST is 28x28.
        # After first pool: 14x14
        # After second conv (k=5, p=0): 10x10
        # After second pool: 5x5
        # 16 * 5 * 5 = 400
        nn.Linear(c2 * 5 * 5, d1),
        act(),
        nn.Linear(d1, d2),
        act(),
        nn.Linear(d2, num_classes),
    )
