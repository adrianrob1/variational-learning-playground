# Copyright (c) 2026 Adrian R. Minut
# 
# SPDX-License-Identifier: GPL-3.0

import numpy as np
import torch
from torch.utils.data import Dataset


class NPZDataset(Dataset):
    """
    A generic Dataset for loading data from .npz files or pre-loaded arrays.
    """

    def __init__(self, x_data, y_data, transform=None):
        """
        Initializes the NPZ dataset.

        Args:
            x_data (np.ndarray): Input data array.
            y_data (np.ndarray): Target label/value array.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.x = x_data
        self.y = y_data
        self.transform = transform

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.x)

    def __getitem__(self, idx):
        """
        Returns a single sample at the given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (input, target)
        """
        xi = self.x[idx]
        yi = self.y[idx]

        if self.transform:
            xi = self.transform(xi)

        return xi, yi
