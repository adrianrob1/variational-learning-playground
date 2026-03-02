# Copyright (c) 2026 Adrian R. Minut
# Copyright (c) 2024 ABI Team
# 
# SPDX-License-Identifier: GPL-3.0

import numpy as np
import torch
from torch.utils.data import Subset, Dataset
from typing import List


def dirichlet_partition(
    targets: np.ndarray,
    num_clients: int,
    num_classes: int,
    alpha1: float = 1e6,
    alpha2: float = 0.5,
    dataset_proportion: float = 1.0,
    seed: int = 42,
) -> List[np.ndarray]:
    """Partition dataset indices among clients using Dirichlet distribution.

    Args:
        targets: Array of labels for each datapoint.
        num_clients: Number of clients.
        num_classes: Number of distinct classes.
        alpha1: Dirichlet parameter for distribution of total points per client.
                High values (>1000) lead to equal total points per client (IID-like total count).
        alpha2: Dirichlet parameter for class distribution within each client.
                High values lead to IID-like class distribution. Low values lead to heterogeneous classes.
        dataset_proportion: Fraction of the dataset to use.
        seed: Random seed.

    Returns:
        List of numpy arrays, where each array contains the indices for one client.
    """
    np.random.seed(seed)

    # 1. Determine how many points per client per class
    # client_splits: Proportion of total data each client gets
    client_splits = np.random.dirichlet([alpha1 for _ in range(num_clients)])

    # client_class_splits: [num_clients, num_classes] matrix.
    # Each row is a Dirichlet sample for that client's class distribution.
    client_class_splits = np.random.dirichlet(
        [alpha2 for _ in range(num_classes)], num_clients
    )

    # Combined proportion: [num_clients, num_classes]
    # Each entry (i, j) is the probability that a point of class j belongs to client i.
    # We need to normalize by class so we don't exceed available samples per class.
    combined_props = client_splits[:, np.newaxis] * client_class_splits
    class_probs = combined_props / combined_props.sum(axis=0)  # Normalize per class

    client_indices = [[] for _ in range(num_clients)]

    for class_idx in range(num_classes):
        indices_in_class = np.where(targets == class_idx)[0]
        num_in_class = int(len(indices_in_class) * dataset_proportion)
        np.random.shuffle(indices_in_class)
        indices_in_class = indices_in_class[:num_in_class]

        # Split indices_in_class among clients based on class_probs[:, class_idx]
        probs = class_probs[:, class_idx]
        # Calculate split points
        counts = np.round(probs * num_in_class).astype(int)

        # Adjust for rounding errors to match exactly num_in_class
        diff = num_in_class - counts.sum()
        if diff != 0:
            # Add/subtract from the client with the largest share to minimize relative impact
            counts[np.argmax(counts)] += diff

        # Assign indices
        offset = 0
        for client_idx in range(num_clients):
            c = counts[client_idx]
            if c > 0:
                client_indices[client_idx].extend(indices_in_class[offset : offset + c])
                offset += c

    return [np.array(idx, dtype=np.int64) for idx in client_indices]


class PartitionedDataset:
    """A wrapper for a dataset that provides subsets for each client."""

    def __init__(
        self,
        dataset: Dataset,
        num_clients: int,
        num_classes: int,
        alpha1: float = 1e6,
        alpha2: float = 0.5,
        dataset_proportion: float = 1.0,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.num_clients = num_clients

        # Try to find targets
        if hasattr(dataset, "targets"):
            t = dataset.targets
            targets = t.numpy() if torch.is_tensor(t) else np.array(t)
        elif hasattr(dataset, "labels"):
            t = dataset.labels
            targets = t.numpy() if torch.is_tensor(t) else np.array(t)
        else:
            # Fallback for TensorDataset or similar
            # This might be slow if we have to iterate
            raise ValueError("Dataset must have 'targets' or 'labels' attribute.")

        self.client_indices = dirichlet_partition(
            targets, num_clients, num_classes, alpha1, alpha2, dataset_proportion, seed
        )

    def get_client_dataset(self, client_idx: int) -> Subset:
        return Subset(self.dataset, self.client_indices[client_idx])

    def __len__(self):
        return self.num_clients
