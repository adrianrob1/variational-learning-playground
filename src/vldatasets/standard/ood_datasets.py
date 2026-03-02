# Copyright (c) 2026 Adrian R. Minut
# Copyright (c) 2024 ABI Team
# 
# SPDX-License-Identifier: GPL-3.0


"""OOD dataset loaders: SVHN and Flowers102.

Provides dataset metadata classes and DataLoader factories for the two
out-of-distribution datasets used in the bdl_ood benchmark (SVHN and
Flowers102).  Images are resized to 32×32 to match the CIFAR-10 in-domain
resolution.

Typical usage::

    from vldatasets.standard.ood_datasets import get_svhn_loader, Flowers102Info

    loader = get_svhn_loader(
        data_dir="data/", workers=4, pin_memory=True,
        batch=512, split="test"
    )
"""

from __future__ import annotations

from os.path import join as pjoin

from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms


from .dataloaders import dup_collate_fn


# ---------------------------------------------------------------------------
# SVHN
# ---------------------------------------------------------------------------


class SVHNInfo:
    """Metadata constants for the Street View House Numbers (SVHN) dataset.

    Attributes:
        outclass: Number of target classes (10 digits).
        split: Valid split names accepted by ``torchvision.datasets.SVHN``.
        count: Number of images per split.
        mean: Per-channel mean used for normalisation.
        std: Per-channel standard deviation used for normalisation.
    """

    outclass = 10
    split = ("train", "test", "extra")
    count = {"train": 73257, "test": 26032, "extra": 531131}
    mean = (0.4376821, 0.4437697, 0.47280442)
    std = (0.19803012, 0.20101562, 0.19703614)


def get_svhn_loader(
    data_dir: str,
    workers: int,
    pin_memory: bool,
    batch: int,
    split: str = "test",
    dups: int = 1,
) -> DataLoader:
    """Build a DataLoader for the SVHN dataset.

    Downloads the dataset to ``<data_dir>/svhn/`` on first call.

    Args:
        data_dir: Root directory under which ``svhn/`` is stored.
        workers: Number of DataLoader worker processes.
        pin_memory: Whether to use pinned (page-locked) memory.
        batch: Batch size.
        split: One of ``SVHNInfo.split`` — ``"train"``, ``"test"``,
            or ``"extra"``.
        dups: If greater than 1, each batch is duplicated *dups* times
            using ``dup_collate_fn``.

    Returns:
        A ``torch.utils.data.DataLoader`` over the requested SVHN split,
        normalised to match the CIFAR-10 32×32 pipeline.

    Raises:
        AssertionError: If *split* is not in ``SVHNInfo.split``.
    """
    assert split in SVHNInfo.split, f"Invalid SVHN split: {split!r}"
    svhn_dir = pjoin(data_dir, "svhn")
    normalize = transforms.Normalize(SVHNInfo.mean, SVHNInfo.std)
    dataset = datasets.SVHN(
        root=svhn_dir,
        split=split,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
    )
    common_kw = dict(batch_size=batch, num_workers=workers, pin_memory=pin_memory)
    if dups > 1:
        return DataLoader(dataset, collate_fn=dup_collate_fn(dups), **common_kw)
    return DataLoader(dataset, **common_kw)


# ---------------------------------------------------------------------------
# Flowers102
# ---------------------------------------------------------------------------


class Flowers102Info:
    """Metadata constants for the Oxford 102 Flower Categories dataset.

    Attributes:
        outclass: Number of flower categories (102).
        split: Valid split names.
        count: Number of images per split.
        mean: Per-channel mean used for normalisation.
        std: Per-channel standard deviation used for normalisation.
    """

    outclass = 102
    split = ("train", "val", "test")
    count = {"train": 1020, "val": 1020, "test": 6149}
    mean = (0.50390434, 0.4516826, 0.494936)
    std = (0.23261614, 0.20974728, 0.2668646)


def get_flowers102_loader(
    data_dir: str,
    workers: int,
    pin_memory: bool,
    batch: int,
    split: str = "test",
    dups: int = 1,
) -> DataLoader:
    """Build a DataLoader for the Flowers102 dataset (resized to 32×32).

    Downloads the dataset to ``<data_dir>/flowers102/`` on first call.

    Args:
        data_dir: Root directory under which ``flowers102/`` is stored.
        workers: Number of DataLoader worker processes.
        pin_memory: Whether to use pinned (page-locked) memory.
        batch: Batch size.
        split: One of ``Flowers102Info.split`` — ``"train"``, ``"val"``,
            or ``"test"``.
        dups: If greater than 1, each batch is duplicated *dups* times
            using ``dup_collate_fn``.

    Returns:
        A ``torch.utils.data.DataLoader`` over the requested Flowers102 split,
        images centre-cropped and resized to 32×32.

    Raises:
        AssertionError: If *split* is not in ``Flowers102Info.split``.
    """
    assert split in Flowers102Info.split, f"Invalid Flowers102 split: {split!r}"
    flowers_dir = pjoin(data_dir, "flowers102")
    normalize = transforms.Normalize(Flowers102Info.mean, Flowers102Info.std)
    dataset = datasets.Flowers102(
        root=flowers_dir,
        split=split,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(32),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    common_kw = dict(batch_size=batch, num_workers=workers, pin_memory=pin_memory)
    if dups > 1:
        return DataLoader(dataset, collate_fn=dup_collate_fn(dups), **common_kw)
    return DataLoader(dataset, **common_kw)
