# Copyright (c) 2026 Adrian R. Minut
# Copyright (c) 2024 ABI Team
#
# SPDX-License-Identifier: GPL-3.0

"""Dataset loaders: CIFAR-10/100, TinyImageNet, SVHN, ImageNet (FFCV).

Refactored from tasks/common/dataloaders.py — relative imports updated.
"""

from typing import Tuple, Any, Callable, Optional
from os.path import join as pjoin
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from .tinyimagenet import TinyImageNet
from .cifar10c import get_cifar10c_loader


# autobatch collate function for dataloader to duplicate batch
def dup_collate_fn(dups: int):
    """Return a collate_fn that duplicates each batch `dups` times along dim 0.

    Args:
        dups: Number of duplications.

    Returns:
        A collate_fn callable compatible with DataLoader.
    """

    def collate_fn(data):
        imgs, gts = tuple(zip(*data))
        t = torch.stack(imgs, dim=0)
        return t.repeat(dups, *(1,) * (t.ndim - 1)), torch.as_tensor(gts)

    return collate_fn


class CIFAR10Info:
    outclass = 10
    imgshape = (3, 32, 32)
    counts = {"train": 50000, "test": 10000}
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)


def get_cifar10_train_loaders(
    data_dir: str,
    train_val_split: float,
    workers: int,
    pin_memory: bool,
    tbatch: int,
    vbatch: int,
    tdups: int = 1,
    vdups: int = 1,
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) for CIFAR-10."""
    cifar10_dir = pjoin(data_dir, "cifar10")
    normalize = transforms.Normalize(mean=CIFAR10Info.mean, std=CIFAR10Info.std)
    train_data = datasets.CIFAR10(
        root=cifar10_dir,
        train=True,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        download=True,
    )
    val_data = datasets.CIFAR10(
        root=cifar10_dir,
        train=True,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
        download=True,
    )
    nb_train = int(len(train_data) * train_val_split)
    train_indices = list(range(nb_train))
    val_indices = list(range(nb_train, len(train_data)))
    train_loader = (
        DataLoader(
            Subset(train_data, train_indices),
            batch_size=tbatch,
            num_workers=workers,
            pin_memory=pin_memory,
            shuffle=True,
            collate_fn=dup_collate_fn(tdups),
        )
        if tdups > 1
        else DataLoader(
            Subset(train_data, train_indices),
            batch_size=tbatch,
            num_workers=workers,
            pin_memory=pin_memory,
            shuffle=True,
        )
    )
    val_loader = (
        DataLoader(
            Subset(val_data, val_indices),
            batch_size=vbatch,
            num_workers=workers,
            pin_memory=pin_memory,
            shuffle=False,
            collate_fn=dup_collate_fn(vdups),
        )
        if vdups > 1
        else DataLoader(
            Subset(val_data, val_indices),
            batch_size=vbatch,
            num_workers=workers,
            pin_memory=pin_memory,
            shuffle=False,
        )
    )
    return train_loader, val_loader


def get_cifar10_test_loader(
    data_dir: str, workers: int, pin_memory: bool, batch: int, dups: int = 1
) -> DataLoader:
    """Return test DataLoader for CIFAR-10."""
    cifar10_dir = pjoin(data_dir, "cifar10")
    normalize = transforms.Normalize(mean=CIFAR10Info.mean, std=CIFAR10Info.std)
    test_data = datasets.CIFAR10(
        root=cifar10_dir,
        train=False,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
        download=True,
    )
    return (
        DataLoader(
            test_data,
            batch_size=batch,
            num_workers=workers,
            shuffle=False,
            pin_memory=pin_memory,
            collate_fn=dup_collate_fn(dups),
        )
        if dups > 1
        else DataLoader(
            test_data,
            batch_size=batch,
            num_workers=workers,
            shuffle=False,
            pin_memory=pin_memory,
        )
    )


class CIFAR100Info:
    outclass = 100
    imgshape = (3, 32, 32)
    counts = {"train": 50000, "test": 10000}
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)


def get_cifar100_train_loaders(
    data_dir: str,
    train_val_split: float,
    workers: int,
    pin_memory: bool,
    tbatch: int,
    vbatch: int,
    tdups: int = 1,
    vdups: int = 1,
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) for CIFAR-100."""
    cifar100_dir = pjoin(data_dir, "cifar100")
    normalize = transforms.Normalize(mean=CIFAR100Info.mean, std=CIFAR100Info.std)
    train_data = datasets.CIFAR100(
        root=cifar100_dir,
        train=True,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        download=True,
    )
    val_data = datasets.CIFAR100(
        root=cifar100_dir,
        train=True,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
        download=True,
    )
    nb_train = int(len(train_data) * train_val_split)
    train_indices = list(range(nb_train))
    val_indices = list(range(nb_train, len(train_data)))
    train_loader = (
        DataLoader(
            Subset(train_data, train_indices),
            batch_size=tbatch,
            num_workers=workers,
            pin_memory=pin_memory,
            shuffle=True,
            collate_fn=dup_collate_fn(tdups),
        )
        if tdups > 1
        else DataLoader(
            Subset(train_data, train_indices),
            batch_size=tbatch,
            num_workers=workers,
            pin_memory=pin_memory,
            shuffle=True,
        )
    )
    val_loader = (
        DataLoader(
            Subset(val_data, val_indices),
            batch_size=vbatch,
            num_workers=workers,
            pin_memory=pin_memory,
            shuffle=False,
            collate_fn=dup_collate_fn(vdups),
        )
        if vdups > 1
        else DataLoader(
            Subset(val_data, val_indices),
            batch_size=vbatch,
            num_workers=workers,
            pin_memory=pin_memory,
            shuffle=False,
        )
    )
    return train_loader, val_loader


def get_cifar100_test_loader(
    data_dir: str, workers: int, pin_memory: bool, batch: int, dups: int = 1
) -> DataLoader:
    """Return test DataLoader for CIFAR-100."""
    cifar100_dir = pjoin(data_dir, "cifar100")
    normalize = transforms.Normalize(mean=CIFAR100Info.mean, std=CIFAR100Info.std)
    test_data = datasets.CIFAR100(
        root=cifar100_dir,
        train=False,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
        download=True,
    )
    return (
        DataLoader(
            test_data,
            batch_size=batch,
            num_workers=workers,
            shuffle=False,
            pin_memory=pin_memory,
            collate_fn=dup_collate_fn(dups),
        )
        if dups > 1
        else DataLoader(
            test_data,
            batch_size=batch,
            num_workers=workers,
            shuffle=False,
            pin_memory=pin_memory,
        )
    )


class SVHNInfo:
    outclass = 10
    imgshape = (3, 32, 32)
    split = ("train", "test", "extra")
    counts = {"train": 73257, "test": 26032, "extra": 531131}
    mean = (0.4376821, 0.4437697, 0.47280442)
    std = (0.19803012, 0.20101562, 0.19703614)


def get_svhn_train_loaders(
    data_dir: str,
    train_val_split: float,
    workers: int,
    pin_memory: bool,
    tbatch: int,
    vbatch: int,
    tdups: int = 1,
    vdups: int = 1,
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) for SVHN."""
    svhn_dir = pjoin(data_dir, "svhn")
    normalize = transforms.Normalize(SVHNInfo.mean, SVHNInfo.std)
    train_data = datasets.SVHN(
        root=svhn_dir,
        split="train",
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        download=True,
    )
    val_data = datasets.SVHN(
        root=svhn_dir,
        split="train",
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
        download=True,
    )
    nb_train = int(len(train_data) * train_val_split)
    train_indices = list(range(nb_train))
    val_indices = list(range(nb_train, len(train_data)))
    train_loader = (
        DataLoader(
            Subset(train_data, train_indices),
            batch_size=tbatch,
            num_workers=workers,
            pin_memory=pin_memory,
            shuffle=True,
            collate_fn=dup_collate_fn(tdups),
        )
        if tdups > 1
        else DataLoader(
            Subset(train_data, train_indices),
            batch_size=tbatch,
            num_workers=workers,
            pin_memory=pin_memory,
            shuffle=True,
        )
    )
    val_loader = (
        DataLoader(
            Subset(val_data, val_indices),
            batch_size=vbatch,
            num_workers=workers,
            pin_memory=pin_memory,
            shuffle=False,
            collate_fn=dup_collate_fn(vdups),
        )
        if vdups > 1
        else DataLoader(
            Subset(val_data, val_indices),
            batch_size=vbatch,
            num_workers=workers,
            pin_memory=pin_memory,
            shuffle=False,
        )
    )
    return train_loader, val_loader


def get_svhn_test_loader(
    data_dir: str, workers: int, pin_memory: bool, batch: int, dups: int = 1
) -> DataLoader:
    """Return test DataLoader for SVHN."""
    svhn_dir = pjoin(data_dir, "svhn")
    normalize = transforms.Normalize(SVHNInfo.mean, SVHNInfo.std)
    test_data = datasets.SVHN(
        root=svhn_dir,
        split="test",
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
        download=True,
    )
    return (
        DataLoader(
            test_data,
            batch_size=batch,
            num_workers=workers,
            shuffle=False,
            pin_memory=pin_memory,
            collate_fn=dup_collate_fn(dups),
        )
        if dups > 1
        else DataLoader(
            test_data,
            batch_size=batch,
            num_workers=workers,
            shuffle=False,
            pin_memory=pin_memory,
        )
    )


class TinyImageNetInfo:
    outclass = 200
    imgshape = (3, 64, 64)
    counts = {"train": 100000, "test": 10000}
    mean = (0.48024865984916687, 0.4480723738670349, 0.3975464701652527)
    std = (0.23022247850894928, 0.22650277614593506, 0.2261698693037033)


def get_tinyimagenet_train_loaders(
    data_dir: str,
    train_val_split: float,
    workers: int,
    pin_memory: bool,
    tbatch: int,
    vbatch: int,
    tdups: int = 1,
    vdups: int = 1,
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) for TinyImageNet."""
    tinyimagenet_dir = pjoin(data_dir, "tinyimagenet")
    normalize = transforms.Normalize(
        mean=TinyImageNetInfo.mean, std=TinyImageNetInfo.std
    )
    train_data = TinyImageNet(
        root=tinyimagenet_dir,
        train=True,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(64, 8),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        download=True,
    )
    val_data = TinyImageNet(
        root=tinyimagenet_dir,
        train=True,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
        download=True,
    )
    nb_train = int(len(train_data) * train_val_split)
    train_indices = list(range(nb_train))
    val_indices = list(range(nb_train, len(train_data)))
    train_loader = (
        DataLoader(
            Subset(train_data, train_indices),
            batch_size=tbatch,
            num_workers=workers,
            pin_memory=pin_memory,
            shuffle=True,
            collate_fn=dup_collate_fn(tdups),
        )
        if tdups > 1
        else DataLoader(
            Subset(train_data, train_indices),
            batch_size=tbatch,
            num_workers=workers,
            pin_memory=pin_memory,
            shuffle=True,
        )
    )
    val_loader = (
        DataLoader(
            Subset(val_data, val_indices),
            batch_size=vbatch,
            num_workers=workers,
            pin_memory=pin_memory,
            shuffle=False,
            collate_fn=dup_collate_fn(vdups),
        )
        if vdups > 1
        else DataLoader(
            Subset(val_data, val_indices),
            batch_size=vbatch,
            num_workers=workers,
            pin_memory=pin_memory,
            shuffle=False,
        )
    )
    return train_loader, val_loader


def get_tinyimagenet_test_loader(
    data_dir: str, workers: int, pin_memory: bool, batch: int, dups: int = 1
) -> DataLoader:
    """Return test DataLoader for TinyImageNet."""
    tinyimagenet_dir = pjoin(data_dir, "tinyimagenet")
    normalize = transforms.Normalize(
        mean=TinyImageNetInfo.mean, std=TinyImageNetInfo.std
    )
    test_data = TinyImageNet(
        root=tinyimagenet_dir,
        train=False,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
        download=True,
    )
    return (
        DataLoader(
            test_data,
            batch_size=batch,
            num_workers=workers,
            shuffle=False,
            pin_memory=pin_memory,
            collate_fn=dup_collate_fn(dups),
        )
        if dups > 1
        else DataLoader(
            test_data,
            batch_size=batch,
            num_workers=workers,
            shuffle=False,
            pin_memory=pin_memory,
        )
    )


class DatasetConfig:
    """Hydra-instantiable dataset metadata and loader provider."""

    def __init__(
        self,
        name: str,
        outclass: int,
        ntrain: int,
        ntest: int,
        insize: int,
        train_loader_f: Optional[Callable] = None,
        test_loader_f: Optional[Callable] = None,
    ):
        self.name = name
        self.outclass = outclass
        self.ntrain = ntrain
        self.ntest = ntest
        self.insize = insize
        self.train_loader_f = train_loader_f
        self.test_loader_f = test_loader_f

    def get_train_loaders(
        self,
        data_dir: str,
        train_val_split: float,
        workers: int,
        pin_memory: bool,
        tbatch: int,
        vbatch: int,
        **kwargs,
    ) -> Tuple[DataLoader, DataLoader]:
        """Return (train_loader, val_loader)."""
        if self.train_loader_f:
            return self.train_loader_f(
                data_dir=data_dir,
                train_val_split=train_val_split,
                workers=workers,
                pin_memory=pin_memory,
                tbatch=tbatch,
                vbatch=vbatch,
                **kwargs,
            )
        # Fallback to registry
        return TRAINDATALOADERS[self.name](
            data_dir, train_val_split, workers, pin_memory, tbatch, vbatch, **kwargs
        )

    def get_test_loader(
        self, data_dir: str, workers: int, pin_memory: bool, batch: int, **kwargs
    ) -> DataLoader:
        """Return test DataLoader."""
        if self.test_loader_f:
            return self.test_loader_f(
                data_dir=data_dir,
                workers=workers,
                pin_memory=pin_memory,
                batch=batch,
                **kwargs,
            )
        # Fallback to registry
        return TESTDATALOADER[self.name](data_dir, workers, pin_memory, batch, **kwargs)


# available datasets and corresponding train/val loaders
TRAINDATALOADERS = {
    "cifar10": get_cifar10_train_loaders,
    "cifar100": get_cifar100_train_loaders,
    "tinyimagenet": get_tinyimagenet_train_loaders,
    "cifar10c": None,
    "cifar100c": None,
    "imagenet": None,
}
# available datasets and corresponding test loader
TESTDATALOADER = {
    "cifar10": get_cifar10_test_loader,
    "cifar100": get_cifar100_test_loader,
    "tinyimagenet": get_tinyimagenet_test_loader,
    "cifar10c": get_cifar10c_loader,
    "cifar100c": get_cifar10c_loader,
    "imagenet": None,  # handled specially in scripts
}


class ImageNetInfo:
    """Metadata for the ImageNet-1k dataset."""

    outclass = 1000
    imgshape = (3, 224, 224)
    counts = {"train": 1281167, "test": 50000}
    mean = np.array([0.485, 0.456, 0.406]) * 255
    std = np.array([0.229, 0.224, 0.225]) * 255
    # Standard torch normalization (0-1)
    mean_torch = (0.485, 0.456, 0.406)
    std_torch = (0.229, 0.224, 0.225)


# number of training data
NTRAIN = {
    "cifar10": CIFAR10Info.counts["train"],
    "cifar100": CIFAR100Info.counts["train"],
    "tinyimagenet": TinyImageNetInfo.counts["train"],
    "cifar10c": 50000,
    "cifar100c": 50000,
    "imagenet": ImageNetInfo.counts["train"],
}
# number of test data
NTEST = {
    "cifar10": CIFAR10Info.counts["test"],
    "cifar100": CIFAR100Info.counts["test"],
    "tinyimagenet": TinyImageNetInfo.counts["test"],
    "cifar10c": 10000,
    "cifar100c": 10000,
    "imagenet": ImageNetInfo.counts["test"],
}


# input image size
INSIZE = {
    "cifar10": CIFAR10Info.imgshape[-1],
    "cifar100": CIFAR100Info.imgshape[-1],
    "tinyimagenet": TinyImageNetInfo.imgshape[-1],
    "imagenet": ImageNetInfo.imgshape[-1],
    "cifar10c": CIFAR10Info.imgshape[-1],
    "cifar100c": CIFAR100Info.imgshape[-1],
}


# number of classes
OUTCLASS = {
    "cifar10": CIFAR10Info.outclass,
    "cifar100": CIFAR100Info.outclass,
    "tinyimagenet": TinyImageNetInfo.outclass,
    "imagenet": ImageNetInfo.outclass,
    "cifar10c": CIFAR10Info.outclass,
    "cifar100c": CIFAR100Info.outclass,
}


def get_imagenet_train_loader(
    imagenet_dir: str,
    workers: int,
    tbatch: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    distributed: bool = True,
    noaugment: bool = False,
    shuffle: bool = True,
) -> Any:
    """Return an FFCV Loader for ImageNet training.

    Args:
        imagenet_dir: Path to the directory containing 'train.ffcv'.
        workers: Number of data loading workers.
        tbatch: Batch size per worker.
        device: Torch device to load onto.
        dtype: Image data type.
        distributed: Whether to use distributed data loading.
        noaugment: If True, disable data augmentation.
        shuffle: If True, shuffle the training data.

    Returns:
        An ffcv.loader.Loader instance.
    """
    from ffcv.loader import Loader, OrderOption
    from ffcv.transforms import (
        ToTensor,
        ToDevice,
        ToTorchImage,
        RandomHorizontalFlip,
        NormalizeImage,
        Squeeze,
    )
    from ffcv.fields.decoders import (
        IntDecoder,
        RandomResizedCropRGBImageDecoder,
        CenterCropRGBImageDecoder,
    )

    if noaugment:
        cropper = CenterCropRGBImageDecoder((224, 224), ratio=224.0 / 256.0)
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(device, non_blocking=True),
            ToTorchImage(),
            NormalizeImage(
                np.asarray(ImageNetInfo.mean),
                np.asarray(ImageNetInfo.std),
                np.float32 if dtype == torch.float32 else dtype,
            ),
        ]
        droplast = False
    else:
        # Random resized crop
        decoder = RandomResizedCropRGBImageDecoder((224, 224))

        # Data decoding and augmentation
        image_pipeline = [
            decoder,
            RandomHorizontalFlip(),
            ToTensor(),
            ToDevice(device, non_blocking=True),
            ToTorchImage(),
            NormalizeImage(
                np.asarray(ImageNetInfo.mean),
                np.asarray(ImageNetInfo.std),
                np.float32 if dtype == torch.float32 else dtype,
            ),
        ]
        droplast = True

    label_pipeline = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(device, non_blocking=True),
    ]

    # Pipeline for each data field
    pipelines = {"image": image_pipeline, "label": label_pipeline}

    loader = Loader(
        pjoin(imagenet_dir, "train.ffcv"),
        batch_size=tbatch,
        num_workers=workers,
        order=OrderOption.RANDOM if shuffle else OrderOption.SEQUENTIAL,
        os_cache=True,
        drop_last=droplast,
        pipelines=pipelines,
        distributed=distributed,
    )

    return loader


def get_imagenet_test_loader(
    imagenet_dir: str,
    workers: int,
    batch: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    distributed: bool = True,
) -> Any:
    """Return an FFCV Loader for ImageNet validation/test.

    Args:
        imagenet_dir: Path to the directory containing 'val.ffcv'.
        workers: Number of data loading workers.
        batch: Batch size per worker.
        device: Torch device to load onto.
        dtype: Image data type.
        distributed: Whether to use distributed data loading.

    Returns:
        An ffcv.loader.Loader instance.
    """
    from ffcv.loader import Loader, OrderOption
    from ffcv.transforms import (
        ToTensor,
        ToDevice,
        ToTorchImage,
        NormalizeImage,
        Squeeze,
    )
    from ffcv.fields.decoders import (
        IntDecoder,
        CenterCropRGBImageDecoder,
    )

    cropper = CenterCropRGBImageDecoder((224, 224), ratio=224.0 / 256.0)
    image_pipeline = [
        cropper,
        ToTensor(),
        ToDevice(device, non_blocking=True),
        ToTorchImage(),
        NormalizeImage(
            np.asarray(ImageNetInfo.mean),
            np.asarray(ImageNetInfo.std),
            np.float32 if dtype == torch.float32 else dtype,
        ),
    ]

    label_pipeline = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(device, non_blocking=True),
    ]

    loader = Loader(
        pjoin(imagenet_dir, "val.ffcv"),
        batch_size=batch,
        num_workers=workers,
        order=OrderOption.SEQUENTIAL,
        drop_last=False,
        pipelines={"image": image_pipeline, "label": label_pipeline},
        distributed=distributed,
    )
    return loader


def get_imagenet_train_loader_torch(
    imagenet_dir: str,
    workers: int,
    tbatch: int,
    pin_memory: bool = True,
    distributed: bool = True,
) -> DataLoader:
    """Return a standard PyTorch DataLoader for ImageNet training."""
    traindir = pjoin(imagenet_dir, "train")
    normalize = transforms.Normalize(
        mean=ImageNetInfo.mean_torch, std=ImageNetInfo.std_torch
    )
    dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    sampler = (
        torch.utils.data.distributed.DistributedSampler(dataset)
        if distributed
        else None
    )
    return DataLoader(
        dataset,
        batch_size=tbatch,
        shuffle=(sampler is None),
        num_workers=workers,
        pin_memory=pin_memory,
        sampler=sampler,
    )


def get_imagenet_test_loader_torch(
    imagenet_dir: str,
    workers: int,
    batch: int,
    pin_memory: bool = True,
    distributed: bool = True,
) -> DataLoader:
    """Return a standard PyTorch DataLoader for ImageNet validation."""
    valdir = pjoin(imagenet_dir, "val")
    normalize = transforms.Normalize(
        mean=ImageNetInfo.mean_torch, std=ImageNetInfo.std_torch
    )
    dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    sampler = (
        torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
        if distributed
        else None
    )
    return DataLoader(
        dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        sampler=sampler,
    )


class HFDatasetWrapper(torch.utils.data.Dataset):
    """Wrapper for Hugging Face datasets to work with PyTorch DataLoaders."""

    def __init__(self, hf_ds, image_col="image", label_col="label", transform=None):
        self.hf_ds = hf_ds
        self.image_col = image_col
        self.label_col = label_col
        self.transform = transform

    def __len__(self):
        return len(self.hf_ds)

    def __getitem__(self, idx):
        item = self.hf_ds[idx]
        image = item[self.image_col]
        # Ensure image is in RGB if it's a PIL image
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = item[self.label_col]
        return image, label


def get_hf_train_loaders(
    data_dir: str,
    train_val_split: float,
    workers: int,
    pin_memory: bool,
    tbatch: int,
    vbatch: int,
    path: str,
    hf_name: Optional[str] = None,
    image_col: str = "image",
    label_col: str = "label",
    train_split: str = "train",
    transform: Optional[Any] = None,
    tdups: int = 1,
    vdups: int = 1,
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) for a Hugging Face dataset."""
    from datasets import load_dataset

    ds = load_dataset(path, hf_name, split=train_split, cache_dir=data_dir)

    if transform is None:
        transform = transforms.ToTensor()

    full_data = HFDatasetWrapper(ds, image_col, label_col, transform)
    nb_train = int(len(full_data) * train_val_split)
    train_indices = list(range(nb_train))
    val_indices = list(range(nb_train, len(full_data)))

    train_loader = DataLoader(
        Subset(full_data, train_indices),
        batch_size=tbatch,
        num_workers=workers,
        pin_memory=pin_memory,
        shuffle=True,
        collate_fn=dup_collate_fn(tdups) if tdups > 1 else None,
    )
    val_loader = DataLoader(
        Subset(full_data, val_indices),
        batch_size=vbatch,
        num_workers=workers,
        pin_memory=pin_memory,
        shuffle=False,
        collate_fn=dup_collate_fn(vdups) if vdups > 1 else None,
    )
    return train_loader, val_loader


def get_hf_test_loader(
    data_dir: str,
    workers: int,
    pin_memory: bool,
    batch: int,
    path: str,
    hf_name: Optional[str] = None,
    image_col: str = "image",
    label_col: str = "label",
    test_split: str = "test",
    transform: Optional[Any] = None,
    dups: int = 1,
) -> DataLoader:
    """Return test DataLoader for a Hugging Face dataset."""
    from datasets import load_dataset

    ds = load_dataset(path, hf_name, split=test_split, cache_dir=data_dir)

    if transform is None:
        transform = transforms.ToTensor()

    test_data = HFDatasetWrapper(ds, image_col, label_col, transform)
    return DataLoader(
        test_data,
        batch_size=batch,
        num_workers=workers,
        pin_memory=pin_memory,
        shuffle=False,
        collate_fn=dup_collate_fn(dups) if dups > 1 else None,
    )
