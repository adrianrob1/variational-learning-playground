# Copyright (c) 2026 Adrian R. Minut
# Copyright (c) 2026 ABI Team
# SPDX-License-Identifier: GPL-3.0

"""Script to prepare ImageNet dataset in FFCV format."""

import os
import argparse
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField
from torchvision.datasets import ImageFolder
from vlbench.train.utils import mkdirp


def prepare_ffcv(imagenet_dir: str, ffcv_dir: str):
    """Convert ImageNet ImageFolder to FFCV files.

    Args:
        imagenet_dir: Path to raw ImageNet dataset (with train/val subdirs).
        ffcv_dir: Directory to save the resulting .ffcv files.
    """
    imagenet_traindir = os.path.join(imagenet_dir, "train")
    imagenet_valdir = os.path.join(imagenet_dir, "val")

    train_dataset = ImageFolder(imagenet_traindir)
    val_dataset = ImageFolder(imagenet_valdir)

    def write_dataset(write_path, dataset):
        writer = DatasetWriter(
            write_path,
            {
                "image": RGBImageField(
                    write_mode="proportion",
                    max_resolution=500,
                    compress_probability=0.50,
                    jpeg_quality=90,
                ),
                "label": IntField(),
            },
            num_workers=16,
        )
        writer.from_indexed_dataset(dataset, chunksize=100)

    mkdirp(ffcv_dir)
    write_dataset(os.path.join(ffcv_dir, "train.ffcv"), train_dataset)
    write_dataset(os.path.join(ffcv_dir, "val.ffcv"), val_dataset)


def main():
    """CLI entry point for FFCV preparation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("imagenetdir", type=str, help="Path to ImageNet dataset")
    parser.add_argument(
        "ffcvdir", type=str, help="Save directory for ffcv format of ImageNet"
    )
    args = parser.parse_args()
    prepare_ffcv(args.imagenetdir, args.ffcvdir)


if __name__ == "__main__":
    main()
