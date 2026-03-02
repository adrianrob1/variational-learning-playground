# Copyright (c) 2026 Adrian R. Minut
# SPDX-License-Identifier: GPL-3.0

import os
from PIL import Image
import tempfile
from vldatasets.standard.dataloaders import (
    get_imagenet_train_loader_torch,
    get_imagenet_test_loader_torch,
)


def create_mock_imagenet(root):
    """Creates a mock ImageNet directory structure with dummy images."""
    for split in ["train", "val"]:
        split_dir = os.path.join(root, split)
        # Create a few class directories
        for cls in ["n01440764", "n01440765"]:
            cls_dir = os.path.join(split_dir, cls)
            os.makedirs(cls_dir, exist_ok=True)
            # Create a dummy image
            img = Image.new("RGB", (256, 256), color="red")
            img.save(os.path.join(cls_dir, "dummy.jpg"))


def test_imagenet_standard_loaders():
    """Verify that the standard PyTorch ImageNet loaders work with raw images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        create_mock_imagenet(tmpdir)

        # Test train loader (non-distributed)
        train_loader = get_imagenet_train_loader_torch(
            tmpdir, workers=0, tbatch=2, pin_memory=False, distributed=False
        )
        assert len(train_loader) > 0
        images, targets = next(iter(train_loader))
        assert images.shape == (2, 3, 224, 224)
        assert targets.shape == (2,)

        # Test test loader (non-distributed)
        test_loader = get_imagenet_test_loader_torch(
            tmpdir, workers=0, batch=2, pin_memory=False, distributed=False
        )
        assert len(test_loader) > 0
        images, targets = next(iter(test_loader))
        assert images.shape == (2, 3, 224, 224)
        assert targets.shape == (2,)


if __name__ == "__main__":
    test_imagenet_standard_loaders()
    print("ImageNet standard loader test passed!")
