# Copyright (c) 2026 Adrian R. Minut
# SPDX-License-Identifier: GPL-3.0

from unittest.mock import MagicMock, patch
from vldatasets.standard.dataloaders import (
    HFDatasetWrapper,
    get_hf_train_loaders,
    get_hf_test_loader,
)


class TestHFDatasets:
    """Verify that Hugging Face dataset integration works."""

    def test_hf_wrapper(self):
        """Test the HFDatasetWrapper with a mock dataset."""
        mock_ds = [
            {"img": "fake_image_1", "lbl": 0},
            {"img": "fake_image_2", "lbl": 1},
        ]

        # Mock convert method for images
        for item in mock_ds:
            item["img"] = MagicMock()
            item["img"].convert.return_value = "converted_image"

        transform = MagicMock(return_value="transformed_image")

        wrapper = HFDatasetWrapper(
            mock_ds, image_col="img", label_col="lbl", transform=transform
        )

        assert len(wrapper) == 2
        img, lbl = wrapper[0]
        assert img == "transformed_image"
        assert lbl == 0
        transform.assert_called_once_with("converted_image")

    @patch("datasets.load_dataset")
    def test_get_hf_train_loaders(self, mock_load):
        """Test get_hf_train_loaders with mocked datasets."""
        mock_ds = MagicMock()
        mock_ds.__len__.return_value = 10
        mock_ds.__getitem__.side_effect = lambda idx: {"image": MagicMock(), "label": 0}
        mock_load.return_value = mock_ds

        train_loader, val_loader = get_hf_train_loaders(
            data_dir="dummy",
            train_val_split=0.8,
            workers=0,
            pin_memory=False,
            tbatch=2,
            vbatch=2,
            path="dummy_path",
        )

        assert len(train_loader.dataset) == 8
        assert len(val_loader.dataset) == 2
        mock_load.assert_called_once_with(
            "dummy_path", None, split="train", cache_dir="dummy"
        )

    @patch("datasets.load_dataset")
    def test_get_hf_test_loader(self, mock_load):
        """Test get_hf_test_loader with mocked datasets."""
        mock_ds = MagicMock()
        mock_ds.__len__.return_value = 5
        mock_ds.__getitem__.side_effect = lambda idx: {"image": MagicMock(), "label": 1}
        mock_load.return_value = mock_ds

        test_loader = get_hf_test_loader(
            data_dir="dummy", workers=0, pin_memory=False, batch=2, path="dummy_path"
        )

        assert len(test_loader.dataset) == 5
        mock_load.assert_called_once_with(
            "dummy_path", None, split="test", cache_dir="dummy"
        )
