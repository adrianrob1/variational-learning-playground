# Copyright (c) 2026 Adrian R. Minut
# SPDX-License-Identifier: GPL-3.0

"""Tests for vldatasets.standard.dataloaders — registry consistency and collate_fn."""

import pytest
import torch
from vldatasets.standard.dataloaders import (
    TRAINDATALOADERS,
    TESTDATALOADER,
    NTRAIN,
    NTEST,
    INSIZE,
    OUTCLASS,
    dup_collate_fn,
)


class TestRegistryConsistency:
    """All dataset dictionaries must share the same key set."""

    @pytest.fixture
    def dataset_keys(self):
        return set(TRAINDATALOADERS.keys())

    def test_testdataloader_same_keys(self, dataset_keys):
        assert set(TESTDATALOADER.keys()) == dataset_keys

    def test_ntrain_same_keys(self, dataset_keys):
        assert set(NTRAIN.keys()) == dataset_keys

    def test_ntest_same_keys(self, dataset_keys):
        assert set(NTEST.keys()) == dataset_keys

    def test_insize_same_keys(self, dataset_keys):
        assert set(INSIZE.keys()) == dataset_keys

    def test_outclass_same_keys(self, dataset_keys):
        assert set(OUTCLASS.keys()) == dataset_keys

    def test_expected_datasets_present(self, dataset_keys):
        for expected in ("cifar10", "cifar100", "tinyimagenet"):
            assert expected in dataset_keys

    def test_outclass_values_positive(self):
        for k, v in OUTCLASS.items():
            assert v > 0, f"OUTCLASS[{k!r}] should be positive"

    def test_ntrain_reasonable(self):
        for k, v in NTRAIN.items():
            assert v >= 1000, f"NTRAIN[{k!r}] = {v} looks too small"


class TestDupCollateFn:
    """dup_collate_fn must duplicate the batch tensor along dim 0."""

    def _make_batch(self, n, nc=3, h=32, w=32):
        imgs = [torch.rand(nc, h, w) for _ in range(n)]
        gts = list(range(n))
        return list(zip(imgs, gts))

    def test_single_dup(self):
        batch = self._make_batch(4)
        collate = dup_collate_fn(1)
        imgs, gts = collate(batch)
        assert imgs.shape[0] == 4

    def test_double_dup(self):
        batch = self._make_batch(4)
        collate = dup_collate_fn(2)
        imgs, gts = collate(batch)
        assert imgs.shape[0] == 8

    def test_triple_dup_shape(self):
        batch = self._make_batch(3)
        collate = dup_collate_fn(3)
        imgs, gts = collate(batch)
        assert imgs.shape[0] == 9

    def test_gt_unchanged(self):
        batch = self._make_batch(4)
        collate = dup_collate_fn(2)
        _, gts = collate(batch)
        assert gts.tolist() == list(range(4))
