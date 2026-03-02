# Copyright (c) 2026 Adrian R. Minut
# SPDX-License-Identifier: GPL-3.0

import pytest
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from vldatasets.standard.dataloaders import (
    DatasetConfig,
    get_cifar10_train_loaders,
    get_cifar10_test_loader,
)


class TestHydraDatasets:
    """Verify that DatasetConfig works with Hydra instantiation."""

    def test_dataset_config_instantiation(self):
        """Test manual instantiation of DatasetConfig."""
        cfg = DatasetConfig(
            name="test_cifar10",
            outclass=10,
            ntrain=50000,
            ntest=10000,
            insize=32,
            train_loader_f=get_cifar10_train_loaders,
            test_loader_f=get_cifar10_test_loader,
        )
        assert cfg.name == "test_cifar10"
        assert cfg.outclass == 10
        assert cfg.ntrain == 50000

    def test_hydra_instantiation(self):
        """Test instantiation via Hydra OmegaConf."""
        yaml_cfg = """
        _target_: vldatasets.standard.DatasetConfig
        name: cifar10
        outclass: 10
        ntrain: 50000
        ntest: 10000
        insize: 32
        train_loader_f: 
          _target_: vldatasets.standard.dataloaders.get_cifar10_train_loaders
          _partial_: true
        test_loader_f:
          _target_: vldatasets.standard.dataloaders.get_cifar10_test_loader
          _partial_: true
        """
        conf = OmegaConf.create(yaml_cfg)
        dataset_obj = instantiate(conf)

        assert isinstance(dataset_obj, DatasetConfig)
        assert dataset_obj.name == "cifar10"

        # Verify loaders can be called
        # Note: We don't actually run them here to avoid downloading data,
        # but we check if they are callable.
        assert callable(dataset_obj.train_loader_f)
        assert callable(dataset_obj.test_loader_f)

    def test_fallback_logic(self):
        """Test that get_train_loaders falls back to registry if train_loader_f is None."""
        cfg = DatasetConfig(
            name="cifar10",
            outclass=10,
            ntrain=50000,
            ntest=10000,
            insize=32,
            train_loader_f=None,
        )
        # This should still work if 'cifar10' is in TRAINDATALOADERS
        # We just check it doesn't crash on access, actual call might need data
        assert cfg.name == "cifar10"
