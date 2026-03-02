# Copyright (c) 2026 Adrian R. Minut
# SPDX-License-Identifier: GPL-3.0

import os
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import shutil
from omegaconf import OmegaConf

# Mock hydra.main decorator before importing the module
import hydra

from vlbench.image_classification.distributed_train import main as train_main
from vlbench.models import MODELS

# Store original functions to restore them if needed, but better use patch
import torch.distributed as dist
import torch.nn.parallel

MODELS["resnet18"] = MagicMock(return_value=MagicMock())


class TestFallbackLogic(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmpdir, "runs"), exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @patch("torch.distributed.init_process_group")
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.destroy_process_group")
    @patch(
        "vlbench.image_classification.distributed_train.DDP",
        side_effect=lambda x, **kw: x,
    )
    @patch("vlbench.image_classification.distributed_train.MODELS")
    @patch("vlbench.image_classification.distributed_train.get_imagenet_train_loader")
    @patch("vlbench.image_classification.distributed_train.get_imagenet_test_loader")
    @patch(
        "vlbench.image_classification.distributed_train.get_imagenet_train_loader_torch"
    )
    @patch(
        "vlbench.image_classification.distributed_train.get_imagenet_test_loader_torch"
    )
    @patch("vlbench.image_classification.distributed_train.loadcheckpoint")
    @patch("vlbench.image_classification.distributed_train.do_epoch")
    @patch("vlbench.image_classification.distributed_train.coro_log_timed")
    @patch("vlbench.image_classification.distributed_train.exists")
    @patch("vlbench.image_classification.distributed_train.mkdirp")
    def test_train_fallback_to_torch(
        self,
        mock_mkdirp,
        mock_exists,
        mock_log,
        mock_epoch,
        mock_load,
        mock_test_loader_torch,
        mock_train_loader_torch,
        mock_test_loader_ffcv,
        mock_train_loader_ffcv,
        mock_models,
        mock_ddp,
        mock_dist_destroy,
        mock_dist_init_val,
        mock_dist_init,
    ):
        """Test that it falls back to torch loader when .ffcv files are missing."""
        mock_models.__getitem__.return_value = MagicMock(return_value=MagicMock())

        def exists_side_effect(path):
            if path.endswith(".ffcv"):
                return False
            return True

        mock_exists.side_effect = exists_side_effect

        mock_load.return_value = (
            0,
            MagicMock(),
            MagicMock(),
            MagicMock(),
            {"modelargs": (), "modelkwargs": {}},
        )

        cfg = OmegaConf.create(
            {
                "data_dir": "/dummy/data",
                "tbatch": 10,
                "vbatch": 10,
                "workers": 1,
                "resume": None,
                "save_dir": os.path.join(self.tmpdir, "runs"),
                "seed": 42,
                "epochs": 0,
                "warmup": 0,
                "method": {"name": "sgd"},
                "model": {"name": "resnet18"},
                "dataset": {"name": "imagenet_ffcv"},
                "printfreq": 10,
                "bins": 20,
            }
        )

        with patch.dict(
            "os.environ", {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1"}
        ):
            train_main(cfg)

        mock_train_loader_torch.assert_called()
        mock_train_loader_ffcv.assert_not_called()

    @patch("torch.distributed.init_process_group")
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.destroy_process_group")
    @patch(
        "vlbench.image_classification.distributed_train.DDP",
        side_effect=lambda x, **kw: x,
    )
    @patch("vlbench.image_classification.distributed_train.MODELS")
    @patch("vlbench.image_classification.distributed_train.get_imagenet_train_loader")
    @patch("vlbench.image_classification.distributed_train.get_imagenet_test_loader")
    @patch(
        "vlbench.image_classification.distributed_train.get_imagenet_train_loader_torch"
    )
    @patch(
        "vlbench.image_classification.distributed_train.get_imagenet_test_loader_torch"
    )
    @patch("vlbench.image_classification.distributed_train.loadcheckpoint")
    @patch("vlbench.image_classification.distributed_train.do_epoch")
    @patch("vlbench.image_classification.distributed_train.coro_log_timed")
    @patch("vlbench.image_classification.distributed_train.exists")
    @patch("vlbench.image_classification.distributed_train.mkdirp")
    def test_train_uses_ffcv(
        self,
        mock_mkdirp,
        mock_exists,
        mock_log,
        mock_epoch,
        mock_load,
        mock_test_loader_torch,
        mock_train_loader_torch,
        mock_test_loader_ffcv,
        mock_train_loader_ffcv,
        mock_models,
        mock_ddp,
        mock_dist_destroy,
        mock_dist_init_val,
        mock_dist_init,
    ):
        """Test that it uses FFCV loader when train.ffcv exists."""
        mock_models.__getitem__.return_value = MagicMock(return_value=MagicMock())
        mock_exists.return_value = True
        mock_load.return_value = (
            0,
            MagicMock(),
            MagicMock(),
            MagicMock(),
            {"modelargs": (), "modelkwargs": {}},
        )

        cfg = OmegaConf.create(
            {
                "data_dir": "/dummy/data",
                "tbatch": 10,
                "vbatch": 10,
                "workers": 1,
                "resume": None,
                "save_dir": os.path.join(self.tmpdir, "runs"),
                "seed": 42,
                "epochs": 0,
                "warmup": 0,
                "method": {"name": "sgd"},
                "model": {"name": "resnet18"},
                "dataset": {"name": "imagenet_ffcv"},
                "printfreq": 10,
                "bins": 20,
            }
        )

        with patch.dict(
            "os.environ", {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1"}
        ):
            train_main(cfg)

        mock_train_loader_ffcv.assert_called()
        mock_train_loader_torch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
