# Copyright (c) 2026 Adrian R. Minut
# SPDX-License-Identifier: GPL-3.0

"""Tests for vlbench.ood — OOD dataset loaders and evaluation pipeline.

All tests run on CPU with no dataset downloads.  External download calls are
patched with ``unittest.mock``.
"""

from __future__ import annotations

import types
import tempfile
import os
from os.path import join as pjoin
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class TinyModel(nn.Module):
    """3-class linear model for smoke tests."""

    def __init__(self, nin=4, nout=3):
        super().__init__()
        self.fc = nn.Linear(nin, nout)

    def forward(self, x):
        return self.fc(x)


def make_ood_args(**overrides):
    """Build a minimal argparse-Namespace for OOD run tests."""
    defaults = dict(
        ood_dataset="svhn",
        workers=0,
        batch=8,
        testsamples=1,
        testrepeat=1,
        valdata=False,
        printfreq=1,
        device="cpu",
        seed=0,
        save_dir="save_temp",
        saveoutput=False,
        data_dir="../data",
        swag_modelsamples=1,
        swag_samplemode="modelwise",
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# TestOodDatasets
# ---------------------------------------------------------------------------


class TestOodDatasets:
    """Tests for SVHNInfo, Flowers102Info, get_svhn_loader, get_flowers102_loader."""

    def test_svhn_info_outclass(self):
        from vldatasets.standard.ood_datasets import SVHNInfo

        assert SVHNInfo.outclass == 10

    def test_svhn_info_splits(self):
        from vldatasets.standard.ood_datasets import SVHNInfo

        assert "train" in SVHNInfo.split
        assert "test" in SVHNInfo.split
        assert "extra" in SVHNInfo.split

    def test_svhn_info_count_keys(self):
        from vldatasets.standard.ood_datasets import SVHNInfo

        assert set(SVHNInfo.count) == set(SVHNInfo.split)

    def test_flowers102_info_outclass(self):
        from vldatasets.standard.ood_datasets import Flowers102Info

        assert Flowers102Info.outclass == 102

    def test_flowers102_info_count_keys(self):
        from vldatasets.standard.ood_datasets import Flowers102Info

        assert set(Flowers102Info.count) == set(Flowers102Info.split)

    def test_get_svhn_loader_invalid_split(self):
        from vldatasets.standard.ood_datasets import get_svhn_loader

        with pytest.raises((AssertionError, ValueError)):
            get_svhn_loader("/fake", workers=0, pin_memory=False, batch=4, split="bad")

    def test_get_flowers102_loader_invalid_split(self):
        from vldatasets.standard.ood_datasets import get_flowers102_loader

        with pytest.raises((AssertionError, ValueError)):
            get_flowers102_loader(
                "/fake", workers=0, pin_memory=False, batch=4, split="bad"
            )

    def test_get_svhn_loader_returns_dataloader(self):
        """With a mocked torchvision dataset, get_svhn_loader returns a DataLoader."""
        from torch.utils.data import DataLoader

        fake_ds = torch.utils.data.TensorDataset(
            torch.randn(10, 3, 32, 32), torch.zeros(10, dtype=torch.long)
        )
        with patch(
            "vldatasets.standard.ood_datasets.datasets.SVHN", return_value=fake_ds
        ):
            from vldatasets.standard.ood_datasets import get_svhn_loader

            loader = get_svhn_loader(
                "/fake", workers=0, pin_memory=False, batch=4, split="test"
            )
        assert isinstance(loader, DataLoader)

    def test_get_flowers102_loader_returns_dataloader(self):
        """With a mocked torchvision dataset, get_flowers102_loader returns a DataLoader."""
        from torch.utils.data import DataLoader

        fake_ds = torch.utils.data.TensorDataset(
            torch.randn(10, 3, 32, 32), torch.zeros(10, dtype=torch.long)
        )
        with patch(
            "vldatasets.standard.ood_datasets.datasets.Flowers102", return_value=fake_ds
        ):
            from vldatasets.standard.ood_datasets import get_flowers102_loader

            loader = get_flowers102_loader(
                "/fake", workers=0, pin_memory=False, batch=4, split="test"
            )
        assert isinstance(loader, DataLoader)

    def test_get_svhn_loader_with_dups_uses_dup_collate(self):
        """When dups > 1, the loader's collate_fn should not be the default."""
        fake_ds = torch.utils.data.TensorDataset(
            torch.randn(2, 3, 32, 32), torch.zeros(2, dtype=torch.long)
        )
        with patch(
            "vldatasets.standard.ood_datasets.datasets.SVHN", return_value=fake_ds
        ):
            from vldatasets.standard.ood_datasets import get_svhn_loader

            loader = get_svhn_loader(
                "/fake", workers=0, pin_memory=False, batch=2, split="test", dups=2
            )
        assert loader.collate_fn is not None


# ---------------------------------------------------------------------------
# TestDupCollateFn
# ---------------------------------------------------------------------------


class TestDupCollateFn:
    """Tests for dup_collate_fn."""

    def test_single_dup_preserves_shape(self):
        from vldatasets.standard.ood_datasets import dup_collate_fn

        fn = dup_collate_fn(1)
        data = [(torch.randn(3, 4, 4), torch.tensor(0)) for _ in range(2)]
        imgs, labels = fn(data)
        assert imgs.shape[0] == 2

    def test_double_dup_doubles_batch(self):
        from vldatasets.standard.ood_datasets import dup_collate_fn

        fn = dup_collate_fn(2)
        data = [(torch.randn(3, 4, 4), torch.tensor(i)) for i in range(3)]
        imgs, labels = fn(data)
        # batch duplicated → 6 images, 3 labels
        assert imgs.shape[0] == 6
        assert labels.shape[0] == 3

    def test_dup_collate_preserves_label_values(self):
        from vldatasets.standard.ood_datasets import dup_collate_fn

        fn = dup_collate_fn(3)
        data = [(torch.zeros(1, 2, 2), torch.tensor(i)) for i in range(4)]
        imgs, labels = fn(data)
        assert labels.tolist() == list(range(4))


# ---------------------------------------------------------------------------
# TestGetArgs
# ---------------------------------------------------------------------------


class TestGetArgs:
    """Tests for vlbench.ood.run.get_args."""

    def test_ood_dataset_default_is_svhn(self):
        from vlbench.ood.run import get_args

        with patch("sys.argv", ["run.py", "/some/traindir"]):
            args = get_args()
        assert args.ood_dataset == "svhn"

    def test_flowers102_choice_accepted(self):
        from vlbench.ood.run import get_args

        with patch(
            "sys.argv", ["run.py", "/some/traindir", "--ood_dataset", "flowers102"]
        ):
            args = get_args()
        assert args.ood_dataset == "flowers102"

    def test_invalid_ood_dataset_raises(self):
        from vlbench.ood.run import get_args

        with patch(
            "sys.argv", ["run.py", "/some/traindir", "--ood_dataset", "imagenet"]
        ):
            with pytest.raises(SystemExit):
                get_args()

    def test_default_device_is_cpu(self):
        from vlbench.ood.run import get_args

        with patch("sys.argv", ["run.py", "/traindir"]):
            args = get_args()
        assert args.device == "cpu"

    def test_saveoutput_flag_off_by_default(self):
        from vlbench.ood.run import get_args

        with patch("sys.argv", ["run.py", "/traindir"]):
            args = get_args()
        assert not args.saveoutput


# ---------------------------------------------------------------------------
# TestGetOodLoader
# ---------------------------------------------------------------------------


class TestGetOodLoader:
    """Tests for vlbench.ood.run.get_ood_loader: correct dispatch."""

    def _fake_ds(self):
        return torch.utils.data.TensorDataset(
            torch.randn(4, 3, 32, 32), torch.zeros(4, dtype=torch.long)
        )

    def test_svhn_dispatches_to_svhn_loader(self):
        from torch.utils.data import DataLoader

        with patch(
            "vldatasets.standard.ood_datasets.datasets.SVHN",
            return_value=self._fake_ds(),
        ):
            from vlbench.ood.run import get_ood_loader

            loader = get_ood_loader(make_ood_args(ood_dataset="svhn"))
        assert isinstance(loader, DataLoader)

    def test_flowers102_dispatches_to_flowers102_loader(self):
        from torch.utils.data import DataLoader

        with patch(
            "vldatasets.standard.ood_datasets.datasets.Flowers102",
            return_value=self._fake_ds(),
        ):
            from vlbench.ood.run import get_ood_loader

            loader = get_ood_loader(make_ood_args(ood_dataset="flowers102"))
        assert isinstance(loader, DataLoader)

    def test_unknown_dataset_raises(self):
        from vlbench.ood.run import get_ood_loader

        with pytest.raises((ValueError, KeyError)):
            get_ood_loader(make_ood_args(ood_dataset="imagenet"))


# ---------------------------------------------------------------------------
# TestRunEvalLoop
# ---------------------------------------------------------------------------


class TestRunEvalLoop:
    """Tests for vlbench.ood.run.run_eval_loop dispatcher."""

    def _make_loader(self, n=8, nin=4, nclass=3):
        ds = torch.utils.data.TensorDataset(
            torch.randn(n, nin), torch.zeros(n, dtype=torch.long)
        )
        return torch.utils.data.DataLoader(ds, batch_size=4)

    def _make_log(self):
        """A simple coroutine mock that accepts sends and can stop on throw."""
        from vlbench.utils.ood_utils import coro_log

        return coro_log(None, 1, "")

    def test_map_optimizer_calls_do_evalbatch(self):
        """For a plain SGD optimizer, run_eval_loop should call do_evalbatch."""
        from vlbench.ood.run import run_eval_loop

        model = TinyModel(nin=4, nout=3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loader = self._make_loader()
        log = self._make_log()

        # do_evalbatch in this test should return just a single Tensor (probabilities)
        with patch(
            "vlbench.ood.run.do_evalbatch",
            side_effect=lambda *a, **kw: torch.zeros(4, 3),
        ) as mock_eval:
            with torch.no_grad():
                run_eval_loop(
                    loader,
                    "ood_test",
                    log,
                    torch.device("cpu"),
                    optimizer,
                    model,
                    make_ood_args(),
                )
        mock_eval.assert_called()

    def test_ivon_optimizer_calls_do_evalbatch_von(self):
        """For IVON, run_eval_loop should call do_evalbatch_von."""
        from vlbench.ood.run import run_eval_loop
        from vloptimizers.ivon import IVON

        model = TinyModel(nin=4, nout=3)
        optimizer = IVON(model.parameters(), lr=0.01, ess=50.0)
        loader = self._make_loader()
        log = self._make_log()

        with patch(
            "vlbench.ood.run.do_evalbatch_von",
            side_effect=lambda *a, **kw: torch.zeros(4, 3),
        ) as mock_von:
            with torch.no_grad():
                run_eval_loop(
                    loader,
                    "ood_test",
                    log,
                    torch.device("cpu"),
                    optimizer,
                    model,
                    make_ood_args(),
                )
        mock_von.assert_called()

    def test_vogn_optimizer_calls_do_evalbatch_von(self):
        """For VOGN, run_eval_loop should call do_evalbatch_von."""
        from vlbench.ood.run import run_eval_loop
        from vloptimizers.vogn import VOGN

        model = TinyModel(nin=4, nout=3)
        optimizer = VOGN(model.parameters(), lr=0.01, prior_precision=1.0, data_size=50)
        loader = self._make_loader()
        log = self._make_log()

        with patch(
            "vlbench.ood.run.do_evalbatch_von",
            side_effect=lambda *a, **kw: torch.zeros(4, 3),
        ) as mock_von:
            with torch.no_grad():
                run_eval_loop(
                    loader,
                    "ood_test",
                    log,
                    torch.device("cpu"),
                    optimizer,
                    model,
                    make_ood_args(),
                )
        mock_von.assert_called()


# ---------------------------------------------------------------------------
# TestComputeAndSaveMetrics
# ---------------------------------------------------------------------------


class TestComputeAndSaveMetrics:
    """Tests for vlbench.ood.run.compute_and_save_metrics."""

    def _make_fake_npys(self, tmpdir, nrun=2, n_in=100, n_ood=80, nclass=10):
        """Write synthetic prediction .npy files and return run ids."""
        np.random.seed(0)
        for i in range(nrun):
            indomain = np.abs(np.random.randn(n_in, nclass))
            indomain = indomain / indomain.sum(axis=1, keepdims=True)
            ood = np.abs(np.random.randn(n_ood, nclass))
            ood = ood / ood.sum(axis=1, keepdims=True)
            np.save(pjoin(tmpdir, f"predictions_indomain_test_{i}.npy"), indomain)
            np.save(pjoin(tmpdir, f"predictions_ood_test_{i}.npy"), ood)
        return [str(i) for i in range(nrun)]

    def test_csv_is_created(self):
        from vlbench.ood.run import compute_and_save_metrics

        with tempfile.TemporaryDirectory() as tmpdir:
            runs = self._make_fake_npys(tmpdir)
            compute_and_save_metrics(tmpdir, "", runs)
            assert os.path.exists(pjoin(tmpdir, "metrics_test.csv"))

    def test_csv_has_correct_columns(self):
        import csv as _csv
        from vlbench.ood.run import compute_and_save_metrics
        from vlbench.utils.ood_utils import OODMetrics

        with tempfile.TemporaryDirectory() as tmpdir:
            runs = self._make_fake_npys(tmpdir)
            compute_and_save_metrics(tmpdir, "", runs)
            with open(pjoin(tmpdir, "metrics_test.csv")) as f:
                reader = _csv.DictReader(f)
                fieldnames = reader.fieldnames
        expected = {"epoch"} | set(OODMetrics.metric_names)
        assert expected <= set(fieldnames)

    def test_csv_has_correct_row_count(self):
        import csv as _csv
        from vlbench.ood.run import compute_and_save_metrics

        with tempfile.TemporaryDirectory() as tmpdir:
            runs = self._make_fake_npys(tmpdir, nrun=3)
            compute_and_save_metrics(tmpdir, "", runs)
            with open(pjoin(tmpdir, "metrics_test.csv")) as f:
                rows = list(_csv.DictReader(f))
        assert len(rows) == 3

    def test_wamode_suffix_applied_to_filename(self):
        from vlbench.ood.run import compute_and_save_metrics

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write files with wamode prefix
            nrun, nclass = 1, 10
            np.random.seed(0)
            for i in range(nrun):
                idm = np.random.dirichlet(np.ones(nclass), size=50)
                ood = np.random.dirichlet(np.ones(nclass), size=40)
                np.save(pjoin(tmpdir, f"predictions_wa_indomain_test_{i}.npy"), idm)
                np.save(pjoin(tmpdir, f"predictions_wa_ood_test_{i}.npy"), ood)
            compute_and_save_metrics(tmpdir, "wa", [str(i) for i in range(nrun)])
            assert os.path.exists(pjoin(tmpdir, "metrics_wa_test.csv"))

    def test_metrics_are_finite(self):
        import csv as _csv
        from vlbench.ood.run import compute_and_save_metrics
        from vlbench.utils.ood_utils import OODMetrics

        with tempfile.TemporaryDirectory() as tmpdir:
            runs = self._make_fake_npys(tmpdir, nrun=2)
            compute_and_save_metrics(tmpdir, "", runs)
            with open(pjoin(tmpdir, "metrics_test.csv")) as f:
                rows = list(_csv.DictReader(f))
        for row in rows:
            for key in OODMetrics.metric_names:
                assert np.isfinite(float(row[key])), f"{key} not finite: {row[key]}"
