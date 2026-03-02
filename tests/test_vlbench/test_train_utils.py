# Copyright (c) 2026 Adrian R. Minut
# SPDX-License-Identifier: GPL-3.0

"""Tests for vlbench.train.utils — coroutines and filesystem helpers."""

import csv
import os
import pytest

from vlbench.train.utils import (
    autoinitcoroutine,
    coro_trackavg,
    coro_trackavg_weighted,
    div0,
    mkdirp,
    coro_dict2csv,
)


# ---------------------------------------------------------------------------
# autoinitcoroutine
# ---------------------------------------------------------------------------
class TestAutoInitCoroutine:
    def test_does_not_require_next_call(self):
        """autoinitcoroutine should start the generator automatically."""

        @autoinitcoroutine
        def _gen():
            v = 0
            val = yield
            while True:
                v += val
                val = yield v

        g = _gen()
        assert g.send(1) == 1
        assert g.send(2) == 3

    def test_returns_a_generator(self):
        @autoinitcoroutine
        def _gen():
            yield

        assert hasattr(_gen(), "send")


# ---------------------------------------------------------------------------
# coro_trackavg
# ---------------------------------------------------------------------------
class TestCoroTrackavg:
    def test_single_value(self):
        g = coro_trackavg()
        assert g.send(5.0) == pytest.approx(5.0)

    def test_running_average(self):
        g = coro_trackavg()
        g.send(2.0)
        g.send(4.0)
        result = g.send(6.0)
        assert result == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# coro_trackavg_weighted
# ---------------------------------------------------------------------------
class TestCoroTrackavgWeighted:
    def test_simple_weighted(self):
        g = coro_trackavg_weighted()
        g.send((10.0, 2))  # total 10, weight 2  => avg 5
        result = g.send((6.0, 2))  # new total 16, weight 4 => avg 4
        assert result == pytest.approx(4.0)

    def test_single_element(self):
        g = coro_trackavg_weighted()
        assert g.send((7.0, 1)) == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# div0
# ---------------------------------------------------------------------------
class TestDiv0:
    def test_normal_division(self):
        assert div0(10.0, 2) == pytest.approx(5.0)

    def test_division_by_zero(self):
        assert div0(5.0, 0) == pytest.approx(0.0)

    def test_zero_numerator(self):
        assert div0(0.0, 3) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# mkdirp
# ---------------------------------------------------------------------------
class TestMkdirp:
    def test_creates_nested_directories(self, tmp_path):
        target = str(tmp_path / "a" / "b" / "c")
        mkdirp(target)
        assert os.path.isdir(target)

    def test_does_not_raise_if_exists(self, tmp_path):
        target = str(tmp_path)
        mkdirp(target)  # already exists — should not raise


# ---------------------------------------------------------------------------
# coro_dict2csv
# ---------------------------------------------------------------------------
class TestCoroDictToCsv:
    def test_writes_rows(self, tmp_path):
        csvfile = str(tmp_path / "out.csv")
        header = ("epoch", "loss", "acc")
        g = coro_dict2csv(csvfile, header)
        g.send({"epoch": 0, "loss": 1.0, "acc": 0.5})
        g.send({"epoch": 1, "loss": 0.8, "acc": 0.6})
        g.close()

        with open(csvfile, "r") as f:
            rows = list(csv.DictReader(f))

        assert rows[0]["epoch"] == "0"
        assert float(rows[0]["loss"]) == pytest.approx(1.0)
        assert float(rows[1]["acc"]) == pytest.approx(0.6)

    def test_header_matches_keys(self, tmp_path):
        csvfile = str(tmp_path / "out2.csv")
        header = ("x", "y")
        g = coro_dict2csv(csvfile, header)
        g.send({"x": 1, "y": 2})
        g.close()
        with open(csvfile, "r") as f:
            content = f.read()
        assert "x" in content and "y" in content
