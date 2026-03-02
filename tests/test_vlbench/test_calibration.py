# Copyright (c) 2026 Adrian R. Minut
# SPDX-License-Identifier: GPL-3.0

"""Tests for vlbench.plotting.calibration — ECE metrics and bin operations."""

import pytest
from vlbench.plotting.calibration import (
    data2bins,
    bins2acc,
    bins2conf,
    bins2ece,
    joinbins,
    coro_binsmerger,
)


# ---------------------------------------------------------------------------
# data2bins
# ---------------------------------------------------------------------------
class TestData2Bins:
    def test_perfect_predictions(self):
        """Perfect predictions: all correct with confidence 1.0 → ece ≈ 0."""
        # 10 samples, each correct, confidence=0.99
        data = [(True, 0.99)] * 10
        bins = data2bins(data, nbin=10)
        bincounts, corrects, cumconf = bins
        assert sum(bincounts) == 10
        assert sum(corrects) == 10
        ece = bins2ece(bins)
        assert ece == pytest.approx(0.0, abs=0.11)  # slight binning rounding

    def test_random_predictions(self):
        """Test that bins have the right total count."""
        data = [(i % 2 == 0, 0.5) for i in range(20)]
        bins = data2bins(data, nbin=10)
        assert sum(bins[0]) == 20

    def test_nbin_affects_bin_count(self):
        data = [(True, 0.5)] * 5
        bins5 = data2bins(data, nbin=5)
        bins10 = data2bins(data, nbin=10)
        assert len(bins5[0]) == 5
        assert len(bins10[0]) == 10


# ---------------------------------------------------------------------------
# bins2acc, bins2conf, bins2ece
# ---------------------------------------------------------------------------
class TestBinsMetrics:
    @pytest.fixture
    def known_bins(self):
        # 2 bins: bin0 has 3 samples (2 correct, cum_conf=2.4)
        #         bin1 has 2 samples (2 correct, cum_conf=1.8)
        bincounts = [3, 2]
        corrects = [2, 2]
        cumconf = [2.4, 1.8]
        return bincounts, corrects, cumconf

    def test_bins2acc(self, known_bins):
        assert bins2acc(known_bins) == pytest.approx(4.0 / 5.0)

    def test_bins2conf(self, known_bins):
        assert bins2conf(known_bins) == pytest.approx((2.4 + 1.8) / 5.0)

    def test_bins2ece(self, known_bins):
        # bins2ece uses |corrects - cumconf| / total_n per bin
        # Bin 0: |2 - 2.4| / 5 = 0.4/5 = 0.08
        # Bin 1: |2 - 1.8| / 5 = 0.2/5 = 0.04
        expected = abs(2 - 2.4) / 5 + abs(2 - 1.8) / 5
        assert bins2ece(known_bins) == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# joinbins
# ---------------------------------------------------------------------------
class TestJoinBins:
    def test_summed_correctly(self):
        bins_a = ([1, 2], [0, 1], [0.3, 0.5])
        bins_b = ([2, 1], [1, 1], [0.4, 0.2])
        merged = joinbins(bins_a, bins_b)
        assert merged[0] == [3, 3]
        assert merged[1] == [1, 2]
        assert merged[2] == pytest.approx([0.7, 0.7])

    def test_length_mismatch_raises(self):
        bins_a = ([1, 2], [0, 1], [0.3, 0.5])
        bins_b = ([1], [1], [0.5])
        with pytest.raises(ValueError):
            joinbins(bins_a, bins_b)


# ---------------------------------------------------------------------------
# coro_binsmerger
# ---------------------------------------------------------------------------
class TestCoroBinsmerger:
    def test_accumulates_bins(self):
        g = coro_binsmerger()
        bins1 = ([1, 0], [1, 0], [0.9, 0.0])
        bins2 = ([0, 1], [0, 1], [0.0, 0.8])
        _ = g.send(bins1)
        result2 = g.send(bins2)
        assert result2[0] == [1, 1]
        assert result2[1] == [1, 1]
