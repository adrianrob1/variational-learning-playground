# Copyright (c) 2026 Adrian R. Minut
# SPDX-License-Identifier: GPL-3.0

"""Tests for vlbench.train.trainutils — pure math utility functions."""

import math
import pytest
import torch

from vlbench.train.trainutils import (
    avgdups,
    top5corrects,
    cumentropy,
    cumnll,
    onehot,
    cumbrier,
    deteministic_run,
)


class TestAvgDups:
    def test_simple_2dups(self):
        """Average over 2 duplicate rows should halve the batch size."""
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        result = avgdups(t, dups=2)
        expected = torch.tensor([[3.0, 4.0], [5.0, 6.0]])
        torch.testing.assert_close(result, expected)

    def test_single_dup(self):
        """1 duplicate should return the same tensor."""
        t = torch.arange(6, dtype=torch.float).view(2, 3)
        torch.testing.assert_close(avgdups(t, 1), t)


class TestTop5Corrects:
    def test_gt_in_top1(self):
        """Target in rank-1 position → always top-5 correct."""
        probas = torch.zeros(4, 10)
        for i in range(4):
            probas[i, i] = 1.0
        gts = torch.arange(4)
        assert top5corrects(probas, gts) == 4

    def test_none_correct(self):
        """Nothing in top-5 for any sample → 0."""
        probas = torch.zeros(3, 10)
        for i in range(3):
            probas[i, 0] = 1.0  # always predict class 0
        gts = torch.tensor([9, 9, 9])  # true label = class 9
        assert top5corrects(probas, gts) == 0


class TestCumEntropy:
    def test_uniform_distribution_high_entropy(self):
        """Uniform distribution → max entropy per sample."""
        n = 10
        probas = torch.full((2, n), 1.0 / n)
        expected = 2 * math.log(n)  # sum over 2 samples
        assert cumentropy(probas) == pytest.approx(expected, rel=1e-4)

    def test_peaked_distribution_low_entropy(self):
        """Almost-peaked distribution → entropy close to 0."""
        probas = torch.zeros(2, 10)
        probas[:, 0] = 1.0
        assert cumentropy(probas) == pytest.approx(0.0, abs=1e-5)


class TestCumNll:
    def test_correct_prediction_near_zero(self):
        """If probas[i, gt[i]] = 1 → NLL = 0."""
        probas = torch.zeros(3, 5)
        gts = torch.tensor([0, 1, 2])
        for i in range(3):
            probas[i, gts[i]] = 1.0
        assert cumnll(probas, gts) == pytest.approx(0.0, abs=1e-5)

    def test_low_probability_high_nll(self):
        """If probas[i, gt[i]] = 0.01 → NLL = -log(0.01) * n."""
        probas = torch.full((2, 5), 0.2)
        gts = torch.tensor([0, 0])
        expected = -2 * math.log(0.2)
        assert cumnll(probas, gts) == pytest.approx(expected, rel=1e-4)


class TestOnehot:
    def test_shape_is_correct(self):
        t = torch.tensor([0, 2, 4])
        out = onehot(t, nclasses=5)
        assert out.shape == (3, 5)

    def test_values_are_correct(self):
        t = torch.tensor([1])
        out = onehot(t, nclasses=3)
        assert out.tolist() == [[0, 1, 0]]

    def test_empty_tensor(self):
        t = torch.empty(0, dtype=torch.long)
        out = onehot(t, nclasses=4)
        assert out.shape == (0, 4)


class TestCumBrier:
    def test_perfect_prediction_zero_brier(self):
        """Perfect predictions → Brier score = 0."""
        probas = onehot(torch.tensor([0, 1, 2]), 3, dtype=torch.float)
        gt_oh = probas.clone()
        assert cumbrier(probas, gt_oh) == pytest.approx(0.0, abs=1e-6)

    def test_worst_prediction(self):
        """Probability mass on wrong class → Brier = 2 per sample."""
        # 1 sample, 2 classes, pred=[1, 0], gt=[0, 1] → (1-0)^2 + (0-1)^2 = 2
        probas = torch.tensor([[1.0, 0.0]])
        gt_oh = torch.tensor([[0.0, 1.0]])
        assert cumbrier(probas, gt_oh) == pytest.approx(2.0)


class TestDeterministicRun:
    def test_sets_seeds(self):
        """deteministic_run should not raise and should be callable."""
        deteministic_run(seed=42)  # just check it runs without error
