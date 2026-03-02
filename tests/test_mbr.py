# Copyright (c) 2026 Adrian R. Minut
# SPDX-License-Identifier: GPL-3.0

import pytest
from vlbench.text_generation.mbr import mbr_corpus, select_best_hypotheses


def dummy_metric(cands, refs, srcs=None):
    """
    A simple exact-match metric for testing.
    Returns 1.0 if candidate == reference, else 0.0.
    """
    scores = [1.0 if c == r else 0.0 for c, r in zip(cands, refs)]
    return scores, sum(scores) / len(scores) if scores else 0.0


def test_mbr_corpus_mean():
    """
    Test standard MBR expected utility using a mean aggregation and exhaustive matching.
    """
    hyps = [["I am happy", "I am very happy", "I am happy"]]
    # For candidate 0 ("I am happy"): matches ref 0 and ref 2 -> expected mean = 2/3
    # For candidate 1 ("I am very happy"): matches ref 1 -> expected mean = 1/3
    # For candidate 2 ("I am happy"): matches ref 0 and ref 2 -> expected mean = 2/3

    utilities = mbr_corpus(
        hyps, metric=dummy_metric, num_subsamples=None, aggregation="mean"
    )

    assert len(utilities) == 1
    assert len(utilities[0]) == 3

    assert utilities[0][0] == pytest.approx(2 / 3)
    assert utilities[0][1] == pytest.approx(1 / 3)
    assert utilities[0][2] == pytest.approx(2 / 3)


def test_select_best_hypotheses():
    """
    Test that the hypothesis with the highest utility is chosen.
    """
    hyps = [["A", "B", "C"], ["X", "Y"]]
    utilities = [[0.1, 0.9, 0.5], [0.8, 0.2]]
    best = select_best_hypotheses(hyps, utilities)
    assert best == ["B", "X"]
