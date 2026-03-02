# Copyright (c) 2026 Adrian R. Minut
# Copyright (c) 2026 ABI Team
# 
# SPDX-License-Identifier: GPL-3.0
# SPDX-License-Identifier: Apache 2.0

import numpy as np
from typing import List, Callable, Optional


def mbr_corpus(
    hyps: List[List[str]],
    metric: Callable,
    srcs: Optional[List[str]] = None,
    num_subsamples: Optional[int] = None,
    aggregation: str = "mean",
    scores: Optional[List[List[float]]] = None,
) -> List[List[float]]:
    """
    Computes per-sample Minimum Bayes Risk (MBR) utility scores for a corpus.
    MBR decoding selects the hypothesis that minimizes expected loss (or maximizes expected utility)
    given a set of candidate hypotheses and reference hypotheses drawn from the model's posterior.

    Args:
        hyps: List of hypotheses for each sample. Shape: (num_samples, num_hypotheses)
        metric: Metric function to compute the utility. It should take
                (candidates: List[str], references: List[str], srcs: Optional[List[str]])
                and return a tuple (flat_scores, mean_score).
        srcs: Optional list of source texts for each sample. Required for source-dependent metrics like COMET.
        num_subsamples: Optional number of subsamples to use for MBR expectation over hypotheses. If None, uses all hypotheses as references.
        aggregation: Strategy to aggregate subsample metrics. "mean" or "weighted_mean".
        scores: Optional list of confidence scores for each hypothesis, used when aggregation is "weighted_mean".

    Returns:
        List of negative risks (i.e., expected utility scores) for each hypothesis of each sample.
        Shape: (num_samples, num_hypotheses). Higher score is better.
    """
    if len(hyps) == 0:
        return []

    num_samples = len(hyps[0])
    use_subsampling = num_subsamples is not None and num_subsamples < num_samples

    cands = []
    refs = []
    dup_srcs = [] if srcs is not None else None

    for i, samples in enumerate(hyps):
        indices = (
            np.random.choice(num_samples, num_subsamples, replace=False)
            if use_subsampling
            else list(range(num_samples))
        )
        for cand in samples:
            for ref_id in indices:
                cands.append(cand)
                refs.append(samples[ref_id])
                if srcs is not None:
                    dup_srcs.append(srcs[i])

    # metric callable returns a tuple (flat_scores, mean_score), we just need the flat_scores list
    flat_metric_matrixes, _ = metric(
        cands, refs, tuple(dup_srcs) if dup_srcs is not None else None
    )  # using tuple for srcs to avoid passing list if unexpected

    metric_matrixes = []
    for i, _ in enumerate(hyps):
        metric_matrixes.append([])
        for j in range(num_samples):
            metric_matrixes[i].append([])
            for k in range(num_subsamples if use_subsampling else num_samples):
                flat_idx = (
                    i
                    * num_samples
                    * (num_subsamples if use_subsampling else num_samples)
                    + j * (num_subsamples if use_subsampling else num_samples)
                    + k
                )
                metric_matrixes[i][j].append(flat_metric_matrixes[flat_idx])

    metric_matrixes = np.array(metric_matrixes)

    if aggregation == "mean":
        neg_risks = metric_matrixes.mean(axis=2).tolist()
    elif aggregation == "weighted_mean":
        assert scores is not None, (
            "scores must be provided for weighted_mean aggregation"
        )
        raise NotImplementedError("weighted_mean aggregation is not fully ported yet.")
    else:
        raise ValueError(f"aggregation {aggregation} not implemented")

    return neg_risks


def select_best_hypotheses(
    hyps: List[List[str]], utilities: List[List[float]]
) -> List[str]:
    """
    Selects the best hypothesis for each sample based on highest utility score.

    Args:
        hyps: List of hypotheses for each sample. Shape: (num_samples, num_hypotheses)
        utilities: List of utility scores for each hypothesis. Shape: (num_samples, num_hypotheses)

    Returns:
        List of selected hypotheses (one per sample).
    """
    predictions = []
    for sample_hyps, sample_utilities in zip(hyps, utilities):
        predictions.append(sample_hyps[np.argmax(sample_utilities)])
    return predictions
