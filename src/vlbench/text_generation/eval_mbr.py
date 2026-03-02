# Copyright (c) 2026 Adrian R. Minut
# Copyright (c) 2026 ABI Team
# 
# SPDX-License-Identifier: GPL-3.0
# SPDX-License-Identifier: Apache 2.0

import logging
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from vlbench.text_generation.mbr import mbr_corpus, select_best_hypotheses

logger = logging.getLogger(__name__)


def evaluate_mbr(cfg: DictConfig) -> None:
    """
    Evaluate generated hypotheses using Minimum Bayes Risk (MBR) decoding.
    The metric function is dynamically instantiated from the Hydra configuration,
    making it extremely easy for users to plug in custom scoring pipelines!
    """
    logger.info(f"Starting MBR Evaluation with Config:\n{cfg}")

    # 1. Load Hypotheses and References
    # In a real pipeline, these would be loaded from prediction outputs
    # For structure, we expect them to be provided via a hydra dataset or loaded from disk
    logger.info("Loading hypotheses for MBR decoding...")

    if "hypotheses_path" in cfg:
        # Placeholder for loading from disk
        logger.info(f"Loading hypotheses from {cfg.hypotheses_path}")
        hyps = [["dummy hypothesis"]]  # To be replaced with actual jsonl load
        srcs = None
    else:
        hyps = OmegaConf.to_container(
            cfg.get("dummy_hyps", [["hypothesis 1", "hypothesis 2"]])
        )
        srcs = (
            OmegaConf.to_container(cfg.get("dummy_srcs"))
            if "dummy_srcs" in cfg
            else None
        )

    # 2. Dynamically Instantiate the Metric
    # This is the crucial part that allows zero-code extension!
    if "metric" not in cfg:
        raise ValueError("A scoring metric must be provided in cfg.metric")

    logger.info(f"Instantiating custom metric from config: {cfg.metric._target_}")
    try:
        metric_fn = instantiate(cfg.metric)
    except Exception as e:
        raise RuntimeError(f"Failed to instantiate metric from {cfg.metric}: {e}")

    if not callable(metric_fn):
        raise TypeError(
            f"The instantiated metric must be a callable function, got {type(metric_fn)}"
        )

    # 3. Compute MBR Utilities
    logger.info("Computing MBR expected utilities...")
    utilities = mbr_corpus(
        hyps=hyps,
        metric=metric_fn,
        srcs=srcs,
        num_subsamples=cfg.get("num_subsamples", None),
        aggregation=cfg.get("aggregation", "mean"),
    )

    # 4. Select Best Predictions
    logger.info("Selecting best hypotheses...")
    best_predictions = select_best_hypotheses(hyps, utilities)

    logger.info("MBR Evaluation Completed!")
    return best_predictions
