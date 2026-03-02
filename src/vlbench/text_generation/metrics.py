# Copyright (c) 2026 Adrian R. Minut
# Copyright (c) 2026 ABI Team
# 
# SPDX-License-Identifier: GPL-3.0
# SPDX-License-Identifier: Apache 2.0

from typing import List, Tuple
import numpy as np


def bleu(
    hyps: List[str], refs: List[str], srcs: List[str] = None
) -> Tuple[List[float], float]:
    """
    Compute sentence-level and corpus-level BLEU scores using SacreBLEU.

    Args:
        hyps: List of candidate hypotheses.
        refs: List of reference hypotheses.
        srcs: Optional list of source texts (unused by BLEU, kept for metric signature compatibility).

    Returns:
        A tuple (sentence_scores, corpus_score).
    """
    import sacrebleu

    def bleu_fn(hyp, ref):
        return sacrebleu.sentence_bleu(hyp, [ref]).score

    sentence_scores = [bleu_fn(hyp, ref) for hyp, ref in zip(hyps, refs)]

    corpus_score = sacrebleu.corpus_bleu(hyps, [refs]).score

    return sentence_scores, corpus_score


def bleurt(
    hyps: List[str], refs: List[str], srcs: List[str] = None, batch_size: int = 64
) -> Tuple[List[float], float]:
    """
    Compute sentence-level and corpus-level BLEURT scores.

    Args:
        hyps: List of candidate hypotheses.
        refs: List of reference hypotheses.
        srcs: Optional list of source texts.
        batch_size: Batch size for inference.

    Returns:
        A tuple (sentence_scores, corpus_score).
    """
    try:
        from bleurt import score
    except ImportError:
        raise ImportError("bleurt is not installed. Please install it manually.")

    bleurt_scorer = score.LengthBatchingBleurtScorer(
        None
    )  # Assuming default model if no dir
    bleurt_scores = bleurt_scorer.score(
        references=refs,
        candidates=hyps,
        batch_size=batch_size,
    )
    return bleurt_scores, float(np.mean(bleurt_scores))


def comet_metric(
    hyps: List[str],
    refs: List[str],
    srcs: List[str],
    model_name: str = "wmt20-comet-da",
    batch_size: int = 64,
) -> Tuple[List[float], float]:
    """
    Compute sentence-level and corpus-level COMET scores.

    Args:
        hyps: List of candidate hypotheses.
        refs: List of reference hypotheses.
        srcs: List of source texts.
        model_name: Name of the COMET model to use.
        batch_size: Batch size for inference.

    Returns:
        A tuple (sentence_scores, corpus_score).
    """
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError:
        raise ImportError(
            "comet is not installed. Please install 'unbabel-comet' manually."
        )

    comet_path = download_model(model_name, None)
    comet_model = load_from_checkpoint(comet_path)
    comet_input = [
        {"src": src, "mt": mt, "ref": ref} for src, mt, ref in zip(srcs, hyps, refs)
    ]

    pred = comet_model.predict(
        comet_input,
        batch_size=batch_size,
        sort_by_mtlen=True,
        progress_bar=False,
        num_workers=1,
    )
    if hasattr(pred, "scores") and hasattr(pred, "system_score"):
        return pred.scores, pred.system_score
    return pred[0], pred[1]


def bertscore(
    hyps: List[str], refs: List[str], srcs: List[str] = None, batch_size: int = 32
) -> Tuple[List[float], float]:
    """
    Compute sentence-level and corpus-level BERTScore.

    Args:
        hyps: List of candidate hypotheses.
        refs: List of reference hypotheses.
        srcs: Optional list of source texts.
        batch_size: Batch size for inference.

    Returns:
        A tuple (sentence_scores, corpus_score).
    """
    try:
        import bert_score
    except ImportError:
        raise ImportError("bert_score is not installed. Please install it manually.")

    bert_scores = (
        bert_score.score(
            hyps,
            refs,
            model_type="microsoft/deberta-base-mnli",
            batch_size=batch_size,
            verbose=False,
        )[-1]
        .cpu()
        .numpy()
        .tolist()
    )

    return bert_scores, float(np.mean(bert_scores))
