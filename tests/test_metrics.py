# Copyright (c) 2026 Adrian R. Minut
# SPDX-License-Identifier: GPL-3.0

import pytest
from unittest.mock import patch, MagicMock
from vlbench.text_generation.metrics import bleu, bleurt, comet_metric, bertscore


def test_bleu_metric_signature():
    """Test BLEU wrapper."""
    hyps = ["the cat sat", "a dog ran"]
    refs = ["the cat sat on mat", "a dog walked"]

    sent_scores, corpus_score = bleu(hyps, refs)
    assert len(sent_scores) == 2
    assert isinstance(corpus_score, float)


def test_bleurt_metric_signature():
    """Test BLEURT wrapper signature with mock."""
    hyps = ["the cat sat", "a dog ran"]
    refs = ["the cat sat on mat", "a dog walked"]

    mock_bleurt = MagicMock()
    mock_score = MagicMock()
    mock_scorer = MagicMock()
    mock_scorer.score.return_value = [0.1, 0.2]
    mock_score.LengthBatchingBleurtScorer.return_value = mock_scorer
    mock_bleurt.score = mock_score

    with patch.dict("sys.modules", {"bleurt": mock_bleurt, "bleurt.score": mock_score}):
        sent_scores, corpus_score = bleurt(hyps, refs)
        assert len(sent_scores) == 2
        assert isinstance(corpus_score, float)


def test_comet_metric_signature():
    """Test COMET metric signature with mock."""
    hyps = ["the cat sat", "a dog ran"]
    refs = ["the cat sat on mat", "a dog walked"]
    srcs = ["le chat s'est assis", "un chien a couru"]

    mock_comet = MagicMock()
    mock_model = MagicMock()
    mock_pred = MagicMock()
    mock_pred.scores = [0.5, 0.6]
    mock_pred.system_score = 0.55
    mock_model.predict.return_value = mock_pred
    mock_comet.load_from_checkpoint.return_value = mock_model
    mock_comet.download_model.return_value = "dummy"

    with patch.dict("sys.modules", {"comet": mock_comet}):
        sent_scores, corpus_score = comet_metric(hyps, refs, srcs)
        assert len(sent_scores) == 2
        assert isinstance(corpus_score, float)


def test_bertscore_metric_signature():
    """Test BERTScore wrapper signature with mock."""
    hyps = ["the cat sat", "a dog ran"]
    refs = ["the cat sat on mat", "a dog walked"]

    import torch

    mock_bert_score = MagicMock()
    mock_bert_score.score.return_value = (None, None, torch.tensor([0.8, 0.9]))

    with patch.dict("sys.modules", {"bert_score": mock_bert_score}):
        sent_scores, corpus_score = bertscore(hyps, refs)
        assert len(sent_scores) == 2
        assert isinstance(corpus_score, float)
