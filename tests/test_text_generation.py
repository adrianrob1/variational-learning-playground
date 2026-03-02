# Copyright (c) 2026 Adrian R. Minut
# SPDX-License-Identifier: GPL-3.0

import torch
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf
from vlbench.text_generation.train import train
from vlbench.text_generation.predict import predict
from vlbench.text_generation.eval_mbr import evaluate_mbr


@patch("vlbench.text_generation.train.Trainer")
@patch("vlbench.text_generation.train.AutoModelForCausalLM")
@patch("vlbench.text_generation.train.AutoTokenizer")
@patch("vlbench.text_generation.train.instantiate")
def test_train_signature(mock_instantiate, mock_tokenizer, mock_model, mock_trainer):
    """Test that train function accepts a DictConfig and executes its main flow."""
    cfg = OmegaConf.create(
        {
            "model": {"model_name_or_path": "dummy/model", "type": "causal_lm"},
            "dataset": {"train": {"_target_": "dummy.TrainDataset"}},
            "optimizer": {"_target_": "vloptimizers.ivon.IVON", "lr": 1e-3},
            "training_args": {
                "output_dir": "dummy_dir",
                "per_device_train_batch_size": 2,
                "num_train_epochs": 1,
            },
            "output_dir": "dummy_dir",
            "save_ivon_hessian": False,
        }
    )

    mock_model_instance = MagicMock()
    mock_model_instance.named_parameters.return_value = [
        ("dummy", torch.nn.Parameter(torch.zeros(1)))
    ]
    mock_model.from_pretrained.return_value = mock_model_instance
    mock_tokenizer.from_pretrained.return_value = MagicMock(
        pad_token="[PAD]", eos_token="[EOS]"
    )
    mock_instantiate.return_value = MagicMock(__len__=lambda self: 10)

    # Should not raise any errors
    train(cfg)


@patch("vlbench.text_generation.predict.Trainer")
@patch("vlbench.text_generation.predict.AutoModelForCausalLM")
@patch("vlbench.text_generation.predict.AutoTokenizer")
@patch("vlbench.text_generation.predict.instantiate")
def test_predict_signature(mock_instantiate, mock_tokenizer, mock_model, mock_trainer):
    """Test that predict function accepts a DictConfig and executes its main flow."""
    cfg = OmegaConf.create(
        {
            "model": {
                "model_name_or_path": "dummy/model",
                "type": "causal_lm",
            },
            "dataset": {"test": {"_target_": "dummy.TestDataset"}},
            "generation": {"num_return_sequences": 10},
            "output_file": "preds.json",
            "output_dir": "dummy_dir",
        }
    )

    # Needs to return a mock predicting object
    mock_trainer_instance = mock_trainer.return_value
    mock_trainer_instance.predict.return_value = MagicMock()

    mock_model.from_pretrained.return_value = MagicMock()
    mock_tokenizer.from_pretrained.return_value = MagicMock(
        pad_token="[PAD]", eos_token="[EOS]"
    )
    mock_instantiate.return_value = MagicMock(__len__=lambda self: 10)

    # Should not raise any errors
    predict(cfg)


def test_evaluate_mbr_signature():
    """Test that evaluate_mbr dynamically instantiates metrics."""
    cfg = OmegaConf.create(
        {
            "dummy_hyps": [["a", "b"]],
            "metric": {
                "_target_": "vlbench.text_generation.metrics.bleu",
                "_partial_": True,
            },
        }
    )

    preds = evaluate_mbr(cfg)
    assert len(preds) == 1
    assert preds[0] in ["a", "b"]
