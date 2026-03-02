# Copyright (c) 2026 Adrian R. Minut
# Copyright (c) 2026 ABI Team
# 
# SPDX-License-Identifier: GPL-3.0
# SPDX-License-Identifier: Apache 2.0

import logging
import torch
from omegaconf import DictConfig
from hydra.utils import instantiate
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


def predict(cfg: DictConfig) -> None:
    """
    Run inference/prediction for a text generation model.
    Loads a fine-tuned model (potentially with IVON weights) and generates predictions.
    """
    logger.info(f"Starting Text Generation Prediction with Config:\n{cfg}")

    # 1. Load Dataset
    logger.info("Loading test dataset")
    test_dataset = instantiate(cfg.dataset.test) if "test" in cfg.dataset else None
    if test_dataset is None:
        raise ValueError("A test dataset must be provided in cfg.dataset.test")

    # 2. Extract Data Collator
    data_collator = None
    if "data_collator" in cfg:
        data_collator = instantiate(cfg.data_collator)

    # 3. Load Model and Tokenizer
    model_name_or_path = cfg.model.model_name_or_path
    logger.info(f"Loading tokenizer & model from {model_name_or_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_type = cfg.model.get("type", "causal_lm")
    if model_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    model.to(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # Generation parameters
    generation_config = cfg.get("generation", {})
    model.config.num_beams = generation_config.get("beam_size", 1)
    model.config.max_length = generation_config.get("max_length", 512)
    model.config.do_sample = generation_config.get("do_sample", False)
    model.config.top_k = generation_config.get("top_k", 50)
    model.config.length_penalty = generation_config.get("length_penalty", 1.0)
    model.config.num_return_sequences = generation_config.get("num_return_sequences", 1)

    # 4. Trainer Setup for Prediction
    training_args = TrainingArguments(
        output_dir=cfg.get("output_dir", "./output"),
        per_device_eval_batch_size=cfg.get("batch_size", 8),
        **cfg.get("training_args", {}),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Note: original repo has custom trainer logic for IVON inference (MC sampling)
    if cfg.get("ivon_inference", False):
        trainer.mc_samples = cfg.get("mc_samples", 1)

    logger.info("Starting prediction")
    results = trainer.predict(test_dataset)

    # Post processing logic can be appended here (e.g., converting token IDs to strings and saving)
    logger.info("Prediction completed")

    return results
