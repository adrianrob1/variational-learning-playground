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
    get_linear_schedule_with_warmup,
)
from transformers.optimization import get_cosine_schedule_with_warmup
from vloptimizers.ivon import IVON

logger = logging.getLogger(__name__)


def get_optimizer_and_scheduler(model, cfg, dataset_len):
    """
    Returns the optimizer and learning rate scheduler.
    Instantiates IVON natively if specified in hydra config.
    """
    opt_cfg = cfg.optimizer

    if "ivon" in opt_cfg._target_.lower():
        optimizer = IVON(
            [p for n, p in model.named_parameters() if p.requires_grad],
            lr=opt_cfg.get("lr", opt_cfg.get("learning_rate")),
            mc_samples=opt_cfg.get("mc_samples", 1),
            beta1=opt_cfg.get("beta1", 0.9),
            beta2=opt_cfg.get("beta2", 0.999),
            weight_decay=opt_cfg.get("weight_decay", 1e-4),
            hess_init=opt_cfg.get("hess_init", 1.0),
            clip_radius=opt_cfg.get("clip_radius", float("inf")),
            ess=opt_cfg.get("ess", 5e4),
        )
    else:
        # Fallback to standard instantiation for other optimizers
        optimizer = instantiate(
            opt_cfg, params=[p for n, p in model.named_parameters() if p.requires_grad]
        )

    lr_scheduler = None
    if cfg.get("lr_scheduler"):
        steps_per_epoch = dataset_len // (
            cfg.training_args.get("per_device_train_batch_size", 8)
            * cfg.training_args.get("gradient_accumulation_steps", 1)
        )
        max_steps = steps_per_epoch * cfg.training_args.get("num_train_epochs", 3)

        if cfg.lr_scheduler == "cosine":
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=cfg.training_args.get("warmup_steps", 0),
                num_training_steps=max_steps,
            )
        elif cfg.lr_scheduler == "linear":
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=cfg.training_args.get("warmup_steps", 0),
                num_training_steps=max_steps,
            )

    return optimizer, lr_scheduler


def train(cfg: DictConfig) -> None:
    """
    Train a HuggingFace Seq2Seq or CausalLM model strictly using the HF Trainer.
    """
    logger.info(f"Starting Text Generation Training with Config:\n{cfg}")

    # 1. Load Dataset
    logger.info("Loading datasets via Hydra instantiation")
    train_dataset = instantiate(cfg.dataset.train)
    eval_dataset = instantiate(cfg.dataset.val) if "val" in cfg.dataset else None

    # 2. Extract Data Collator
    data_collator = None
    if "data_collator" in cfg:
        data_collator = instantiate(cfg.data_collator)

    # 3. Load Model and Tokenizer
    model_name_or_path = cfg.model.model_name_or_path
    logger.info(f"Loading tokenizer from {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading model from {model_name_or_path}")
    model_type = cfg.model.get("type", "causal_lm")
    if model_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    # 4. Optimizer and LR Scheduler setup
    optimizer, lr_scheduler = get_optimizer_and_scheduler(
        model, cfg, len(train_dataset)
    )

    # Pre-sample if IVON
    if isinstance(optimizer, IVON) and cfg.get("sample_params", False):
        optimizer._sample_params()

    # 5. Trainer Setup
    logger.info("Setting up HF Trainer")
    t_args = dict(cfg.get("training_args", {}))
    if "output_dir" not in t_args:
        t_args["output_dir"] = cfg.get("output_dir", "./output")

    training_args = TrainingArguments(**t_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        optimizers=(optimizer, lr_scheduler),
        tokenizer=tokenizer,
    )

    if isinstance(optimizer, IVON):
        # Configure inference mc_samples similar to original repo
        trainer.mc_samples = cfg.optimizer.get("inference_mc_samples", 1)

    # 6. Train
    logger.info("Starting training loop")
    train_result = trainer.train(
        resume_from_checkpoint=cfg.get("resume_from_checkpoint", None)
    )
    logger.info("Training completed.")

    trainer.save_state()

    # Save model normally
    trainer.save_model(cfg.get("output_dir", "./output"))

    # Specifically save IVON Hessian properly if configured
    if cfg.get("save_ivon_hessian", True) and isinstance(
        trainer.optimizer.optimizer
        if hasattr(trainer.optimizer, "optimizer")
        else trainer.optimizer,
        IVON,
    ):
        logger.info("Saving IVON Hessian logic.")
        # Re-apply hessian for saving (ported from `main.py`)
        lower_bound = 0
        opt = (
            trainer.optimizer.optimizer
            if hasattr(trainer.optimizer, "optimizer")
            else trainer.optimizer
        )
        hessian = opt.param_groups[0]["hess"]
        for n, p in trainer.model.named_parameters():
            if p.requires_grad:
                length = p.flatten().shape[0]
                local_hessian = hessian[lower_bound : lower_bound + length]
                local_hessian = local_hessian.reshape(p.shape)
                p.data.copy_(local_hessian)
                lower_bound += length

        hessian_dir = f"{cfg.get('output_dir', './output')}/hessian"
        trainer.save_model(hessian_dir)
        logger.info(f"Hessian saved to {hessian_dir}")

    return train_result
