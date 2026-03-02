# Copyright (c) 2026 Adrian R. Minut
# Copyright (c) 2026 ABI Team
# SPDX-License-Identifier: GPL-3.0

"""Script for Monte Carlo samples ablation during inference."""

import os
import torch
import torch.nn.functional as nnf
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import List, Optional

from vldatasets.standard import TESTDATALOADER, OUTCLASS
from vlbench.train.trainutils import (
    loadcheckpoint,
    do_epoch,
    do_evalbatch,
    coro_log_auroc,
    check_cuda,
    summarize_csv,
)
from vlbench.train.utils import mkdirp


def run_evaluation(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    dataloader: torch.utils.data.DataLoader,
    mc_samples_list: List[int],
    device: torch.device,
    save_dir: str,
    prefix: str = "eval_mc",
):
    """
    Run evaluation for multiple MC sample counts.
    """
    mkdirp(save_dir)
    log_ece = coro_log_auroc(logfreq=10, nbin=20, save_dir=save_dir)

    for mc in mc_samples_list:
        print(f"\n>>> Evaluating with {mc} MC samples <<<")
        # Note: we use 'repeat' in do_evalbatch to handle MC samples if no sampled_params
        # OR if we want to explicitly control samples from here.
        # If optimizer is IVON, repeat=mc will loop correctly over sampled_params inside do_evalbatch.

        log_ece.send((mc, f"{prefix}", len(dataloader), None))
        with torch.no_grad():
            model.eval()
            do_epoch(
                dataloader,
                do_evalbatch,
                log_ece,
                device,
                model=model,
                optimizer=optimizer,
                repeat=mc,
            )
        log_ece.throw(StopIteration)

    log_ece.close()

    print(f"\n>>> Summary of results in {save_dir}/{prefix}.csv <<<")
    summarize_csv(os.path.join(save_dir, f"{prefix}.csv"))


@hydra.main(config_path="../conf/indomain", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main entry point for MC samples ablation."""
    print("--- Configuration ---")
    print(OmegaConf.to_yaml(cfg))

    device = torch.device(cfg.device)
    if device.type == "cuda":
        check_cuda()

    # Path to checkpoint to evaluate
    # Use save_dir/checkpoint.pt as default if not explicitly provided
    checkpoint_path = cfg.get(
        "checkpoint_path", os.path.join(cfg.save_dir, "checkpoint.pt")
    )
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    print(f"Loading checkpoint from {checkpoint_path}...")
    # Load checkpoint
    try:
        _, model, optimizer, _, dic = loadcheckpoint(checkpoint_path, device=device)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        # Fallback to model only if optimizer load fails (e.g. key mismatch)
        from vlbench.models import MODELS

        dataset_name = cfg.dataset.name
        outclass = OUTCLASS[dataset_name]
        model = MODELS[cfg.model.name](outclass).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["modelstates"])
        optimizer = None
        print("Loaded model only (no optimizer state).")

    model.to(device)

    # Load test dataset
    dataset_name = cfg.dataset.name
    print(f"Loading test dataloader for {dataset_name}...")
    test_loader = TESTDATALOADER[dataset_name](
        cfg.data_dir, cfg.workers, device.type == "cuda", cfg.vbatch
    )

    # MC samples to test
    mc_list = cfg.get("mc_samples_list", [1, 2, 4, 8, 16, 32, 64])
    if isinstance(mc_list, str):
        mc_list = [int(x) for x in mc_list.split(",")]

    run_evaluation(
        model,
        optimizer,
        test_loader,
        mc_list,
        device,
        cfg.save_dir,
        prefix="test_mc_ablation",
    )


if __name__ == "__main__":
    main()
