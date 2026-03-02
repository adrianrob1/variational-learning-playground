# Copyright (c) 2026 Adrian R. Minut
# Copyright (c) 2024 ABI Team
# 
# SPDX-License-Identifier: GPL-3.0

import os
import torch
import torch.nn.functional as F
import numpy as np
import scipy.stats
import hydra
from omegaconf import DictConfig, OmegaConf
from vlbench.models import loadmodel
from vldatasets.standard import get_bdl_loaders


def agreement(predictions: np.array, reference: np.array):
    """Returns 1 if predictions match and 0 otherwise."""
    return (predictions.argmax(axis=-1) == reference.argmax(axis=-1)).mean()


def total_variation_distance(predictions: np.array, reference: np.array):
    """Returns total variation distance."""
    return np.abs(predictions - reference).sum(axis=-1).mean() / 2.0


def w2_distance(predictions: np.array, reference: np.array):
    """Returns W-2 distance"""
    num_samples_required = 1000
    assert predictions.shape[0] == reference.shape[0], "wrong predictions shape"
    assert predictions.shape[1] == num_samples_required, "wrong number of samples"
    return -np.mean(
        [
            scipy.stats.wasserstein_distance(pred, ref)
            for pred, ref in zip(predictions, reference)
        ]
    )


def list_pt_files(directory):
    pt_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".pt"):
                pt_files.append(os.path.join(root, file))
    return pt_files


@hydra.main(
    config_path="../../../conf/bdl_competition",
    config_name="config",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    """Main testing entry point for BDL competition task."""
    print(">>> Testing initiated <<<\n")
    print(OmegaConf.to_yaml(cfg))

    device = torch.device(cfg.device)
    testmc = cfg.get("testmc", 64)

    # Load data
    _, test_loader = get_bdl_loaders(
        cfg.dataset.name, cfg.data_dir, cfg.tbatch, cfg.vbatch, cfg.workers, cfg.device
    )

    # Load GT
    gt_paths = {
        "cifar10": "cifar_probs.csv",
        "medmnist": "medmnist_probs.csv",
        "uci": "uci_samples.csv",
    }
    gt_path = os.path.join(cfg.data_dir, gt_paths[cfg.dataset.name])
    if not os.path.exists(gt_path):
        print(f"Warning: GT file not found at {gt_path}. Skipping metrics vs GT.")
        gt = None
    else:
        gt = np.genfromtxt(gt_path)

    model_filenames = list_pt_files(cfg.save_dir)
    if not model_filenames:
        print(f"No checkpoint files found in {cfg.save_dir}.")
        return

    if cfg.dataset.name in ("cifar10", "medmnist"):
        all_probs = []
        for fn in model_filenames:
            print(f"Processing model {fn}...")
            # We use loadmodel from the registry which handles architecture metadata
            model, dic = loadmodel(fn, device=device)
            # Re-instantiate optimizer to get sampled_params if it was IVON
            # In legacy code, they just used the saved optimizer state.
            # Here we need the optimizer object to sample params.
            # We'll try to re-instantiate it using the saved state if available.
            # For now, let's assume we can use the model as is if it's not a Bayesian sampler.
            # But the task specifically used IVON sampling.

            # Since we saved the optimizer state in train.py, we can re-instantiate it.
            # Actually, loadmodel currently only returns the model and the extra dict.
            # I'll need to instantiate the optimizer manually.

            opt_cfg = OmegaConf.to_container(cfg.method, resolve=True)
            opt_cfg.pop("name", None)
            optimizer = hydra.utils.instantiate(opt_cfg, params=model.parameters())
            if "optimizer_state_dict" in dic:
                optimizer.load_state_dict(dic["optimizer_state_dict"])

            for _ in range(testmc):
                # Check if optimizer has sampled_params (e.g. IVON)
                sampler = getattr(optimizer, "sampled_params", None)
                context = sampler(train=False) if sampler else torch.no_grad()

                with context:
                    probs_per_seed = []
                    for batch in test_loader:
                        images = batch[0].to(device)
                        logits = model(images)
                        probs = torch.softmax(logits, dim=1)
                        probs_per_seed.append(probs.detach().cpu().numpy())

                    probs_per_seed = np.vstack(probs_per_seed)
                    all_probs.append(probs_per_seed.reshape(1, *probs_per_seed.shape))

        probs = np.concatenate(all_probs, axis=0)
        predictions = probs.mean(axis=0)

        if gt is not None:
            if cfg.dataset.name == "medmnist":
                agreements = (
                    agreement(predictions[:1000], gt[:1000]),
                    agreement(predictions, gt),
                )
                tvs = (
                    total_variation_distance(predictions[:1000], gt[:1000]),
                    total_variation_distance(predictions, gt),
                )
            else:  # cifar10
                agreements = (
                    agreement(predictions[:10000], gt[:10000]),
                    agreement(predictions, gt),
                )
                tvs = (
                    total_variation_distance(predictions[:10000], gt[:10000]),
                    total_variation_distance(predictions, gt),
                )

            print(f"Agreement: {agreements}")
            print(f"TV: {tvs}")

            with open(os.path.join(cfg.save_dir, "results.txt"), "w") as f:
                f.write(f"Agreement: {agreements}\nTV: {tvs}\n")

    else:  # uci
        num_samples = 1000
        all_samples = []

        # Pre-load all checkpoints to avoid repeated IO
        checkpoints = []
        for fn in model_filenames:
            checkpoints.append(loadmodel(fn, device=device))

        for _ in range(num_samples):
            comp_idx = np.random.randint(0, len(checkpoints))
            model, dic = checkpoints[comp_idx]

            opt_cfg = OmegaConf.to_container(cfg.method, resolve=True)
            opt_cfg.pop("name", None)
            optimizer = hydra.utils.instantiate(opt_cfg, params=model.parameters())
            if "optimizer_state_dict" in dic:
                optimizer.load_state_dict(dic["optimizer_state_dict"])

            sampler = getattr(optimizer, "sampled_params", None)
            context = sampler(train=False) if sampler else torch.no_grad()

            with context:
                # Load all test data
                x_test = []
                for b in test_loader:
                    x_test.append(b[0])
                x_test = torch.cat(x_test, dim=0).to(device)

                test_preds = model(x_test).detach().cpu()
                mu, sigma_raw = test_preds.split([1, 1], dim=-1)
                sigma = F.softplus(sigma_raw)
                mu, sigma = mu[:, 0].numpy(), sigma[:, 0].numpy()
                eps = np.random.randn(x_test.shape[0])
                samples = mu + eps * sigma
                all_samples.append(samples.reshape(-1, 1))

        samples = np.hstack(all_samples)
        if gt is not None:
            result_public = w2_distance(samples[:100,], gt.T[:100,])
            result_full = w2_distance(samples, gt.T)
            print(f"Public W2: {result_public}\nFull W2: {result_full}")
            with open(os.path.join(cfg.save_dir, "results.txt"), "w") as f:
                f.write(f"Public W2: {result_public}\nFull W2: {result_full}")


if __name__ == "__main__":
    main()
