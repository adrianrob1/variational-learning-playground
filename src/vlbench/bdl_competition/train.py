# Copyright (c) 2026 Adrian R. Minut
# Copyright (c) 2024 ABI Team
# 
# SPDX-License-Identifier: GPL-3.0

import os
import torch
import torch.nn.functional as nnf
import numpy as np
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR

import hydra
from omegaconf import DictConfig, OmegaConf

from vlbench.models import MODELS
from vldatasets.standard import get_bdl_loaders
from vlbench.train.utils import coro_timer, mkdirp
from vlbench.train.trainutils import (
    coro_log,
    do_epoch,
    do_evalbatch,
    SummaryWriter,
    check_cuda,
    deteministic_run,
    savecheckpoint,
)


def avneg_loglik_gaussian(output, y):
    """
    Computes the negative log-likelihood for Gaussian regression.
    First output is predictive mean, second is inverse-softplus of predictive std.
    """
    predictions_mean, predictions_std_raw = output.split([1, 1], dim=-1)
    predictions_std = nnf.softplus(predictions_std_raw)

    se = (predictions_mean - y) ** 2
    log_likelihood = -0.5 * se / predictions_std**2 - 0.5 * torch.log(
        predictions_std**2 * 2 * torch.pi
    )
    return -torch.mean(log_likelihood)


def do_trainbatch_von(
    batchinput, model, optimizer, lossfun, mc_samples=1, create_graph=False
):
    """Training batch update for IVON with posterior sampling."""
    images, target = batchinput
    loss_samples = []
    prob_samples = []

    for _ in range(mc_samples):
        with optimizer.sampled_params(train=True):
            optimizer.zero_grad(set_to_none=True)
            output = model(images)
            loss = lossfun(output, target)
            if create_graph:
                params = [p for p in model.parameters() if p.requires_grad]
                grads = torch.autograd.grad(loss, params, create_graph=True)
                for p, g in zip(params, grads):
                    p.grad = g
            else:
                loss.backward()
        loss_samples.append(loss.detach())
        if output.shape[-1] > 1:  # Classification
            prob_samples.append(nnf.softmax(output.detach(), -1))
        else:  # Regression
            prob_samples.append(output.detach())

    optimizer.step()
    loss = torch.mean(torch.stack(loss_samples, dim=0), dim=0)
    prob = torch.mean(torch.stack(prob_samples, dim=0), dim=0)
    return prob, target, loss.item()


@hydra.main(
    config_path="../../../conf/bdl_competition",
    config_name="config",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    """Main training entry point for BDL competition task."""
    timer = coro_timer()
    t_init = next(timer)
    print(f">>> Training initiated at {t_init.isoformat()} <<<\n")
    print(OmegaConf.to_yaml(cfg))

    if cfg.seed is not None:
        deteministic_run(seed=cfg.seed)

    device = torch.device(cfg.device)
    if device.type == "cuda":
        check_cuda()

    mkdirp(cfg.save_dir)
    sw = SummaryWriter(cfg.tensorboard_dir) if cfg.tensorboard_dir else None

    # Load data
    train_loader, val_loader = get_bdl_loaders(
        cfg.dataset.name, cfg.data_dir, cfg.tbatch, cfg.vbatch, cfg.workers, cfg.device
    )

    # Dataset specific info
    model_kwargs = {k: v for k, v in cfg.model.items() if k != "name"}
    if cfg.dataset.name == "uci":
        num_features = next(iter(train_loader))[0].shape[1]
        model = MODELS[cfg.model.name](num_features=num_features, **model_kwargs).to(
            device
        )
        lossfun = avneg_loglik_gaussian
        modelargs, modelkwargs = (num_features,), model_kwargs
    else:
        # For CIFAR and MedMNIST, classes are 10 and 8 (for dermamnist)
        # We'll just pass outclass from config
        model = MODELS[cfg.model.name](
            outclass=cfg.dataset.outclass, **model_kwargs
        ).to(device)
        lossfun = nnf.cross_entropy
        modelargs, modelkwargs = (cfg.dataset.outclass,), model_kwargs

    # Instantiate optimizer via Hydra
    opt_cfg = OmegaConf.to_container(cfg.method, resolve=True)
    # We follow the indomain.train pattern of popping 'name' if it's there
    opt_cfg.pop("name", None)
    create_graph = opt_cfg.pop("create_graph", False)

    optimizer = hydra.utils.instantiate(opt_cfg, params=model.parameters())

    scheduler = (
        LinearLR(optimizer, start_factor=1.0 / cfg.warmup, total_iters=cfg.warmup)
        if cfg.warmup > 0
        else None
    )

    if cfg.dataset.name != "uci":
        log_ece = coro_log(sw, cfg.printfreq, cfg.bins, cfg.save_dir)
    else:
        log_ece = None
    print(f">>> Training starts at {next(timer)[0].isoformat()} <<<\n")

    for e in range(cfg.epochs):
        if e == cfg.warmup:
            print("Warmup complete, starting CosineAnnealingLR")
            scheduler = CosineAnnealingLR(
                optimizer, T_max=cfg.epochs, eta_min=cfg.lr_final
            )

        if log_ece:
            log_ece.send((e, "train", len(train_loader), None))
        else:
            print(f"Epoch {e} Train:")

        model.train()

        def batch_fn(b, **kw):
            return do_trainbatch_von(
                b,
                lossfun=lossfun,
                mc_samples=cfg.mc_samples,
                create_graph=create_graph,
                **kw,
            )

        if log_ece:
            do_epoch(
                train_loader,
                batch_fn,
                log_ece,
                device,
                model=model,
                optimizer=optimizer,
            )
        else:
            losses = []
            for i, batch in enumerate(train_loader):
                _, _, loss = batch_fn(batch, model=model, optimizer=optimizer)
                losses.append(loss)
                if i % cfg.printfreq == 0:
                    print(f"Batch {i}/{len(train_loader)} Loss: {loss:.4f}")
            print(f"Mean Train Loss: {np.mean(losses):.4f}")

        if scheduler:
            scheduler.step()
        if log_ece:
            log_ece.throw(StopIteration)

        # Save checkpoint
        savecheckpoint(
            os.path.join(cfg.save_dir, f"checkpoint_seed{cfg.seed}.pt"),
            cfg.model.name,
            modelargs,
            modelkwargs,
            model,
            optimizer,
            scheduler,
        )

        # Validation
        if log_ece:
            log_ece.send((e, "val", len(val_loader), None))
        else:
            print(f"Epoch {e} Val:")

        model.eval()

        if log_ece:
            do_epoch(val_loader, do_evalbatch, log_ece, device, model=model)
            log_ece.throw(StopIteration)
        else:
            losses = []
            for i, batch in enumerate(val_loader):
                with torch.no_grad():
                    output = model(batch[0].to(device))
                    loss = lossfun(output, batch[1].to(device))
                    losses.append(loss.item())
            print(f"Mean Val Loss: {np.mean(losses):.4f}")

        print(f">>> Time elapsed: {next(timer)[1]} <<<\n")

    if log_ece:
        log_ece.close()
    print(f">>> Training completed at {next(timer)[0].isoformat()} <<<\n")


if __name__ == "__main__":
    main()
