# Copyright (c) 2026 Adrian R. Minut
# Copyright (c) 2024 ABI Team
#
# SPDX-License-Identifier: GPL-3.0

"""Distributed training script for large-scale image classification (ImageNet)."""

import os
from os.path import join as pjoin, exists
import torch
import torch.nn.functional as nnf
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR

import hydra
from omegaconf import DictConfig, OmegaConf

from vlbench.models import MODELS
from vldatasets.standard import NTRAIN, OUTCLASS
from vldatasets.standard.dataloaders import (
    ImageNetInfo,
    get_imagenet_train_loader,
    get_imagenet_test_loader,
    get_imagenet_train_loader_torch,
    get_imagenet_test_loader_torch,
)
from vlbench.train.utils import coro_timer, mkdirp
from vlbench.train.trainutils import (
    coro_log_timed,
    do_epoch,
    do_evalbatch,
    check_cuda,
    savecheckpoint,
    loadcheckpoint,
)


def do_trainbatch(batchinput, model, optimizer):
    """Standard training batch update."""
    images, target = batchinput
    optimizer.zero_grad(set_to_none=True)
    output = model(images)
    loss = nnf.cross_entropy(output, target)
    loss.backward()
    optimizer.step()
    return nnf.softmax(output.detach(), -1), target, loss.item()


def do_trainbatch_von(batchinput, model, optimizer, mc_samples=1):
    """Training batch update for IVON with posterior sampling and DDP synchronization."""
    images, target = batchinput
    loss_samples = []
    prob_samples = []

    for _ in range(mc_samples):
        # model.no_sync() is crucial for shared gradient computation across samples
        with optimizer.sampled_params(train=True), model.no_sync():
            optimizer.zero_grad(set_to_none=True)
            output = model(images)
            loss = nnf.cross_entropy(output, target)
            loss.backward()

        loss_samples.append(loss.detach())
        prob_samples.append(nnf.softmax(output.detach(), -1))

    optimizer.step()

    loss = torch.mean(torch.stack(loss_samples, dim=0), dim=0)
    prob = torch.mean(torch.stack(prob_samples, dim=0), dim=0)

    return prob, target, loss.item()


@hydra.main(
    config_path="../../../conf/image_classification",
    config_name="distributed_config",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    """Main entry point for distributed training.

    Args:
        cfg: Hydra configuration object.
    """
    global_rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    timer = coro_timer()
    t_init = next(timer)

    if local_rank == 0:
        print(f">>> Training initiated at {t_init.isoformat()} <<<\n")
        print(OmegaConf.to_yaml(cfg))
        check_cuda()
        mkdirp(cfg.save_dir)

    if cfg.seed is not None:
        torch.manual_seed(cfg.seed * world_size + global_rank)
    else:
        torch.manual_seed(42 * world_size + global_rank)

    if world_size > 1:
        dist.init_process_group(backend="nccl")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Model instantiation
    if cfg.resume:
        startepoch, model, optimizer, scheduler, dic = loadcheckpoint(
            cfg.resume, device, epochs=cfg.epochs
        )
        if hasattr(optimizer, "debias"):
            optimizer.debias = False  # don't debias gradients when resuming
        if hasattr(optimizer, "sync"):
            optimizer.sync = True
        modelargs, modelkwargs = dic["modelargs"], dic["modelkwargs"]
        if local_rank == 0:
            print(f"Resumed from {cfg.resume}")
    else:
        startepoch = 0
        outclass = OUTCLASS.get(cfg.dataset.name, ImageNetInfo.outclass)
        modelargs, modelkwargs = (outclass,), {}
        model = MODELS[cfg.model.name](*modelargs, **modelkwargs).to(device)

    model = DDP(model, device_ids=[local_rank])

    # Optimizer instantiation
    if not cfg.resume:
        opt_cfg = OmegaConf.to_container(cfg.method, resolve=True)
        opt_cfg.pop("name", None)
        optimizer = hydra.utils.instantiate(opt_cfg, params=model.parameters())

        scheduler = (
            LinearLR(
                optimizer,
                start_factor=1.0 / cfg.warmup,
                end_factor=1.0,
                total_iters=cfg.warmup,
            )
            if cfg.warmup > 0
            else None
        )

    # Dataloaders
    local_tbatch = int(cfg.tbatch / world_size)
    local_vbatch = int(cfg.vbatch / world_size)

    use_ffcv = exists(pjoin(cfg.data_dir, "train.ffcv"))
    if use_ffcv:
        if local_rank == 0:
            print("Using FFCV loaders for ImageNet")
        train_loader = get_imagenet_train_loader(
            cfg.data_dir,
            workers=cfg.workers,
            tbatch=local_tbatch,
            device=device,
            distributed=(world_size > 1),
        )

        val_loader = get_imagenet_test_loader(
            cfg.data_dir,
            workers=cfg.workers,
            batch=local_vbatch,
            device=device,
            distributed=(world_size > 1),
        )
    else:
        if local_rank == 0:
            print("FFCV files not found, using standard PyTorch loaders for ImageNet")
        train_loader = get_imagenet_train_loader_torch(
            cfg.data_dir,
            workers=cfg.workers,
            tbatch=local_tbatch,
            pin_memory=(device.type == "cuda"),
            distributed=(world_size > 1),
        )

        val_loader = get_imagenet_test_loader_torch(
            cfg.data_dir,
            workers=cfg.workers,
            batch=local_vbatch,
            pin_memory=(device.type == "cuda"),
            distributed=(world_size > 1),
        )

    log_ece = coro_log_timed(
        None,
        cfg.printfreq,
        cfg.bins,
        cfg.save_dir,
        global_rank,
        append=bool(cfg.resume),
    )

    if local_rank == 0:
        print(
            f"Data size: {NTRAIN.get(cfg.dataset.name, ImageNetInfo.counts['train'])}"
        )
        print(f"Parameter size: {sum(p.nelement() for p in model.parameters())}")
        print(f">>> Training starts at {next(timer)[0].isoformat()} <<<\n")

    for e in range(startepoch, cfg.epochs):
        log_ece.send((e, "train", len(train_loader), None))

        if e == cfg.warmup:
            if local_rank == 0:
                print("End of warmup epochs, starting cosine annealing")
            scheduler = CosineAnnealingLR(optimizer, eta_min=0.0, T_max=cfg.epochs)

        model.train()

        # Dispatch training batch
        if cfg.method.name in ["ivon", "vogn"]:

            def batch_fn(b, **kw):
                return do_trainbatch_von(b, mc_samples=cfg.mc_samples, **kw)
        else:
            batch_fn = do_trainbatch

        do_epoch(
            train_loader,
            batch_fn,
            log_ece,
            device,
            model=model,
            optimizer=optimizer,
        )

        if scheduler:
            scheduler.step()
        log_ece.throw(StopIteration)

        if local_rank == 0:
            savecheckpoint(
                pjoin(cfg.save_dir, "checkpoint.pt"),
                cfg.model.name,
                modelargs,
                modelkwargs,
                model.module,
                optimizer,
                scheduler,
            )

            # Periodic checkpoints
            if e % 50 == 0 or e == cfg.epochs - 1:
                savecheckpoint(
                    pjoin(cfg.save_dir, f"checkpoint_{e:03d}.pt"),
                    cfg.model.name,
                    modelargs,
                    modelkwargs,
                    model.module,
                    optimizer,
                    scheduler,
                )

            time_per_epoch = next(timer)[1]
            print(f">>> Epoch {e} Time elapsed: {time_per_epoch} <<<\n")

        # Evaluation
        if local_rank == 0:
            log_ece.send((e, "test", len(val_loader), None))
            with torch.no_grad():
                model.eval()
                do_epoch(val_loader, do_evalbatch, log_ece, device, model=model)
            log_ece.throw(StopIteration)

    log_ece.close()
    if local_rank == 0:
        print(f">>> Training completed at {next(timer)[0].isoformat()} <<<\n")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
