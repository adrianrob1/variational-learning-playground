# Copyright (c) 2026 Adrian R. Minut
# Copyright (c) 2024 ABI Team
#
# SPDX-License-Identifier: GPL-3.0

"""Consolidated training script for in-domain BDL tasks (CIFAR-10/100)."""

import os
import torch
import torch.nn.functional as nnf
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR

import hydra
from omegaconf import DictConfig, OmegaConf

from vlbench.models import MODELS
from vldatasets.standard import TRAINDATALOADERS, NTRAIN, OUTCLASS, INSIZE
from vlbench.train.utils import coro_timer, mkdirp
from vlbench.train.trainutils import (
    coro_log,
    do_epoch,
    do_evalbatch,
    SummaryWriter,
    check_cuda,
    deteministic_run,
    savecheckpoint,
    bn_update,
)
from bayesian_torch.models.dnn_to_bnn import get_kl_loss


def do_trainbatch(batchinput, model, optimizer, create_graph=False):
    """Standard training batch update."""
    images, target = batchinput
    optimizer.zero_grad(set_to_none=True)
    output = model(images)
    loss = nnf.cross_entropy(output, target)
    if create_graph:
        params = [p for p in model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(loss, params, create_graph=True)
        for p, g in zip(params, grads):
            p.grad = g
    else:
        loss.backward()
    optimizer.step()
    return nnf.softmax(output.detach(), -1), target, loss.item()


def do_trainbatch_von(batchinput, model, optimizer, mc_samples=1, create_graph=False):
    """Training batch update for IVON/VOGN with posterior sampling."""
    images, target = batchinput
    loss_samples = []
    prob_samples = []
    for _ in range(mc_samples):
        with optimizer.sampled_params(train=True):
            optimizer.zero_grad(set_to_none=True)
            output = model(images)
            loss = nnf.cross_entropy(output, target)
            if create_graph:
                params = [p for p in model.parameters() if p.requires_grad]
                grads = torch.autograd.grad(loss, params, create_graph=True)
                for p, g in zip(params, grads):
                    p.grad = g
            else:
                loss.backward()
        loss_samples.append(loss.detach())
        prob_samples.append(nnf.softmax(output.detach(), -1))
    optimizer.step()
    loss = torch.mean(torch.stack(loss_samples, dim=0), dim=0)
    prob = torch.mean(torch.stack(prob_samples, dim=0), dim=0)
    return prob, target, loss.item()


def do_trainbatch_bbb(batchinput, model, optimizer, train_size, temperature=1.0):
    """Training batch update for Bayes-by-Backprop with KL loss."""
    optimizer.zero_grad(set_to_none=True)
    kl = temperature * get_kl_loss(model) / train_size
    kl.backward()
    images, target = batchinput
    output = model(images)
    loss = nnf.cross_entropy(output, target)
    loss.backward()
    optimizer.step()
    return nnf.softmax(output.detach(), -1), target, loss.item() + kl.item()


@hydra.main(
    config_path="../../../conf/indomain", config_name="config", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    """Main training entry point."""
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

    # Prep tensorboard
    sw = SummaryWriter(cfg.tensorboard_dir) if cfg.tensorboard_dir else None

    # Load data
    if "_target_" in cfg.dataset:
        dataset_obj = hydra.utils.instantiate(cfg.dataset)
        train_loader, val_loader = dataset_obj.get_train_loaders(
            data_dir=cfg.data_dir,
            train_val_split=cfg.tvsplit,
            workers=cfg.workers,
            pin_memory=device.type == "cuda",
            tbatch=cfg.tbatch,
            vbatch=cfg.vbatch,
        )
        train_size = int(dataset_obj.ntrain * cfg.tvsplit)
        outclass = dataset_obj.outclass
        dataset_name = dataset_obj.name
        insize = dataset_obj.insize
    else:
        dataset_name = cfg.dataset.name
        train_loader, val_loader = TRAINDATALOADERS[dataset_name](
            cfg.data_dir,
            cfg.tvsplit,
            cfg.workers,
            device.type == "cuda",
            cfg.tbatch,
            cfg.vbatch,
        )
        train_size = int(NTRAIN[dataset_name] * cfg.tvsplit)
        outclass = OUTCLASS[dataset_name]
        insize = INSIZE[dataset_name]

    # Initialize model
    swag_model = None
    if cfg.method.name == "swag":
        from vlbench.models.swag import SWAG

        model = MODELS[cfg.model.name](outclass, input_size=insize).to(device)
        swag_model = SWAG(model, cfg.method.swag_devrank).to(device)
        modelargs, modelkwargs = (
            (outclass, cfg.method.swag_devrank),
            {"input_size": insize},
        )
        arch_name = f"{cfg.model.name}_swag"
    elif cfg.method.name == "bbb":
        # BBB models are converted in registry usually
        model = MODELS[f"{cfg.model.name}_bbb"](
            outclass,
            input_size=insize,
            prior_precision=cfg.method.weight_decay * train_size,
            std_init=cfg.method.std_init / torch.sqrt(torch.tensor(train_size)),
        ).to(device)
        modelargs, modelkwargs = (
            (outclass,),
            {
                "input_size": insize,
                "prior_precision": cfg.method.weight_decay * train_size,
                "std_init": cfg.method.std_init,
            },
        )
        arch_name = f"{cfg.model.name}_bbb"
    else:  # sgd / mcdropout / ivon / vogn
        arch = (
            f"{cfg.model.name}_mcdrop"
            if cfg.method.name == "mcdropout"
            else cfg.model.name
        )
        model = MODELS[arch](outclass, input_size=insize).to(device)
        modelargs, modelkwargs = (outclass,), {"input_size": insize}
        arch_name = arch

    # Instantiate optimizer via Hydra
    opt_cfg = OmegaConf.to_container(cfg.method, resolve=True)
    opt_cfg.pop("name", None)
    create_graph = opt_cfg.pop("create_graph", False)

    extra_kwargs = {}
    if cfg.method.name == "vogn":
        extra_kwargs["data_size"] = train_size

    optimizer = hydra.utils.instantiate(
        opt_cfg, params=model.parameters(), **extra_kwargs
    )

    scheduler = (
        LinearLR(optimizer, start_factor=1.0 / cfg.warmup, total_iters=cfg.warmup)
        if cfg.warmup > 0
        else None
    )

    # Logging
    log_ece = coro_log(sw, cfg.printfreq, cfg.bins, cfg.save_dir)
    print(f">>> Training starts at {next(timer)[0].isoformat()} <<<\n")

    for e in range(cfg.epochs):
        # SWAG phase transition
        if cfg.method.name == "swag" and e == cfg.method.swag_start:
            print(
                f"Starting SWAG collection at epoch {e}, setting LR to {cfg.method.swag_lr}"
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = cfg.method.swag_lr
            # No scheduler for SWAG collection phase in legacy
            scheduler = None

        # Warmup to Cosine transition
        if e == cfg.warmup:
            print("Warmup complete, starting CosineAnnealingLR")
            scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=0.0)

        log_ece.send((e, "train", len(train_loader), None))
        model.train()

        # Dispatch training batch
        if cfg.method.name in ["ivon", "vogn"]:

            def batch_fn(b, **kw):
                return do_trainbatch_von(
                    b, mc_samples=cfg.mc_samples, create_graph=create_graph, **kw
                )
        elif cfg.method.name == "bbb":

            def batch_fn(b, **kw):
                return do_trainbatch_bbb(
                    b, train_size=train_size, temperature=cfg.temperature, **kw
                )
        else:

            def batch_fn(b, **kw):
                return do_trainbatch(b, create_graph=create_graph, **kw)

        do_epoch(
            train_loader, batch_fn, log_ece, device, model=model, optimizer=optimizer
        )
        if scheduler:
            scheduler.step()
        log_ece.throw(StopIteration)

        # SWAG collection
        if cfg.method.name == "swag" and e >= cfg.method.swag_start:
            if (e - cfg.method.swag_start) % cfg.swag_collectfreq == 0:
                swag_model.collect_model()

        # Save checkpoint
        save_obj = swag_model if cfg.method.name == "swag" else model
        savecheckpoint(
            os.path.join(cfg.save_dir, "checkpoint.pt"),
            arch_name,
            modelargs,
            modelkwargs,
            save_obj,
            optimizer,
            scheduler,
        )

        # Validation
        if len(val_loader) > 0:
            log_ece.send((e, "val", len(val_loader), None))
            model.eval()
            do_epoch(val_loader, do_evalbatch, log_ece, device, model=model)
            log_ece.throw(StopIteration)

            # SWAG specific validation
            if (
                cfg.method.name == "swag"
                and e >= cfg.method.swag_start
                and (e - cfg.method.swag_start) % cfg.swag_valfreq
                == cfg.swag_valfreq - 1
            ):
                log_ece.send((e, "swaval", len(val_loader), None))
                swamodel = swag_model.averaged_model()
                if cfg.swag_bnupdate:
                    bn_update(train_loader, swamodel, device=device)
                swamodel.eval()
                do_epoch(val_loader, do_evalbatch, log_ece, device, model=swamodel)
                log_ece.throw(StopIteration)

        print(f">>> Time elapsed: {next(timer)[1]} <<<\n")

    log_ece.close()
    print(f">>> Training completed at {next(timer)[0].isoformat()} <<<\n")


if __name__ == "__main__":
    main()
