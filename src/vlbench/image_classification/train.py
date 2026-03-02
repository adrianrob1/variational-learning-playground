# Copyright (c) 2026 Adrian R. Minut
# Copyright (c) 2024 ABI Team
#
# SPDX-License-Identifier: GPL-3.0

"""Image classification training script.

Run as:
    uv run python -m vlbench.image_classification.train \\
        optimizer=ivon dataset=cifar10 arch=resnet20

Refactored from tasks/image_classification/train.py:
- Removed sys.path hacks
- Uses absolute vlbench/vloptimizers imports
- Accepts Hydra cfg object; argparse kept as fallback for CLI compat.
"""

import argparse
from os.path import join as pjoin
import torch
import torch.nn.functional as nnf

from vlbench.train.utils import coro_timer, mkdirp
from vlbench.models._registry import STANDARDMODELS
from vldatasets.standard.dataloaders import (
    TRAINDATALOADERS,
    TESTDATALOADER,
    NTRAIN,
    OUTCLASS,
    INSIZE,
)
from vlbench.train.trainutils import (
    coro_log_timed,
    do_epoch,
    do_trainbatch,
    do_evalbatch,
    SummaryWriter,
    check_cuda,
    deteministic_run,
    savecheckpoint,
    loadcheckpoint,
)
from vloptimizers.adahessian import AdaHessian
from vloptimizers.ivon import IVON
from vloptimizers.vogn import VOGN
from vloptimizers.variational_adam import VariationalAdam


def get_args():
    """Parse CLI arguments for the image classification training script.

    Returns:
        argparse.Namespace with all hyperparameters.
    """
    parser = argparse.ArgumentParser(description="CIFAR10/100 IVON training")
    parser.add_argument("arch", choices=STANDARDMODELS)
    parser.add_argument("dataset", choices=TRAINDATALOADERS)
    parser.add_argument("-j", "--workers", default=0, type=int)
    parser.add_argument("-tb", "--tbatch", default=512, type=int)
    parser.add_argument("-vb", "--vbatch", default=512, type=int)
    parser.add_argument("-sp", "--tvsplit", default=0.9, type=float)
    parser.add_argument("-e", "--epochs", default=400, type=int)
    parser.add_argument("-lr", "--learning_rate", default=1.0, type=float)
    parser.add_argument("--lr_final", default=0.0, type=float)
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        dest="weight_decay",
    )
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("-pf", "--printfreq", default=200, type=int)
    parser.add_argument("-r", "--resume", default="", type=str)
    parser.add_argument("-d", "--device", default="cpu", type=str)
    parser.add_argument("-s", "--seed", type=int)
    parser.add_argument("-sd", "--save_dir", default="save_temp", type=str)
    parser.add_argument("-dd", "--data_dir", default="../data", type=str)
    parser.add_argument("-nb", "--bins", default=20, type=int)
    parser.add_argument("-pd", "--plotdiagram", action="store_true")
    parser.add_argument("-tbd", "--tensorboard_dir", default="", type=str)
    parser.add_argument("--mc_samples", default=1, type=int)
    parser.add_argument("--momentum_hess", default=0.999, type=float)
    parser.add_argument("--hess_init", default=1.0, type=float)
    parser.add_argument("--ess", default=5e4, type=float)
    parser.add_argument("--clip_radius", default=float("inf"), type=float)
    parser.add_argument("--warmup", default=5, type=int)
    parser.add_argument(
        "-opt",
        "--optimizer",
        default="ivon",
        choices=[
            "ivon",
            "sgd",
            "adamw",
            "adahessian",
            "vogn",
            "variational_adam",
        ],
        type=str,
    )
    return parser.parse_args()


def do_trainbatch_ivon(batchinput, model, optimizer: IVON, mc_samples: int = 1):
    """IVON training step with MC sampling.

    Args:
        batchinput: (images, target) tuple.
        model: nn.Module.
        optimizer: IVON instance.
        mc_samples: Number of Monte Carlo samples per step.

    Returns:
        Tuple of (mean_prob, target, mean_loss).
    """
    images, target = batchinput
    loss_samples = []
    prob_samples = []
    for _ in range(mc_samples):
        with optimizer.sampled_params(train=True):
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


def do_trainbatch_adahessian(batchinput, model, optimizer: AdaHessian):
    """AdaHessian training step (requires create_graph=True backward).

    Uses autograd.grad to avoid memory leak warning from create_graph=True.
    """
    images, target = batchinput
    optimizer.zero_grad(set_to_none=True)
    output = model(images)
    loss = nnf.cross_entropy(output, target)

    # Use autograd.grad to avoid the memory leak warning from backward(create_graph=True)
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, create_graph=True)
    for p, g in zip(params, grads):
        p.grad = g

    optimizer.step()
    prob = nnf.softmax(output.detach(), -1)
    return prob, target, loss.item()


train_functions = {
    "sgd": do_trainbatch,
    "adamw": do_trainbatch,
    "adahessian": do_trainbatch_adahessian,
    "ivon": do_trainbatch_ivon,
    "vogn": do_trainbatch_ivon,
    "variational_adam": do_trainbatch_ivon,
}


def get_optimizer(args, model):
    """Construct and return the optimizer specified by args.optimizer.

    Args:
        args: Namespace (or any object) with optimizer hyperparameter attrs.
        model: nn.Module whose parameters to optimise.

    Returns:
        A torch.optim.Optimizer instance.
    """
    if args.optimizer == "ivon":
        return IVON(
            model.parameters(),
            lr=args.learning_rate,
            mc_samples=args.mc_samples,
            beta1=args.momentum,
            beta2=args.momentum_hess,
            weight_decay=args.weight_decay,
            hess_init=args.hess_init,
            ess=args.ess,
            clip_radius=args.clip_radius,
        )
    elif args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adahessian":
        return AdaHessian(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "vogn":
        return VOGN(
            model.parameters(),
            lr=args.learning_rate,
            data_size=args.ess,  # Using ess as data_size for consistency in simple CLI
            mc_samples=args.mc_samples,
        )
    elif args.optimizer == "variational_adam":
        return VariationalAdam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    raise ValueError(f"Unknown optimizer: {args.optimizer}")


if __name__ == "__main__":
    timer = coro_timer()
    t_init = next(timer)
    print(f">>> Training initiated at {t_init.isoformat()} <<<\n")

    args = get_args()
    print(args, end="\n\n")

    if args.seed is not None:
        deteministic_run(seed=args.seed)

    device = torch.device(args.device)
    if device != torch.device("cpu"):
        check_cuda()

    mkdirp(args.save_dir)

    if args.resume:
        startepoch, model, optimizer, scheduler, dic = loadcheckpoint(
            args.resume, device
        )
        modelargs, modelkwargs = dic["modelargs"], dic["modelkwargs"]
        print(f"resumed from {args.resume}\n")
    else:
        startepoch = 0
        modelargs, modelkwargs = (OUTCLASS[args.dataset], INSIZE[args.dataset]), {}
        model = STANDARDMODELS[args.arch](*modelargs, **modelkwargs).to(args.device)
        optimizer = get_optimizer(args, model)
        scheduler = (
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0 / args.warmup,
                end_factor=1.0,
                total_iters=args.warmup,
            )
            if args.warmup > 0
            else None
        )

    data_size = int(NTRAIN[args.dataset] * args.tvsplit)

    if args.tensorboard_dir:
        mkdirp(args.tensorboard_dir)
        sw = SummaryWriter(args.tensorboard_dir)
    else:
        sw = None

    train_loader, val_loader = TRAINDATALOADERS[args.dataset](
        args.data_dir,
        args.tvsplit,
        args.workers,
        (device != torch.device("cpu")),
        args.tbatch,
        args.vbatch,
    )
    test_loader = TESTDATALOADER[args.dataset](
        args.data_dir,
        args.workers,
        (device != torch.device("cpu")),
        args.tbatch,
    )

    log_ece = coro_log_timed(sw, args.printfreq, args.bins, args.save_dir)

    print(
        f"datasize {int(data_size * args.tvsplit)}, paramsize "
        f"{sum(p.nelement() for p in model.parameters())}"
    )
    print(f">>> Training starts at {next(timer)[0].isoformat()} <<<\n")

    # Build a closure that injects mc_samples for IVON
    _train_fn = train_functions[args.optimizer]
    _extra = (
        {"mc_samples": args.mc_samples}
        if args.optimizer in ["ivon", "vogn", "variational_adam"]
        else {}
    )

    for e in range(startepoch, args.epochs):
        log_ece.send((e, "train", len(train_loader), None))
        if e == args.warmup:
            print("End of warmup epochs, starting cosine annealing")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, eta_min=args.lr_final, T_max=args.epochs
            )
        model.train()
        do_epoch(
            train_loader,
            _train_fn,
            log_ece,
            device,
            model=model,
            optimizer=optimizer,
            **_extra,
        )
        scheduler.step()
        log_ece.throw(StopIteration)

        savecheckpoint(
            pjoin(args.save_dir, "checkpoint.pt"),
            args.arch,
            modelargs,
            modelkwargs,
            model,
            optimizer,
            scheduler,
        )
        checkpoint_epochs = [0, 1, 2, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200]
        if e in checkpoint_epochs:
            savecheckpoint(
                pjoin(args.save_dir, "checkpoint%03d.pt" % (e + 1)),
                args.arch,
                modelargs,
                modelkwargs,
                model,
                optimizer,
                scheduler,
            )
        print(f"Max memory usage {torch.cuda.max_memory_allocated()}")
        time_per_epoch = next(timer)[1]
        print(f">>> Time elapsed: {time_per_epoch} <<<\n")
        with open(pjoin(args.save_dir, "time.csv"), "a+") as file:
            file.write("%d,%f\n" % (e, time_per_epoch.total_seconds()))

        log_ece.send((e, "test", len(test_loader), None))
        with torch.no_grad():
            model.eval()
            do_epoch(test_loader, do_evalbatch, log_ece, device, model=model)
        log_ece.throw(StopIteration)

        if len(val_loader) == 0:
            continue

        log_ece.send((e, "val", len(val_loader), None))
        with torch.no_grad():
            model.eval()
            do_epoch(val_loader, do_evalbatch, log_ece, device, model=model)
        log_ece.throw(StopIteration)
        print(f">>> Time elapsed: {next(timer)[1]} <<<\n")

    log_ece.close()
    print(f">>> Training completed at {next(timer)[0].isoformat()} <<<\n")
