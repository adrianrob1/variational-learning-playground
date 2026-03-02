# Copyright (c) 2026 Adrian R. Minut
# Copyright (c) 2024 ABI Team
# 
# SPDX-License-Identifier: GPL-3.0

"""Image classification test/evaluation script.

Run as:
    uv run python -m vlbench.image_classification.test \\
        traindir=runs/cifar10_ivon dataset=cifar10

Refactored from tasks/image_classification/test.py:
- Removed sys.path hacks and `from ivon import IVON`
- Uses absolute vlbench/vloptimizers imports
"""

import argparse
from os.path import join as pjoin, exists
import torch
import torch.nn.functional as nnf

from vlbench.train.utils import coro_timer, mkdirp
from vlbench.plotting.calibration import bins2diagram
from vldatasets.standard.dataloaders import (
    TRAINDATALOADERS,
    TESTDATALOADER,
    OUTCLASS,
    NTRAIN,
    NTEST,
)
from vlbench.train.trainutils import (
    coro_log_auroc,
    do_epoch,
    do_evalbatch,
    check_cuda,
    deteministic_run,
    summarize_csv,
    get_outputsaver,
    loadcheckpoint,
)
from vloptimizers.ivon import IVON


def get_args():
    """Parse CLI arguments for the image classification test script.

    Returns:
        argparse.Namespace with all test hyperparameters.
    """
    parser = argparse.ArgumentParser(description="CIFAR10/100 IVON test")
    parser.add_argument(
        "traindir", type=str, help="path that collects all trained runs."
    )
    parser.add_argument("dataset", type=str, choices=TESTDATALOADER)
    parser.add_argument(
        "-j",
        "--workers",
        default=1,
        type=int,
        metavar="N",
        help="number of data loading workers",
    )
    parser.add_argument(
        "-b",
        "--batch",
        default=512,
        type=int,
        metavar="N",
        help="test mini-batch size",
    )
    parser.add_argument(
        "-tr",
        "--testrepeat",
        default=0,
        type=int,
        help="number of posterior samples for Bayesian evaluation (0=MAP)",
    )
    parser.add_argument(
        "-vd",
        "--valdata",
        action="store_true",
        help="use validation instead of test data",
    )
    parser.add_argument(
        "-sp",
        "--tvsplit",
        default=0.9,
        type=float,
        metavar="RATIO",
        help="ratio of data used for training (only needed with --valdata)",
    )
    parser.add_argument(
        "-pf",
        "--printfreq",
        default=10,
        type=int,
        metavar="N",
        help="print frequency",
    )
    parser.add_argument(
        "-d",
        "--device",
        default="cpu",
        type=str,
        metavar="DEV",
        help="run on cpu/cuda",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0,
        help="fixes seed for reproducibility",
    )
    parser.add_argument(
        "-sd",
        "--save_dir",
        default="save_temp",
        type=str,
        help="directory to save test results",
    )
    parser.add_argument(
        "-so",
        "--saveoutput",
        action="store_true",
        help="save output probability arrays as .npy",
    )
    parser.add_argument(
        "-dd",
        "--data_dir",
        default="../data",
        type=str,
        help="directory to find/store datasets",
    )
    parser.add_argument(
        "-nb",
        "--bins",
        default=20,
        type=int,
        help="number of bins for ECE and reliability diagram",
    )
    parser.add_argument(
        "-pd",
        "--plotdiagram",
        action="store_true",
        help="plot reliability diagram and save as PDF",
    )
    return parser.parse_args()


def do_evalbatch_ivon(batchinput, model, optimizer: IVON, repeat: int = 1):
    """Evaluation step using posterior predictive averaging for IVON.

    Samples `repeat` weights from the posterior and averages the predictions.

    Args:
        batchinput: (images, target) tuple.
        model: nn.Module.
        optimizer: IVON instance with sampled_params context manager.
        repeat: Number of posterior samples to average.

    Returns:
        Tuple of (mean_prob, target, mean_loss).
    """
    inputs, gt = batchinput[:-1], batchinput[-1]
    cumloss = 0.0
    cumprob = torch.zeros([], device=inputs[0].device, dtype=inputs[0].dtype)
    for _ in range(repeat):
        with optimizer.sampled_params():
            output = model(*inputs)
        ll = nnf.log_softmax(output, 1)
        loss = nnf.nll_loss(ll, gt) / repeat
        cumloss += loss.item()
        prob = nnf.softmax(output, 1)
        cumprob = cumprob + prob / repeat
    return cumprob, gt, cumloss


def get_dataloader(args, device):
    """Return the evaluation DataLoader based on args.

    Args:
        args: Parsed args with dataset, data_dir, workers, tvsplit, batch.
        device: torch.device for pin_memory determination.

    Returns:
        DataLoader for either the validation or test split.
    """
    pin = device != torch.device("cpu")
    if args.valdata:
        _, data_loader = TRAINDATALOADERS[args.dataset](
            args.data_dir,
            args.tvsplit,
            args.workers,
            pin,
            args.batch,
            args.batch,
        )
    elif args.dataset == "imagenet":
        from vldatasets.standard.dataloaders import (
            get_imagenet_test_loader,
            get_imagenet_test_loader_torch,
        )

        if exists(pjoin(args.data_dir, "val.ffcv")):
            data_loader = get_imagenet_test_loader(
                args.data_dir, args.workers, args.batch, device, distributed=False
            )
        else:
            data_loader = get_imagenet_test_loader_torch(
                args.data_dir,
                args.workers,
                args.batch,
                pin_memory=(device.type == "cuda"),
                distributed=False,
            )
    else:
        data_loader = TESTDATALOADER[args.dataset](
            args.data_dir,
            args.workers,
            pin,
            args.batch,
        )
    return data_loader


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    timer = coro_timer()
    t_init = next(timer)
    print(f">>> Test initiated at {t_init.isoformat()} <<<\n")

    args = get_args()
    print(args, end="\n\n")

    if args.seed is not None:
        deteministic_run(seed=args.seed)

    device = torch.device(args.device)
    if device != torch.device("cpu"):
        check_cuda()

    mkdirp(args.save_dir)

    log_ece = coro_log_auroc(None, args.printfreq, args.bins, args.save_dir)
    prefix = "val" if args.valdata else "test"

    for runfolder in [str(i) for i in range(5)]:
        model_path = pjoin(args.traindir, runfolder, "checkpoint.pt")
        if not exists(model_path):
            print(f"skipping {pjoin(args.traindir, runfolder)}\n")
            continue

        print(f"loading model from {model_path} ...\n")
        _, model, optimizer = loadcheckpoint(model_path, device)[:3]
        print(optimizer.defaults)

        data_loader = get_dataloader(args, device)
        dataset = args.dataset
        ndata = (
            NTRAIN[dataset] - int(args.tvsplit * NTRAIN[dataset])
            if args.valdata
            else NTEST[dataset]
        )

        if isinstance(optimizer, IVON):
            if args.testrepeat:
                prefix = "val_bayes" if args.valdata else "test_bayes"
            else:
                prefix = "val_map" if args.valdata else "test_map"
        else:
            prefix = "val" if args.valdata else "test"

        print(f">>> Test starts at {next(timer)[0].isoformat()} <<<\n")

        outputsaver = None
        if args.saveoutput:
            outputsaver = get_outputsaver(
                args.save_dir,
                ndata,
                OUTCLASS[dataset],
                f"predictions_{prefix}_{runfolder}.npy",
            )

        log_ece.send((runfolder, prefix, len(data_loader), outputsaver))
        with torch.no_grad():
            model.eval()
            if isinstance(optimizer, IVON) and args.testrepeat:
                do_epoch(
                    data_loader,
                    do_evalbatch_ivon,
                    log_ece,
                    device,
                    model=model,
                    optimizer=optimizer,
                    repeat=args.testrepeat,
                )
            else:
                do_epoch(data_loader, do_evalbatch, log_ece, device, model=model)

        bins, *_ = log_ece.throw(StopIteration)

        if outputsaver is not None:
            outputsaver.close()

        del model

        if args.plotdiagram:
            bins2diagram(
                bins,
                False,
                pjoin(args.save_dir, f"calibration_{prefix}_{runfolder}.pdf"),
            )

        print(f">>> Time elapsed: {next(timer)[1]} <<<\n")

    summarize_csv(pjoin(args.save_dir, f"{prefix}.csv"))
    log_ece.close()
    print(f">>> Test completed at {next(timer)[0].isoformat()} <<<\n")
