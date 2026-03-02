# Copyright (c) 2026 Adrian R. Minut
# Copyright (c) 2024 ABI Team
#
# SPDX-License-Identifier: GPL-3.0

"""OOD evaluation entry point: run CIFAR-10 models against SVHN or Flowers102.

Run as::

    uv run python -m vlbench.ood.run <traindir> [--ood_dataset svhn] ...

Refactored from ``tasks/bdl_ood/run.py``:
- Dropped ``sys.path`` hacks and ``from ivon import IVON``
- Absolute imports from ``vlbench.*`` and ``vloptimizers.*``
- Extracted ``run_eval_loop`` to de-duplicate the in-domain / OOD eval blocks
"""

from __future__ import annotations

import argparse
from os.path import join as pjoin, exists

import torch

from vlbench.train.utils import coro_timer, mkdirp, coro_dict2csv
from vlbench.train.trainutils import (
    do_epoch,
    check_cuda,
    deteministic_run,
    loadcheckpoint,
)
from vldatasets.standard.dataloaders import get_cifar10_test_loader
from vldatasets.standard.ood_datasets import (
    get_svhn_loader,
    get_flowers102_loader,
    SVHNInfo,
    Flowers102Info,
)
from vlbench.utils.ood_utils import (
    do_evalbatch,
    do_evalbatch_von,
    do_evalbatch_swag,
    get_outputsaver,
    summarize_csv,
    coro_log,
    confidence_from_prediction_npy,
    OODMetrics,
)
from vloptimizers.ivon import IVON
from vloptimizers.vogn import VOGN
from vlbench.models.swag import SWAG


OOD_INFO = {
    "svhn": SVHNInfo,
    "flowers102": Flowers102Info,
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def get_args() -> argparse.Namespace:
    """Parse command-line arguments for the OOD evaluation script.

    Returns:
        ``argparse.Namespace`` with fields:

        - ``traindir`` (str): directory containing run sub-folders 0..4.
        - ``ood_dataset`` (str): ``"svhn"`` or ``"flowers102"``.
        - ``workers`` (int): DataLoader worker count.
        - ``batch`` (int): Test batch size.
        - ``testsamples`` (int): Duplication factor for MC-Dropout batches.
        - ``testrepeat`` (int): Posterior-sample repeat count (IVON/VOGN).
        - ``valdata`` (bool): Use validation split instead of test.
        - ``printfreq`` (int): Logging frequency (batches).
        - ``device`` (str): ``"cpu"``, ``"cuda"``, or ``"cuda:N"``.
        - ``seed`` (int | None): RNG seed for deterministic runs.
        - ``save_dir`` (str): Directory to write CSV / npy outputs.
        - ``saveoutput`` (bool): Whether to save raw probability arrays.
        - ``data_dir`` (str): Root dataset directory.
        - ``swag_modelsamples`` (int): SWAG ensemble size.
        - ``swag_samplemode`` (str): SWAG sample granularity.
    """
    parser = argparse.ArgumentParser(
        description="BDL OOD evaluation (SVHN / Flowers102)"
    )
    parser.add_argument(
        "traindir", type=str, help="path that collects all trained runs."
    )
    parser.add_argument(
        "--ood_dataset",
        default="svhn",
        choices=list(OOD_INFO),
    )
    parser.add_argument("-j", "--workers", default=1, type=int, metavar="N")
    parser.add_argument("-b", "--batch", default=512, type=int, metavar="N")
    parser.add_argument("-ts", "--testsamples", default=1, type=int)
    parser.add_argument("-tr", "--testrepeat", default=1, type=int)
    parser.add_argument("-vd", "--valdata", action="store_true")
    parser.add_argument("-pf", "--printfreq", default=10, type=int, metavar="N")
    parser.add_argument("-d", "--device", default="cpu", type=str, metavar="DEV")
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-sd", "--save_dir", default="save_temp", type=str)
    parser.add_argument("-so", "--saveoutput", action="store_true")
    parser.add_argument("-dd", "--data_dir", default="../data", type=str)
    parser.add_argument("-sms", "--swag_modelsamples", type=int, default=1)
    parser.add_argument(
        "-ssm", "--swag_samplemode", default="modelwise", choices=SWAG.sample_mode
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# OOD DataLoader factory
# ---------------------------------------------------------------------------


def get_ood_loader(args: argparse.Namespace):
    """Return the OOD DataLoader selected by ``args.ood_dataset``.

    Args:
        args: Parsed CLI args.  ``args.ood_dataset`` must be a key of
            ``OOD_INFO``.

    Returns:
        ``torch.utils.data.DataLoader`` for the chosen OOD split (test).

    Raises:
        ValueError: If ``args.ood_dataset`` is not ``"svhn"`` or
            ``"flowers102"``.
    """
    pin = args.device != "cpu"
    if args.ood_dataset == "svhn":
        return get_svhn_loader(
            args.data_dir, args.workers, pin, args.batch, "test", args.testsamples
        )
    elif args.ood_dataset == "flowers102":
        return get_flowers102_loader(
            args.data_dir, args.workers, pin, args.batch, "test", args.testsamples
        )
    raise ValueError(f"Unknown ood_dataset: {args.ood_dataset!r}")


# ---------------------------------------------------------------------------
# Per-run evaluation loop
# ---------------------------------------------------------------------------


def run_eval_loop(
    loader,
    prefix: str,
    log,
    device: torch.device,
    optimizer,
    model,
    args: argparse.Namespace,
    outputsaver=None,
) -> None:
    """Run one full evaluation epoch and stream results to *log*.

    Dispatches to the appropriate per-batch function based on optimizer type:

    - ``IVON`` or ``VOGN`` → ``do_evalbatch_von`` (posterior sampling)
    - ``SWAG`` model → ``do_evalbatch_swag`` (model-ensemble)
    - Everything else → ``do_evalbatch`` (MAP / MC-Dropout duplication)

    Args:
        loader: DataLoader to iterate over.
        prefix: String prefix for logging (e.g. ``"indomain_test"``).
        log: Coroutine logger (``coro_log`` instance).
        device: Torch device.
        optimizer: Optimizer loaded from checkpoint (used for isinstance checks
            and as context manager for IVON/VOGN).
        model: ``nn.Module`` to evaluate (may itself be a ``SWAG`` wrapper).
        args: Parsed CLI args (provides ``testrepeat``, ``testsamples``,
            ``swag_modelsamples``, ``swag_samplemode``).
        outputsaver: Optional coroutine that accumulates probability arrays.

    Returns:
        None

    Side effects:
        Sends batches to *log* and optionally writes ``.npy`` output via
        *outputsaver*.
    """
    log.send((None, prefix, len(loader), outputsaver))
    with torch.no_grad():
        if isinstance(optimizer, (IVON, VOGN)):
            model.eval()
            do_epoch(
                loader,
                do_evalbatch_von,
                log,
                device,
                model=model,
                optimizer=optimizer,
                repeat=args.testrepeat,
            )
        elif isinstance(model, SWAG):
            sampledmodels = [
                model.sampled_model(mode=args.swag_samplemode)
                for _ in range(args.swag_modelsamples)
            ]
            for m in sampledmodels:
                m.eval()
            do_epoch(loader, do_evalbatch_swag, log, device, models=sampledmodels)
        else:
            model.eval()
            do_epoch(
                loader,
                do_evalbatch,
                log,
                device,
                model=model,
                dups=args.testsamples,
                repeat=args.testrepeat,
            )
    log.throw(StopIteration)
    if outputsaver is not None:
        outputsaver.close()


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------


def compute_and_save_metrics(
    test_folder: str,
    wamode: str = "",
    runs: tuple = (),
) -> None:
    """Read saved ``.npy`` prediction files and write OOD metrics to CSV.

    For each run index in *runs*, loads ``predictions_indomain_test_<idx>.npy``
    and ``predictions_ood_test_<idx>.npy``, computes ``OODMetrics``, prints a
    summary, and appends a row to ``metrics_test.csv`` (or
    ``metrics_<wamode>_test.csv`` when *wamode* is non-empty).

    Args:
        test_folder: Directory containing the prediction ``.npy`` files and
            where the output CSV will be written.
        wamode: Weight-averaging mode string used to suffix file names
            (empty string for no suffix).
        runs: Sequence of run identifiers (strings matching the saved file
            indices).

    Returns:
        None

    Side effects:
        Writes / appends to ``<test_folder>/metrics_test.csv``.
    """
    starts_with = "predictions" if not wamode else f"predictions_{wamode}"
    csv_name = "metrics_test.csv" if not wamode else f"metrics_{wamode}_test.csv"
    csvcorolog = coro_dict2csv(
        pjoin(test_folder, csv_name), ("epoch",) + OODMetrics.metric_names
    )
    for e in runs:
        in_name = pjoin(test_folder, f"{starts_with}_indomain_test_{e}.npy")
        out_name = pjoin(test_folder, f"{starts_with}_ood_test_{e}.npy")
        in_conf = confidence_from_prediction_npy(in_name)
        out_conf = confidence_from_prediction_npy(out_name)
        metrics = OODMetrics(in_conf, out_conf).get_all()
        print(
            ", ".join([f"epoch: {e}"] + [f"{k}: {v:.4f}" for k, v in metrics.items()])
        )
        csvcorolog.send({"epoch": e, **metrics})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for OOD evaluation.

    Iterates over run sub-folders ``0`` through ``4`` inside ``args.traindir``,
    loads each checkpoint, evaluates the model on both the in-domain CIFAR-10
    test split and the chosen OOD dataset, and writes per-run metrics to CSV.

    Returns:
        None
    """
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

    log_ece = coro_log(None, args.printfreq, args.save_dir)

    indomain_prefix = "indomain_test"
    pin = device != torch.device("cpu")
    indomain_loader = get_cifar10_test_loader(
        args.data_dir, args.workers, pin, args.batch, args.testsamples
    )
    ood_prefix = "ood_test"
    ood_loader = get_ood_loader(args)
    outclass = 10

    runs = [str(i) for i in range(5)]
    valid_runs = []

    for runfolder in runs:
        model_path = pjoin(args.traindir, runfolder, "checkpoint.pt")
        if not exists(model_path):
            print(f"skipping {pjoin(args.traindir, runfolder)}\n")
            continue
        valid_runs.append(runfolder)

        _, model, optimizer, _, _ddat = loadcheckpoint(model_path, device)
        optimizer.mc_samples = args.testrepeat

        print(f">>> Test starts at {next(timer)[0].isoformat()} <<<\n")

        # In-domain test run
        outputsaver = None
        if args.saveoutput:
            outputsaver = get_outputsaver(
                args.save_dir,
                10000,
                outclass,
                f"predictions_{indomain_prefix}_{runfolder}.npy",
            )
        run_eval_loop(
            indomain_loader,
            indomain_prefix,
            log_ece,
            device,
            optimizer,
            model,
            args,
            outputsaver,
        )

        # OOD test run
        outputsaver = None
        if args.saveoutput:
            outputsaver = get_outputsaver(
                args.save_dir,
                OOD_INFO[args.ood_dataset].count["test"],
                outclass,
                f"predictions_{ood_prefix}_{runfolder}.npy",
            )
        run_eval_loop(
            ood_loader, ood_prefix, log_ece, device, optimizer, model, args, outputsaver
        )

        del model
        print(f">>> Time elapsed: {next(timer)[1]} <<<\n")

    compute_and_save_metrics(args.save_dir, "", valid_runs)
    summarize_csv(pjoin(args.save_dir, "metrics_test.csv"))
    print(f">>> Test completed at {next(timer)[0].isoformat()} <<<\n")
    log_ece.close()


if __name__ == "__main__":
    main()
