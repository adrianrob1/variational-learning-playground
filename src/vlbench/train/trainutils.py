# Copyright (c) 2026 Adrian R. Minut
# Copyright (c) 2026 ABI Team
#
# SPDX-License-Identifier: GPL-3.0
# SPDX-License-Identifier: Apache 2.0

"""Training loop coroutines, epoch runners, checkpoint I/O.

Refactored from tasks/common/trainutils.py — all relative imports updated
to use absolute vlbench/vloptimizers paths.
"""

import math
from typing import Callable, Any, Optional, Mapping, Iterable
import warnings
import collections
import csv
import statistics
import os
from os.path import join as pjoin
import random
from timeit import default_timer as timer
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch import Tensor, LongTensor, nn
import torch.nn.functional as nnf
from torch.utils.data import DataLoader
from torch.optim import SGD, AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler, LinearLR, CosineAnnealingLR

from vlbench.train.utils import (
    coro_trackavg_weighted,
    coro_dict2csv,
    coro_npybatchgatherer,
    autoinitcoroutine,
)
from vloptimizers.ivon import IVON
from vloptimizers.adahessian import AdaHessian
from vloptimizers.vogn import VOGN
from vloptimizers.variational_adam import VariationalAdam
import vlbench.models as models


def _cal():
    from vlbench.plotting import calibration as _c

    return _c


SummaryWriter = None  # optional tensorboard


def deteministic_run(seed=0):
    """Set all random seeds and CUDA flags for fully deterministic runs.

    Args:
        seed: Integer seed for random, numpy, and torch.
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def check_cuda() -> None:
    """Assert CUDA is available and print available device names."""
    if not torch.cuda.is_available():
        raise Exception("No CUDA device available")
    cuda_count = torch.cuda.device_count()
    print("{0} CUDA device(s) available:".format(cuda_count))
    for i in range(cuda_count):
        print(
            "- {0}: {1} ({2})".format(
                i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i)
            )
        )
    curr_idx = torch.cuda.current_device()
    print("Currently using device {0}".format(curr_idx))


def avgdups(t: Tensor, dups: int) -> Tensor:
    """Average a tensor over its first `dups` duplicated rows.

    Args:
        t: Tensor of shape (dups * batch, ...).
        dups: Number of duplicates to average over.

    Returns:
        Tensor of shape (batch, ...).
    """
    return torch.mean(t.view(dups, -1, *t.size()[1:]), dim=0)


def apply_batch(batch, fn: Callable[[Tensor], Any]):
    """Recursively apply fn to every tensor in a batch (tensor/dict/list)."""
    if isinstance(batch, Tensor):
        return fn(batch)
    elif isinstance(batch, collections.abc.Mapping):
        return {k: apply_batch(sample, fn) for k, sample in batch.items()}
    elif isinstance(batch, collections.abc.Sequence):
        return [apply_batch(sample, fn) for sample in batch]
    else:
        return batch


def top5corrects(outprobas: Tensor, gt: LongTensor) -> int:
    """Count samples whose ground-truth label is in the top-5 predictions.

    Args:
        outprobas: (batch, nclasses) probability tensor.
        gt: (batch,) ground-truth label tensor.

    Returns:
        Number of top-5 correct predictions (int).
    """
    preds = outprobas.topk(5)[1].t()
    return torch.sum(preds.eq(gt.view(1, -1)), dtype=torch.long).item()


def cumentropy(probas: Tensor) -> float:
    """Sum of per-sample entropies over the batch.

    Args:
        probas: (batch, nclasses) probability tensor.

    Returns:
        Cumulative entropy as float.
    """
    return torch.sum(
        -probas * torch.log(probas + torch.finfo(probas.dtype).tiny)
    ).item()


def cumnll(probas: Tensor, gts: LongTensor) -> float:
    """Cumulative negative log-likelihood across the batch.

    Args:
        probas: (batch, nclasses) probability tensor.
        gts: (batch,) ground-truth label tensor.

    Returns:
        Sum of NLL values as float.
    """
    nlls = -torch.log(torch.gather(probas, -1, gts.unsqueeze(-1)).squeeze(-1))
    return torch.sum(nlls).item()


def onehot(t: LongTensor, nclasses: int, dtype=torch.long):
    """Convert a label tensor to one-hot encoding.

    Args:
        t: (batch,) or (batch, ...) label tensor.
        nclasses: Number of classes.
        dtype: Output dtype.

    Returns:
        One-hot tensor of shape (*t.shape, nclasses).
    """
    if torch.numel(t) == 0:
        return torch.empty(0, nclasses, device=t.device)
    t_onehot = torch.zeros(*t.size(), nclasses, device=t.device, dtype=dtype)
    return t_onehot.scatter(t.dim(), t.unsqueeze(-1), 1)


def cumbrier(probas: Tensor, onehotgts: Tensor) -> float:
    """Cumulative Brier score across the batch.

    Args:
        probas: (batch, nclasses) probability tensor.
        onehotgts: (batch, nclasses) one-hot ground-truth tensor.

    Returns:
        Sum of Brier scores as float.
    """
    return torch.sum((probas - onehotgts) ** 2).item()


def coro_epochlog(
    total: int, logfreq: int = 100, nbin: int = 10, outputsaver=None, global_rank=None
):
    """Coroutine that accumulates per-batch metrics for one epoch.

    Yields: None on first send (initialisation).
    Sends: ((outprobas, gt, loss), batch_index).
    Returns: (bins, loss, nll, brier, acc5, ent) on StopIteration.
    """
    cal = _cal()
    data2bins = cal.data2bins
    bins2acc = cal.bins2acc
    bins2conf = cal.bins2conf
    bins2ece = cal.bins2ece
    losstracker = coro_trackavg_weighted()
    nlltracker = coro_trackavg_weighted()
    briertracker = coro_trackavg_weighted()
    binsmerger = cal.coro_binsmerger()
    top5tracker = coro_trackavg_weighted()
    enttracker = coro_trackavg_weighted()
    bins, loss, nll, brier, acc5, ent = (None,) + (float("nan"),) * 5
    try:
        yield
        while True:
            (outprobas, gt, loss), i = yield
            if outputsaver is not None:
                outputsaver.send(outprobas.cpu().numpy())
            bs = outprobas.size(0)
            probas, preds = torch.max(outprobas, dim=1)
            bins = binsmerger.send(
                data2bins(zip((preds == gt).tolist(), probas.tolist()), nbin)
            )
            loss = losstracker.send((loss * bs, bs))
            nll = nlltracker.send((cumnll(outprobas, gt), bs))
            brier = briertracker.send(
                (
                    cumbrier(outprobas, onehot(gt, outprobas.size(1), outprobas.dtype)),
                    bs,
                )
            )
            acc5 = top5tracker.send((top5corrects(outprobas, gt), bs))
            ent = enttracker.send((cumentropy(outprobas), bs))
            if (not global_rank) and (i % logfreq == 0):
                print(
                    f"  {i}/{total}: loss={loss:.4f}, nll={nll:.4f}, "
                    f"brier={brier:.4f}, acc={bins2acc(bins):.4f}, "
                    f"conf={bins2conf(bins):.4f}, ece={bins2ece(bins):.4f}, "
                    f"acc@5={acc5:.4f}, entropy={ent:.4f}"
                )
    except StopIteration:
        return bins, loss, nll, brier, acc5, ent


@autoinitcoroutine
def coro_log(
    sw=None, logfreq: int = 100, nbin: int = 10, save_dir="", global_rank=None
):
    """Auto-initialised coroutine wrapping coro_epochlog with CSV logging."""
    cal = _cal()
    bins2acc, bins2conf, bins2ece = cal.bins2acc, cal.bins2conf, cal.bins2ece
    bins, loss, nll, brier, acc5, ent = (None,) + (float("nan"),) * 5
    if save_dir:
        csvhead = (
            "epoch",
            "loss",
            "nll",
            "brier",
            "acc",
            "confidence",
            "ece",
            "acc@5",
            "entropy",
        )
        csvcorologs = dict()
    try:
        epoch, prefix, total, outputsaver = yield
        while True:
            print(f"*** Epoch {epoch} {prefix} ***\n")
            (bins, loss, nll, brier, acc5, ent) = yield from coro_epochlog(
                total, logfreq, nbin, outputsaver, global_rank
            )
            acc, conf, ece = bins2acc(bins), bins2conf(bins), bins2ece(bins)
            if not global_rank:
                print(
                    f"\nEpoch {epoch}: loss={loss:.4f}, nll={nll:.4f}, "
                    f"brier={brier:.4f}, acc={acc:.4f}, conf={conf:.4f}, "
                    f"ece={ece:.4f}, acc@5={acc5:.4f}, entropy={ent:.4f};\n"
                )
            if save_dir:
                if prefix not in csvcorologs:
                    csvcorologs[prefix] = coro_dict2csv(
                        pjoin(save_dir, f"{prefix}.csv"), csvhead
                    )
                csvcorologs[prefix].send(
                    {
                        "epoch": epoch,
                        "loss": loss,
                        "nll": nll,
                        "brier": brier,
                        "acc": acc,
                        "confidence": conf,
                        "ece": ece,
                        "acc@5": acc5,
                        "entropy": ent,
                    }
                )
            (epoch, prefix, total, outputsaver) = yield (
                bins,
                loss,
                nll,
                brier,
                acc5,
                ent,
            )
    except StopIteration:
        return bins, loss, nll, brier, acc5, ent


class AUROC:
    """Streaming AUROC accumulator."""

    def __init__(self):
        self.positive = []
        self.confidence = []

    def collect(self, positives, confidences):
        """Append lists of (is_correct, confidence) pairs."""
        self.positive += positives
        self.confidence += confidences

    def compute(self) -> float:
        """Compute AUROC from collected data. Returns nan if undefined."""
        try:
            return roc_auc_score(np.asarray(self.positive), np.asarray(self.confidence))
        except ValueError:
            return float("nan")


def coro_epochlog_auroc(
    total: int, logfreq: int = 100, nbin: int = 10, outputsaver=None, global_rank=None
):
    """Like coro_epochlog but also computes AUROC.

    Returns: (bins, loss, nll, brier, acc5, ent, auroc) on StopIteration.
    """
    cal = _cal()
    losstracker = coro_trackavg_weighted()
    nlltracker = coro_trackavg_weighted()
    briertracker = coro_trackavg_weighted()
    binsmerger = cal.coro_binsmerger()
    top5tracker = coro_trackavg_weighted()
    enttracker = coro_trackavg_weighted()
    auroctracker = AUROC()
    data2bins = cal.data2bins
    bins2acc = cal.bins2acc
    bins2conf = cal.bins2conf
    bins2ece = cal.bins2ece
    bins, loss, nll, brier, acc5, ent, auroc = (None,) + (float("nan"),) * 6
    try:
        yield
        while True:
            (outprobas, gt, loss), i = yield
            if outputsaver is not None:
                outputsaver.send(outprobas.cpu().numpy())
            bs = outprobas.size(0)
            probas, preds = torch.max(outprobas, dim=1)
            bins = binsmerger.send(
                data2bins(zip((preds == gt).tolist(), probas.tolist()), nbin)
            )
            auroctracker.collect((preds == gt).tolist(), probas.tolist())
            loss = losstracker.send((loss * bs, bs))
            nll = nlltracker.send((cumnll(outprobas, gt), bs))
            brier = briertracker.send(
                (
                    cumbrier(outprobas, onehot(gt, outprobas.size(1), outprobas.dtype)),
                    bs,
                )
            )
            acc5 = top5tracker.send((top5corrects(outprobas, gt), bs))
            ent = enttracker.send((cumentropy(outprobas), bs))
            if i % logfreq == 0 and (not global_rank):
                print(
                    f"  {i}/{total}: loss={loss:.4f}, nll={nll:.4f}, "
                    f"brier={brier:.4f}, acc={bins2acc(bins):.4f}, "
                    f"conf={bins2conf(bins):.4f}, ece={bins2ece(bins):.4f}, "
                    f"acc@5={acc5:.4f}, entropy={ent:.4f}"
                )
    except StopIteration:
        return bins, loss, nll, brier, acc5, ent, auroctracker.compute()


@autoinitcoroutine
def coro_log_auroc(sw=None, logfreq: int = 100, nbin: int = 10, save_dir=""):
    """Auto-initialised coroutine wrapping coro_epochlog_auroc with CSV logging."""
    cal = _cal()
    bins2acc, bins2conf, bins2ece = cal.bins2acc, cal.bins2conf, cal.bins2ece
    bins, loss, nll, brier, acc5, ent, auroc = (None,) + (float("nan"),) * 6
    if save_dir:
        csvhead = (
            "epoch",
            "loss",
            "nll",
            "brier",
            "acc",
            "confidence",
            "ece",
            "acc@5",
            "entropy",
            "auroc",
        )
        csvcorologs = dict()
    try:
        epoch, prefix, total, outputsaver = yield
        while True:
            print(f"*** Epoch {epoch} {prefix} ***\n")
            (bins, loss, nll, brier, acc5, ent, auroc) = yield from coro_epochlog_auroc(
                total, logfreq, nbin, outputsaver
            )
            acc, conf, ece = bins2acc(bins), bins2conf(bins), bins2ece(bins)
            print(
                f"\nEpoch {epoch}: loss={loss:.4f}, nll={nll:.4f}, "
                f"brier={brier:.4f}, acc={acc:.4f}, conf={conf:.4f}, "
                f"ece={ece:.4f}, acc@5={acc5:.4f}, entropy={ent:.4f}, "
                f"auroc={auroc:.4f};\n"
            )
            if save_dir:
                if prefix not in csvcorologs:
                    csvcorologs[prefix] = coro_dict2csv(
                        pjoin(save_dir, f"{prefix}.csv"), csvhead
                    )
                csvcorologs[prefix].send(
                    {
                        "epoch": epoch,
                        "loss": loss,
                        "nll": nll,
                        "brier": brier,
                        "acc": acc,
                        "confidence": conf,
                        "ece": ece,
                        "acc@5": acc5,
                        "entropy": ent,
                        "auroc": auroc,
                    }
                )
            (epoch, prefix, total, outputsaver) = yield (
                bins,
                loss,
                nll,
                brier,
                acc5,
                ent,
                auroc,
            )
    except StopIteration:
        return bins, loss, nll, brier, acc5, ent, auroc


@autoinitcoroutine
def coro_log_timed(
    sw=None,
    logfreq: int = 100,
    nbin: int = 10,
    save_dir="",
    global_rank=None,
    append: bool = False,
):
    """Auto-initialised coroutine wrapping coro_epochlog_auroc with timing + CSV logging."""
    cal = _cal()
    bins2acc, bins2conf, bins2ece = cal.bins2acc, cal.bins2conf, cal.bins2ece
    bins, loss, nll, brier, acc5, ent, auroc = (None,) + (float("nan"),) * 6
    if save_dir:
        csvhead = (
            "time",
            "epoch",
            "loss",
            "nll",
            "brier",
            "acc",
            "confidence",
            "ece",
            "acc@5",
            "entropy",
            "auroc",
        )
        csvcorologs = dict()
        start = timer()
    else:
        csvcorologs = None
        csvhead = None
        start = None
    try:
        epoch, prefix, total, outputsaver = yield
        while True:
            if not global_rank:
                print(f"*** Epoch {epoch} {prefix} ***\n")
            (bins, loss, nll, brier, acc5, ent, auroc) = yield from coro_epochlog_auroc(
                total, logfreq, nbin, outputsaver, global_rank
            )
            acc, conf, ece = bins2acc(bins), bins2conf(bins), bins2ece(bins)
            duration = timer() - start
            if not global_rank:
                print(
                    f"\nEpoch {epoch}: loss={loss:.4f}, nll={nll:.4f}, "
                    f"brier={brier:.4f}, acc={acc:.4f}, conf={conf:.4f}, "
                    f"ece={ece:.4f}, acc@5={acc5:.4f}, entropy={ent:.4f}, "
                    f"auroc={auroc:.4f};\nCurrent elapsed time: {duration:.2f} s\n"
                )
            if save_dir:
                if prefix not in csvcorologs:
                    csvcorologs[prefix] = coro_dict2csv(
                        pjoin(save_dir, f"{prefix}.csv"), csvhead, append=append
                    )
                csvcorologs[prefix].send(
                    {
                        "time": duration,
                        "epoch": epoch,
                        "loss": loss,
                        "nll": nll,
                        "brier": brier,
                        "acc": acc,
                        "confidence": conf,
                        "ece": ece,
                        "acc@5": acc5,
                        "entropy": ent,
                        "auroc": auroc,
                    }
                )
            (epoch, prefix, total, outputsaver) = yield (
                bins,
                loss,
                nll,
                brier,
                acc5,
                ent,
                auroc,
            )
    except StopIteration:
        return bins, loss, nll, brier, acc5, ent, auroc


def do_epoch(
    loader: DataLoader,
    compbatch,
    corolog,
    device=torch.device("cpu"),
    **comp_kwargs,
):
    """Run a full epoch: iterate loader, send batches through compbatch, feed corolog.

    Args:
        loader: DataLoader for the epoch.
        compbatch: Callable(batchinput, **comp_kwargs) -> output.
        corolog: Running coroutine that receives (output, batch_idx).
        device: Target device for batch tensors.
        **comp_kwargs: Extra keyword args passed to compbatch.
    """
    i = -1
    batchoutput = None
    for i, batchinput in enumerate(loader):
        batchinput = apply_batch(batchinput, lambda t: t.to(device, non_blocking=True))
        corolog.send((batchoutput, i))
        batchoutput = compbatch(batchinput, **comp_kwargs)
    corolog.send((batchoutput, i + 1))


def do_trainbatch(batchinput, model, optimizer, dups: int = 1, repeat: int = 1):
    """Standard SGD/AdamW training step for one mini-batch.

    Args:
        batchinput: List/tuple where last element is ground-truth labels.
        model: nn.Module.
        optimizer: Torch optimizer.
        dups: Number of duplicates in the batch (used with dup_collate_fn).
        repeat: Number of gradient accumulation steps.

    Returns:
        Tuple of (output_probas, ground_truth, loss_scalar).
    """
    optimizer.zero_grad(set_to_none=True)
    inputs, gt = batchinput[:-1], batchinput[-1]
    cumloss = 0.0
    cumprob = torch.zeros([], device=inputs[0].device, dtype=inputs[0].dtype)
    for _ in range(repeat):
        output = model(*inputs)
        ll = nnf.log_softmax(output, 1)
        ll = avgdups(ll, dups) if dups > 1 else ll
        loss = nnf.nll_loss(ll, gt) / repeat
        loss.backward()
        cumloss += loss.item()
        prob = nnf.softmax(output.detach(), 1)
        prob = avgdups(prob, dups) if dups > 1 else prob
        cumprob = cumprob + prob / repeat
    optimizer.step()
    return cumprob, gt, cumloss


def do_evalbatch(
    batchinput,
    model,
    dups: int = 1,
    repeat: int = 1,
    optimizer: Optional[Optimizer] = None,
):
    """Standard evaluation step for one mini-batch (call inside torch.no_grad()).

    Args:
        batchinput: List/tuple where last element is ground-truth labels.
        model: nn.Module.
        dups: Number of duplicates in the batch.
        repeat: Number of forward passes to average.
        optimizer: Optional optimizer for posterior sampling (e.g. IVON).

    Returns:
        Tuple of (output_probas, ground_truth, loss_scalar).
    """
    inputs, gt = batchinput[:-1], batchinput[-1]
    cumloss = 0.0
    cumprob = torch.zeros([], device=inputs[0].device, dtype=inputs[0].dtype)
    for _ in range(repeat):
        if optimizer is not None and hasattr(optimizer, "sampled_params"):
            with optimizer.sampled_params():
                output = model(*inputs)
        else:
            output = model(*inputs)
        ll = nnf.log_softmax(output, 1)
        ll = avgdups(ll, dups) if dups > 1 else ll
        loss = nnf.nll_loss(ll, gt) / repeat
        cumloss += loss.item()
        prob = nnf.softmax(output, 1)
        prob = avgdups(prob, dups) if dups > 1 else prob
        cumprob = cumprob + prob / repeat
    return cumprob, gt, cumloss


BatchNorm = nn.modules.batchnorm._BatchNorm


def _check_bn(module, flag):
    if isinstance(module, BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if isinstance(module, BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if isinstance(module, BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if isinstance(module, BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model, device=None, **kwargs):
    """Update BatchNorm running statistics using the training set.

    Args:
        loader: Train DataLoader.
        model: nn.Module with BatchNorm layers.
        device: Target device.
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    with torch.no_grad():
        for t, _ in loader:
            b = t.size(0)
            t = t.to(device=device, non_blocking=True)
            momentum = float(b) / (n + b)
            for module in momenta.keys():
                module.momentum = momentum
            model(t, **kwargs)
            n += b
    model.apply(lambda module: _set_momenta(module, momenta))


def savecheckpoint(
    to,
    modelname: str,
    modelargs: Iterable[Any],
    modelkwargs: Mapping[str, Any],
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    **kwargs,
) -> None:
    """Save model, optimizer, and scheduler state to a .pt file.

    Args:
        to: Path to save to.
        modelname: String key identifying the model constructor.
        modelargs: Positional args used to construct the model.
        modelkwargs: Keyword args used to construct the model.
        model: The nn.Module to save.
        optimizer: The optimizer whose state to save.
        scheduler: The LR scheduler whose state to save.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        models.savemodel(
            to,
            modelname,
            modelargs,
            modelkwargs,
            model,
            **{
                "optimname": type(optimizer).__name__,
                "optimargs": optimizer.defaults,
                "optimstates": optimizer.state_dict(),
                "schedulername": type(scheduler).__name__,
                "schedulerstates": scheduler.state_dict(),
            },
            **kwargs,
        )


def loadcheckpoint(fromfile, device=torch.device("cpu"), epochs=200):
    """Load and reconstruct model, optimizer, and scheduler from a checkpoint.

    Args:
        fromfile: Path to the checkpoint .pt file.
        device: Device to load tensors onto.
        epochs: T_max for CosineAnnealingLR if resuming.

    Returns:
        (startepoch, model, optimizer, scheduler, extra_dict)
    """
    model, dic = models.loadmodel(fromfile, device)
    optimizer = {
        "SGD": SGD,
        "AdamW": AdamW,
        "VOGN": VOGN,
        "AdaHessian": AdaHessian,
        "IVON": IVON,
        "VariationalAdam": VariationalAdam,
    }[dic["optimname"]](model.parameters(), **dic.pop("optimargs"))
    optimizer.load_state_dict(dic.pop("optimstates"))
    schedulername = dic["schedulername"]
    if schedulername == "LinearLR":
        scheduler = LinearLR(optimizer)
    elif schedulername == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer, eta_min=0.0, T_max=epochs)
    else:
        raise NotImplementedError
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        scheduler.load_state_dict(dic.pop("schedulerstates"))
    startepoch = scheduler.last_epoch
    return startepoch, model, optimizer, scheduler, dic


def get_outputsaver(save_dir, ndata, outclass, predictionfile):
    """Create a coroutine that gathers model output predictions to a .npy file.

    Args:
        save_dir: Directory to save the prediction file.
        ndata: Total number of data points.
        outclass: Number of output classes.
        predictionfile: Filename for the saved predictions.

    Returns:
        Initialised coro_npybatchgatherer coroutine.
    """
    return coro_npybatchgatherer(
        pjoin(save_dir, predictionfile),
        ndata,
        (outclass,),
        True,
        str(torch.get_default_dtype())[6:],
    )


def summarize_csv(csvfile):
    """Print mean and std of each metric column in a CSV log file.

    Args:
        csvfile: Path to the CSV file.
    """
    with open(csvfile, "r") as csvfp:
        reader = csv.DictReader(csvfp)
        criteria = [k for k in reader.fieldnames if k != "epoch"]
        maxlen = max(len(k) for k in criteria)
        values = {k: [] for k in criteria}
        for row in reader:
            for k, v in row.items():
                if k != "epoch":
                    values[k].append(float(v))
        for k, vals in values.items():
            # Filter matches to remove NaN values
            vals = [v for v in vals if not math.isnan(v)]
            if not vals:
                print(f"{k:>{maxlen}}:\tAll values are NaN")
                continue

            print(
                f"{k:>{maxlen}}:\tmean {statistics.mean(vals):.4f}, "
                f"std={statistics.stdev(vals):.4f}"
                if len(vals) > 1
                else "std=0.0000"
            )


def group_params_by_layer(model: torch.nn.Module):
    """Group model parameters by leaf module for per-layer optimizer groups.

    Args:
        model: nn.Module to group.

    Returns:
        List of dicts with {"params": ...} for each leaf module.
    """
    param_groups = []
    for _, m in model.named_modules():
        children = tuple(m.children())
        if len(children) > 0:
            continue
        params = tuple(m.parameters())
        if len(params) > 0:
            param_groups.append({"params": params})
    return param_groups
