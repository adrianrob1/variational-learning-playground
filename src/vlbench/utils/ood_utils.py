# Copyright (c) 2026 Adrian R. Minut
# Copyright (c) 2026 ABI Team
# 
# SPDX-License-Identifier: GPL-3.0
# SPDX-License-Identifier: Apache 2.0

from typing import Tuple, Iterable, Dict
from os.path import join as pjoin
import csv
import statistics
from functools import cached_property, lru_cache
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import torch
from torch import Tensor
import torch.nn.functional as nnf
from torch.utils.data import DataLoader
from vldatasets.standard.ood_datasets import (  # noqa: F401 (re-exported for backward compat)
    SVHNInfo,
    get_svhn_loader,
    Flowers102Info,
    get_flowers102_loader,
    dup_collate_fn,
)
from vlbench.train.utils import (
    coro_npybatchgatherer,
    coro_trackavg_weighted,
    coro_dict2csv,
    autoinitcoroutine,
)
from vlbench.train.trainutils import cumentropy, avgdups


def confidence_from_prediction_npy(npyfile: str) -> np.ndarray:
    probas = np.load(npyfile)
    return np.amax(probas, axis=1)


def cumconfidence(probas: Tensor) -> float:
    return torch.sum(torch.max(probas, dim=1)[0]).item()


def do_evalbatch(batchinput, model, dups: int = 1, repeat: int = 1):
    inputs = batchinput[:-1]
    cumprob = torch.zeros([], device=inputs[0].device, dtype=inputs[0].dtype)
    for _ in range(repeat):
        output = model(*inputs)
        prob = nnf.softmax(output, 1)  # get likelihood
        prob = avgdups(prob, dups) if dups > 1 else prob
        cumprob = cumprob + prob / repeat
    return cumprob


# generic boilerplate to eval/test a minibatch
# should be wrapped within torch.no_grad()
def do_evalbatch_von(batchinput, model, optimizer, repeat: int = 1):
    inputs = batchinput[:-1]
    cumprob = torch.zeros([])
    for _ in range(repeat):
        with optimizer.sampled_params():
            output = model(*inputs)
        prob = nnf.softmax(output, 1)  # get likelihood
        cumprob = cumprob + prob / repeat
    return cumprob


def do_evalbatch_swag(batchinput, models):
    inputs = batchinput[:-1]
    cumprob = torch.zeros([])
    nmodel = len(models)
    for model in models:
        output = model(*inputs)
        cumprob = cumprob + nnf.softmax(output, 1) / nmodel
    return cumprob


# coroutine to monitor top 1/5 acc, loss, nll, entropy and brier score
# during each epoch, saves output probability prediction
def coro_epochlog(total: int, logfreq: int = 100, outputsaver=None):
    conftracker = coro_trackavg_weighted()
    enttracker = coro_trackavg_weighted()
    conf, ent = float("nan"), float("nan")
    try:
        yield  # skip first dud input
        while True:
            outprobas, i = yield
            if outputsaver is not None:
                outputsaver.send(outprobas.cpu().numpy())
            bs = outprobas.size(0)
            ent = enttracker.send((cumentropy(outprobas), bs))
            conf = conftracker.send((cumconfidence(outprobas), bs))
            if i % logfreq == 0:
                print(f"  {i}/{total}: conf={conf:.4f}, entropy={ent:.4f}")
    except StopIteration:  # on manual stop, return final accumulations
        return conf, ent


# coroutine to monitor top 1/5 acc, loss, entropy and brier score
@autoinitcoroutine
def coro_log(sw=None, logfreq: int = 100, save_dir=""):
    ent, conf = float("nan"), float("nan")
    csvhead = ("epoch", "confidence", "entropy")
    csvcorologs = dict()
    try:
        epoch, prefix, total, outputsaver = yield
        while True:
            print(f"*** Epoch {epoch} {prefix} ***\n")
            conf, ent = yield from coro_epochlog(total, logfreq, outputsaver)
            print(f"\nEpoch {epoch}: conf={conf:.4f}, entropy={ent:.4f};\n")
            # write to csv if asked for
            if save_dir:
                if prefix not in csvcorologs:
                    csvcorologs[prefix] = coro_dict2csv(
                        pjoin(save_dir, f"{prefix}.csv"), csvhead
                    )
                csvcorologs[prefix].send(
                    {"epoch": epoch, "confidence": conf, "entropy": ent}
                )
            # update tensorboard if asked for
            if sw is not None:
                sw.add_scalar(f"{prefix}/uncertainty", 1 - conf, epoch)
                sw.add_scalar(f"{prefix}/entropy", ent, epoch)
                sw.flush()
            # coroutine I/O
            epoch, prefix, total, outputsaver = yield (conf, ent)
    except StopIteration:  # on exit, return result from last epoch
        return conf, ent


def get_outputsaver(save_dir, ndata, outclass, predictionfile):
    return coro_npybatchgatherer(
        pjoin(save_dir, predictionfile),
        ndata,
        (outclass,),
        True,
        str(torch.get_default_dtype())[6:],
    )


def mean_std(vals: Iterable[float]) -> Tuple[float, float]:
    return statistics.mean(vals), statistics.stdev(vals)


def summarize_csv(csvfile):
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
            mean, std = mean_std(vals)
            print(f"{k:>{maxlen}}:\tmean {mean:.4f}, std={std:.4f}")


def auroc(indomain_confidence: np.ndarray, ood_confidence: np.ndarray):
    confidence = np.concatenate((indomain_confidence, ood_confidence))
    is_indomain = np.concatenate(
        (np.ones_like(indomain_confidence), np.zeros_like(ood_confidence))
    )
    return roc_auc_score(is_indomain, confidence)


class OODMetrics:
    metric_names = ("auroc", "aupr-in", "aupr-out", "fpr95", "dterr")

    def __init__(
        self,
        indomain_confidence: np.ndarray,
        ood_confidence: np.ndarray,
        eps: float = 0.0005,
    ):
        self.indomain_confidence = indomain_confidence
        self.ood_confidence = ood_confidence
        self.eps = eps

    @cached_property
    def _confidence(self) -> np.ndarray:
        return np.concatenate((self.indomain_confidence, self.ood_confidence))

    @cached_property
    def _is_indomain(self) -> np.ndarray:
        return np.concatenate(
            (
                np.ones_like(self.indomain_confidence),
                np.zeros_like(self.ood_confidence),
            )
        )

    @cached_property
    def _is_ood(self) -> np.ndarray:
        return np.concatenate(
            (
                np.zeros_like(self.indomain_confidence),
                np.ones_like(self.ood_confidence),
            )
        )

    @cached_property
    def _fpr_tpr(self) -> Tuple[np.ndarray, np.ndarray]:
        return roc_curve(self._is_indomain, self._confidence)[:2]

    @cached_property
    def auroc(self) -> float:
        return roc_auc_score(self._is_indomain, self._confidence)

    @cached_property
    def aupr_in(self) -> float:
        return average_precision_score(self._is_indomain, self._confidence)

    @cached_property
    def aupr_out(self) -> float:
        return average_precision_score(self._is_ood, -self._confidence)

    @cached_property
    def fpr_at_tpr95(self) -> float:
        # c.f. https://github.com/facebookresearch/odin
        fpr, tpr = self._fpr_tpr
        eps = self.eps
        idx_tpr95 = (tpr <= (0.95 + eps)) >= (0.95 - eps)
        if not np.any(idx_tpr95):
            raise ValueError(
                f"no tpr between [{0.95 - eps}, {0.95 + eps}], increase eps!"
            )
        return fpr[idx_tpr95].mean()

    @cached_property
    def detection_error(self) -> float:
        fpr, tpr = self._fpr_tpr
        detection_error = (fpr - tpr + 1.0) / 2.0
        return detection_error.min()

    @lru_cache(maxsize=None)
    def get_all(self) -> Dict[str, float]:
        return {
            "auroc": self.auroc,
            "aupr-in": self.aupr_in,
            "aupr-out": self.aupr_out,
            "fpr95": self.fpr_at_tpr95,
            "dterr": self.detection_error,
        }
