# Copyright (c) 2026 Adrian R. Minut
# Copyright (c) 2024 ABI Team
# 
# SPDX-License-Identifier: GPL-3.0

"""Consolidated testing script for in-domain BDL tasks (CIFAR-10/100)."""

import os
import torch
import torch.nn.functional as nnf
import hydra
from omegaconf import DictConfig, OmegaConf

from vldatasets.standard import (
    TRAINDATALOADERS,
    TESTDATALOADER,
    NTRAIN,
    NTEST,
    OUTCLASS,
)
from vlbench.train.utils import coro_timer, mkdirp
from vlbench.plotting.calibration import bins2diagram
from vlbench.train.trainutils import (
    coro_log_auroc,
    do_epoch,
    check_cuda,
    deteministic_run,
    SummaryWriter,
    get_outputsaver,
    summarize_csv,
    loadcheckpoint,
)


def do_evalbatch(batchinput, model, optimizer=None, repeat: int = 1):
    """Evaluation batch with optional posterior sampling."""
    inputs, gt = batchinput[:-1], batchinput[-1]
    cumloss = 0.0
    cumprob = torch.zeros([])
    for _ in range(repeat):
        if optimizer is not None and hasattr(optimizer, "sampled_params"):
            with optimizer.sampled_params():
                output = model(*inputs)
        else:
            output = model(*inputs)

        ll = nnf.log_softmax(output, 1)
        loss = nnf.nll_loss(ll, gt) / repeat
        cumloss += loss.item()
        prob = nnf.softmax(output, 1)
        cumprob = cumprob + prob / repeat
    return cumprob, gt, cumloss


def do_evalbatch_swag(
    batchinput, model, swag_model, samples=1, sample_mode="modelwise"
):
    """Evaluation batch for SWAG with weight sampling."""
    inputs, gt = batchinput[:-1], batchinput[-1]
    cumloss = 0.0
    cumprob = torch.zeros([])
    for _ in range(samples):
        swag_model.sample(mode=sample_mode)
        output = model(*inputs)
        ll = nnf.log_softmax(output, 1)
        loss = nnf.nll_loss(ll, gt) / samples
        cumloss += loss.item()
        prob = nnf.softmax(output, 1)
        cumprob = cumprob + prob / samples
    return cumprob, gt, cumloss


@hydra.main(
    config_path="../../../conf/indomain", config_name="config", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    """Main testing entry point."""
    timer = coro_timer()
    t_init = next(timer)
    print(f">>> Test initiated at {t_init.isoformat()} <<<\n")
    print(OmegaConf.to_yaml(cfg))

    if cfg.seed is not None:
        deteministic_run(seed=cfg.seed)

    device = torch.device(cfg.device)
    if device.type == "cuda":
        check_cuda()

    mkdirp(cfg.save_dir)

    # Prefix for results
    valdata = getattr(cfg, "valdata", False)
    prefix = "val_bayes" if valdata else "test_bayes"
    # Load data
    if "_target_" in cfg.dataset:
        dataset_obj = hydra.utils.instantiate(cfg.dataset)
        dataset_name = dataset_obj.name
        outclass = dataset_obj.outclass
        if valdata:
            ndata = dataset_obj.ntrain - int(cfg.tvsplit * dataset_obj.ntrain)
            _, data_loader = dataset_obj.get_train_loaders(
                data_dir=cfg.data_dir,
                train_val_split=cfg.tvsplit,
                workers=cfg.workers,
                pin_memory=device.type == "cuda",
                tbatch=cfg.vbatch,
                vbatch=cfg.vbatch,
            )
        else:
            ndata = dataset_obj.ntest
            data_loader = dataset_obj.get_test_loader(
                data_dir=cfg.data_dir,
                workers=cfg.workers,
                pin_memory=device.type == "cuda",
                batch=cfg.vbatch,
            )
    else:
        dataset_name = cfg.dataset.name
        ndata = (
            (NTRAIN[dataset_name] - int(cfg.tvsplit * NTRAIN[dataset_name]))
            if cfg.valdata
            else NTEST[dataset_name]
        )
        outclass = OUTCLASS[dataset_name]

        if valdata:
            _, data_loader = TRAINDATALOADERS[dataset_name](
                cfg.data_dir,
                cfg.tvsplit,
                cfg.workers,
                device.type == "cuda",
                cfg.vbatch,
                cfg.vbatch,
            )
        else:
            data_loader = TESTDATALOADER[dataset_name](
                cfg.data_dir, cfg.workers, device.type == "cuda", cfg.vbatch
            )

    sw = SummaryWriter(cfg.tensorboard_dir) if cfg.tensorboard_dir else None
    ensemble_mode = getattr(cfg, "ensemble", False)
    log_ece = coro_log_auroc(
        sw, cfg.printfreq, cfg.bins, "" if ensemble_mode else cfg.save_dir
    )
    plot_diagram = getattr(cfg, "plotdiagram", False)

    runs_range = range(5, 25) if ensemble_mode else range(5)

    for runfolder in [str(i) for i in runs_range]:
        model_path = os.path.join(cfg.traindir, runfolder, "checkpoint.pt")
        if not os.path.exists(model_path):
            print(f"skipping {os.path.join(cfg.traindir, runfolder)}\n")
            continue

        print(f"loading model from {model_path} ...\n")
        # Load model and optimizer/extra info
        # Note: loadmodel in _registry.py returns (model, extra_dict)
        # loadcheckpoint in trainutils.py returns (startepoch, model, optimizer, scheduler, extra_dict)
        _, model, optimizer, _, extra = loadcheckpoint(model_path, device)

        outputsaver = None
        if getattr(cfg, "saveoutput", False):
            outputsaver = get_outputsaver(
                cfg.save_dir, ndata, outclass, f"predictions_{prefix}_{runfolder}.npy"
            )
        else:
            outputsaver = None

        log_ece.send((runfolder, prefix, len(data_loader), outputsaver))

        with torch.no_grad():
            model.eval()

            # Decide on batch function
            if cfg.method.name == "swag":
                # For SWAG, the model loaded is the SWAG wrapper
                swag_model = model

                output_samples = getattr(cfg, "testrepeat", 1)

                def batch_fn(b, m=model, s_m=swag_model, **kw):
                    # kw might contain 'model' from do_epoch, which we ignore here
                    # as we use m from closure (or explicitly provided m)
                    return do_evalbatch_swag(
                        b,
                        m.basemodel,
                        s_m,
                        samples=output_samples,
                        sample_mode=getattr(cfg, "swag_samplemode", "modelwise"),
                    )
            else:
                repeats = getattr(cfg, "testrepeat", 1)

                def batch_fn(b, m=model, opt=optimizer, **kw):
                    return do_evalbatch(b, m, repeat=repeats, optimizer=opt)

            do_epoch(data_loader, batch_fn, log_ece, device)

        bins, *_ = log_ece.throw(StopIteration)
        if bins is not None and plot_diagram:
            bins2diagram(
                bins,
                False,
                os.path.join(cfg.save_dir, f"calibration_{prefix}_{runfolder}.pdf"),
            )

        if outputsaver:
            outputsaver.close()
        del model
        print(f">>> Run {runfolder} time elapsed: {next(timer)[1]} <<<\n")

    summarize_csv(os.path.join(cfg.save_dir, f"{prefix}.csv"))
    log_ece.close()
    print(f">>> Test completed at {next(timer)[0].isoformat()} <<<\n")


if __name__ == "__main__":
    main()
