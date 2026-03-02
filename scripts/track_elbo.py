# Copyright (c) 2026 Adrian R. Minut
# SPDX-License-Identifier: GPL-3.0

import hydra
import torch
import functools
import inspect
from omegaconf import DictConfig, OmegaConf
from vlbench.federated.worker import FederatedWorker
from vlbench.federated.orchestrator import FederatedOrchestrator
from vlbench.utils.federated_utils import get_parameters_vector


@hydra.main(config_path="../conf/federated", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    if cfg.method.name != "fedivon":
        print("ELBO tracking is only supported for FedIVON.")
        return

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # 1. Load partitioned data
    data_loader_fn = hydra.utils.instantiate(cfg.dataset.loader)
    partitioned_train, test_loader = data_loader_fn(
        data_dir=cfg.data_dir,
        num_clients=cfg.num_clients,
        alpha1=cfg.alpha1,
        alpha2=cfg.alpha2,
        dataset_proportion=cfg.dataset_proportion,
        seed=cfg.seed,
        batch_size=cfg.dataset.batch_size,
        workers=cfg.dataset.get("workers", 2),
    )

    # 2. Setup workers
    workers = []
    num_params = 0
    # Dummy model to get param count
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg.pop("name", None)
    dummy_model = hydra.utils.instantiate(model_cfg).to(device)
    num_params = get_parameters_vector(dummy_model).numel()

    global_model = hydra.utils.instantiate(model_cfg).to(device)

    for i in range(cfg.num_clients):
        local_model = hydra.utils.instantiate(model_cfg).to(device)
        train_loader = torch.utils.data.DataLoader(
            partitioned_train.get_client_dataset(i),
            batch_size=cfg.dataset.batch_size,
            shuffle=True,
        )

        opt_cfg = OmegaConf.to_container(cfg.method, resolve=True)
        opt_cfg.pop("name", None)
        opt_cfg.pop("num_comm_rounds", None)
        opt_cfg.pop("local_epochs", None)
        opt_cfg.pop("rho", None)
        opt_cfg.pop("mu", None)
        opt_cfg.pop("temperature", None)

        optimizer_fn = hydra.utils.instantiate(opt_cfg)

        kwargs = {}
        sig = inspect.signature(
            optimizer_fn.func
            if isinstance(optimizer_fn, functools.partial)
            else optimizer_fn
        )

        if "ess" in sig.parameters:
            tau = opt_cfg.get("temperature", 1.0)
            kwargs["ess"] = len(train_loader.dataset) / tau
        if "prior_mean" in sig.parameters:
            wd = opt_cfg.get("weight_decay", 0.01)
            num_params = get_parameters_vector(local_model).numel()
            prior_mean = torch.zeros(num_params, device=device)
            kwargs["prior_mean"] = prior_mean
            kwargs["prior_prec"] = wd * torch.ones_like(prior_mean)
        if "dual_mean" in sig.parameters:
            num_params = get_parameters_vector(local_model).numel()
            kwargs["dual_mean"] = torch.zeros(num_params, device=device)
            kwargs["dual_prec"] = torch.zeros(num_params, device=device)

        optimizer = optimizer_fn(local_model.parameters(), **kwargs)

        worker = FederatedWorker(
            model=local_model,
            optimizer=optimizer,
            method=cfg.method.name,
            client_idx=i,
            device=device,
            train_loader=train_loader,
            test_loader=test_loader,
            method_params=cfg.method,
        )
        workers.append(worker)

    # 3. Setup orchestrator
    orchestrator = FederatedOrchestrator(
        global_model=global_model,
        workers=workers,
        device=device,
        method=cfg.method.name,
        method_params=cfg.method,
    )

    # 4. Training loop with ELBO tracking
    elbos = []
    for r in range(cfg.num_comm_rounds):
        orchestrator.run_round()
        elbo = orchestrator.compute_variational_objective()
        loss, acc = orchestrator.evaluate()
        elbos.append(elbo)
        print(f"Round {r}: ELBO={elbo:.2f}, Loss={loss:.4f}, Acc={acc:.4f}")

    # Final results
    print("Final ELBO tracking complete.")


if __name__ == "__main__":
    main()
