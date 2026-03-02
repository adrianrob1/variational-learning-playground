# Copyright (c) 2026 Adrian R. Minut
# Copyright (c) 2026 ABI Team
# SPDX-License-Identifier: GPL-3.0

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from vlbench.federated.worker import FederatedWorker
from vlbench.federated.orchestrator import FederatedOrchestrator
from vldatasets.partitioned.toy import ToyDataGenerator
from vlbench.plotting.federated import plot_2d_federated
from vlbench.utils.federated_utils import get_parameters_vector
import functools
import inspect


@hydra.main(config_path="../conf/federated", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    device = torch.device("cpu")  # Toy examples are small

    # 1. Generate Toy Data
    toy_generator = ToyDataGenerator(
        num_clients=cfg.num_clients,
        num_samples=200,
        seed=cfg.seed,
        multiclass=(cfg.dataset.num_classes > 2),
    )

    # 2. Setup workers
    workers = []

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg.pop("name", None)
    global_model = hydra.utils.instantiate(model_cfg).to(device)

    for i in range(cfg.num_clients):
        local_model = hydra.utils.instantiate(model_cfg).to(device)
        (X, y), _ = toy_generator.data_split(i)
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, y),
            batch_size=len(X),  # Full batch for toy
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
            test_loader=None,
            method_params=cfg.method,
        )
        workers.append(worker)

    orchestrator = FederatedOrchestrator(
        global_model=global_model,
        workers=workers,
        device=device,
        method=cfg.method.name,
        method_params=cfg.method,
    )

    # 3. Training and plotting
    for r in range(cfg.num_comm_rounds):
        orchestrator.run_round()
        if (r + 1) % 5 == 0 or r == 0:
            print(f"Round {r + 1} completed.")

    # Final plot
    print("Generating final illustration...")
    # Get model outputs on test grid
    X_test = toy_generator.inputs_test.reshape(-1, 2)
    with torch.no_grad():
        outputs = global_model(X_test)

    plot_2d_federated(
        model_outputs_list=[outputs],
        toy_generator=toy_generator,
        worker_indices=[-1],
        title=f"{cfg.method.name} Global Model",
        save_path="toy_illustration.png",
    )


if __name__ == "__main__":
    main()
