# Reproducing Results from `bayes-admm`

This benchmark natively supports the `bayes-admm` components port inside `src/vlbench/federated`. So the results from [bayes-admm](https://github.com/team-approx-bayes/bayes-admm) can be reproduced using `vlbench`.

For example, you can use the federated demonstration script to replicate the 2D toy dataset results from the `bayes-admm` paper, evaluating convergence and visualization of the global model learned using variations of ADMM and other optimization strategies.

## Overview

The experiments are located in `scripts/illustrate_federated.py`. This script dynamically instances the provided methods via `hydra`.

The methods commonly leveraged in these results are:
- **FedADMM**: Alternating Direction Method of Multipliers.
- **FedDyn**: Dynamic Regularization for Federated Learning.
- **FedProx**: Proximal Term Regularization.

### Prerequisites

All methods will leverage the predefined toy dataset configuration and run via CPU. The toy dataset defaults to producing data for up to `5` clients.

---

## Reproducing Toy Results

Run the following scripts from the project root using `uv`:

### 1. FedADMM

To reproduce the `FedADMM` illustration with 5 workers running for 100 communication rounds:

```bash
uv run python scripts/illustrate_federated.py method=fedadmm num_clients=5 model=linear dataset.num_classes=2 model.D_in=2 model.D_out=2
```

This updates the configurations, pops parameters handled directly by the orchestrator (like `rho`), and correctly invokes SGD as the local optimizer under the hood with local dual updates executed per communication round. The output will be a generic visualization: `toy_illustration.png`.

### 2. FedDyn

To run with `FedDyn`, simply change the target method:

```bash
uv run python scripts/illustrate_federated.py method=feddyn num_clients=5 model=linear dataset.num_classes=2 model.D_in=2 model.D_out=2
```

### 3. FedProx

For `FedProx`:

```bash
uv run python scripts/illustrate_federated.py method=fedprox num_clients=5 model=linear dataset.num_classes=2 model.D_in=2 model.D_out=2
```

> [!NOTE]
> Additional hyperparameters like the regularizer weight (`rho` or `mu`), local epochs, or weight decay can be directly overridden in the CLI, for example by appending `method.rho=0.01` or `method.local_epochs=5`.

---

## Reproducing Real-World Dataset Results

To run full experiments with datasets like MNIST, FashionMNIST, or CIFAR-10, you can use the `train_federated.py` script. This handles training and accuracy evaluation iteratively across communication rounds without creating a 2D plot.

### 1. MNIST / FashionMNIST

To train using `FedADMM` on MNIST (with a linear model) over 100 communication rounds for 5 clients:

```bash
uv run python scripts/train_federated.py method=fedadmm dataset=mnist model=linear num_clients=5 method.num_comm_rounds=100
```
*(Swap `dataset=mnist` with `dataset=fashionmnist` for FashionMNIST).*

### 2. CIFAR-10

For CIFAR-10, the benchmark supports CNN models such as `FedDynCifarCNN` or `ResNet20` configured properly.

```bash
uv run python scripts/train_federated.py method=feddyn dataset=cifar10 model=feddyncifarcnn num_clients=10 method.num_comm_rounds=100
```

> [!TIP]
> The orchestrator outputs the average global model Evaluation Loss and Accuracy correctly formatted at the end of each round.
