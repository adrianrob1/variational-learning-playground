# Federated Learning Benchmark

This module provides tools for testing optimizers and models in a federated learning (FL) setting with partitioned datasets (IID and non-IID). It supports several standard aggregation methods and tracks relevant metrics like Variational Objective (ELBO).

## Features

- **Partitioned Datasets:** Dirichlet-based partitioners for `MNIST`, `FashionMNIST`, `CIFAR-10`, and `CIFAR-100`.
- **Toy Data:** 2D Gaussian generator for illustrating decision boundaries.
- **Methods:** `FedAvg`, `FedProx`, `FedADMM`, `FedDyn`, `FedLap`, and `FedIVON`.
- **Models:** Specialized FL architectures (`CifarNet`, `FedDynCifarCNN`, `ResNet20` with `GroupNorm`/`FilterResponseNorm`).

## Running an Experiment

Use the `train_federated.py` script. The configuration is managed by Hydra under `conf/federated` (see the [Hydra Configuration Guide](../hydra.md)).

### Basic FedAvg on MNIST
```bash
uv run python scripts/train_federated.py method=fedavg dataset=mnist num_clients=10 num_comm_rounds=50
```

### FedIVON on non-IID CIFAR-10
```bash
uv run python scripts/train_federated.py method=fedivon dataset=cifar10 alpha2=0.1 num_clients=20
```

## Specialized Scripts

### Tracking ELBO
To track the Variational Objective/ELBO specifically for `FedIVON`:
```bash
uv run python scripts/track_elbo.py method=fedivon dataset=mnist num_clients=5
```

### Toy 2D Illustrations
To generate 2D decision boundary plots for a global federated model:
```bash
uv run python scripts/illustrate_federated.py method=fedivon dataset=toy num_clients=3
```
This will save a `toy_illustration.png` file to your current directory.

## Configuration Structure
- **Directories:** Configurations are in `conf/federated/`.
- **Datasets:** `conf/federated/dataset/` contains `mnist.yaml`, `cifar10.yaml`, etc.
- **Methods:** `conf/federated/method/` contains hyperparameters for each FL technique.
