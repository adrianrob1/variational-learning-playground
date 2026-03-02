# BDL Competition Task

This task refactors the NeurIPS 2021 Bayesian Deep Learning (BDL) Competition benchmarks for use with `vlbench`. It covers small-to-medium scale tasks across different domains (Image, Medical, Tabular).

## Prerequisites

- **Data Files**: Requires `.npz` files for the datasets to be present in the `data/` directory.

## Training

```bash
uv run python -m vlbench.bdl_competition.train method=ivon dataset=cifar10 model=cifar_alexnet
```

### MedMNIST
```bash
uv run python -m vlbench.bdl_competition.train dataset=medmnist model=medmnist_lenet
```

### UCI (Energy)
```bash
uv run python -m vlbench.bdl_competition.train dataset=uci model=uci_mlp
```

## Customizing Architectures

Architectures can be modified via Hydra overrides (see [Command-Line Overrides](../hydra.md#command-line-overrides)):

```bash
# Deeper MLP for UCI
uv run python -m vlbench.bdl_competition.train dataset=uci model=uci_mlp model.hidden_dims="[100, 100, 100]"

# Wider AlexNet for CIFAR-10
uv run python -m vlbench.bdl_competition.train dataset=cifar10 model=cifar_alexnet model.width_mult=2
```

## Evaluation

Calculate competition metrics including Accuracy, Agreement, Total Variation (TV), and Wasserstein-2 (W-2) distance.

```bash
uv run python -m vlbench.bdl_competition.test \
    dataset=[dataset] \
    model=[model] \
    save_dir=[path_to_checkpoints]
```
