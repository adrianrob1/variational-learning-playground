# Variational Learning Benchmark

A comprehensive toolkit for benchmarking variational inference and standard optimization methods in deep learning.

## Quick Start

### 1. Installation
Ensure you have `uv` installed, then sync the environment:
```bash
uv sync
```
*For FFCV/ImageNet support, see [Full Installation Guide](docs/installation.md).*

### 2. Run a Benchmark
```bash
uv run python -m vlbench.indomain.train method=ivon model=resnet20 dataset=cifar10
```

## Supported Tasks

| Task            | Description                                  | Dependencies            | Documentation                            |
| :-------------- | :------------------------------------------- | :---------------------- | :--------------------------------------- |
| **In-Domain**   | Standard Training/Eval (CIFAR, TinyImageNet) | None                    | [Link](docs/tasks/in_domain.md)          |
| **OOD**         | Out-of-Distribution (SVHN, Flowers102)       | In-Domain Checkpoints   | [Link](docs/tasks/ood.md)                |
| **Robustness**  | Data Corruptions (CIFAR-C)                   | In-Domain Checkpoints   | [Link](docs/tasks/robustness.md)         |
| **BDL Comp**    | NeurIPS BDL Competition Tasks                | Dataset `.npz` files    | [Link](docs/tasks/bdl_competition.md)    |
| **ImageNet**    | Large-scale Distributed Training             | **FFCV**, ImageNet Data | [Link](docs/tasks/imagenet.md)           |
| **MC Ablation** | Monte Carlo Sample counts study              | In-Domain Checkpoints   | [Link](docs/tasks/mc_ablation.md)        |
| **Federated**   | Federated Learning across clients            | None                    | [Link](docs/tasks/federated.md)          |
| **Text Gen**    | Text Generation & MBR Decoding               | None                    | [Link](docs/results-reproduction/mbr.md) |


## Documentation

- [Installation Guide](docs/installation.md)
- [Getting Started](docs/getting_started.md)
- [Hydra Configuration Guide](docs/hydra.md)
- [Task Guides](docs/tasks/index.md)
- [**How to Extend (Add New Components)**](docs/extending/index.md)

## License
GPL-3.0, see [LICENSE](LICENSE) for details.
