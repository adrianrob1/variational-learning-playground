# Getting Started

The `variational-learning-benchmark` (vlbench) is a comprehensive toolkit for evaluating variational inference and standard optimization methods in deep learning.

## Core Philosophy

1. **Hydra-Powered Configuration**: All tasks use [Hydra](https://hydra.cc/) for flexible configuration and overrides. See the [Hydra Quickstart Guide](hydra.md) for details.
2. **Standardized Interfaces**: Whether you are training an MLP on UCI or a ResNet-50 on ImageNet, the interfaces remain consistent.
3. **Reproducibility**: Benchmarks are designed to be run across multiple seeds, with aggregation scripts provided.

## Basic Workflow

Most tasks in `vlbench` follow this pattern:

1. **Setup**: Install dependencies as described in [Installation](installation.md).
2. **Train**: Run a training script for your desired method/model/dataset.
   ```bash
   uv run python -m vlbench.<task_area>.train method=... model=... dataset=...
   ```
3. **Evaluate**: Run the corresponding test script.
   ```bash
   uv run python -m vlbench.<task_area>.test traindir=...
   ```

## Next Steps

Explore the specific documentation for each task area in the [Tasks Index](tasks/index.md):

- [In-Domain Benchmarks](tasks/in_domain.md)
- [Out-Of-Distribution Evaluation](tasks/ood.md)
- [Robustness (CIFAR-C)](tasks/robustness.md)
- [BDL Competition Tasks](tasks/bdl_competition.md)
- [Large-scale ImageNet](tasks/imagenet.md)
- [MC Ablation Studies](tasks/mc_ablation.md)
- [Federated Learning](tasks/federated.md)
- [Text Generation (MBR)](results-reproduction/mbr.md)


## Optimizers and Methods

Learn more about the specific variational and standard optimizers implemented in `vlbench`:

- [IVON (Improved Variational Online Newton)](methods/ivon.md)
- [VOGN (Variational Online Gauss-Newton)](methods/vogn.md)
- [AdaHessian (Adaptive Second-Order)](methods/adahessian.md)
- [Variational Adam](methods/variational_adam.md)
