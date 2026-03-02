# In-Domain Training and Evaluation

This task involves training models on standard datasets like CIFAR-10, CIFAR-100, and TinyImageNet, followed by evaluation on their respective test sets.

## Prerequisites

- None (all data is downloaded automatically by PyTorch/Torchvision).

## Training

Use the training script with Hydra-style configuration (see [Command-Line Overrides](../hydra.md#command-line-overrides)):

```bash
uv run python -m vlbench.indomain.train \
    method=ivon \
    model=resnet20 \
    dataset=cifar10 \
    seed=0 \
    save_dir=runs/cifar10/ivon/resnet20/0
```

### Supported Options

- **Methods**: `sgd`, `adamw`, `adahessian`, `ivon`, `vogn`, `bbb`, `swag`, `mcdropout`.
- **Models**: `resnet20`, `resnet18wide`, `preresnet110`, `densenet121`.
- **Datasets**: `cifar10`, `cifar100`, `tinyimagenet`, or any registered dataset via Hydra (e.g., `mnist_hf`).

## Evaluation

The evaluation script aggregates results across multiple seeds. It expects the `traindir` to contain subfolders named `0`, `1`, `2`, `3`, `4`.

### Mean Inference
Standard single-forward pass using the posterior mean (for variational methods).

```bash
uv run python -m vlbench.indomain.test \
    traindir=runs/cifar10/ivon/resnet20 \
    testrepeat=1 \
    save_dir=results/cifar10/ivon/resnet20
```

### Bayesian Inference
Monte Carlo ensemble by sampling weights from the posterior.

```bash
uv run python -m vlbench.indomain.test \
    traindir=runs/cifar10/ivon/resnet20 \
    testrepeat=64 \
    save_dir=results/cifar10/ivon_bayes/resnet20
```

### Calibration
To calculate Expected Calibration Error (ECE) and plot reliability diagrams, use the `--plotdiagram` flag:

```bash
uv run python -m vlbench.indomain.test \
    traindir=runs/cifar10/ivon/resnet20 \
    save_dir=results/cifar10/ivon/resnet20 \
    --plotdiagram
```

---

## New Dataset Registration

As of the latest refactor, you can add any dataset (including from **Hugging Face**) by simply creating a YAML configuration in `conf/indomain/dataset/`. For more details, see [How to Extend (Add New Components)](../extending.md).
