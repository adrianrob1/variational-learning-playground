# Robustness Benchmark (CIFAR-C)

Evaluates model performance and calibration under various dataset corruptions using the CIFAR-10 Corruptions (C) and CIFAR-100 Corruptions (C) datasets.

## Prerequisites

- **Training Checkpoints**: Requires models previously trained using the [In-Domain Training](in_domain.md) benchmark on **CIFAR-10** or **CIFAR-100**.

## Running Evaluation

Use the image classification test script with the corruption dataset, specifying overrides via Hydra (see [Command-Line Overrides](../hydra.md#command-line-overrides)):

```bash
uv run python -m vlbench.image_classification.test \
    traindir=runs/cifar10/ivon/resnet20 \
    dataset=cifar10c \
    save_dir=results/cifar10c/ivon/resnet20
```

### Supported Datasets

- `cifar10c`
- `cifar100c`

The script will evaluate the model across all 15 types of corruptions and 5 severity levels, providing an aggregate robustness metric.
