# Out-Of-Distribution (OOD) Evaluation

Evaluates models trained on CIFAR-10 against out-of-distribution datasets like SVHN and Flowers102.

## Prerequisites

- **Training Checkpoints**: Requires models previously trained using the [In-Domain Training](in_domain.md) benchmark on **CIFAR-10**.

## Evaluation using Shell Scripts

Generic scripts are provided in `scripts/ood/`:

```bash
# Evaluate SVHN
bash scripts/ood/test_svhn.sh <method> <arch> <seed>

# Evaluate Flowers102
bash scripts/ood/test_flowers102.sh <method> <arch> <seed>
```

**Example:**
```bash
bash scripts/ood/test_svhn.sh ivon resnet20 0
```

## Manual Evaluation

For more control, use the `vlbench.ood.run` module directly:

```bash
uv run python -m vlbench.ood.run <traindir> --ood_dataset {svhn,flowers102} [options]
```

Run `uv run python -m vlbench.ood.run --help` for all available options.
