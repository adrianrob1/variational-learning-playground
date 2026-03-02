# Variational Adam Optimizer

`VariationalAdam` is a variational extension of the standard `AdamW` optimizer. It maintains a Gaussian posterior over the model parameters, adaptively updating the posterior mean and variance based on the second-order information provided by Adam's squared gradient moving average.

## Key Features

- **Bayesian Framework**: Provides a principled way to estimate model uncertainty.
- **AdamW Integration**: Leverages the robust convergence properties of AdamW.
- **Variational Updates**: Maintains a posterior mean and variance, sampling from the distribution for each forward pass.

## Parameters

| Parameter        | Default        | Description                                                         |
| :--------------- | :------------- | :------------------------------------------------------------------ |
| `lr`             | `1e-3`         | Learning rate.                                                      |
| `prior_variance` | `1e-2`         | Variance of the Gaussian prior.                                     |
| `betas`          | `(0.9, 0.999)` | Coefficients for Adam's moving averages (first and second moments). |
| `eps`            | `1e-8`         | Term added to the denominator for numerical stability.              |
| `weight_decay`   | `0.0`          | Decoupled weight decay coefficient (AdamW).                         |

## Running Variational Adam on Tasks

`VariationalAdam` is available in the `vloptimizers` library and can be used in custom training loops or by registering it in the benchmark configurations.

### 1. In-Domain Training

```bash
uv run python -m vlbench.indomain.train \
    method=variational_adam \
    model=resnet20 \
    dataset=cifar10
```

### 2. Image Classification

```bash
uv run python -m vlbench.image_classification.train \
    optimizer=variational_adam \
    dataset=cifar10 \
    arch=resnet20
```

### 3. BDL Competition

```bash
uv run python -m vlbench.bdl_competition.train \
    method=variational_adam \
    dataset=uci
```

### 4. OOD Evaluation

```bash
uv run python -m vlbench.ood.run \
    runs/cifar10/variational_adam/resnet20 \
    --ood_dataset svhn \
    -tr 64
```

### 5. MC Samples Ablation

```bash
uv run python scripts/eval_mcsamples.py \
    save_dir=runs/cifar10/variational_adam/resnet20/0 \
    mc_samples_list="1,2,4,8,16,32,64"
```

## Visualization

You can generate **reliability diagrams** (calibration plots) for in-domain benchmarks using the `--plotdiagram` flag. This will save a PDF file in the `save_dir`.

### In-Domain Training Reliability Diagram
```bash
uv run python -m vlbench.indomain.test \
    traindir=runs/cifar10/variational_adam/resnet20 \
    dataset=cifar10 \
    --plotdiagram
```

### Image Classification Reliability Diagram
```bash
uv run python -m vlbench.image_classification.test \
    traindir=runs/imagenet/variational_adam/resnet50 \
    dataset=imagenet \
    --plotdiagram
```

---

## References

For more implementation details, see [src/vloptimizers/variational_adam.py](../../src/vloptimizers/variational_adam.py).
