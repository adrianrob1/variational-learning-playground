# VOGN Optimizer

`VOGN` (Variational Online Gauss-Newton) is a natural gradient variational optimizer that estimates the Gauss-Newton approximation of the Hessian. It is particularly effective for training Bayesian Neural Networks (BNNs) and provides robust uncertainty estimates through posterior sampling.

## Key Features

- **Natural Gradient Updates**: Follows the natural gradient direction for faster convergence in parameter space.
- **Gauss-Newton Approximation**: Efficiently approximates the Hessian using the Gauss-Newton method.
- **Variational Inference**: Maintains a Gaussian posterior and enables Monte Carlo (MC) sampling during training and inference.

## Parameters

| Parameter         | Default    | Description                                                               |
| :---------------- | :--------- | :------------------------------------------------------------------------ |
| `lr`              | Required   | Learning rate.                                                            |
| `data_size`       | Required   | Total number of samples in the dataset (used for scaling the likelihood). |
| `mc_samples`      | `1`        | Number of MC samples used during training.                                |
| `momentum_grad`   | `0.9`      | Momentum coefficient for the gradient.                                    |
| `momentum_hess`   | `1.0 - lr` | Momentum coefficient for the Hessian approximation.                       |
| `prior_precision` | `1.0`      | Precision (inverse variance) of the isotropic Gaussian prior.             |
| `temperature`     | `1.0`      | Temperature parameter for the posterior.                                  |

## Running VOGN on Tasks

### 1. In-Domain Training

```bash
uv run python -m vlbench.indomain.train \
    method=vogn \
    model=resnet20 \
    dataset=cifar10
```

### 2. Image Classification

```bash
uv run python -m vlbench.image_classification.train \
    optimizer=vogn \
    dataset=cifar10 \
    arch=resnet20
```

### 3. BDL Competition

```bash
uv run python -m vlbench.bdl_competition.train \
    method=vogn \
    dataset=uci
```

### 4. OOD Evaluation

```bash
uv run python -m vlbench.ood.run \
    runs/cifar10/vogn/resnet20 \
    --ood_dataset svhn \
    -tr 64
```

### 5. MC Samples Ablation

```bash
uv run python scripts/eval_mcsamples.py \
    save_dir=runs/cifar10/vogn/resnet20/0 \
    mc_samples_list="1,2,4,8,16,32,64"
```

## Visualization

You can generate **reliability diagrams** (calibration plots) for in-domain benchmarks using the `--plotdiagram` flag. This will save a PDF file in the `save_dir`.

### In-Domain Training Reliability Diagram
```bash
uv run python -m vlbench.indomain.test \
    traindir=runs/cifar10/vogn/resnet20 \
    dataset=cifar10 \
    --plotdiagram
```

### Image Classification Reliability Diagram
```bash
uv run python -m vlbench.image_classification.test \
    traindir=runs/imagenet/vogn/resnet50 \
    dataset=imagenet \
    --plotdiagram
```

---

## References

For more implementation details, see [src/vloptimizers/vogn.py](../../src/vloptimizers/vogn.py).
