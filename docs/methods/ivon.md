# IVON Optimizer

`IVON` (Improved Variational Online Newton) is a variational optimizer that adaptively estimates the diagonal Hessian using either the "Price" or "Gradient Square" approximation. It provides a Bayesian framework for standard deep learning models, enabling uncertainty estimation through posterior sampling.

## Key Features

- **Adaptive Hessian Estimation**: Uses moving averages of gradients and noise-gradient products (Price estimator) to approximate the posterior variance.
- **Variational Inference**: Enables Monte Carlo (MC) sampling of weights from the Gaussian posterior during both training and inference.
- **Efficient Second-Order Updates**: Provides faster convergence than standard first-order methods like SGD or Adam by leveraging Hessian information.
- **Compatibility**: Integrates seamlessly with standard PyTorch modules and allows for easy quantification of model uncertainty.

## Parameters

`IVON` uses several hyperparameters to control the variational update and the Hessian estimation:

| Parameter      | Default   | Description                                                              |
| :------------- | :-------- | :----------------------------------------------------------------------- |
| `lr`           | Required  | Learning rate.                                                           |
| `ess`          | Required  | Effective Sample Size (controls the strength of the variational update). |
| `mc_samples`   | `1`       | Number of MC samples used during training.                               |
| `weight_decay` | `1e-4`    | Weight decay coefficient (often interpreted as prior precision).         |
| `hess_init`    | `1.0`     | Initial value for the diagonal Hessian approximation.                    |
| `beta1`        | `0.9`     | Coefficient for the moving average of the gradient.                      |
| `beta2`        | `0.99999` | Coefficient for the moving average of the Hessian.                       |
| `hess_approx`  | `"price"` | Method for Hessian estimation (`"price"` or `"gradsq"`).                 |

## Running IVON on Tasks

`IVON` is the default variational optimizer for most tasks in this benchmark.

### 1. In-Domain Training

```bash
uv run python -m vlbench.indomain.train \
    method=ivon \
    model=resnet20 \
    dataset=cifar10
```

### 2. Image Classification

```bash
uv run python -m vlbench.image_classification.train \
    optimizer=ivon \
    dataset=cifar10 \
    arch=resnet20
```

### 3. BDL Competition

```bash
uv run python -m vlbench.bdl_competition.train \
    method=ivon \
    dataset=uci
```

### 4. OOD Evaluation

Run the OOD evaluation script with a high number of posterior samples (`-tr` / testrepeat) to get better uncertainty estimates:

```bash
uv run python -m vlbench.ood.run \
    runs/cifar10/ivon/resnet20 \
    --ood_dataset svhn \
    -tr 64
```

### 5. MC Samples Ablation

Evaluate the impact of varying the number of Monte Carlo samples during inference:

```bash
uv run python scripts/eval_mcsamples.py \
    save_dir=runs/cifar10/ivon/resnet20/0 \
    mc_samples_list="1,2,4,8,16,32,64"
```

## Visualization

You can generate **reliability diagrams** (calibration plots) for in-domain benchmarks using the `--plotdiagram` flag. This will save a PDF file in the `save_dir`.

### In-Domain Training Reliability Diagram
```bash
uv run python -m vlbench.indomain.test \
    traindir=runs/cifar10/ivon/resnet20 \
    dataset=cifar10 \
    --plotdiagram
```

### Image Classification Reliability Diagram
```bash
uv run python -m vlbench.image_classification.test \
    traindir=runs/imagenet/ivon/resnet50 \
    dataset=imagenet \
    --plotdiagram
```

---

## References

For more implementation details, see [src/vloptimizers/ivon.py](../../src/vloptimizers/ivon.py).
