# AdaHessian Optimizer

`AdaHessian` is an adaptive second-order optimizer that uses a Hessian-based scaling of the learning rate. It approximates the diagonal of the Hessian using the Hutchinson method, providing more robust convergence in complex landscapes compared to standard first-order methods.

## Key Features

- **Second-Order Adaptive Learning Rate**: Scales the learning rate based on an approximation of the Hessian diagonal.
- **Hutchinson Hessian Approximation**: Efficiently estimates the Hessian trace without requiring the full matrix.
- **Stability**: Often provides more stable training for large-scale models and architectures with sharp loss landscapes.

## Parameters

| Parameter       | Default        | Description                                                                  |
| :-------------- | :------------- | :--------------------------------------------------------------------------- |
| `lr`            | `0.1`          | Learning rate.                                                               |
| `betas`         | `(0.9, 0.999)` | Coefficients for moving averages of gradients and the squared Hessian trace. |
| `eps`           | `1e-8`         | Term added to the denominator to improve numerical stability.                |
| `weight_decay`  | `0.0`          | Weight decay (L2 penalty) coefficient.                                       |
| `hessian_power` | `1.0`          | Exponent of the Hessian trace.                                               |
| `update_each`   | `1`            | Frequency (in steps) of Hessian trace approximation.                         |
| `n_samples`     | `1`            | Number of samples for Hutchinson approximation.                              |

## Usage Note

`AdaHessian` requires the computation of second-order derivatives. When calling `backward()` in your training loop, you must use `create_graph=True`:

```python
loss = criterion(output, target)
loss.backward(create_graph=True)
optimizer.step()
```

## Running AdaHessian on Tasks

### 1. In-Domain Training

```bash
uv run python -m vlbench.indomain.train \
    method=adahessian \
    model=resnet20 \
    dataset=cifar10
```

### 2. Image Classification

```bash
uv run python -m vlbench.image_classification.train \
    optimizer=adahessian \
    dataset=cifar10 \
    arch=resnet20
```

### 3. BDL Competition

```bash
uv run python -m vlbench.bdl_competition.train \
    method=adahessian \
    dataset=uci
```

### 4. OOD Evaluation

`AdaHessian` is a second-order optimizer but does not perform variational sampling by default. For OOD evaluation, it provides a single point estimate:

```bash
uv run python -m vlbench.ood.run \
    runs/cifar10/adahessian/resnet20 \
    --ood_dataset svhn \
    -tr 1
```

### 5. MC Samples Ablation

Since `AdaHessian` is deterministic during inference, the MC samples ablation will show constant performance regardless of the number of samples:

```bash
uv run python scripts/eval_mcsamples.py \
    save_dir=runs/cifar10/adahessian/resnet20/0 \
    mc_samples_list="1,2,4,8,16,32,64"
```

## Visualization

You can generate **reliability diagrams** (calibration plots) for in-domain benchmarks using the `--plotdiagram` flag. This will save a PDF file in the `save_dir`.

### In-Domain Training Reliability Diagram
```bash
uv run python -m vlbench.indomain.test \
    traindir=runs/cifar10/adahessian/resnet20 \
    dataset=cifar10 \
    --plotdiagram
```

### Image Classification Reliability Diagram
```bash
uv run python -m vlbench.image_classification.test \
    traindir=runs/imagenet/adahessian/resnet50 \
    dataset=imagenet \
    --plotdiagram
```

---

## References

For more implementation details, see [src/vloptimizers/adahessian.py](../../src/vloptimizers/adahessian.py).
