# MC Samples Ablation Benchmark

This benchmark evaluates the effect of Monte Carlo (MC) sample counts during both training and inference.

## Prerequisites

- **Training**: None.
- **Inference Ablation**: Requires pre-trained models from the [In-Domain Training](in_domain.md) task.

## Training with Varying MC Samples

To train a model with a specific number of MC samples (e.g., 2), use the `mc_samples` override (see [Command-Line Overrides](../hydra.md#command-line-overrides)):

```bash
uv run python -m vlbench.indomain.train \
    method=ivon \
    model=resnet20 \
    dataset=cifar10 \
    mc_samples=2 \
    save_dir=runs/cifar10/ivon_mc2/resnet20/0
```

## Inference Ablation Study

To evaluate how increasing the number of samples during inference impacts performance on a pre-trained model:

```bash
uv run python scripts/eval_mcsamples.py \
    save_dir=runs/cifar10/ivon_mc1/resnet20/0 \
    mc_samples_list="1,2,4,8,16,32,64"
```

The script will generate a `test_mc_ablation.csv` in the specified `save_dir`.
