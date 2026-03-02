# Distributed ImageNet Training

Large-scale training on ImageNet using Distributed Data Parallel (DDP) and FFCV for optimized data loading.

## Prerequisites

- **FFCV**: Requires FFCV to be installed. See [Installation](../installation.md).
- [Installation](../installation.md).
- **ImageNet Dataset**: The raw ImageFolder-style dataset (with `train/` and `val/` subdirectories) must be available on your system.

## Option 1: Optimized FFCV Training (Recommended)

FFCV provides significantly faster data loading, which is crucial for large-scale training.

### 1. Prepare Dataset
Convert the raw ImageNet dataset into FFCV format:

```bash
uv run python scripts/prepare_ffcv.py /path/to/imagenet /path/to/save/ffcv
```

### 2. Launch Training
Use `torchrun` and point to the directory containing the `.ffcv` files, using Hydra overrides to select datasets and methods (see [Command-Line Overrides](../hydra.md#command-line-overrides)):

```bash
uv run torchrun --nproc_per_node=GPU_COUNT -m vlbench.image_classification.distributed_train \
    dataset=imagenet_ffcv \
    model=resnet50_imagenet \
    method=ivon_imagenet \
    data_dir=/path/to/save/ffcv \
    tbatch=1024
```

## Option 2: Standard PyTorch Training

If you do not want to use FFCV, you can run the benchmark using standard PyTorch `DataLoader`.

### 1. Launch Training
Simply point `data_dir` to the root of the raw ImageNet dataset (containing `train/` and `val/`). The script will automatically detect the absence of `.ffcv` files and fallback to the standard loader.

```bash
uv run torchrun --nproc_per_node=GPU_COUNT -m vlbench.image_classification.distributed_train \
    dataset=imagenet_ffcv \
    model=resnet50_imagenet \
    method=ivon_imagenet \
    data_dir=/path/to/raw/imagenet \
    tbatch=1024
```

> [!WARNING]
> Standard training will be significantly slower due to I/O bottlenecks. Ensure you have high-performance storage or a high number of CPU workers.

### Automatic Checkpointing

The script supports automatic checkpointing and resumption. By default, it uses high-level configurations in `conf/image_classification/distributed_config.yaml`.

## 3. Resuming Training

To resume from a specific checkpoint:

```bash
uv run torchrun --nproc_per_node=GPU_COUNT -m vlbench.image_classification.distributed_train \
    dataset=imagenet_ffcv \
    model=resnet50_imagenet \
    method=ivon_imagenet \
    data_dir=/path/to/save/ffcv \
    resume=/path/to/checkpoint.pt
```
