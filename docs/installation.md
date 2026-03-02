# Installation

This guide covers the prerequisites and steps to set up the `variational-learning-benchmark` environment.

## Prerequisites

- **Python 3.10+**
- **`uv`** — [Install uv](https://docs.astral.sh/uv/getting-started/installation/)
- **CUDA 11.x or 12.x** (Required for GPU acceleration and FFCV)

## Standard Setup

Clone the repository and install dependencies using `uv`:

```bash
git clone https://github.com/adrianrob1/variational-learning-benchmark.git
cd variational-learning-benchmark
uv sync
```

## Optional: FFCV Support

[FFCV](https://ffcv.io/) is a fast data-loading backend used for large-scale benchmarks.

### When is FFCV Required?

- **Distributed ImageNet Training**: FFCV is **mandatory** for the ImageNet task to achieve high-performance data loading.
- **CIFAR/TinyImageNet**: FFCV is **optional**. The standard setup uses standard PyTorch dataloaders, but you can opt-in to FFCV for faster training on these datasets if desired.

### Prerequisites for FFCV

- **`apt`** package manager (Debian/Ubuntu)
- **`build-essential`** (for compiling FFCV if no wheels are available)

### Automated Installation

We provide a helper script to install FFCV and its system dependencies:

```bash
bash scripts/install_ffcv.sh
```

The script will:
1. Detect your CUDA version and install the matching `cupy` wheel.
2. Install required system libraries (`libjpeg-turbo`, `pkg-config`) via `apt`.
3. Add `ffcv`, `numba`, `cupy`, and `opencv-python-headless` to the project as an optional dependency group via `uv add --extra ffcv`.

> [!NOTE]
> If you encounter build errors, ensure you have the required headers:
> ```bash
> sudo apt-get install -y build-essential libjpeg-turbo8-dev
> ```
