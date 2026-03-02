#!/usr/bin/env bash
# install_ffcv.sh
# Installs ffcv and its system/Python dependencies.
# Requires: uv (https://docs.astral.sh/uv/), CUDA toolkit, apt (Debian/Ubuntu)
set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Detect CUDA version for the correct cupy variant
# ---------------------------------------------------------------------------
if command -v nvcc &>/dev/null; then
    CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
    CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
else
    echo "WARNING: nvcc not found. Defaulting to CUDA 12.x cupy variant."
    CUDA_MAJOR=12
fi

if   [ "$CUDA_MAJOR" -ge 12 ]; then CUPY_PKG="cupy-cuda12x"
elif [ "$CUDA_MAJOR" -eq 11 ]; then CUPY_PKG="cupy-cuda11x"
else
    echo "ERROR: Unsupported CUDA major version: $CUDA_MAJOR"
    exit 1
fi

echo "Detected CUDA major version: ${CUDA_MAJOR} → installing ${CUPY_PKG}"

# ---------------------------------------------------------------------------
# 2. System-level dependencies (libjpeg-turbo, pkg-config)
# ---------------------------------------------------------------------------
echo "Installing system dependencies via apt..."
sudo apt-get update -qq
sudo apt-get install -y pkg-config libturbojpeg0-dev libopencv-dev

# ---------------------------------------------------------------------------
# 3. Python dependencies via uv
# ---------------------------------------------------------------------------
echo "Installing Python packages via uv..."
uv add --extra ffcv ffcv numba "$CUPY_PKG" opencv-python-headless

echo ""
echo "ffcv installation complete."
