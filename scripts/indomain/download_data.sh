#!/bin/bash
# Download datasets: bash download_data.sh <dataset_name>

DATASET=${1:-cifar10}
DATA_DIR=${2:-data}

uv run python -c "from vldatasets.standard import TRAINDATALOADERS; TRAINDATALOADERS['$DATASET']('$DATA_DIR', 1.0, 1, False, 1, 1)"
