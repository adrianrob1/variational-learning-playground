#!/bin/bash
# Evaluates ImageNet models using mean inference
# Usage: bash scripts/image_classification/test_imagenet_mean.sh <traindir> <datadir> [seed]

TRAINDIR=$1
DATADIR=$2
SEED=${3:-0}

if [ -z "$TRAINDIR" ] || [ -z "$DATADIR" ]; then
    echo "Usage: $0 <traindir> <datadir> [seed]"
    exit 1
fi

uv run python -m vlbench.image_classification.test \
    "$TRAINDIR" \
    imagenet \
    --data_dir "$DATADIR" \
    --testrepeat 0 \
    --seed "$SEED" \
    --device cuda \
    --plotdiagram \
    --save_dir "$TRAINDIR"
