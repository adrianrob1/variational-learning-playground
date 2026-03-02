#!/bin/bash
# Generic training runner: bash train.sh <optimizer> <model> <dataset> <seed>

METHOD=${1:-sgd}
MODEL=${2:-resnet20}
DATASET=${3:-cifar10}
SEED=${4:-0}

uv run python -m vlbench.indomain.train \
    method=$METHOD \
    model=$MODEL \
    dataset=$DATASET \
    seed=$SEED \
    save_dir=runs/$DATASET/$METHOD/$MODEL/$SEED \
    "$@"
