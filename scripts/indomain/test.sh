#!/bin/bash
# Generic testing runner: bash test.sh <optimizer> <model> <dataset> <traindir>

METHOD=${1:-sgd}
MODEL=${2:-resnet20}
DATASET=${3:-cifar10}
TRAINDIR=${4:-runs/cifar10/sgd/resnet20}

uv run python -m vlbench.indomain.test \
    method=$METHOD \
    model=$MODEL \
    dataset=$DATASET \
    traindir=$TRAINDIR \
    save_dir=$TRAINDIR/test_results
