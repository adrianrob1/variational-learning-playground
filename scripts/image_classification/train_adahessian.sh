#!/bin/bash
# scripts/image_classification/train_adahessian.sh
# Train with AdaHessian optimizer.
# Usage: bash train_adahessian.sh <dataset> <arch> <seed>

set -euo pipefail
ts=$(date +"%Y%m%dT%H%M%S")

dataset=${1:?Usage: train_adahessian.sh <dataset> <arch> <seed>}
arch=${2:?Usage: train_adahessian.sh <dataset> <arch> <seed>}
seed=${3:-0}
datadir=${DATADIR:-data}

epochs=200
device=cuda
tbatch=128
vbatch=128
split=1.0
savedir=runs/${dataset}/adahessian/${arch}/${seed}
mkdir -p ${savedir}

uv run python -m vlbench.image_classification.train \
    optimizer=adahessian \
    dataset=${dataset} \
    arch=${arch} \
    seed=${seed} \
    device=${device} \
    epochs=${epochs} \
    tbatch=${tbatch} \
    vbatch=${vbatch} \
    tvsplit=${split} \
    data_dir=${datadir} \
    save_dir=${savedir} \
    plotdiagram=true \
    |& tee -a ${savedir}/stdout-${ts}.log
