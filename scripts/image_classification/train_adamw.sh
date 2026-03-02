#!/bin/bash
# scripts/image_classification/train_adamw.sh
# Train with AdamW optimizer.
# Usage: bash train_adamw.sh <dataset> <arch> <seed>

set -euo pipefail
ts=$(date +"%Y%m%dT%H%M%S")

dataset=${1:?Usage: train_adamw.sh <dataset> <arch> <seed>}
arch=${2:?Usage: train_adamw.sh <dataset> <arch> <seed>}
seed=${3:-0}
datadir=${DATADIR:-data}

epochs=200
device=cuda
tbatch=128
vbatch=128
split=1.0
savedir=runs/${dataset}/adamw/${arch}/${seed}
mkdir -p ${savedir}

uv run python -m vlbench.image_classification.train \
    optimizer=adamw \
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
