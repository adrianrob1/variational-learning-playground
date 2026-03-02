#!/bin/bash
# scripts/image_classification/train_ivon.sh
# Train with IVON optimizer.
# Usage: bash train_ivon.sh <dataset> <arch> <seed>
#   dataset: cifar10 | cifar100 | tinyimagenet
#   arch:    resnet20 | resnet18wide | preresnet110 | densenet121
#   seed:    integer

set -euo pipefail
ts=$(date +"%Y%m%dT%H%M%S")

dataset=${1:?Usage: train_ivon.sh <dataset> <arch> <seed>}
arch=${2:?Usage: train_ivon.sh <dataset> <arch> <seed>}
seed=${3:-0}
datadir=${DATADIR:-data}

epochs=200
device=cuda
tbatch=50
vbatch=50
split=1.0

case $dataset in
  cifar10 | cifar100)
    ess=50000
    ;;
  tinyimagenet)
    ess=200000
    ;;
  *)
    echo "Unknown dataset: ${dataset}" >&2
    exit 1
    ;;
esac

savedir=runs/${dataset}/ivon/${arch}/${seed}
mkdir -p ${savedir}

uv run python -m vlbench.image_classification.train \
    optimizer=ivon \
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
    optimizer.ess=${ess} \
    plotdiagram=true \
    |& tee -a ${savedir}/stdout-${ts}.log
