#!/bin/bash
# scripts/image_classification/test_standard.sh
# Evaluate standard (SGD/AdamW etc.) checkpoints.
# Usage: bash test_standard.sh <dataset> <arch> <optimizer_name> <seed>

set -euo pipefail

dataset=${1:?Usage: test_standard.sh <dataset> <arch> <optimizer_name> <seed>}
arch=${2:?}
optname=${3:-sgd}
seed=${4:-0}
datadir=${DATADIR:-data}
device=cuda

traindir=runs/${dataset}/${optname}/${arch}/${seed}
savedir=${traindir}/test
mkdir -p ${savedir}

uv run python -m vlbench.image_classification.test \
    ${traindir} \
    ${dataset} \
    --device ${device} \
    --data_dir ${datadir} \
    --save_dir ${savedir} \
    --plotdiagram \
    --saveoutput
