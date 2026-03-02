#!/bin/bash
# scripts/image_classification/test_ivon.sh
# Evaluate IVON checkpoints with Bayesian averaging.
# Usage: bash test_ivon.sh <dataset> <arch> <seed> [testrepeat]
#   testrepeat defaults to 10 (number of posterior samples)

set -euo pipefail

dataset=${1:?Usage: test_ivon.sh <dataset> <arch> <seed> [testrepeat]}
arch=${2:?Usage: test_ivon.sh <dataset> <arch> <seed> [testrepeat]}
seed=${3:-0}
testrepeat=${4:-10}
datadir=${DATADIR:-data}
device=cuda

traindir=runs/${dataset}/ivon/${arch}/${seed}
savedir=${traindir}/test
mkdir -p ${savedir}

uv run python -m vlbench.image_classification.test \
    ${traindir} \
    ${dataset} \
    --device ${device} \
    --data_dir ${datadir} \
    --save_dir ${savedir} \
    --testrepeat ${testrepeat} \
    --plotdiagram \
    --saveoutput
