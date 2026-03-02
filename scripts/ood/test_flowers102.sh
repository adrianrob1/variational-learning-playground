#!/bin/bash
# scripts/ood/test_flowers102.sh
# Evaluate a trained CIFAR-10 checkpoint against the Flowers102 OOD dataset.
#
# Usage: bash test_flowers102.sh <optimizer> <arch> <seed>
#   optimizer: sgd | mcdrop | swag | ivon | vogn | bbb
#   arch:      resnet20 | resnet20_mcdrop | resnet20_swag | ...
#   seed:      integer run seed (default: 0)
#
# Example:
#   bash test_flowers102.sh ivon resnet20 0

set -euo pipefail

optname=${1:?Usage: test_flowers102.sh <optimizer> <arch> <seed>}
arch=${2:?Usage: test_flowers102.sh <optimizer> <arch> <seed>}
seed=${3:-0}
datadir=${DATADIR:-data}
device=cuda

traindir=runs/cifar10/${optname}/${arch}/${seed}
savedir=${traindir}/ood_flowers102
mkdir -p "${savedir}"

ts=$(date +"%Y%m%dT%H%M%S")

uv run python -m vlbench.ood.run \
    "${traindir}" \
    --ood_dataset flowers102 \
    --device "${device}" \
    --data_dir "${datadir}" \
    --save_dir "${savedir}" \
    --saveoutput \
    $( [[ "${optname}" == "mcdrop" ]]  && echo "--testsamples 32" ) \
    $( [[ "${optname}" == "swag" ]]    && echo "--swag_modelsamples 64 --swag_samplemode modelwise" ) \
    $( [[ "${optname}" =~ ^(ivon|vogn|bbb)$ ]] && echo "--testrepeat 64" ) \
    |& tee -a "${savedir}/stdout-${ts}.log"
