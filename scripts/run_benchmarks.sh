#!/usr/bin/env bash
# Run the full benchmark suite: every dataset x every standard horizon.
#   ./scripts/run_benchmarks.sh                 # all datasets, H in {96,192,336,720}
#   ./scripts/run_benchmarks.sh ETTh1 weather   # subset of datasets
# Extra flags are forwarded to src.train, e.g.:
#   ./scripts/run_benchmarks.sh traffic --freq_encoder mamba
set -euo pipefail
cd "$(dirname "$0")/.."

ALL_DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 weather electricity solar exchange_rate traffic)
HORIZONS=(96 192 336 720)

DATASETS=()
EXTRA=()
for arg in "$@"; do
  if [[ " ${ALL_DATASETS[*]} " == *" ${arg} "* ]]; then
    DATASETS+=("$arg")
  else
    EXTRA+=("$arg")
  fi
done
[[ ${#DATASETS[@]} -eq 0 ]] && DATASETS=("${ALL_DATASETS[@]}")

for ds in "${DATASETS[@]}"; do
  for h in "${HORIZONS[@]}"; do
    echo "=== ${ds} | pred_len=${h} ==="
    python -m src.train \
      --config "configs/${ds}.yaml" \
      --pred_len "${h}" \
      --experiment.name "${ds}_h${h}" \
      "${EXTRA[@]+"${EXTRA[@]}"}"
  done
done
