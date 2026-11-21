#!/usr/bin/env bash
set -euo pipefail

workers="${1:-8}"
python=".venv/bin/python"
mkdir -p results/logs

run() {
  local experiment=$1 model=$2
  echo "=== $experiment $model ==="
  "$python" -m src.evaluate --experiment "$experiment" --model "$model" --device cpu --workers "$workers" \
    --out "results/${experiment}_${model}.csv" 2>&1 | tee -a "results/logs/${experiment}_${model}.log"
}

run within csp_lda
run within fbcsp
run cross fbcsp
run within eegnet
run cross eegnet
run calibration eegnet
