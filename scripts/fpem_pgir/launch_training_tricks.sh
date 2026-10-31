#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/OuXiaoyu/Time-Series-Library-FPEM
PYTHON=/data/OuXiaoyu/miniconda3/envs/basicts/bin/python
OUTPUT="$ROOT/results/fpem_pgir/training_tricks_20261024"
mkdir -p "$OUTPUT"

launch() {
  local case="$1" gpu="$2" name="$3"
  if screen -ls | grep -Fq ".${name}"; then
    printf 'already running: %s\n' "$name"
    return
  fi
  screen -dmS "$name" bash -lc "cd '$ROOT' && '$PYTHON' tools/run_fpem_pgir_training_search.py --case '$case' --gpu '$gpu' > '$OUTPUT/${case}_worker.log' 2>&1"
  printf 'launched: %s on GPU %s; log %s\n' "$name" "$gpu" "$OUTPUT/${case}_worker.log"
}

launch moderntcn_ETTm2_720 0 pgirtt_modern
launch patchtst_ETTh1_720 1 pgirtt_patch
launch itransformer_ETTm1_720 2 pgirtt_itrans
