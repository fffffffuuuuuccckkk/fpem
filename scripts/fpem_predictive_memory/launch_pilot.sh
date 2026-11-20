#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/OuXiaoyu/Time-Series-Library-FPEM
OUT="$ROOT/results/fpem_predictive_memory/pilot_20261024"
PY=/data/OuXiaoyu/miniconda3/envs/basicts/bin/python
mkdir -p "$OUT"
if [[ "${1:-}" == worker ]]; then
  case_name="$2"
  gpu="$3"
  mkdir -p "$OUT/$case_name"
  cd "$ROOT"
  export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
  exec "$PY" tools/run_fpem_predictive_memory_queue.py "$case_name" \
    --gpu "$gpu" --output results/fpem_predictive_memory/pilot_20261024 \
    >> "$OUT/$case_name/launcher.log" 2>&1
fi
screen -dmS fpem_mem_patch bash "$ROOT/scripts/fpem_predictive_memory/launch_pilot.sh" \
  worker patchtst_ETTh1_720 0
screen -dmS fpem_mem_itrans bash "$ROOT/scripts/fpem_predictive_memory/launch_pilot.sh" \
  worker itransformer_ETTm1_720 1
screen -dmS fpem_mem_modern bash "$ROOT/scripts/fpem_predictive_memory/launch_pilot.sh" \
  worker moderntcn_ETTm2_720 2
screen -ls
