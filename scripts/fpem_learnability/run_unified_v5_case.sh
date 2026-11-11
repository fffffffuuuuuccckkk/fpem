#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/OuXiaoyu/Time-Series-Library-FPEM
PYTHON=/data/OuXiaoyu/miniconda3/envs/basicts/bin/python
CASE="${1:?case required}"
GPU="${2:?gpu required}"
RUN_ROOT=results/fpem_learnability/unified_consensus_v5_20261026
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$ROOT"

"$PYTHON" tools/run_fpem_learnability_queue.py "$CASE" --gpu "$GPU" \
  --protocol unified --topk 5 --output "$RUN_ROOT/point" --attempts 3
"$PYTHON" tools/run_fpem_residual_difference_queue.py "$CASE" --gpu "$GPU" \
  --protocol unified --point_root "$RUN_ROOT/point" \
  --output "$RUN_ROOT/probability"
"$PYTHON" tools/summarize_fpem_learnability_v5.py "$CASE" --root "$RUN_ROOT"
