#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_EXEC="${PYTHON_BIN:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
SEQ_LEN="${SEQ_LEN:-96}"
PRED_LEN="${PRED_LEN:-96}"
SETTING="${SETTING:-deformable_pattern_bank_ETTh1_${SEQ_LEN}_${PRED_LEN}}"
SMOKE="${SMOKE:-0}"
EXTRA_ARGS=()
if [[ "$SMOKE" == "1" ]]; then
  EXTRA_ARGS+=(--max_windows 512)
  SETTING="${SETTING}_smoke"
fi

"$PYTHON_EXEC" -u tools/run_deformable_pattern_bank.py \
  --root_path ./dataset/ETT-small/ \
  --seq_len "$SEQ_LEN" --pred_len "$PRED_LEN" \
  --patch_len 16 --stride 8 --setting "$SETTING" \
  "${EXTRA_ARGS[@]}"
