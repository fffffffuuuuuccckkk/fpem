#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/fpem_environment_semantic_analysis}"

cd "$ROOT"
exec "$PYTHON" tools/analyze_predictive_environment_semantics.py \
  --repo "$ROOT" \
  --output_root "$OUTPUT_ROOT" \
  --permutations 1000 \
  "$@"
