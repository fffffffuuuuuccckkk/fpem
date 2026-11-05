#!/usr/bin/env bash
# Launch one of three non-overlapping, four-GPU invariant-head LR shards.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/OuXiaoyu/Time-Series-Library-FPEM/results/fpem_statistical_invhead_lr_search_20261018}"
MANIFEST="${MANIFEST:-$PROJECT_DIR/scripts/fpem_statistical/inv_head_manifest.csv}"
: "${FP_STAT_SHARD:?Set FP_STAT_SHARD to 0, 1 or 2}"
mkdir -p "$OUTPUT_ROOT"
cd "$PROJECT_DIR"
exec "$PYTHON" tools/run_fpem_stat_invhead_search.py run \
  --manifest "$MANIFEST" --server-index "$FP_STAT_SHARD" \
  --project-dir "$PROJECT_DIR" --output-root "$OUTPUT_ROOT" \
  >"$OUTPUT_ROOT/launcher.log" 2>&1
