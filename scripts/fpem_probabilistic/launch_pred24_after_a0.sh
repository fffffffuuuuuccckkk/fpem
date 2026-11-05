#!/usr/bin/env bash
# Server3: finish A0 first, then run the three backbone-specific pred=24 banks.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/fpem_best_a0_code_20261013}"
DATA_PROJECT="${DATA_PROJECT:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/tslib/bin/python}"
ROOT="$DATA_PROJECT/results/fpem_probabilistic/test_best_backbones_a0_20261013"
PATCH_ROOT="$DATA_PROJECT/results/fpem_probabilistic/test_best_patchtst_a0_20261013"
datasets=(ETTh1 ETTh2 ETTm1 ETTm2 ExchangeRate)
while true; do
  ready=1
  for dataset in "${datasets[@]}"; do
    [[ -s "$ROOT/tslib_24_baselines/iTransformer/$dataset/pred_24/metrics.json" &&
       -e "$ROOT/tslib_24_baselines/iTransformer/$dataset/pred_24/run_complete" &&
       -s "$ROOT/moderntcn_a0_pred24/$dataset/A0/metrics_and_diagnostics.json" &&
       -s "$ROOT/references/moderntcn/$dataset/pred_24.pt" ]] || ready=0
  done
  (( ready )) && break
  echo "WAIT pred24 A0 $(date -Is)"
  sleep 30
done
"$PYTHON" "$PROJECT_DIR/tools/prepare_pred24_test_best_a0_catalogs.py" \
  --project-root "$DATA_PROJECT" --output-root "$ROOT"
for dataset in "${datasets[@]}"; do
  [[ -s "$PATCH_ROOT/baseline_checkpoints/$dataset/pred_24/checkpoint.pth" ]] || {
    echo "Missing PatchTST pred24 checkpoint for $dataset" >&2; exit 1;
  }
done
manifest="$PROJECT_DIR/scripts/fpem_probabilistic/test_best_a0_pred24_manifest.csv"
runner="$PROJECT_DIR/scripts/fpem_probabilistic/run_test_best_patchtst_a0.sh"
PROJECT_DIR="$PROJECT_DIR" DATA_PROJECT="$DATA_PROJECT" PYTHON="$PYTHON" \
  BACKBONE=patchtst GPU_LIST=0 bash "$runner" queue "$manifest" \
  >"$PATCH_ROOT/pred24_queue.log" 2>&1 & patch_pid=$!
PROJECT_DIR="$PROJECT_DIR" DATA_PROJECT="$DATA_PROJECT" PYTHON="$PYTHON" \
  BACKBONE=itransformer GPU_LIST=1 bash "$runner" queue "$manifest" \
  >"$ROOT/itransformer_pred24_queue.log" 2>&1 & itra_pid=$!
PROJECT_DIR="$PROJECT_DIR" DATA_PROJECT="$DATA_PROJECT" PYTHON="$PYTHON" \
  BACKBONE=moderntcn GPU_LIST=2,3 bash "$runner" queue "$manifest" \
  >"$ROOT/moderntcn_pred24_queue.log" 2>&1 & modern_pid=$!
wait "$patch_pid"
wait "$itra_pid"
wait "$modern_pid"
echo "PRED24_QUEUES_FINISHED $(date -Is)"
