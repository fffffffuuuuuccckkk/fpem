#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/tslib/bin/python}"
source "$PROJECT_DIR/scripts/fpem_final/common.sh"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/fpem_backbone_a0_completion_20261013}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"
DRY_RUN="${DRY_RUN:-0}"
DATASETS="${DATASETS:-ETTh1 ETTh2 ETTm1 ETTm2 ExchangeRate}"
PRED_LENGTHS="${PRED_LENGTHS:-24 48 96 192 336 720}"
cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/status"
ARCHIVE_SHA256="$(sha256sum dataset/all_datasets.zip | cut -d ' ' -f1)"

complete_case() {
  [[ -s "$1/A0/metrics_and_diagnostics.json" ]]
}

jobs=()
for dataset in $DATASETS; do
  for pred in $PRED_LENGTHS; do
    old="results/fpem_final_backbones/cyclenet/$dataset/pred_$pred/A0"
    dest="$OUTPUT_ROOT/cyclenet/$dataset/pred_$pred/A0"
    if complete_case "$old" || complete_case "$dest"; then
      echo "reuse cyclenet/$dataset/pred_$pred"
    else
      jobs+=("$dataset:$pred")
    fi
  done
done
echo "CycleNet A0 missing=${#jobs[@]} GPUs=$GPU_LIST output=$OUTPUT_ROOT"
if [[ "$DRY_RUN" == 1 ]]; then printf '%s\n' "${jobs[@]}"; exit 0; fi

run_case() {
  local spec="$1" gpu="$2" dataset pred data_root data_class data_path freq batch cycle channels
  IFS=: read -r dataset pred <<<"$spec"
  read -r data_root data_class data_path freq batch cycle channels <<<"$(dataset_config "$dataset")"
  local base="$OUTPUT_ROOT/cyclenet/$dataset/pred_$pred"
  local out="$base/A0"
  local reference="$base/shared_reference.pt"
  if complete_case "$out"; then echo "reuse $spec"; return 0; fi
  if [[ -e "$out" ]]; then
    mv "$out" "${out}.incomplete.$(date +%Y%m%d_%H%M%S)"
  fi
  mkdir -p "$out"
  echo "start cyclenet $spec gpu=$gpu reference=$reference" | tee -a "$OUTPUT_ROOT/status/started.txt"
  env PYTHON="$PYTHON" GPU="$gpu" BACKBONE=cyclenet SEED=2021 ENV_NUM=2 EXPERIMENTS=A0 \
    DATASET_NAME="$dataset" DATASET_ARCHIVE_SHA256="$ARCHIVE_SHA256" \
    DATA_ROOT="$data_root" DATA_CLASS="$data_class" DATA_PATH="$data_path" \
    TARGET=OT FREQ="$freq" CYCLENET_CYCLE_LEN="$cycle" SEQ_LEN=96 PRED_LEN="$pred" \
    BATCH_SIZE="$batch" NUM_WORKERS=0 EPOCHS=10 WARMUP_EPOCHS=3 \
    D_MODEL=512 D_FF=2048 N_HEADS=2 E_LAYERS=1 DROPOUT=0.1 LR=1e-4 \
    SAVE_FINAL_CHECKPOINT=0 REFERENCE_CHECKPOINT="$reference" OUTPUT="$out" \
    bash scripts/run_predictive_env_iv_patchtst.sh >"$out/run.log" 2>&1 || return 1
  complete_case "$out" || return 1
  touch "$out/run_complete"
  echo "$spec" >> "$OUTPUT_ROOT/status/completed.txt"
}

scheduler="$(mktemp -d /tmp/fpem_cycle_a0_scheduler.XXXXXX)"
trap 'rm -rf "$scheduler"' EXIT
echo 0 > "$scheduler/next"
touch "$scheduler/lock"
worker() {
  local gpu="$1" i spec
  while true; do
    exec 9>"$scheduler/lock"
    flock 9
    i="$(<"$scheduler/next")"
    if (( i >= ${#jobs[@]} )); then flock -u 9; exec 9>&-; break; fi
    spec="${jobs[$i]}"
    echo $((i+1)) > "$scheduler/next"
    flock -u 9
    exec 9>&-
    if ! run_case "$spec" "$gpu"; then
      echo "$spec" >> "$OUTPUT_ROOT/status/failed.txt"
      echo "FAILED $spec" >&2
    fi
  done
}
for gpu in $GPU_LIST; do worker "$gpu" & done
wait
"$PYTHON" - "$OUTPUT_ROOT" <<'PY'
import csv, json, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = []
for path in root.glob('cyclenet/*/pred_*/A0/A0/metrics_and_diagnostics.json'):
    data = json.loads(path.read_text())
    rows.append(dict(backbone='CycleNet', dataset=path.parents[3].name,
                     pred_len=int(path.parents[2].name[5:]),
                     MSE=data['MSE'], MAE=data['MAE'], source=str(path)))
with (root / 'baseline_summary.csv').open('w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['backbone','dataset','pred_len','MSE','MAE','source'])
    writer.writeheader()
    writer.writerows(sorted(rows, key=lambda row: (row['dataset'],row['pred_len'])))
print('CycleNet A0 completed:', len(rows))
PY
