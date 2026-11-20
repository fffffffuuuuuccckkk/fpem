#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
MANIFEST="${MANIFEST:?set MANIFEST to CSV backbone,dataset,pred_len}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/fpem_final_patchtst_search}"
DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-1}"
SEARCH_PROFILE="${SEARCH_PROFILE:-compact}"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/status"
ARCHIVE_SHA256="${ARCHIVE_SHA256:-$(sha256sum "${ARCHIVE:-dataset/all_datasets.zip}" | awk '{print $1}')}"
export ARCHIVE_SHA256
mapfile -t CASES < <(awk -F, 'NF>=3 && $1 !~ /^#/ {print $1":"$2":"$3}' "$MANIFEST")

if [[ "$DRY_RUN" == 1 ]]; then
  i=0
  for spec in "${CASES[@]}"; do
    IFS=: read -r backbone dataset pred_len <<<"$spec"
    gpu_array=($GPU_LIST); gpu="${gpu_array[$((i % ${#gpu_array[@]}))]}"; i=$((i+1))
    BACKBONE="$backbone" DATASET="$dataset" PRED_LEN="$pred_len" GPU="$gpu" \
      OUTPUT_ROOT="$OUTPUT_ROOT" DRY_RUN=1 RESUME="$RESUME" SEARCH_PROFILE="$SEARCH_PROFILE" \
      bash scripts/fpem_final/run_fpem_search_case.sh
  done
  exit 0
fi

SCHEDULER_DIR="$(mktemp -d "$OUTPUT_ROOT/.scheduler.XXXXXX")"
printf '0\n' > "$SCHEDULER_DIR/next"; touch "$SCHEDULER_DIR/lock"
trap 'rm -rf "$SCHEDULER_DIR"' EXIT

worker() {
  local gpu="$1" idx spec backbone dataset pred_len
  while true; do
    exec 9>"$SCHEDULER_DIR/lock"; flock 9
    idx="$(<"$SCHEDULER_DIR/next")"
    if (( idx >= ${#CASES[@]} )); then flock -u 9; exec 9>&-; break; fi
    spec="${CASES[$idx]}"; printf '%s\n' "$((idx+1))" > "$SCHEDULER_DIR/next"
    flock -u 9; exec 9>&-
    IFS=: read -r backbone dataset pred_len <<<"$spec"
    if BACKBONE="$backbone" DATASET="$dataset" PRED_LEN="$pred_len" GPU="$gpu" \
      OUTPUT_ROOT="$OUTPUT_ROOT" RESUME="$RESUME" SEARCH_PROFILE="$SEARCH_PROFILE" \
      bash scripts/fpem_final/run_fpem_search_case.sh \
      > >(tee -a "$OUTPUT_ROOT/logs/${backbone}_${dataset}_${pred_len}.log") 2>&1; then
      echo "$spec" >> "$OUTPUT_ROOT/status/completed.txt"
    else
      echo "$spec" >> "$OUTPUT_ROOT/status/failed.txt"
    fi
  done
}

echo "queue=${#CASES[@]} GPUs=$GPU_LIST output=$OUTPUT_ROOT"
pids=(); for gpu in $GPU_LIST; do worker "$gpu" & pids+=("$!"); done
for pid in "${pids[@]}"; do wait "$pid" || true; done
"$PYTHON" tools/summarize_fpem_final_matrix.py --root "$OUTPUT_ROOT" \
  --csv "$OUTPUT_ROOT/fpem_best_configs.csv" --txt "$OUTPUT_ROOT/fpem_best_configs.txt"
