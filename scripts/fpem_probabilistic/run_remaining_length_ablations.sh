#!/usr/bin/env bash
# Fixed-configuration long-horizon controls, with one job per free GPU.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/results/fpem_decomposition_controls_20261012}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"
cd "$PROJECT_DIR"

case_files() {
  find results/fpem_no_future_anchor_patchtst_search \
       results/fpem_no_future_anchor_patchtst_search_h48_h192 \
       results/fpem_no_future_anchor_moderntcn_search \
       -name best_config.json -type f 2>/dev/null \
    | grep -E '/pred_(96|192|336|720)/best_config.json$' | sort -u
}

run_case() {
  local best="$1" gpu="$2" arm
  for arm in no_domain environment_shuffled; do
    "$PYTHON" tools/run_fixed_config_ablation.py \
      --best-config "$best" --arm "$arm" --gpu "$gpu" \
      --output-root "$OUTPUT_ROOT"
  done
}

queue() {
  mkdir -p "$OUTPUT_ROOT/queue_logs" "$OUTPUT_ROOT/queue_status"
  local scheduler
  scheduler="$(mktemp -d "$OUTPUT_ROOT/.queue.XXXXXX")"
  printf '0\n' > "$scheduler/next"
  touch "$scheduler/lock"
  mapfile -t cases < <(case_files)
  echo "QUEUE host=$HOSTNAME cases=${#cases[@]} GPUs=$GPU_LIST root=$OUTPUT_ROOT"
  worker() {
    local gpu="$1" used index best label
    while true; do
      used="$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')"
      if [[ ! "$used" =~ ^[0-9]+$ ]] || (( used > 256 )); then sleep 30; continue; fi
      exec 9>"$scheduler/lock"; flock 9
      index="$(<"$scheduler/next")"
      if (( index >= ${#cases[@]} )); then flock -u 9; exec 9>&-; break; fi
      best="${cases[$index]}"
      printf '%s\n' "$((index+1))" > "$scheduler/next"
      flock -u 9; exec 9>&-
      label="$(echo "$best" | sed -E 's#^.*/(patchtst|moderntcn)/([^/]+)/pred_([0-9]+)/best_config.json#\1_\2_\3#')"
      if run_case "$best" "$gpu" >"$OUTPUT_ROOT/queue_logs/$label.log" 2>&1; then
        printf '%s,%s\n' "$best" "$gpu" >>"$OUTPUT_ROOT/queue_status/completed.txt"
        echo "DONE $label GPU=$gpu"
      else
        printf '%s,%s\n' "$best" "$gpu" >>"$OUTPUT_ROOT/queue_status/failed.txt"
        echo "FAILED $label GPU=$gpu" >&2
      fi
    done
  }
  local pids=() gpu status=0
  for gpu in $GPU_LIST; do worker "$gpu" & pids+=("$!"); done
  for gpu in "${pids[@]}"; do wait "$gpu" || status=1; done
  return "$status"
}

case "${1:-}" in
  case) run_case "${2:?best-config}" "${3:?gpu}" ;;
  plan) case_files ;;
  queue) queue ;;
  *) echo "usage: $0 plan | queue | case BEST_CONFIG GPU" >&2; exit 2 ;;
esac
