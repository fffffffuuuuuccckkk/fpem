#!/usr/bin/env bash
set -euo pipefail

repo="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
python="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
root="${OUTPUT_ROOT:-$repo/results/fpem_periodic_token_a0_global_20261118}"
gpu_list="${GPU_LIST:-0 1 2 3}"
shard_count="${SHARD_COUNT:-2}"
shard_index="${SHARD_INDEX:?SHARD_INDEX is required}"

cd "$repo"
mkdir -p "$root/logs" "$root/status"
"$python" tools/prepare_fpem_periodic_token_a0_full.py --root "$root"
mapfile -t all_cases < "$root/cases.txt"
cases=()
for index in "${!all_cases[@]}"; do
  if (( index % shard_count == shard_index )); then
    cases+=("${all_cases[$index]}")
  fi
done
printf 'host=%s shard=%s/%s cases=%s gpus=%s\n' \
  "$(hostname)" "$shard_index" "$shard_count" "${#cases[@]}" "$gpu_list"

run_case() {
  local case_name="$1" gpu="$2"
  local out="$root/pilot_$case_name"
  if [[ -f "$out/complete" ]]; then
    echo "reuse $case_name"
    return 0
  fi
  if [[ -e "$out" ]]; then
    mv "$out" "${out}.incomplete.$(date +%Y%m%d_%H%M%S)"
  fi
  echo "start case=$case_name gpu=$gpu time=$(date -Is)"
  "$python" tools/run_fpem_periodic_token_a0.py \
    --repo-root "$repo" \
    --token-stage-root "$repo" \
    --base-run-config "$root/configs/$case_name.json" \
    --skip-old-a0 \
    --periodic-mode both \
    --output-dir "$out" \
    --gpu "$gpu" \
    > "$root/logs/$case_name.log" 2>&1
  echo "complete case=$case_name gpu=$gpu time=$(date -Is)"
}

scheduler="$(mktemp -d "$root/.scheduler.XXXXXX")"
trap 'rm -rf "$scheduler"' EXIT
echo 0 > "$scheduler/next"
touch "$scheduler/lock"
worker() {
  local gpu="$1" index case_name
  while true; do
    exec 9>"$scheduler/lock"
    flock 9
    index="$(<"$scheduler/next")"
    if (( index >= ${#cases[@]} )); then
      flock -u 9
      exec 9>&-
      break
    fi
    echo $((index + 1)) > "$scheduler/next"
    flock -u 9
    exec 9>&-
    case_name="${cases[$index]}"
    if run_case "$case_name" "$gpu"; then
      echo "$case_name" >> "$root/status/completed_shard_${shard_index}.txt"
    else
      echo "$case_name" >> "$root/status/failed_shard_${shard_index}.txt"
    fi
  done
}

pids=()
for gpu in $gpu_list; do
  worker "$gpu" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid" || true
done
"$python" tools/aggregate_fpem_periodic_token_a0.py --root "$root"
echo "queue complete host=$(hostname) shard=$shard_index time=$(date -Is)"
