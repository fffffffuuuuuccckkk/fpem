#!/usr/bin/env bash
# Launch only locally available exact-A0 cases.  Server 3's /data is full,
# so the current full matrix uses servers 1 and 2 without risking old results.
set -euo pipefail
root=/data/OuXiaoyu/Time-Series-Library-FPEM
source_root="$root/results/fpem_eiil3_fulltrain_testselect_20261113/cases"
result_root="$root/results/fpem_eiil3_exog_full_20261117"
host=$(hostname)
cd "$root"
mkdir -p "$result_root/queue_logs"
mapfile -t cases < <(find "$source_root" -name complete -printf '%h\n' |
  sed "s|^$source_root/||" | sort)
gpu_count="${FPEM_EXOG_GPU_COUNT:-3}"
if [[ "$gpu_count" != 3 && "$gpu_count" != 4 ]]; then
  echo "FPEM_EXOG_GPU_COUNT must be 3 or 4" >&2
  exit 2
fi
# Main host's GPU3 is owned by the pre-existing dual queue.  It is not used.
printf 'case_key\thost\tgpu\tq_modes\n' > "$result_root/case_plan.tsv"
for gpu in $(seq 0 $((gpu_count-1))); do
  selected=()
  for i in "${!cases[@]}"; do
    if (( i % gpu_count == gpu )); then
      case_key="${cases[$i]}"
      selected+=("$case_key")
      printf '%s\t%s\t%s\trediscover,fixed_original\n' "$case_key" "$host" "$gpu" >> "$result_root/case_plan.tsv"
    fi
  done
  # Server 2's existing A0 queue may add four ModernTCN/Traffic sources later.
  # This GPU3 queue will wait for that queue, then pick up the new cases.
  if (( gpu == 3 )); then
    for horizon in 96 192 336 720; do
      case_key="moderntcn/Traffic/pred_${horizon}/seed_2021"
      selected+=("$case_key")
      printf '%s\t%s\t%s\trediscover,fixed_original\n' "$case_key" "$host" "$gpu" >> "$result_root/case_plan.tsv"
    done
  fi
  ((${#selected[@]})) || continue
  name="fpem_exog_gpu${gpu}"
  if screen -ls 2>/dev/null | grep -q "[.]${name}[[:space:]]"; then
    echo "ALREADY_RUNNING $name"
    continue
  fi
  screen -dmS "$name" bash scripts/run_fpem_eiil3_exog_queue.sh "$gpu" "${selected[@]}"
  echo "LAUNCHED $name cases=${#selected[@]} host=$host"
done
