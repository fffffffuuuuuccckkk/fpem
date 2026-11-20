#!/usr/bin/env bash
# Launch one resumable queue per GPU for locally completed, identity-checked cases.
set -euo pipefail
root=/data/OuXiaoyu/Time-Series-Library-FPEM
source_root="$root/results/fpem_eiil3_fulltrain_testselect_20261113/cases"
batch="$root/results/fpem_eiil3_r3_f03_full_dual_20261116"
exclude="${1:-}"
cd "$root"
mkdir -p "$batch/queue_logs"
mapfile -t cases < <(find "$source_root" -name complete -printf '%h\n' | sed "s|^$source_root/||" | sort)
printf 'case_key\tgpu\thost\n' > "$batch/case_plan.tsv"
for gpu in 0 1 2 3; do
  selected=()
  for i in "${!cases[@]}"; do
    case_key="${cases[$i]}"
    [[ "$case_key" == "$exclude" ]] && continue
    if (( i % 4 == gpu )); then
      selected+=("$case_key")
      printf '%s\t%s\t%s\n' "$case_key" "$gpu" "$(hostname)" >> "$batch/case_plan.tsv"
    fi
  done
  ((${#selected[@]})) || continue
  screen_name="fpem_dual_gpu${gpu}"
  if screen -ls 2>/dev/null | grep -q "[.]${screen_name}[[:space:]]"; then
    echo "ALREADY_RUNNING $screen_name"
    continue
  fi
  screen -dmS "$screen_name" bash scripts/run_fpem_eiil3_r3_f03_full_dual_queue.sh "$gpu" "${selected[@]}"
  echo "LAUNCHED $screen_name cases=${#selected[@]}"
done
