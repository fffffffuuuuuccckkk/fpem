#!/usr/bin/env bash
# Restartable three-control EIIL3 search; one control per idle GPU.
set -euo pipefail
host="${1:?server1 or server2}"
case "$host" in server1|server2) ;; *) echo "invalid host: $host" >&2; exit 2 ;; esac
repo=/data/OuXiaoyu/Time-Series-Library-FPEM
python=/data/OuXiaoyu/miniconda3/envs/basicts/bin/python
source_root="$repo/results/fpem_eiil3_fulltrain_testselect_20261113"
root="$repo/results/fpem_eiil3_environment_ablation_20261115"
mkdir -p "$root/plans" "$root/queue_logs"
cd "$repo"
plan="$root/plans/${host}_all.csv"
tmp="$root/plans/${host}_all.tmp"
printf 'case_key,manifest\n' > "$tmp"
for source_plan in "$source_root"/plans/"${host}"_gpu*.csv; do
  [[ -s "$source_plan" ]] || continue
  tail -n +2 "$source_plan" >> "$tmp"
done
{ head -n 1 "$tmp"; tail -n +2 "$tmp" | sort -u; } > "$plan"
rm "$tmp"
expected=32
[[ "$host" == server2 ]] && expected=64
actual=$(tail -n +2 "$plan" | wc -l)
[[ "$actual" -eq "$expected" ]] || { echo "plan count $actual != $expected" >&2; exit 3; }

for mode in shuffled random uniform; do
  while IFS=, read -r case_key manifest; do
    [[ "$case_key" == case_key || -z "$case_key" ]] && continue
    dest="$root/$mode/cases/$case_key"
    source_case="$source_root/cases/$case_key"
    mkdir -p "$dest"
    for name in teacher_q.npz teacher_q.json; do
      if [[ -s "$source_case/$name" && ! -e "$dest/$name" ]]; then
        cp "$source_case/$name" "$dest/$name"
      fi
    done
  done < "$plan"
done

gpu=0
for mode in shuffled random uniform; do
  name="fpem_eiil3_${mode}_${host}"
  if screen -ls | grep -Fq ".${name}"; then
    echo "REUSED_SCREEN $name"
  else
    task="while nvidia-smi -i $gpu --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '^[0-9]+'; do echo WAIT_GPU $gpu; sleep 300; done; PYTHONUNBUFFERED=1 '$python' tools/run_fpem_eiil3_safe_full_queue.py --plan '$plan' --output-root '$root/$mode/cases' --gpu $gpu --full-train --selection-split test --exhaustive-r3 --ready-first --environment-supervision $mode"
    screen -dmS "$name" bash -lc "cd '$repo'; { $task; } >> '$root/queue_logs/${name}.log' 2>&1"
    echo "START_SCREEN $name gpu=$gpu cases=$actual"
  fi
  gpu=$((gpu+1))
done
