#!/usr/bin/env bash
# Launch one durable screen per free GPU; A0 creation precedes dependent trials.
set -euo pipefail
host="${1:?server1 or server2 required}"
case "$host" in server1|server2) ;; *) echo "invalid host $host" >&2; exit 2 ;; esac
root=/data/OuXiaoyu/Time-Series-Library-FPEM
python=/data/OuXiaoyu/miniconda3/envs/basicts/bin/python
batch="$root/results/fpem_eiil3_safe_full_20261112"
mkdir -p "$batch/plans" "$batch/queue_logs" "$batch/cases"
cd "$root"
for gpu in 0 1 2 3; do
  name="fpem_safe_${host}_gpu${gpu}"
  if screen -ls | grep -Fq ".${name}"; then
    echo "REUSED_SCREEN $name"
    continue
  fi
  plan="$batch/plans/${host}_gpu${gpu}.csv"
  a0plan="$batch/plans/${host}_a0_gpu${gpu}.csv"
  [[ -s "$plan" ]] || { echo "MISSING_PLAN $plan" >&2; exit 2; }
  if [[ -s "$a0plan" ]]; then
    task="bash scripts/run_eiil3_missing_a0_matrix.sh $gpu '$a0plan'; PYTHONUNBUFFERED=1 '$python' tools/run_fpem_eiil3_safe_full_queue.py --plan '$plan' --output-root '$batch/cases' --gpu $gpu"
  else
    task="PYTHONUNBUFFERED=1 '$python' tools/run_fpem_eiil3_safe_full_queue.py --plan '$plan' --output-root '$batch/cases' --gpu $gpu"
  fi
  screen -dmS "$name" bash -lc "cd '$root'; $task >> '$batch/queue_logs/${name}.log' 2>&1"
  echo "START_SCREEN $name"
done
