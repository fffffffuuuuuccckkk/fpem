#!/usr/bin/env bash
# Durable full-TRAIN EIIL3 queues. Never share a GPU with an existing process.
set -euo pipefail
host="${1:?server1 or server2}"
shift
case "$host" in server1|server2) ;; *) echo "invalid host" >&2; exit 2 ;; esac
root=/data/OuXiaoyu/Time-Series-Library-FPEM
python=/data/OuXiaoyu/miniconda3/envs/basicts/bin/python
batch="$root/results/fpem_eiil3_fulltrain_20261113"
mkdir -p "$batch/queue_logs" "$batch/cases"
cd "$root"
for gpu in "$@"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "invalid GPU $gpu" >&2; exit 2; }
  plan="$batch/plans/${host}_gpu${gpu}.csv"
  [[ -s "$plan" ]] || { echo "missing $plan" >&2; exit 2; }
  name="fpem_fulltrain_${host}_gpu${gpu}"
  if screen -ls | grep -Fq ".${name}"; then
    echo "REUSED_SCREEN $name"
    continue
  fi
  task="while nvidia-smi -i $gpu --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '^[0-9]+'; do echo WAIT_GPU $gpu; sleep 300; done; PYTHONUNBUFFERED=1 '$python' tools/run_fpem_eiil3_safe_full_queue.py --plan '$plan' --output-root '$batch/cases' --gpu $gpu --full-train"
  screen -dmS "$name" bash -lc "cd '$root'; { $task; } >> '$batch/queue_logs/${name}.log' 2>&1"
  echo "START_SCREEN $name"
done
