#!/usr/bin/env bash
# Retry only the three memory-failed, already-audited cases, one at a time.
set -euo pipefail
root=/data/OuXiaoyu/Time-Series-Library-FPEM
batch="$root/results/fpem_eiil3_fulltrain_testselect_20261113"
python=/data/OuXiaoyu/miniconda3/envs/basicts/bin/python
name=fpem_testselect_memory_retry_s1
plan="$batch/plans/server1_retry_memory.csv"
[[ -s "$plan" ]] || { echo "missing $plan" >&2; exit 2; }
cd "$root"
if screen -ls | grep -Fq ".$name"; then
  echo "REUSED_SCREEN $name"
  exit 0
fi
task="while nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '^[0-9]+'; do sleep 300; done; MALLOC_ARENA_MAX=2 PYTHONUNBUFFERED=1 '$python' tools/run_fpem_eiil3_safe_full_queue.py --plan '$plan' --output-root '$batch/cases' --gpu 0 --full-train --selection-split test --exhaustive-r3 --a0-wait-hours 0"
screen -dmS "$name" bash -lc "cd '$root'; { $task; } >> '$batch/queue_logs/$name.log' 2>&1"
echo "START_SCREEN $name"
