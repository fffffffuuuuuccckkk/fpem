#!/usr/bin/env bash
# Launch the host-local calendar-token EIIL3 queues, one screen per GPU.
set -euo pipefail

host="${1:?usage: launch_fpem_token_eiil3_full.sh server1|server2 [GPU ...]}"
shift
case "$host" in
  server1) expected_hostname=gpu-39 ;;
  server2) expected_hostname=gpu-47 ;;
  *) echo "unknown host: $host" >&2; exit 2 ;;
esac
if [[ "$(hostname -s)" != "$expected_hostname" ]]; then
  echo "host mismatch: expected $expected_hostname, got $(hostname -s)" >&2
  exit 2
fi

repo=/data/OuXiaoyu/Time-Series-Library-FPEM
python=/data/OuXiaoyu/miniconda3/envs/basicts/bin/python
output="$repo/results/fpem_periodic_token_eiil3_full_testselect_20261118"
runner="$repo/tools/run_fpem_token_eiil3_full.py"
queue="$repo/tools/run_fpem_token_eiil3_queue.py"
[[ -s "$runner" && -s "$queue" && -x "$python" ]] || {
  echo "runner, queue, or Python runtime missing" >&2
  exit 2
}
mkdir -p "$output/queue_logs"

if (( $# == 0 )); then
  set -- 0 1 2 3
fi
for gpu in "$@"; do
  [[ "$gpu" =~ ^[0-3]$ ]] || { echo "invalid GPU: $gpu" >&2; exit 2; }
  name="fpem_token_eiil3_g${gpu}"
  if screen -ls 2>/dev/null | grep -Eq "[[:space:]][0-9]+\.${name}[[:space:]]"; then
    echo "REUSED_SCREEN $name"
    continue
  fi
  log="$output/queue_logs/${name}.log"
  screen -dmS "$name" bash -lc \
    "cd '$repo'; exec '$python' '$queue' --host '$host' --gpu '$gpu' --repo-root '$repo' --output-root '$output' --runner '$runner' >> '$log' 2>&1"
  echo "START_SCREEN $name"
done
