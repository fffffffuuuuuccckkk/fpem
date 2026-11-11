#!/usr/bin/env bash
set -euo pipefail

# One host-local invocation launches three isolated, resumable GPU queues.
# Host 3 is not mutated while its four GPUs run unrelated jobs and its checkout
# lacks the required current F03/Flow dependencies. Re-plan it after capacity
# and code/data identity have been audited.
ROOT=/data/OuXiaoyu/Time-Series-Library-FPEM
PY=/data/OuXiaoyu/miniconda3/envs/basicts/bin/python
MODE=${1:-all}
cd "$ROOT"
mkdir -p results/fpem_eiil3_20261110/queue_logs

if [[ "$MODE" == audit-only ]]; then
  exec "$PY" tools/audit_fpem_eiil3_assets.py
fi
if [[ "$MODE" == aggregate ]]; then
  exec "$PY" tools/aggregate_fpem_eiil3.py
fi

case "$(hostname)" in
  gpu-39)
    QUEUES=("patchtst short ETTh1 0" "patchtst long ETTh1 1" "moderntcn short ETTm2 2")
    ;;
  gpu-47)
    QUEUES=("itransformer short ETTm1 0" "itransformer long ETTm1 1" "moderntcn long ETTm2 2")
    ;;
  gpu-41)
    echo 'SERVER3_PENDING: all GPUs occupied and current checkout lacks required modules; preserve running jobs.'
    exit 0
    ;;
  *)
    echo "Unsupported host $(hostname); no queue launched" >&2
    exit 2
    ;;
esac

if [[ "$MODE" == flow-only ]]; then
  for case_dir in results/fpem_eiil3_20261110/*/*/pred_*/seed_*; do
    [[ -f "$case_dir/point_complete" ]] || continue
    "$PY" tools/run_fpem_eiil3_flow.py --case-dir "$case_dir" --resume
  done
  exit 0
fi

for spec in "${QUEUES[@]}"; do
  read -r backbone group dataset gpu <<< "$spec"
  name="fpem_eiil3_${backbone}_${group}"
  args=(--backbone "$backbone" --horizon-group "$group" --pilot-dataset "$dataset"
        --gpu "$gpu" --generate-missing-q --flow)
  if [[ "$MODE" == dry-run ]]; then
    "$PY" tools/run_fpem_eiil3_queue.py "${args[@]}" --dry-run
    continue
  fi
  if [[ "$MODE" == pilot ]]; then args+=(--pilot-only); fi
  if [[ "$MODE" == full ]]; then args+=(--full-only); fi
  if [[ "$MODE" != all && "$MODE" != resume && "$MODE" != pilot && "$MODE" != full ]]; then
    echo "Unknown mode: $MODE" >&2; exit 2
  fi
  if screen -ls 2>/dev/null | grep -q "[.]${name}[[:space:]]"; then
    echo "EXISTS $name"
    continue
  fi
  screen -dmS "$name" env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
    "$PY" tools/run_fpem_eiil3_queue.py "${args[@]}"
  echo "START $name GPU=$gpu pilot=$dataset-$group"
done
