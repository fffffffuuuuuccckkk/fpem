#!/usr/bin/env bash
set -uo pipefail
ROOT=/data/OuXiaoyu/Time-Series-Library-FPEM
PY=/data/OuXiaoyu/miniconda3/envs/basicts/bin/python
OUT="$ROOT/results/fpem_dual_pattern_memory/extractor_pilot"
mkdir -p "$OUT"
cd "$ROOT" || exit 1
export PYTHONPATH="$ROOT"
export OMP_NUM_THREADS=4
run_case() {
  local case="$1" gpu="$2" modes="$3" mode attempt status
  for mode in $modes; do
    mkdir -p "$OUT/$case/$mode"
    if [[ -f "$OUT/$case/$mode/extractor_complete" ]]; then
      echo "skip complete $case $mode"
      continue
    fi
    for attempt in 1 2; do
      echo "START case=$case mode=$mode gpu=$gpu attempt=$attempt $(date -Is)"
      CUDA_VISIBLE_DEVICES="$gpu" "$PY" tools/run_fpem_dppm_extractors.py "$case" \
        --mode "$mode" --gpu 0 --controls \
        > "$OUT/$case/$mode/queue.log" 2>&1
      status=$?
      if [[ "$status" -eq 0 ]]; then
        echo "END case=$case mode=$mode status=$status $(date -Is)"
        break
      fi
      echo "FAIL case=$case mode=$mode status=$status $(date -Is)"
      if [[ "$attempt" -eq 1 ]]; then sleep 60; fi
    done
    "$PY" tools/summarize_fpem_dppm_extractors.py || true
  done
}
run_case patchtst_ETTh1_720 0 'P0 P1 P2 P7 P8' > "$OUT/patchtst_queue.log" 2>&1 &
pid0=$!
run_case itransformer_ETTm1_720 1 'P0 P1 P2 P7 P8' > "$OUT/itransformer_queue.log" 2>&1 &
pid1=$!
run_case moderntcn_ETTm2_720 2 'P0 P1 P2 P7 P8 P3 P4 P5 P6' > "$OUT/moderntcn_queue.log" 2>&1 &
pid2=$!
wait "$pid0"; s0=$?
wait "$pid1"; s1=$?
wait "$pid2"; s2=$?
"$PY" tools/summarize_fpem_dppm_extractors.py
echo "queue end $(date -Is) exit=$s0,$s1,$s2"
