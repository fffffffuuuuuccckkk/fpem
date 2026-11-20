#!/usr/bin/env bash
set -u

ROOT=/data/OuXiaoyu/Time-Series-Library-FPEM
PY=/data/OuXiaoyu/miniconda3/envs/basicts/bin/python
CASE=${1:?case required}
GPU=${2:?gpu required}
OUT="$ROOT/results/fpem_dual_pattern_memory/validation_pilot/$CASE/probability"
mkdir -p "$OUT"
cd "$ROOT" || exit 1

attempt=0
while [ ! -f "$OUT/E3_complete" ]; do
    attempt=$((attempt+1))
    echo "E3 $CASE attempt=$attempt gpu=$GPU"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" tools/run_fpem_dppm_probability.py "$CASE" --gpu 0 \
        --epochs 10 --samples 100 --flow_steps 12 \
        >> "$OUT/run.log" 2>&1
    code=$?
    if [ "$code" -eq 0 ] && [ -f "$OUT/E3_complete" ]; then
        echo "E3 $CASE complete"
        exit 0
    fi
    echo "E3 $CASE failed code=$code; waiting 60 seconds before safe resume"
    if [ "$attempt" -ge 5 ]; then
        echo "E3 $CASE stopped after 5 attempts; inspect $OUT/run.log"
        exit 1
    fi
    sleep 60
done
