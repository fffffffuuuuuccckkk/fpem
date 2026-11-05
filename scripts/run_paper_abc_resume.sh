#!/usr/bin/env bash
set -u
ROOT=/data/OuXiaoyu/Time-Series-Library-FPEM
PY=/data/OuXiaoyu/miniconda3/envs/basicts/bin/python
SOURCE="$ROOT/results/fpem_dual_pattern_memory/g1_r3_full_testlr_search/patchtst/ETTh1/pred_96"
OLD="$ROOT/results/fpem_abc_mechanism_search_20261031/patchtst/ETTh1/pred_96"
OUT="$ROOT/results/fpem_paper_completion_20261101/abc_patchtst_etth1_96"
LOG="$OUT/queue.log"
mkdir -p "$OUT"
cd "$ROOT" || exit 1
if [ ! -f "$OLD/A_all_results.csv" ] || [ ! -f "$OLD/records.pt" ]; then
  echo "Missing prior A and record cache" | tee -a "$LOG"
  exit 1
fi
# The old directory has a misleading overall complete marker. Reuse only its
# verified A result and cache, leaving that directory read-only.
cp -n "$OLD/A_all_results.csv" "$OUT/A_all_results.csv"
cp -n "$OLD/A_ranking.csv" "$OUT/A_ranking.csv"
cp -n "$OLD/records.pt" "$OUT/records.pt"
cp -n "$OLD/A.complete" "$OUT/A.complete"
for phase in B C; do
  if [ -f "$OUT/$phase.complete" ]; then continue; fi
  success=0
  for attempt in 1 2 3; do
    echo "RUN $phase attempt=$attempt $(date -Is)" | tee -a "$LOG"
    "$PY" tools/run_fpem_abc_mechanism.py \
      --checkpoint_dir "$SOURCE" --output "$OUT" --gpu 2 \
      --phase "$phase" --repeats 20 >> "$LOG" 2>&1
    code=$?
    if [ "$code" -eq 0 ] && [ -f "$OUT/$phase.complete" ]; then success=1; break; fi
    echo "RETRY $phase exit=$code" | tee -a "$LOG"
    sleep 60
  done
  if [ "$success" -ne 1 ]; then exit 1; fi
done
if [ -s "$OUT/B_all_results.csv" ] && [ -s "$OUT/C_probability_results.csv" ] && [ -f "$OUT/C12.complete" ]; then
  touch "$OUT/paper_verified.complete"
fi
echo "ABC END $(date -Is)" | tee -a "$LOG"
