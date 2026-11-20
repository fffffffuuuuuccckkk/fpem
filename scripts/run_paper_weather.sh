#!/usr/bin/env bash
set -u
ROOT=/data/OuXiaoyu/Time-Series-Library-FPEM
PY=/data/OuXiaoyu/miniconda3/envs/basicts/bin/python
HORIZON="${1:?horizon required}"
GPU="${2:?gpu required}"
OLD="$ROOT/results/fpem_dual_pattern_memory/g1_r3_extended_test_search/patchtst/Weather/pred_$HORIZON"
CASE="$ROOT/results/fpem_paper_completion_20261101/patchtst/Weather/pred_$HORIZON"
LOG="$CASE/queue.log"
mkdir -p "$CASE"
cd "$ROOT" || exit 1
"$PY" tools/fpem_paper_prepare_weather.py --source "$OLD" --destination "$CASE" || exit 1
for phase in reference point flow full_eval; do
  case "$phase" in
    reference) marker="$CASE/reference_ready" ;;
    point) marker="$CASE/search/model.pt" ;;
    flow) marker="$CASE/probability_search_ema_lr/metrics.json" ;;
    full_eval) marker="$CASE/full_test_eval/verified.complete" ;;
  esac
  if [ -s "$marker" ]; then echo "SKIP $phase" | tee -a "$LOG"; continue; fi
  success=0
  for attempt in 1 2 3; do
    echo "RUN $phase Weather/$HORIZON GPU$GPU attempt=$attempt $(date -Is)" | tee -a "$LOG"
    case "$phase" in
      reference)
        "$PY" tools/prepare_fpem_g1r3_extension.py --ensure_case "$CASE" --gpu "$GPU" >> "$LOG" 2>&1 ;;
      point)
        "$PY" tools/run_fpem_dppm_g1_r3_test_search.py \
          --base_run_config "$CASE/base_run_config.json" --output "$CASE/search" \
          --gpu "$GPU" --allow_legacy_reference_signature >> "$LOG" 2>&1 ;;
      flow)
        "$PY" tools/run_fpem_g1r3_probability_search.py --case_dir "$CASE" \
          --gpu "$GPU" --output "$CASE/probability_search_ema_lr" \
          --samples 100 --epochs 10 --train_stride 32 --test_stride 16 >> "$LOG" 2>&1 ;;
      full_eval)
        "$PY" tools/fpem_paper_full_eval.py --case_dir "$CASE" \
          --prob_dir "$CASE/probability_search_ema_lr" \
          --output "$CASE/full_test_eval" --gpu "$GPU" \
          --samples 100 --batch_size 2 --val_stride 4 >> "$LOG" 2>&1 ;;
    esac
    code=$?
    if [ "$phase" = reference ] && [ "$code" -eq 0 ]; then
      printf 'ready\n' > "$marker"
    fi
    if [ "$code" -eq 0 ] && [ -s "$marker" ]; then success=1; break; fi
    echo "RETRY $phase code=$code" | tee -a "$LOG"
    sleep 60
  done
  if [ "$success" -ne 1 ]; then echo "FAILED $phase" | tee -a "$LOG"; exit 1; fi
done
echo "PAPER WEATHER COMPLETE $HORIZON $(date -Is)" | tee -a "$LOG"
