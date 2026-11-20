#!/usr/bin/env bash
set -u

# Usage: bash scripts/run_paper_final_calibration_queue.sh GPU PY DATASET:SOURCE:24,48 [...]
# SOURCE is "existing" for g1_r3_extended_test_search, "paper" for Weather,
# or "pilot" for new ETT cases. Results are always isolated in final evidence.
gpu=${1:?GPU required}; shift
python_bin=${1:?Python required}; shift
root=/data/OuXiaoyu/Time-Series-Library-FPEM
final="$root/results/fpem_paper_final_evidence_20261101"
cd "$root" || exit 2
if [[ -n "${WAIT_FOR_SCREEN:-}" ]]; then
  while screen -ls 2>/dev/null | grep -q "$WAIT_FOR_SCREEN"; do
    echo "WAIT GPU owner $WAIT_FOR_SCREEN $(date -u +%FT%TZ)"
    sleep 60
  done
fi
for entry in "$@"; do
  IFS=: read -r dataset source horizons <<< "$entry"
  IFS=, read -ra pred_lengths <<< "$horizons"
  for horizon in "${pred_lengths[@]}"; do
    case "$source" in
      existing) case_dir="$root/results/fpem_dual_pattern_memory/g1_r3_extended_test_search/patchtst/$dataset/pred_$horizon" ;;
      paper) case_dir="$root/results/fpem_paper_completion_20261101/patchtst/$dataset/pred_$horizon" ;;
      pilot) case_dir="$final/probability_pilot/patchtst/$dataset/pred_$horizon" ;;
      *) echo "Invalid source: $source" >&2; exit 2 ;;
    esac
    prob_dir="$case_dir/probability_search_ema_lr"
    output="$final/calibration/patchtst/$dataset/pred_$horizon"
    mkdir -p "$output"
    if [[ -f "$output/verified.complete" ]]; then
      echo "SKIP $dataset $horizon"
      continue
    fi
    # New pilots can still be running. Never calibrate a partial Flow.
    while [[ ! -f "$prob_dir/complete" ]]; do
      echo "WAIT Flow $dataset $horizon $(date -u +%FT%TZ)"
      sleep 60
    done
    attempt=0
    until [[ -f "$output/verified.complete" ]]; do
      attempt=$((attempt+1))
      echo "CALIBRATE $dataset $horizon attempt=$attempt $(date -u +%FT%TZ)"
      CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" tools/fpem_paper_final_calibration.py \
        --case_dir "$case_dir" --prob_dir "$prob_dir" --output "$output" \
        --gpu 0 --batch_size 2 --samples 100 --flow_steps 12 \
        >> "$output/run.log" 2>&1 && break
      if [[ "$attempt" -ge 3 ]]; then
        echo "FAILED $dataset $horizon after $attempt attempts" >&2
        exit 1
      fi
      sleep 60
    done
  done
done
echo "CALIBRATION QUEUE COMPLETE $(date -u +%FT%TZ)"
