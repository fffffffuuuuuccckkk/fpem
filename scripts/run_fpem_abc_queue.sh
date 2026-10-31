#!/usr/bin/env bash
# Resumable single-GPU queue. Run inside screen; never edits old result trees.
set -u
ROOT=/data/OuXiaoyu/Time-Series-Library-FPEM
PYTHON=/data/OuXiaoyu/miniconda3/envs/basicts/bin/python
RESULT="$ROOT/results/fpem_abc_mechanism_search_20261031"
SOURCE="$ROOT/results/fpem_dual_pattern_memory/g1_r3_full_testlr_search"
GPU="$1"
shift
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=4
mkdir -p "$RESULT/queue_logs"
cd "$ROOT" || exit 1
for item in "$@"; do
  mode=full
  case_path="$item"
  if [[ "$item" == low:* ]]; then
    mode=low
    case_path="${item#low:}"
  fi
  output="$RESULT/$case_path"
  checkpoint="$SOURCE/$case_path"
  mkdir -p "$output"
  if [[ "$mode" == full && -f "$output/complete" ]] || \
     [[ "$mode" == low && -f "$output/B_low_cost.complete" ]]; then
    echo "SKIP $item"
    continue
  fi
  if [[ ! -f "$checkpoint/model.pt" ]]; then
    printf '%s\t%s\t%s\n' "$item" missing_checkpoint "$(date -Is)" >> "$RESULT/failures_raw.tsv"
    continue
  fi
  flags=()
  if [[ "$mode" == low ]]; then flags+=(--low_cost_only --test_stride 4 --repeats 1); fi
  success=0
  for attempt in 1 2 3; do
    echo "START $item attempt=$attempt gpu=$GPU $(date -Is)"
    "$PYTHON" tools/run_fpem_abc_mechanism.py \
      --checkpoint_dir "$checkpoint" --output "$output" --gpu 0 \
      --phase all --repeats 20 "${flags[@]}" \
      > "$RESULT/queue_logs/${case_path//\//_}.log" 2>&1
    status=$?
    if [[ $status -eq 0 ]]; then success=1; break; fi
    printf '%s\t%s\t%s\t%s\n' "$item" "$attempt" "$status" "$(date -Is)" >> "$RESULT/failures_raw.tsv"
    echo "RETRY $item code=$status after 60 seconds"
    sleep 60
  done
  if [[ $success -eq 0 ]]; then
    echo "FAILED $item after 3 attempts; next case continues"
  else
    echo "FINISHED $item $(date -Is)"
  fi
done
"$PYTHON" tools/summarize_fpem_abc_mechanism.py --root "$RESULT" --source "$SOURCE" \
  > "$RESULT/queue_logs/aggregate_gpu${GPU}.log" 2>&1 || true
touch "$RESULT/queue_gpu${GPU}.done"
echo "QUEUE GPU$GPU COMPLETE $(date -Is)"
