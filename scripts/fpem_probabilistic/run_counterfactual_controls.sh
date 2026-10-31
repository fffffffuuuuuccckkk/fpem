#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
DATASET="${DATASET:?set DATASET}"
PRED_LEN="${PRED_LEN:?set PRED_LEN}"
GPU="${GPU:-0}"
SPLIT="${SPLIT:-val}"
CHECKPOINT="${CHECKPOINT:-results/fpem_probabilistic/pilot/patchtst/$DATASET/pred_$PRED_LEN/full/A2/trained_checkpoint.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/fpem_probabilistic/controls/$DATASET/pred_$PRED_LEN}"
cd "$PROJECT_DIR"
[[ -s "$CHECKPOINT" ]] || { echo "missing checkpoint: $CHECKPOINT" >&2; exit 2; }
for ablation in deterministic_affine_center affine_flow shuffled_zvar unconditional_flow; do
  destination="$OUTPUT_ROOT/$ablation"
  mkdir -p "$destination"
  echo "starting $ablation $SPLIT"
  "$PYTHON" tools/evaluate_fpem_probabilistic.py \
    --checkpoint "$CHECKPOINT" --ablation "$ablation" \
    --split "$SPLIT" --gpu "$GPU" --num_samples "${PROB_NUM_SAMPLES:-100}" \
    --max_batches "${MAX_PROB_EVAL_BATCHES:-0}" --output "$destination" \
    >"$destination/evaluate.log" 2>&1
  echo "complete $ablation $SPLIT"
done
