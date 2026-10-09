#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
DATASET="${DATASET:?set DATASET}"
PRED_LEN="${PRED_LEN:?set PRED_LEN}"
GPU="${GPU:-0}"
source "$PROJECT_DIR/scripts/fpem_final/common.sh"
read -r _DATA_ROOT _DATA_CLASS _DATA_PATH _FREQ FPEM_BATCH_SIZE _CYCLE _CHANNELS \
  <<<"$(dataset_config "$DATASET")"
MICRO_BATCH="${CSDI_MICRO_BATCH:-4}"
if (( FPEM_BATCH_SIZE % MICRO_BATCH != 0 )); then
  echo "CSDI microbatch must divide FPEM batch size" >&2
  exit 2
fi
ACCUMULATION=$((FPEM_BATCH_SIZE / MICRO_BATCH))
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/results/fpem_probabilistic/csdi_probts_aligned}"
OUTPUT="$OUTPUT_ROOT/$DATASET/pred_$PRED_LEN"
if [[ -e "$OUTPUT" ]]; then
  echo "Existing aligned CSDI result; inspect before rerun: $OUTPUT" >&2
  exit 3
fi
mkdir -p "$OUTPUT_ROOT/$DATASET"
cd "$PROJECT_DIR"

# FPEM shared-reference warmup (3) plus its 10 joint epochs give 13 TRAIN
# passes. The independent CSDI baseline receives the same exposure budget,
# effective batch size, seed, split and 100-sample full-TEST evaluation.
# CSDI's architecture and diffusion schedule remain the ProbTS configuration.
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u tools/run_csdi_probts.py \
  --dataset "$DATASET" --seq_len 96 --pred_len "$PRED_LEN" \
  --data_root "$PROJECT_DIR/dataset/all_datasets" \
  --output "$OUTPUT" --gpu 0 --seed 2021 \
  --epochs "${CSDI_EPOCHS:-13}" \
  --batch_size "$MICRO_BATCH" --accumulate_grad_batches "$ACCUMULATION" \
  --max_train_batches 0 --eval_every 0 \
  --max_test_windows 0 --num_samples 100 --sample_chunk 2 \
  --num_steps 50 --hidden_channels 64 --num_layers 4 --num_heads 8 \
  --lr "${CSDI_LR:-1e-4}" --beta_start 0.001 --beta_end 0.5
