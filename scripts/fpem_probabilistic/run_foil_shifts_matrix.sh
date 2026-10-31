#!/usr/bin/env bash
# Resumable 12-GPU matrix: each of three servers owns four deterministic shards.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
DATA_PROJECT="${DATA_PROJECT:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_PROJECT/results/foil_shifts_a0_protocol_20261011}"
SERVER_INDEX="${SERVER_INDEX:-0}"
BACKBONES="${BACKBONES:-patchtst moderntcn}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"
WAIT_SCREENS_PATTERN="${WAIT_SCREENS_PATTERN:-}"
EPOCHS="${EPOCHS:-10}"
ENV_EPOCHS="${ENV_EPOCHS:-2}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-0}"
SEED=2021

if (( SERVER_INDEX < 0 || SERVER_INDEX > 2 )); then
  echo "SERVER_INDEX must be 0, 1 or 2" >&2; exit 2
fi
source "$PROJECT_DIR/scripts/fpem_final/common.sh"
mkdir -p "$OUTPUT_ROOT/queue_status" "$OUTPUT_ROOT/queue_logs"

run_method() {
  local backbone="$1" dataset="$2" horizon="$3" method="$4" gpu="$5"
  local case_root="$OUTPUT_ROOT/$backbone/$dataset/pred_$horizon"
  local out="$case_root/$method" log="$OUTPUT_ROOT/queue_logs/${backbone}_${dataset}_${horizon}_${method}.log"
  if [[ -s "$out/metrics.json" && -f "$out/run_complete" ]]; then
    echo "SKIP complete $backbone $dataset $horizon $method"; return 0
  fi
  # A queued baseline must not collide with an unrelated GPU experiment.
  local used
  while true; do
    used="$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')"
    [[ "$used" =~ ^[0-9]+$ ]] && (( used <= 256 )) && break
    sleep 30
  done
  read -r DATA_ROOT DATA_CLASS DATA_PATH FREQ BATCH_SIZE CYCLE_LEN CHANNELS <<<"$(dataset_config "$dataset")"
  DATA_ROOT="$DATA_PROJECT/${DATA_ROOT#./}"
  local D_MODEL=512 D_FF=2048 DROPOUT=0.1
  local MPATCH=8 MSTRIDE=4 MSTAGES=1 MFFN=1 MHEAD=0
  if [[ "$backbone" == moderntcn ]]; then
    D_MODEL=64
    read -r MPATCH MSTRIDE MSTAGES MFFN DROPOUT MHEAD <<<"$(modern_tcn_config "$dataset")"
    D_FF=$((D_MODEL * MFFN))
  fi
  local LR
  LR="$(backbone_lr "$backbone")"
  mkdir -p "$out"
  if "$PYTHON" -u "$PROJECT_DIR/tools/run_foil_shifts_baseline.py" \
    --method "$method" --backbone "$backbone" --dataset "$dataset" \
    --pred-len "$horizon" --data-root "$DATA_ROOT" \
    --data-class "$DATA_CLASS" --data-path "$DATA_PATH" --freq "$FREQ" \
    --d-model "$D_MODEL" --d-ff "$D_FF" --dropout "$DROPOUT" \
    --modern-patch-size "$MPATCH" --modern-patch-stride "$MSTRIDE" \
    --modern-stages "$MSTAGES" --modern-ffn-ratio "$MFFN" \
    --modern-head-dropout "$MHEAD" --seq-len 96 \
    --epochs "$EPOCHS" --env-epochs "$ENV_EPOCHS" --warmup-epochs 3 \
    --env-num 3 --batch-size "$BATCH_SIZE" --lr "$LR" --seed "$SEED" \
    --gpu "$gpu" --num-workers 0 \
    --max-train-batches "$MAX_TRAIN_BATCHES" --max-eval-batches "$MAX_EVAL_BATCHES" \
    --reference-root "$DATA_PROJECT/results" --output "$out" \
    > "$log" 2>&1; then
    echo "$backbone,$dataset,$horizon,$method,$gpu" >> "$OUTPUT_ROOT/queue_status/completed_server${SERVER_INDEX}.csv"
    echo "DONE $backbone $dataset $horizon $method GPU=$gpu"
  else
    echo "$backbone,$dataset,$horizon,$method,$gpu" >> "$OUTPUT_ROOT/queue_status/failed_server${SERVER_INDEX}.csv"
    echo "FAILED $backbone $dataset $horizon $method GPU=$gpu (see $log)" >&2
  fi
}

worker() {
  local gpu="$1" index=0 slot owner target
  for backbone in $BACKBONES; do
    for dataset in ETTh1 ETTh2 ETTm1 ETTm2 ExchangeRate; do
      for horizon in 24 48 96 192 336 720; do
        slot=$((index % 12)); owner=$((slot / 4)); target=$((slot % 4))
        if (( owner == SERVER_INDEX && target == gpu )); then
          for method in foil shifts; do
            run_method "$backbone" "$dataset" "$horizon" "$method" "$gpu"
          done
        fi
        index=$((index + 1))
      done
    done
  done
}

cd "$PROJECT_DIR"
if [[ -n "$WAIT_SCREENS_PATTERN" ]]; then
  echo "Waiting for existing screens matching $WAIT_SCREENS_PATTERN"
  while screen -ls | grep -Eq "$WAIT_SCREENS_PATTERN"; do
    sleep 30
  done
fi
for gpu in $GPU_LIST; do
  worker "$gpu" > "$OUTPUT_ROOT/queue_logs/server${SERVER_INDEX}_gpu${gpu}.log" 2>&1 &
done
wait
echo "server $SERVER_INDEX queue finished"
