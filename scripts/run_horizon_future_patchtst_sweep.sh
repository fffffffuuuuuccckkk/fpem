#!/usr/bin/env bash
set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
PRED_LEN="${PRED_LEN:?set PRED_LEN to 96, 336, or 720}"
ENV_NUM="${ENV_NUM:-3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/predictive_env_future_patch_gradrel_patchtst_${PRED_LEN}_k${ENV_NUM}}"
REFERENCE_ROOT="${REFERENCE_ROOT:-results/predictive_env_patchtst_multihorizon_96}"
ARCHIVE="${ARCHIVE:-dataset/all_datasets.zip}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-60}"

case "$PRED_LEN" in 96|336|720) ;; *) echo "invalid PRED_LEN=$PRED_LEN" >&2; exit 2 ;; esac
cd "$PROJECT_DIR" || exit 1
mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/status"
ARCHIVE_SHA256="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
DATASETS=(Traffic Electricity Weather ETTm1 ETTm2 ETTh1 ETTh2 ExchangeRate)
MODES=(horizon_future_var inv_only film y_film_decay)
printf '%s\n' \
  "backbone=patchtst" "seq_len=96" "pred_len=$PRED_LEN" \
  "datasets=${DATASETS[*]}" "modes=${MODES[*]}" "seed=2021" \
  "environment_count=$ENV_NUM" "representation_constraint=classification" \
  "decomposition_type=complementary_gate" "lambda_var_gain=0.1" \
  "lambda_var_conditional_gain=0.2" "lambda_future_h=0.1" \
  "lambda_horizon_reliability=0.0" \
  "variant_training_schedule=gradient_relative_invariant_anchored" \
  "horizon_reliability_temperature=0.05" \
  "future_patch_len=16" "future_teacher_patches_per_batch=2" \
  "future_teacher_eval_patch_count=3" "archive_sha256=$ARCHIVE_SHA256" \
  > "$OUTPUT_ROOT/protocol.txt"

dataset_config() {
  case "$1" in
    ETTh1)        echo "./dataset/all_datasets/ETT-small ett_hour ETTh1.csv h 32 24 32" ;;
    ETTh2)        echo "./dataset/all_datasets/ETT-small ett_hour ETTh2.csv h 32 24 32" ;;
    ETTm1)        echo "./dataset/all_datasets/ETT-small ett_minute ETTm1.csv 15min 32 96 32" ;;
    ETTm2)        echo "./dataset/all_datasets/ETT-small ett_minute ETTm2.csv 15min 32 96 32" ;;
    Electricity)  echo "./dataset/all_datasets/electricity custom electricity.csv h 4 168 8" ;;
    ExchangeRate) echo "./dataset/all_datasets/exchange_rate custom exchange_rate.csv d 32 7 32" ;;
    Weather)      echo "./dataset/all_datasets/weather custom weather.csv 10min 16 144 16" ;;
    Traffic)      echo "./dataset/all_datasets/traffic custom traffic.csv h 2 168 8" ;;
  esac
}

reference_path() {
  local dataset="$1"
  if [[ "$PRED_LEN" == 96 ]]; then
    echo "results/predictive_env_featurewise_gate_all_backbones_k3/patchtst/$dataset/shared_reference.pt"
  elif [[ -s "$REFERENCE_ROOT/pred_$PRED_LEN/$dataset/shared_reference.pt" ]]; then
    echo "$REFERENCE_ROOT/pred_$PRED_LEN/$dataset/shared_reference.pt"
  elif [[ -s "results/predictive_env_patchtst_film_featurewise_multihorizon_k3/pred_$PRED_LEN/$dataset/shared_reference.pt" ]]; then
    echo "results/predictive_env_patchtst_film_featurewise_multihorizon_k3/pred_$PRED_LEN/$dataset/shared_reference.pt"
  else
    echo "$OUTPUT_ROOT/$dataset/shared_reference.pt"
  fi
}

wait_for_gpu() {
  local gpu="$1" memory utilization
  while true; do
    IFS=, read -r memory utilization < <(
      nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null | tr -d ' '
    )
    memory="${memory:-999999}"; utilization="${utilization:-100}"
    if (( memory < 1000 && utilization < 10 )); then return 0; fi
    echo "[gpu $gpu] waiting memory=${memory}MiB utilization=${utilization}%"
    sleep "$GPU_WAIT_SECONDS"
  done
}

prepare_reference() {
  local dataset="$1" gpu="$2" reference="$3"
  [[ -s "$reference" ]] && return 0
  local root klass path freq batch cycle chunk
  read -r root klass path freq batch cycle chunk <<<"$(dataset_config "$dataset")"
  mkdir -p "$(dirname "$reference")"
  echo "[$dataset/$PRED_LEN] preparing shared reference"
  GPU="$gpu" BACKBONE=patchtst SEED=2021 ENV_NUM="$ENV_NUM" EXPERIMENTS=A2 \
  DATASET_NAME="$dataset" DATASET_ARCHIVE_SHA256="$ARCHIVE_SHA256" \
  DATA_ROOT="$root" DATA_CLASS="$klass" DATA_PATH="$path" TARGET=OT \
  FREQ="$freq" CYCLENET_CYCLE_LEN="$cycle" SEQ_LEN=96 PRED_LEN="$PRED_LEN" \
  BATCH_SIZE="$batch" NUM_WORKERS=0 EPOCHS=10 WARMUP_EPOCHS=3 STAGE_EPOCHS=2 \
  PREPARE_REFERENCE_ONLY=1 REFERENCE_CHECKPOINT="$reference" \
  OUTPUT="$OUTPUT_ROOT/$dataset/reference_prepare" \
  bash scripts/run_predictive_env_iv_patchtst.sh
}

result_valid() {
  local metrics="$1" mode="$2"
  [[ -s "$metrics" ]] || return 1
  "$PYTHON" - "$metrics" "$mode" "$ENV_NUM" <<'PY'
import json, sys
from pathlib import Path
row = json.loads(Path(sys.argv[1]).read_text())
expected = {
    "backbone": "patchtst", "experiment": "A2",
    "representation_constraint": "classification",
    "decomposition_type": "complementary_gate",
    "variant_fusion_mode": sys.argv[2], "environment_count": int(sys.argv[3]),
    "seed": 2021, "optimization_epochs": 10, "stage_epochs": 2,
    "future_patch_len": 16,
}
if sys.argv[2] == "horizon_future_var":
    expected["variant_training_schedule"] = "gradient_relative_invariant_anchored"
if any(row.get(k) != v for k, v in expected.items()):
    raise SystemExit(1)
PY
}

run_mode() {
  local dataset="$1" mode="$2" gpu="$3" reference="$4"
  local destination="$OUTPUT_ROOT/$dataset/$mode"
  local metrics="$destination/A2/metrics_and_diagnostics.json"
  local root klass path freq batch cycle chunk future_weight reliability_weight
  local var_gain_weight conditional_gain_weight
  if result_valid "$metrics" "$mode"; then
    echo "[$dataset/$PRED_LEN/$mode] reuse verified result"; return 0
  fi
  read -r root klass path freq batch cycle chunk <<<"$(dataset_config "$dataset")"
  future_weight=0.0; reliability_weight=0.0
  var_gain_weight=0.1; conditional_gain_weight=0.2
  if [[ "$mode" == horizon_future_var ]]; then
    future_weight=0.1
    var_gain_weight=0.0; conditional_gain_weight=0.0
  fi
  mkdir -p "$destination"
  echo "[$dataset/$PRED_LEN/$mode] start GPU=$gpu $(date --iso-8601=seconds)"
  GPU="$gpu" BACKBONE=patchtst SEED=2021 ENV_NUM="$ENV_NUM" EXPERIMENTS=A2 \
  DATASET_NAME="$dataset" DATASET_ARCHIVE_SHA256="$ARCHIVE_SHA256" \
  DATA_ROOT="$root" DATA_CLASS="$klass" DATA_PATH="$path" TARGET=OT \
  FREQ="$freq" CYCLENET_CYCLE_LEN="$cycle" SEQ_LEN=96 PRED_LEN="$PRED_LEN" \
  BATCH_SIZE="$batch" NUM_WORKERS=0 REPRESENTATION_CONSTRAINT=classification \
  DECOMPOSITION_TYPE=complementary_gate VARIANT_FUSION_MODE="$mode" \
  VARIANT_FUSION_GATE_TYPE=feature EPOCHS=10 WARMUP_EPOCHS=3 STAGE_EPOCHS=2 \
  LAMBDA_INVPRED=1.0 LAMBDA_VAR_GAIN="$var_gain_weight" VAR_GAIN_TEMPERATURE=0.01 \
  LAMBDA_VAR_CONDITIONAL_GAIN="$conditional_gain_weight" LAMBDA_FUTURE_VAR=0.0 \
  LAMBDA_FUTURE_H="$future_weight" \
  LAMBDA_HORIZON_RELIABILITY="$reliability_weight" \
  HORIZON_RELIABILITY_TEMPERATURE=0.05 FUTURE_PATCH_LEN=16 \
  FUTURE_TEACHER_PATCHES_PER_BATCH=2 FUTURE_TEACHER_EVAL_PATCH_COUNT=3 \
  HORIZON_FUTURE_DIM=32 HORIZON_FUTURE_CHUNK_SIZE="$chunk" \
  HORIZON_FUTURE_GAMMA_SCALE=0.1 HORIZON_FUTURE_BETA_SCALE=0.1 \
  FILM_GAMMA_SCALE=0.1 FILM_BETA_SCALE=0.1 \
  Y_FILM_GAMMA_SCALE=0.1 Y_FILM_BETA_SCALE=0.1 Y_FILM_DECAY_BIAS=-4.0 \
  PREDICTIVE_ENV_REFACTOR_MODE=current FUSION_SCALE_CALIBRATION=none \
  LAMBDA_H_ANCHOR=0.0 LAMBDA_VAR_PREDICTIVE=0.0 LAMBDA_VAR_UTILITY=0.0 \
  LAMBDA_COMPLEMENTARY_GATE_ACTIVITY=0.0 ENVIRONMENT_MATCHING=overlap \
  REQUIRE_REFERENCE_CHECKPOINT=1 SAVE_FINAL_CHECKPOINT=1 \
  REFERENCE_CHECKPOINT="$reference" OUTPUT="$destination" \
  bash scripts/run_predictive_env_iv_patchtst.sh
}

run_dataset() {
  local dataset="$1" gpu="$2" reference mode failures=0
  reference="$(reference_path "$dataset")"
  exec > >(tee -a "$OUTPUT_ROOT/logs/${dataset}_pred${PRED_LEN}.log") 2>&1
  if ! prepare_reference "$dataset" "$gpu" "$reference"; then
    echo "$dataset reference" >> "$OUTPUT_ROOT/status/failures.txt"; return 1
  fi
  for mode in "${MODES[@]}"; do
    if run_mode "$dataset" "$mode" "$gpu" "$reference"; then
      echo "$dataset $mode" >> "$OUTPUT_ROOT/status/completed.txt"
    else
      echo "$dataset $mode" >> "$OUTPUT_ROOT/status/failures.txt"
      failures=$((failures + 1))
    fi
  done
  return "$failures"
}

SCHEDULER_DIR="$(mktemp -d "$OUTPUT_ROOT/.scheduler.XXXXXX")"
echo 0 > "$SCHEDULER_DIR/next"; touch "$SCHEDULER_DIR/lock"
trap 'rm -f "$SCHEDULER_DIR/next" "$SCHEDULER_DIR/lock"; rmdir "$SCHEDULER_DIR" 2>/dev/null || true' EXIT
worker() {
  local gpu="$1" index dataset
  while true; do
    exec 9>"$SCHEDULER_DIR/lock"; flock 9; index="$(<"$SCHEDULER_DIR/next")"
    if (( index >= ${#DATASETS[@]} )); then flock -u 9; exec 9>&-; break; fi
    dataset="${DATASETS[$index]}"; echo $((index + 1)) > "$SCHEDULER_DIR/next"
    flock -u 9; exec 9>&-
    wait_for_gpu "$gpu"; run_dataset "$dataset" "$gpu" || true
  done
}

echo "queued ${#DATASETS[@]} datasets x ${#MODES[@]} modes, pred_len=$PRED_LEN"
pids=(); for gpu in $GPU_LIST; do worker "$gpu" & pids+=("$!"); done
for pid in "${pids[@]}"; do wait "$pid" || true; done
date --iso-8601=seconds > "$OUTPUT_ROOT/completion.txt"
