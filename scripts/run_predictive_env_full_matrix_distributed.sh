#!/usr/bin/env bash
set -uo pipefail

# Distributed full matrix for the three FPEM hosts.
# SERVER_ROLE=server1: PatchTST + CycleNet, pred_len=96
# SERVER_ROLE=server2: PatchTST, pred_len=168/336/720
# SERVER_ROLE=server3: CycleNet, pred_len=168/336/720

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
SERVER_ROLE="${SERVER_ROLE:?set SERVER_ROLE=server1|server2|server3}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/predictive_env_film_decay_full_matrix/$SERVER_ROLE}"
ARCHIVE="${ARCHIVE:-dataset/all_datasets.zip}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"
NUM_WORKERS="${NUM_WORKERS:-0}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-60}"
GPU_FREE_MEMORY_MIB="${GPU_FREE_MEMORY_MIB:-1000}"
GPU_FREE_UTIL_PERCENT="${GPU_FREE_UTIL_PERCENT:-10}"

LAMBDA_VAR_GAIN="${LAMBDA_VAR_GAIN:-0.1}"
VAR_GAIN_TEMPERATURE="${VAR_GAIN_TEMPERATURE:-0.01}"
LAMBDA_DECAY_MONO="${LAMBDA_DECAY_MONO:-0.1}"
LAMBDA_DECAY_FAR="${LAMBDA_DECAY_FAR:-0.1}"
HORIZON_DECAY_BINS="${HORIZON_DECAY_BINS:-4}"

cd "$PROJECT_DIR" || exit 1
mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/status"
test -s "$ARCHIVE" || { echo "missing archive: $ARCHIVE" >&2; exit 2; }
ARCHIVE_SHA256="$(sha256sum "$ARCHIVE" | awk '{print $1}')"

case "$SERVER_ROLE" in
  server1)
    BACKBONES=(patchtst cyclenet)
    PRED_LENGTHS=(96)
    ;;
  server2)
    BACKBONES=(patchtst)
    PRED_LENGTHS=(168 336 720)
    ;;
  server3)
    BACKBONES=(cyclenet)
    PRED_LENGTHS=(168 336 720)
    ;;
  *) echo "unknown SERVER_ROLE=$SERVER_ROLE" >&2; exit 2 ;;
esac

DATASETS=(Traffic Electricity Weather ETTm1 ETTm2 ETTh1 ETTh2 ExchangeRate)
MODES=(film film_decay_reg y_film_decay direct_gated inv_only)

cat > "$OUTPUT_ROOT/protocol.txt" <<EOF
server_role=$SERVER_ROLE
backbones=${BACKBONES[*]}
seq_len=96
pred_lengths=${PRED_LENGTHS[*]}
datasets=${DATASETS[*]}
modes=${MODES[*]}
environment_count=3
seed=2021
representation_constraint=classification
decomposition_type=complementary_gate
gate_type=feature
lambda_var_gain=$LAMBDA_VAR_GAIN
var_gain_temperature=$VAR_GAIN_TEMPERATURE
lambda_decay_mono=$LAMBDA_DECAY_MONO
lambda_decay_far=$LAMBDA_DECAY_FAR
horizon_decay_bins=$HORIZON_DECAY_BINS
save_final_checkpoint=false
archive_sha256=$ARCHIVE_SHA256
EOF

dataset_config() {
  case "$1" in
    ETTh1)        echo "./dataset/all_datasets/ETT-small ett_hour ETTh1.csv h 32 24" ;;
    ETTh2)        echo "./dataset/all_datasets/ETT-small ett_hour ETTh2.csv h 32 24" ;;
    ETTm1)        echo "./dataset/all_datasets/ETT-small ett_minute ETTm1.csv 15min 32 96" ;;
    ETTm2)        echo "./dataset/all_datasets/ETT-small ett_minute ETTm2.csv 15min 32 96" ;;
    Electricity)  echo "./dataset/all_datasets/electricity custom electricity.csv h 4 168" ;;
    ExchangeRate) echo "./dataset/all_datasets/exchange_rate custom exchange_rate.csv d 32 7" ;;
    Weather)      echo "./dataset/all_datasets/weather custom weather.csv 10min 16 144" ;;
    Traffic)      echo "./dataset/all_datasets/traffic custom traffic.csv h 2 168" ;;
    *) echo "unknown dataset: $1" >&2; return 2 ;;
  esac
}

first_existing_reference() {
  local candidate
  for candidate in "$@"; do
    if [[ -s "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  printf '%s\n' "${@: -1}"
}

reference_path() {
  local backbone="$1" dataset="$2" pred_len="$3"
  if [[ "$pred_len" == 96 && "$backbone" == patchtst ]]; then
    first_existing_reference \
      "results/predictive_env_featurewise_gate_all_backbones_k3/patchtst/$dataset/shared_reference.pt" \
      "$OUTPUT_ROOT/$backbone/pred_$pred_len/$dataset/shared_reference.pt"
  elif [[ "$pred_len" == 96 && "$backbone" == cyclenet ]]; then
    first_existing_reference \
      "results/predictive_env_cyclenet_itransformer_all_datasets_k3/cyclenet/$dataset/shared_reference.pt" \
      "$OUTPUT_ROOT/$backbone/pred_$pred_len/$dataset/shared_reference.pt"
  elif [[ "$backbone" == patchtst ]]; then
    first_existing_reference \
      "results/predictive_env_patchtst_multihorizon_96/pred_$pred_len/$dataset/shared_reference.pt" \
      "results/predictive_env_patchtst_film_featurewise_multihorizon_k3/pred_$pred_len/$dataset/shared_reference.pt" \
      "$OUTPUT_ROOT/$backbone/pred_$pred_len/$dataset/shared_reference.pt"
  else
    first_existing_reference \
      "results/predictive_env_cyclenet_film_featurewise_multihorizon_k3/pred_$pred_len/$dataset/shared_reference.pt" \
      "$OUTPUT_ROOT/$backbone/pred_$pred_len/$dataset/shared_reference.pt"
  fi
}

wait_for_gpu() {
  local gpu="$1" memory utilization
  while true; do
    IFS=, read -r memory utilization < <(
      nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null | tr -d ' '
    )
    memory="${memory:-999999}"
    utilization="${utilization:-100}"
    if (( memory < GPU_FREE_MEMORY_MIB && utilization < GPU_FREE_UTIL_PERCENT )); then
      echo "[gpu $gpu] available: memory=${memory}MiB utilization=${utilization}%"
      return 0
    fi
    echo "[gpu $gpu] waiting: memory=${memory}MiB utilization=${utilization}%"
    sleep "$GPU_WAIT_SECONDS"
  done
}

prepare_reference() {
  local backbone="$1" dataset="$2" pred_len="$3" gpu="$4" reference="$5"
  [[ -s "$reference" ]] && return 0
  local data_root data_class data_path freq batch cycle ref_output
  read -r data_root data_class data_path freq batch cycle <<<"$(dataset_config "$dataset")" || return 1
  mkdir -p "$(dirname "$reference")"
  ref_output="$OUTPUT_ROOT/$backbone/pred_$pred_len/$dataset/reference_prepare"
  echo "[$backbone/$dataset/96->$pred_len] creating shared reference"
  GPU="$gpu" BACKBONE="$backbone" SEED=2021 ENV_NUM=3 EXPERIMENTS=A2 \
  DATASET_NAME="$dataset" DATASET_ARCHIVE_SHA256="$ARCHIVE_SHA256" \
  DATA_ROOT="$data_root" DATA_CLASS="$data_class" DATA_PATH="$data_path" \
  TARGET=OT FREQ="$freq" CYCLENET_CYCLE_LEN="$cycle" \
  SEQ_LEN=96 PRED_LEN="$pred_len" BATCH_SIZE="$batch" NUM_WORKERS="$NUM_WORKERS" \
  VARIANT_FUSION_MODE=film VARIANT_FUSION_GATE_TYPE=feature \
  EPOCHS=10 WARMUP_EPOCHS=3 STAGE_EPOCHS=2 PREPARE_REFERENCE_ONLY=1 \
  REFERENCE_CHECKPOINT="$reference" OUTPUT="$ref_output" \
  bash scripts/run_predictive_env_iv_patchtst.sh
}

result_valid() {
  local metrics="$1" config="$2" backbone="$3" dataset="$4" pred_len="$5" mode="$6" reference="$7"
  [[ -s "$metrics" && -s "$config" && -s "$reference" ]] || return 1
  "$PYTHON" - "$metrics" "$config" "$backbone" "$dataset" "$pred_len" "$mode" "$reference" "$ARCHIVE_SHA256" <<'PY'
import hashlib, json, sys
from pathlib import Path

metrics_path, config_path, backbone, dataset, pred_len, mode, reference, archive_hash = sys.argv[1:]
metrics = json.loads(Path(metrics_path).read_text())
config = json.loads(Path(config_path).read_text())
expected = {
    "backbone": backbone,
    "dataset_name": dataset,
    "dataset_archive_sha256": archive_hash,
    "experiment": "A2",
    "representation_constraint": "classification",
    "decomposition_type": "complementary_gate",
    "variant_fusion_mode": mode,
    "variant_fusion_gate_type": "feature",
    "environment_count": 3,
    "seed": 2021,
    "optimization_epochs": 10,
    "stage_epochs": 2,
    "reference_checkpoint_sha256": hashlib.sha256(Path(reference).read_bytes()).hexdigest(),
    "reference_source": "loaded",
}
if any(metrics.get(k) != v for k, v in expected.items()):
    raise SystemExit(1)
if config.get("seq_len") != 96 or config.get("pred_len") != int(pred_len):
    raise SystemExit(1)
PY
}

run_mode() {
  local backbone="$1" dataset="$2" pred_len="$3" mode="$4" gpu="$5" reference="$6"
  local destination="$OUTPUT_ROOT/$backbone/pred_$pred_len/$dataset/$mode"
  local metrics="$destination/A2/metrics_and_diagnostics.json"
  local config="$destination/run_config.json"
  local data_root data_class data_path freq batch cycle decay_mono decay_far
  mkdir -p "$destination"
  if result_valid "$metrics" "$config" "$backbone" "$dataset" "$pred_len" "$mode" "$reference"; then
    echo "[$backbone/$dataset/96->$pred_len/$mode] reuse verified result"
    return 0
  fi
  read -r data_root data_class data_path freq batch cycle <<<"$(dataset_config "$dataset")" || return 1
  decay_mono=0.0
  decay_far=0.0
  if [[ "$mode" == film_decay_reg ]]; then
    decay_mono="$LAMBDA_DECAY_MONO"
    decay_far="$LAMBDA_DECAY_FAR"
  fi
  echo "[$backbone/$dataset/96->$pred_len/$mode] start GPU=$gpu $(date --iso-8601=seconds)"
  GPU="$gpu" BACKBONE="$backbone" SEED=2021 ENV_NUM=3 EXPERIMENTS=A2 \
  DATASET_NAME="$dataset" DATASET_ARCHIVE_SHA256="$ARCHIVE_SHA256" \
  DATA_ROOT="$data_root" DATA_CLASS="$data_class" DATA_PATH="$data_path" \
  TARGET=OT FREQ="$freq" CYCLENET_CYCLE_LEN="$cycle" \
  SEQ_LEN=96 PRED_LEN="$pred_len" BATCH_SIZE="$batch" NUM_WORKERS="$NUM_WORKERS" \
  REPRESENTATION_CONSTRAINT=classification DECOMPOSITION_TYPE=complementary_gate \
  VARIANT_FUSION_MODE="$mode" VARIANT_FUSION_GATE_TYPE=feature \
  FILM_GAMMA_SCALE=0.1 FILM_BETA_SCALE=0.1 \
  Y_FILM_GAMMA_SCALE=0.1 Y_FILM_BETA_SCALE=0.1 Y_FILM_DECAY_BIAS=-4.0 \
  LAMBDA_DECAY_MONO="$decay_mono" LAMBDA_DECAY_FAR="$decay_far" \
  HORIZON_DECAY_BINS="$HORIZON_DECAY_BINS" \
  PREDICTIVE_ENV_REFACTOR_MODE=current FUSION_SCALE_CALIBRATION=none \
  EPOCHS=10 WARMUP_EPOCHS=3 STAGE_EPOCHS=2 LAMBDA_H_ANCHOR=0.0 \
  LAMBDA_INVPRED=1.0 LAMBDA_VAR_GAIN="$LAMBDA_VAR_GAIN" \
  VAR_GAIN_TEMPERATURE="$VAR_GAIN_TEMPERATURE" \
  LAMBDA_VAR_CONDITIONAL_GAIN=0.2 VAR_CONDITIONAL_MARGIN=0.0 \
  LAMBDA_VAR_PREDICTIVE=0.0 LAMBDA_VAR_UTILITY=0.0 LAMBDA_FUTURE_VAR=0.0 \
  LAMBDA_COMPLEMENTARY_GATE_ACTIVITY=0.0 ENVIRONMENT_MATCHING=overlap \
  REQUIRE_REFERENCE_CHECKPOINT=1 SAVE_FINAL_CHECKPOINT=0 \
  REFERENCE_CHECKPOINT="$reference" OUTPUT="$destination" \
  bash scripts/run_predictive_env_iv_patchtst.sh
}

run_case() {
  local case_spec="$1" gpu="$2" backbone pred_len dataset reference mode failures=0
  IFS=: read -r backbone pred_len dataset <<<"$case_spec"
  reference="$(reference_path "$backbone" "$dataset" "$pred_len")"
  exec > >(tee -a "$OUTPUT_ROOT/logs/${backbone}_${dataset}_pred${pred_len}.log") 2>&1
  if ! prepare_reference "$backbone" "$dataset" "$pred_len" "$gpu" "$reference"; then
    echo "[$case_spec] reference preparation failed" >&2
    printf '%s\n' "$case_spec reference" >> "$OUTPUT_ROOT/status/failures.txt"
    return 1
  fi
  for mode in "${MODES[@]}"; do
    if run_mode "$backbone" "$dataset" "$pred_len" "$mode" "$gpu" "$reference"; then
      printf '%s\n' "$case_spec $mode" >> "$OUTPUT_ROOT/status/completed.txt"
    else
      echo "[$case_spec/$mode] FAILED; continuing worker queue" >&2
      printf '%s\n' "$case_spec $mode" >> "$OUTPUT_ROOT/status/failures.txt"
      failures=$((failures + 1))
    fi
  done
  return "$failures"
}

CASES=()
for pred_len in "${PRED_LENGTHS[@]}"; do
  for backbone in "${BACKBONES[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      CASES+=("$backbone:$pred_len:$dataset")
    done
  done
done

SCHEDULER_DIR="$(mktemp -d "$OUTPUT_ROOT/.scheduler.XXXXXX")"
printf '0\n' > "$SCHEDULER_DIR/next"
touch "$SCHEDULER_DIR/lock"
trap 'rm -f "$SCHEDULER_DIR/next" "$SCHEDULER_DIR/lock"; rmdir "$SCHEDULER_DIR" 2>/dev/null || true' EXIT

worker() {
  local gpu="$1" index case_spec
  while true; do
    exec 9>"$SCHEDULER_DIR/lock"
    flock 9
    index="$(<"$SCHEDULER_DIR/next")"
    if (( index >= ${#CASES[@]} )); then
      flock -u 9
      exec 9>&-
      break
    fi
    case_spec="${CASES[$index]}"
    printf '%s\n' "$((index + 1))" > "$SCHEDULER_DIR/next"
    flock -u 9
    exec 9>&-
    wait_for_gpu "$gpu"
    run_case "$case_spec" "$gpu" || true
  done
}

echo "[$SERVER_ROLE] queued ${#CASES[@]} cases x ${#MODES[@]} modes on GPUs: $GPU_LIST"
pids=()
for gpu in $GPU_LIST; do
  worker "$gpu" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid" || true
done

date --iso-8601=seconds > "$OUTPUT_ROOT/completion.txt"
echo "[$SERVER_ROLE] queue complete; failures=$(wc -l < "$OUTPUT_ROOT/status/failures.txt" 2>/dev/null || echo 0)"
