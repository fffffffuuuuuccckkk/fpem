#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/predictive_env_patchtst_film_featurewise_multihorizon_k3}"
REFERENCE_ROOT="${REFERENCE_ROOT:-results/predictive_env_patchtst_multihorizon_96}"
ARCHIVE="${ARCHIVE:-dataset/all_datasets.zip}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"
PRED_LENGTHS="${PRED_LENGTHS:-168 336 720}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LAMBDA_VAR_GAIN="${LAMBDA_VAR_GAIN:-0.1}"
VAR_GAIN_TEMPERATURE="${VAR_GAIN_TEMPERATURE:-0.01}"
FILM_GAMMA_SCALE="${FILM_GAMMA_SCALE:-0.1}"
FILM_BETA_SCALE="${FILM_BETA_SCALE:-0.1}"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_ROOT/logs"
test -s "$ARCHIVE"
ARCHIVE_SHA256="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
printf 'server_role=secondary\nbackbone=patchtst\nseq_len=96\npred_lengths=%s\ndatasets=8\nfusion=film\nfeature_wise=true\nlambda_var_gain=%s\nfilm_gamma_scale=%s\nfilm_beta_scale=%s\narchive_sha256=%s\n' \
  "$PRED_LENGTHS" "$LAMBDA_VAR_GAIN" "$FILM_GAMMA_SCALE" \
  "$FILM_BETA_SCALE" "$ARCHIVE_SHA256" > "$OUTPUT_ROOT/protocol.txt"

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

reference_path() {
    local dataset="$1" pred_len="$2"
    local established="$REFERENCE_ROOT/pred_$pred_len/$dataset/shared_reference.pt"
    local local_reference="$OUTPUT_ROOT/pred_$pred_len/$dataset/shared_reference.pt"
    if [[ -s "$established" ]]; then
        echo "$established"
    else
        echo "$local_reference"
    fi
}

prepare_reference() {
    local dataset="$1" pred_len="$2" gpu="$3" reference="$4"
    [[ -s "$reference" ]] && return
    local dataset_root="$OUTPUT_ROOT/pred_$pred_len/$dataset"
    local data_root data_class data_path freq batch cycle
    mkdir -p "$dataset_root"
    read -r data_root data_class data_path freq batch cycle <<<"$(dataset_config "$dataset")"
    echo "[$dataset 96->$pred_len] create shared reference on GPU=$gpu"
    GPU="$gpu" BACKBONE=patchtst SEED=2021 ENV_NUM=3 EXPERIMENTS=A2 \
    DATASET_NAME="$dataset" DATASET_ARCHIVE_SHA256="$ARCHIVE_SHA256" \
    DATA_ROOT="$data_root" DATA_CLASS="$data_class" DATA_PATH="$data_path" \
    TARGET=OT FREQ="$freq" CYCLENET_CYCLE_LEN="$cycle" \
    SEQ_LEN=96 PRED_LEN="$pred_len" BATCH_SIZE="$batch" NUM_WORKERS="$NUM_WORKERS" \
    VARIANT_FUSION_MODE=film VARIANT_FUSION_GATE_TYPE=feature \
    FILM_GAMMA_SCALE="$FILM_GAMMA_SCALE" FILM_BETA_SCALE="$FILM_BETA_SCALE" \
    EPOCHS=10 WARMUP_EPOCHS=3 STAGE_EPOCHS=2 PREPARE_REFERENCE_ONLY=1 \
    REFERENCE_CHECKPOINT="$reference" OUTPUT="$dataset_root/reference_prepare" \
    bash scripts/run_predictive_env_iv_patchtst.sh
}

result_valid() {
    local metrics="$1" config="$2" dataset="$3" pred_len="$4" reference="$5"
    [[ -s "$metrics" && -s "$config" && -s "$reference" ]] || return 1
    "$PYTHON" - "$metrics" "$config" "$dataset" "$pred_len" "$reference" \
      "$ARCHIVE_SHA256" "$LAMBDA_VAR_GAIN" "$VAR_GAIN_TEMPERATURE" \
      "$FILM_GAMMA_SCALE" "$FILM_BETA_SCALE" <<'PY'
import hashlib, json, sys
from pathlib import Path

(metrics_path, config_path, dataset, pred_len, reference_path, archive_hash,
 gain_weight, gain_temperature, gamma_scale, beta_scale) = sys.argv[1:]
metrics = json.loads(Path(metrics_path).read_text())
config = json.loads(Path(config_path).read_text())
expected_metrics = {
    "dataset_name": dataset,
    "dataset_archive_sha256": archive_hash,
    "backbone": "patchtst",
    "experiment": "A2",
    "representation_constraint": "classification",
    "decomposition_type": "complementary_gate",
    "variant_fusion_mode": "film",
    "environment_count": 3,
    "seed": 2021,
    "optimization_epochs": 10,
    "stage_epochs": 2,
    "lambda_var_gain": float(gain_weight),
    "reference_checkpoint_sha256": hashlib.sha256(Path(reference_path).read_bytes()).hexdigest(),
    "reference_source": "loaded",
}
expected_config = {
    "seq_len": 96,
    "pred_len": int(pred_len),
    "variant_fusion_gate_type": "feature",
    "var_gain_temperature": float(gain_temperature),
    "film_gamma_scale": float(gamma_scale),
    "film_beta_scale": float(beta_scale),
}
if any(metrics.get(k) != v for k, v in expected_metrics.items()):
    raise SystemExit(1)
if any(config.get(k) != v for k, v in expected_config.items()):
    raise SystemExit(1)
PY
}

run_one() (
    local dataset="$1" pred_len="$2" gpu="$3"
    local dataset_root="$OUTPUT_ROOT/pred_$pred_len/$dataset"
    local destination="$dataset_root/film"
    local metrics="$destination/A2/metrics_and_diagnostics.json"
    local config="$destination/run_config.json"
    local reference
    reference="$(reference_path "$dataset" "$pred_len")"
    mkdir -p "$destination"
    exec > >(tee -a "$OUTPUT_ROOT/logs/${dataset}_pred${pred_len}_film.log") 2>&1
    prepare_reference "$dataset" "$pred_len" "$gpu" "$reference"
    if result_valid "$metrics" "$config" "$dataset" "$pred_len" "$reference"; then
        echo "[$dataset 96->$pred_len film] reuse verified result"
        exit 0
    fi
    local data_root data_class data_path freq batch cycle
    read -r data_root data_class data_path freq batch cycle <<<"$(dataset_config "$dataset")"
    echo "[$dataset 96->$pred_len film] start GPU=$gpu $(date --iso-8601=seconds)"
    GPU="$gpu" BACKBONE=patchtst SEED=2021 ENV_NUM=3 EXPERIMENTS=A2 \
    DATASET_NAME="$dataset" DATASET_ARCHIVE_SHA256="$ARCHIVE_SHA256" \
    DATA_ROOT="$data_root" DATA_CLASS="$data_class" DATA_PATH="$data_path" \
    TARGET=OT FREQ="$freq" CYCLENET_CYCLE_LEN="$cycle" \
    SEQ_LEN=96 PRED_LEN="$pred_len" BATCH_SIZE="$batch" NUM_WORKERS="$NUM_WORKERS" \
    REPRESENTATION_CONSTRAINT=classification DECOMPOSITION_TYPE=complementary_gate \
    VARIANT_FUSION_MODE=film VARIANT_FUSION_GATE_TYPE=feature \
    FILM_GAMMA_SCALE="$FILM_GAMMA_SCALE" FILM_BETA_SCALE="$FILM_BETA_SCALE" \
    PREDICTIVE_ENV_REFACTOR_MODE=current FUSION_SCALE_CALIBRATION=none \
    EPOCHS=10 WARMUP_EPOCHS=3 STAGE_EPOCHS=2 LAMBDA_H_ANCHOR=0.0 \
    LAMBDA_INVPRED=1.0 LAMBDA_VAR_GAIN="$LAMBDA_VAR_GAIN" \
    VAR_GAIN_TEMPERATURE="$VAR_GAIN_TEMPERATURE" \
    LAMBDA_VAR_CONDITIONAL_GAIN=0.2 VAR_CONDITIONAL_MARGIN=0.0 \
    LAMBDA_VAR_PREDICTIVE=0.0 LAMBDA_VAR_UTILITY=0.0 LAMBDA_FUTURE_VAR=0.0 \
    LAMBDA_COMPLEMENTARY_GATE_ACTIVITY=0.0 ENVIRONMENT_MATCHING=overlap \
    REQUIRE_REFERENCE_CHECKPOINT=1 SAVE_FINAL_CHECKPOINT=1 \
    REFERENCE_CHECKPOINT="$reference" OUTPUT="$destination" \
    bash scripts/run_predictive_env_iv_patchtst.sh
    echo "[$dataset 96->$pred_len film] complete $(date --iso-8601=seconds)"
)

TASKS=(
    Traffic:720 Electricity:720 Traffic:336 Electricity:336
    Traffic:168 Electricity:168 Weather:720 ETTm1:720 ETTm2:720
    ETTh1:720 ETTh2:720 ExchangeRate:720 Weather:336 ETTm1:336
    ETTm2:336 ETTh1:336 ETTh2:336 ExchangeRate:336 Weather:168
    ETTm1:168 ETTm2:168 ETTh1:168 ETTh2:168 ExchangeRate:168
)
SCHEDULER_DIR="$(mktemp -d "$OUTPUT_ROOT/.scheduler.XXXXXX")"
printf '0\n' > "$SCHEDULER_DIR/next"
touch "$SCHEDULER_DIR/lock"
trap 'rm -f "$SCHEDULER_DIR/next" "$SCHEDULER_DIR/lock"; rmdir "$SCHEDULER_DIR" 2>/dev/null || true' EXIT

worker() {
    local gpu="$1" index task
    while true; do
        exec 9>"$SCHEDULER_DIR/lock"
        flock 9
        index="$(<"$SCHEDULER_DIR/next")"
        if (( index >= ${#TASKS[@]} )); then
            flock -u 9; exec 9>&-; break
        fi
        task="${TASKS[$index]}"
        printf '%s\n' "$((index + 1))" > "$SCHEDULER_DIR/next"
        flock -u 9; exec 9>&-
        run_one "${task%%:*}" "${task#*:}" "$gpu"
    done
}

pids=()
for gpu in $GPU_LIST; do worker "$gpu" & pids+=("$!"); done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[[ "$status" -eq 0 ]] || { echo "one or more FiLM multihorizon jobs failed" >&2; exit "$status"; }
"$PYTHON" tools/summarize_predictive_env_film_multihorizon.py --root "$OUTPUT_ROOT"
date --iso-8601=seconds > "$OUTPUT_ROOT/completion.txt"
