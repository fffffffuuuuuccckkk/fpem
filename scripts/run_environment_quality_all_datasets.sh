#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/environment_quality_patchtst_all_datasets_k3}"
ARCHIVE="${ARCHIVE:-dataset/all_datasets.zip}"
BACKBONE="${BACKBONE:-patchtst}"
RANDOM_PARTITION_REPEATS="${RANDOM_PARTITION_REPEATS:-1000}"
NUM_WORKERS="${NUM_WORKERS:-0}"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_ROOT"
test -s "$ARCHIVE"
ARCHIVE_SHA256="$(sha256sum "$ARCHIVE" | awk '{print $1}')"

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

run_dataset() (
    dataset="$1"; gpu="$2"
    read -r root data_class data_path freq batch cycle \
      <<<"$(dataset_config "$dataset")"
    destination="$OUTPUT_ROOT/$dataset"
    reference="$destination/shared_reference.pt"
    mkdir -p "$destination"
    exec > >(tee -a "$destination/run.log") 2>&1
    if [[ ! -s "$reference" ]]; then
        GPU="$gpu" BACKBONE="$BACKBONE" EXPERIMENTS=A2 \
        DATASET_NAME="$dataset" DATASET_ARCHIVE_SHA256="$ARCHIVE_SHA256" \
        DATA_ROOT="$root" DATA_CLASS="$data_class" DATA_PATH="$data_path" \
        TARGET=OT FREQ="$freq" CYCLENET_CYCLE_LEN="$cycle" BATCH_SIZE="$batch" \
        NUM_WORKERS="$NUM_WORKERS" SEED=2021 ENV_NUM=3 EPOCHS=10 \
        WARMUP_EPOCHS=3 STAGE_EPOCHS=2 PREPARE_REFERENCE_ONLY=1 \
        REFERENCE_CHECKPOINT="$reference" OUTPUT="$destination/reference_prepare" \
        bash scripts/run_predictive_env_iv_patchtst.sh
    fi
    GPU="$gpu" BACKBONE="$BACKBONE" EXPERIMENTS=A2 \
    DATASET_NAME="$dataset" DATASET_ARCHIVE_SHA256="$ARCHIVE_SHA256" \
    DATA_ROOT="$root" DATA_CLASS="$data_class" DATA_PATH="$data_path" \
    TARGET=OT FREQ="$freq" CYCLENET_CYCLE_LEN="$cycle" BATCH_SIZE="$batch" \
    NUM_WORKERS="$NUM_WORKERS" SEED=2021 ENV_NUM=3 EPOCHS=10 \
    WARMUP_EPOCHS=3 STAGE_EPOCHS=2 REPRESENTATION_CONSTRAINT=classification \
    DECOMPOSITION_TYPE=complementary_gate VARIANT_FUSION_MODE=direct_gated \
    PREDICTIVE_ENV_REFACTOR_MODE=current FUSION_SCALE_CALIBRATION=none \
    LAMBDA_INVPRED=1.0 LAMBDA_VAR_CONDITIONAL_GAIN=0.2 \
    LAMBDA_VAR_PREDICTIVE=0.0 LAMBDA_VAR_UTILITY=0.0 LAMBDA_FUTURE_VAR=0.0 \
    LAMBDA_COMPLEMENTARY_GATE_ACTIVITY=0.0 ENVIRONMENT_MATCHING=overlap \
    ENVIRONMENT_QUALITY_DIAGNOSTICS=1 \
    RANDOM_PARTITION_REPEATS="$RANDOM_PARTITION_REPEATS" \
    REQUIRE_REFERENCE_CHECKPOINT=1 REFERENCE_CHECKPOINT="$reference" \
    OUTPUT="$destination" bash scripts/run_predictive_env_iv_patchtst.sh
)

run_queue() {
    gpu="$1"; shift
    for dataset in "$@"; do run_dataset "$dataset" "$gpu"; done
}

run_queue 0 Traffic & p0=$!
run_queue 1 Electricity & p1=$!
run_queue 2 Weather ETTm1 ETTh2 & p2=$!
run_queue 3 ETTm2 ETTh1 ExchangeRate & p3=$!
status=0
for pid in "$p0" "$p1" "$p2" "$p3"; do wait "$pid" || status=1; done
[[ "$status" -eq 0 ]] || exit "$status"
"$PYTHON" tools/summarize_environment_quality.py "$OUTPUT_ROOT"
