#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/environment_quality_selected_patchtst_k3}"
REFERENCE_ROOT="${REFERENCE_ROOT:-results/predictive_env_reference_all_datasets_k3}"
RANDOM_PARTITION_REPEATS="${RANDOM_PARTITION_REPEATS:-1000}"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_ROOT"

dataset_config() {
    case "$1" in
        ETTh2)       echo "./dataset/ETT-small ett_hour ETTh2.csv h 32 24" ;;
        Electricity) echo "./dataset/electricity custom electricity.csv h 4 168" ;;
        Traffic)     echo "./dataset/traffic custom traffic.csv h 2 168" ;;
        *) echo "unsupported selected dataset: $1" >&2; return 2 ;;
    esac
}

run_dataset() (
    dataset="$1"; gpu="$2"
    read -r root data_class data_path freq batch cycle <<<"$(dataset_config "$dataset")"
    destination="$OUTPUT_ROOT/$dataset"
    reference="$REFERENCE_ROOT/$dataset/shared_reference.pt"
    test -s "$reference"
    mkdir -p "$destination"
    exec > >(tee -a "$destination/run.log") 2>&1
    echo "[$dataset] start $(date --iso-8601=seconds) reference=$reference"
    sha256sum "$reference" | tee "$destination/reference_sha256.txt"
    GPU="$gpu" BACKBONE=patchtst EXPERIMENTS=A2 \
    DATASET_NAME="$dataset" DATA_ROOT="$root" DATA_CLASS="$data_class" \
    DATA_PATH="$data_path" TARGET=OT FREQ="$freq" CYCLENET_CYCLE_LEN="$cycle" \
    BATCH_SIZE="$batch" NUM_WORKERS=0 SEED=2021 ENV_NUM=3 EPOCHS=10 \
    WARMUP_EPOCHS=3 STAGE_EPOCHS=2 REPRESENTATION_CONSTRAINT=classification \
    DECOMPOSITION_TYPE=complementary_gate VARIANT_FUSION_MODE=direct_gated \
    PREDICTIVE_ENV_REFACTOR_MODE=current FUSION_SCALE_CALIBRATION=none \
    LAMBDA_INVPRED=1.0 LAMBDA_VAR_CONDITIONAL_GAIN=0.2 \
    LAMBDA_VAR_PREDICTIVE=0.0 LAMBDA_VAR_UTILITY=0.0 LAMBDA_FUTURE_VAR=0.0 \
    LAMBDA_COMPLEMENTARY_GATE_ACTIVITY=0.0 ENVIRONMENT_MATCHING=overlap \
    ENVIRONMENT_QUALITY_DIAGNOSTICS=1 ENVIRONMENT_QUALITY_FINAL_ONLY=1 \
    RANDOM_PARTITION_REPEATS="$RANDOM_PARTITION_REPEATS" \
    REQUIRE_REFERENCE_CHECKPOINT=1 ALLOW_LEGACY_REFERENCE_CHECKPOINT=1 \
    SAVE_FINAL_CHECKPOINT=1 REFERENCE_CHECKPOINT="$reference" \
    OUTPUT="$destination" bash scripts/run_predictive_env_iv_patchtst.sh
    echo "[$dataset] complete $(date --iso-8601=seconds)"
)

run_dataset ETTh2 0 & p0=$!
run_dataset Electricity 1 & p1=$!
run_dataset Traffic 2 & p2=$!
status=0
for pid in "$p0" "$p1" "$p2"; do wait "$pid" || status=1; done
[[ "$status" -eq 0 ]] || exit "$status"
"$PYTHON" tools/summarize_environment_quality.py "$OUTPUT_ROOT"
