#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/predictive_env_featurewise_gate_all_backbones_k3}"
EXISTING_BACKBONE_ROOT="${EXISTING_BACKBONE_ROOT:-results/predictive_env_cyclenet_itransformer_all_datasets_k3}"
ARCHIVE="${ARCHIVE:-dataset/all_datasets.zip}"
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

metrics_valid() {
    local metrics="$1" backbone="$2" dataset="$3" gate_type="$4" reference="$5"
    [[ -s "$metrics" && -s "$reference" ]] || return 1
    "$PYTHON" - "$metrics" "$backbone" "$dataset" "$gate_type" \
      "$reference" "$ARCHIVE_SHA256" <<'PY'
import hashlib, json, sys
from pathlib import Path

metrics, backbone, dataset, gate_type, reference, archive_hash = sys.argv[1:]
row = json.loads(Path(metrics).read_text())
expected = {
    "backbone": backbone,
    "dataset_name": dataset,
    "dataset_archive_sha256": archive_hash,
    "experiment": "A2",
    "representation_constraint": "classification",
    "decomposition_type": "complementary_gate",
    "variant_fusion_mode": "direct_gated",
    "variant_fusion_gate_type": gate_type,
    "environment_count": 3,
    "seed": 2021,
    "optimization_epochs": 10,
    "stage_epochs": 2,
    "lambda_var_conditional_gain": 0.2,
    "reference_checkpoint_sha256": hashlib.sha256(Path(reference).read_bytes()).hexdigest(),
    "reference_source": "loaded",
}
if any(row.get(key) != value for key, value in expected.items()):
    raise SystemExit(1)
PY
}

prepare_patchtst_reference() {
    local dataset="$1" gpu="$2" reference="$3"
    [[ -s "$reference" ]] && return
    read -r data_root data_class data_path freq batch cycle <<<"$(dataset_config "$dataset")"
    mkdir -p "$(dirname "$reference")"
    echo "[patchtst/$dataset] prepare shared reference on GPU $gpu"
    GPU="$gpu" BACKBONE=patchtst SEED=2021 ENV_NUM=3 EXPERIMENTS=A2 \
    DATASET_NAME="$dataset" DATASET_ARCHIVE_SHA256="$ARCHIVE_SHA256" \
    DATA_ROOT="$data_root" DATA_CLASS="$data_class" DATA_PATH="$data_path" \
    TARGET=OT FREQ="$freq" CYCLENET_CYCLE_LEN="$cycle" \
    BATCH_SIZE="$batch" NUM_WORKERS="$NUM_WORKERS" \
    EPOCHS=10 WARMUP_EPOCHS=3 STAGE_EPOCHS=2 PREPARE_REFERENCE_ONLY=1 \
    REFERENCE_CHECKPOINT="$reference" \
    OUTPUT="$OUTPUT_ROOT/patchtst/$dataset/reference_prepare" \
    bash scripts/run_predictive_env_iv_patchtst.sh
}

run_variant() {
    local backbone="$1" dataset="$2" gpu="$3" gate_type="$4" reference="$5"
    local destination="$OUTPUT_ROOT/$backbone/$dataset/${gate_type}_gate"
    local metrics="$destination/A2/metrics_and_diagnostics.json"
    if metrics_valid "$metrics" "$backbone" "$dataset" "$gate_type" "$reference"; then
        echo "[$backbone/$dataset] reuse verified ${gate_type}-wise gate"
        return
    fi
    read -r data_root data_class data_path freq batch cycle <<<"$(dataset_config "$dataset")"
    echo "[$backbone/$dataset] run ${gate_type}-wise gate on GPU $gpu"
    GPU="$gpu" BACKBONE="$backbone" SEED=2021 ENV_NUM=3 EXPERIMENTS=A2 \
    DATASET_NAME="$dataset" DATASET_ARCHIVE_SHA256="$ARCHIVE_SHA256" \
    DATA_ROOT="$data_root" DATA_CLASS="$data_class" DATA_PATH="$data_path" \
    TARGET=OT FREQ="$freq" CYCLENET_CYCLE_LEN="$cycle" BATCH_SIZE="$batch" \
    NUM_WORKERS="$NUM_WORKERS" REPRESENTATION_CONSTRAINT=classification \
    DECOMPOSITION_TYPE=complementary_gate VARIANT_FUSION_MODE=direct_gated \
    VARIANT_FUSION_GATE_TYPE="$gate_type" PREDICTIVE_ENV_REFACTOR_MODE=current \
    FUSION_SCALE_CALIBRATION=none EPOCHS=10 WARMUP_EPOCHS=3 STAGE_EPOCHS=2 \
    LAMBDA_H_ANCHOR=0.0 LAMBDA_INVPRED=1.0 \
    LAMBDA_VAR_CONDITIONAL_GAIN=0.2 VAR_CONDITIONAL_MARGIN=0.0 \
    LAMBDA_VAR_PREDICTIVE=0.0 LAMBDA_VAR_UTILITY=0.0 LAMBDA_FUTURE_VAR=0.0 \
    LAMBDA_COMPLEMENTARY_GATE_ACTIVITY=0.0 ENVIRONMENT_MATCHING=overlap \
    REQUIRE_REFERENCE_CHECKPOINT=1 SAVE_FINAL_CHECKPOINT=1 \
    REFERENCE_CHECKPOINT="$reference" OUTPUT="$destination" \
    bash scripts/run_predictive_env_iv_patchtst.sh
}

run_dataset() (
    local dataset="$1" gpu="$2" backbone reference
    mkdir -p "$OUTPUT_ROOT/logs"
    exec > >(tee -a "$OUTPUT_ROOT/logs/${dataset}.log") 2>&1
    echo "[$dataset] start $(date --iso-8601=seconds) GPU=$gpu"

    reference="$OUTPUT_ROOT/patchtst/$dataset/shared_reference.pt"
    prepare_patchtst_reference "$dataset" "$gpu" "$reference"
    # The ZIP-based PatchTST token baseline is run here because the older
    # PatchTST study predates the archive protocol. Both gates share this ref.
    run_variant patchtst "$dataset" "$gpu" token "$reference"
    run_variant patchtst "$dataset" "$gpu" feature "$reference"

    for backbone in cyclenet itransformer; do
        reference="$EXISTING_BACKBONE_ROOT/$backbone/$dataset/shared_reference.pt"
        test -s "$reference"
        run_variant "$backbone" "$dataset" "$gpu" feature "$reference"
    done
    echo "[$dataset] complete $(date --iso-8601=seconds)"
)

run_queue() {
    local gpu="$1"; shift
    local dataset
    for dataset in "$@"; do run_dataset "$dataset" "$gpu"; done
}

run_queue 0 Traffic & p0=$!
run_queue 1 Electricity & p1=$!
run_queue 2 Weather ETTm1 ETTh2 & p2=$!
run_queue 3 ETTm2 ETTh1 ExchangeRate & p3=$!
status=0
for pid in "$p0" "$p1" "$p2" "$p3"; do wait "$pid" || status=1; done
[[ "$status" -eq 0 ]] || { echo "one or more feature-gate queues failed" >&2; exit "$status"; }
"$PYTHON" tools/summarize_predictive_env_featurewise_gate.py \
  --feature-root "$OUTPUT_ROOT" --existing-backbone-root "$EXISTING_BACKBONE_ROOT"
echo "all token-vs-feature gate comparisons complete $(date --iso-8601=seconds)"
