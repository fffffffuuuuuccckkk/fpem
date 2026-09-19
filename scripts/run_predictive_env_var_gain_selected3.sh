#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/predictive_env_var_gain_selected3_all_backbones_k3}"
FEATURE_BASE_ROOT="${FEATURE_BASE_ROOT:-results/predictive_env_featurewise_gate_all_backbones_k3}"
BACKBONE_BASE_ROOT="${BACKBONE_BASE_ROOT:-results/predictive_env_cyclenet_itransformer_all_datasets_k3}"
ARCHIVE="${ARCHIVE:-dataset/all_datasets.zip}"
TEMPERATURE="${VAR_GAIN_TEMPERATURE:-0.01}"
NUM_WORKERS="${NUM_WORKERS:-0}"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_ROOT/logs"
ARCHIVE_SHA256="$(sha256sum "$ARCHIVE" | awk '{print $1}')"

dataset_config() {
    case "$1" in
        ETTh2)       echo "./dataset/all_datasets/ETT-small ett_hour ETTh2.csv h 32 24" ;;
        Electricity) echo "./dataset/all_datasets/electricity custom electricity.csv h 4 168" ;;
        Traffic)     echo "./dataset/all_datasets/traffic custom traffic.csv h 2 168" ;;
        *) echo "unsupported dataset: $1" >&2; return 2 ;;
    esac
}

reference_path() {
    local backbone="$1" dataset="$2"
    if [[ "$backbone" == patchtst ]]; then
        echo "$FEATURE_BASE_ROOT/patchtst/$dataset/shared_reference.pt"
    else
        echo "$BACKBONE_BASE_ROOT/$backbone/$dataset/shared_reference.pt"
    fi
}

prepare_patch_reference() {
    local dataset="$1" gpu="$2" reference="$3"
    [[ -s "$reference" ]] && return
    read -r root data_class data_path freq batch cycle <<<"$(dataset_config "$dataset")"
    mkdir -p "$(dirname "$reference")"
    GPU="$gpu" BACKBONE=patchtst SEED=2021 ENV_NUM=3 EXPERIMENTS=A2 \
    DATASET_NAME="$dataset" DATASET_ARCHIVE_SHA256="$ARCHIVE_SHA256" \
    DATA_ROOT="$root" DATA_CLASS="$data_class" DATA_PATH="$data_path" \
    TARGET=OT FREQ="$freq" CYCLENET_CYCLE_LEN="$cycle" BATCH_SIZE="$batch" \
    NUM_WORKERS="$NUM_WORKERS" EPOCHS=10 WARMUP_EPOCHS=3 STAGE_EPOCHS=2 \
    PREPARE_REFERENCE_ONLY=1 REFERENCE_CHECKPOINT="$reference" \
    OUTPUT="$FEATURE_BASE_ROOT/patchtst/$dataset/reference_prepare" \
    bash scripts/run_predictive_env_iv_patchtst.sh
}

baseline_metrics() {
    local backbone="$1" dataset="$2" gate="$3"
    if [[ "$backbone" == patchtst || "$gate" == feature ]]; then
        echo "$FEATURE_BASE_ROOT/$backbone/$dataset/${gate}_gate/A2/metrics_and_diagnostics.json"
    else
        echo "$BACKBONE_BASE_ROOT/$backbone/$dataset/A2_predictive_env/A2/metrics_and_diagnostics.json"
    fi
}

lambda_label() { printf '%s' "$1" | tr -d '.'; }

metrics_valid() {
    local metrics="$1" backbone="$2" dataset="$3" gate="$4" lambda="$5" reference="$6"
    [[ -s "$metrics" && -s "$reference" ]] || return 1
    "$PYTHON" - "$metrics" "$backbone" "$dataset" "$gate" "$lambda" \
      "$TEMPERATURE" "$reference" "$ARCHIVE_SHA256" <<'PY'
import hashlib, json, sys
from pathlib import Path

metrics, backbone, dataset, gate, weight, temperature, reference, archive_hash = sys.argv[1:]
row = json.loads(Path(metrics).read_text())
expected = {
    "backbone": backbone,
    "dataset_name": dataset,
    "dataset_archive_sha256": archive_hash,
    "experiment": "A2",
    "representation_constraint": "classification",
    "decomposition_type": "complementary_gate",
    "variant_fusion_mode": "direct_gated",
    "environment_count": 3,
    "seed": 2021,
    "optimization_epochs": 10,
    "stage_epochs": 2,
    "lambda_var_conditional_gain": 0.2,
    "reference_checkpoint_sha256": hashlib.sha256(Path(reference).read_bytes()).hexdigest(),
    "reference_source": "loaded",
}
actual_gate = row.get("variant_fusion_gate_type", "token")
actual_weight = float(row.get("lambda_var_gain", 0.0))
if any(row.get(key) != value for key, value in expected.items()):
    raise SystemExit(1)
if actual_gate != gate or abs(actual_weight - float(weight)) > 1e-12:
    raise SystemExit(1)
if float(weight) != 0 and abs(float(row.get("var_gain_temperature", -1)) - float(temperature)) > 1e-12:
    raise SystemExit(1)
PY
}

run_setting() {
    local backbone="$1" dataset="$2" gpu="$3" gate="$4" weight="$5" reference="$6"
    local label destination metrics baseline
    label="$(lambda_label "$weight")"
    destination="$OUTPUT_ROOT/$backbone/$dataset/${gate}_gate/lambda_${label}"
    metrics="$destination/A2/metrics_and_diagnostics.json"
    if metrics_valid "$metrics" "$backbone" "$dataset" "$gate" "$weight" "$reference"; then
        echo "[$backbone/$dataset] reuse $gate lambda=$weight"
        return
    fi
    if [[ "$weight" == 0 ]]; then
        baseline="$(baseline_metrics "$backbone" "$dataset" "$gate")"
        if metrics_valid "$baseline" "$backbone" "$dataset" "$gate" 0 "$reference"; then
            echo "[$backbone/$dataset] reuse external $gate lambda=0 baseline"
            return
        fi
    fi
    read -r root data_class data_path freq batch cycle <<<"$(dataset_config "$dataset")"
    echo "[$backbone/$dataset] run $gate lambda=$weight GPU=$gpu"
    GPU="$gpu" BACKBONE="$backbone" SEED=2021 ENV_NUM=3 EXPERIMENTS=A2 \
    DATASET_NAME="$dataset" DATASET_ARCHIVE_SHA256="$ARCHIVE_SHA256" \
    DATA_ROOT="$root" DATA_CLASS="$data_class" DATA_PATH="$data_path" \
    TARGET=OT FREQ="$freq" CYCLENET_CYCLE_LEN="$cycle" BATCH_SIZE="$batch" \
    NUM_WORKERS="$NUM_WORKERS" REPRESENTATION_CONSTRAINT=classification \
    DECOMPOSITION_TYPE=complementary_gate VARIANT_FUSION_MODE=direct_gated \
    VARIANT_FUSION_GATE_TYPE="$gate" PREDICTIVE_ENV_REFACTOR_MODE=current \
    FUSION_SCALE_CALIBRATION=none EPOCHS=10 WARMUP_EPOCHS=3 STAGE_EPOCHS=2 \
    LAMBDA_INVPRED=1.0 LAMBDA_VAR_CONDITIONAL_GAIN=0.2 \
    LAMBDA_VAR_GAIN="$weight" VAR_GAIN_TEMPERATURE="$TEMPERATURE" \
    LAMBDA_VAR_PREDICTIVE=0.0 LAMBDA_VAR_UTILITY=0.0 LAMBDA_FUTURE_VAR=0.0 \
    LAMBDA_COMPLEMENTARY_GATE_ACTIVITY=0.0 ENVIRONMENT_MATCHING=overlap \
    REQUIRE_REFERENCE_CHECKPOINT=1 REFERENCE_CHECKPOINT="$reference" \
    OUTPUT="$destination" bash scripts/run_predictive_env_iv_patchtst.sh
}

run_item() (
    local backbone="$1" dataset="$2" gpu="$3" reference gate weight
    exec > >(tee -a "$OUTPUT_ROOT/logs/${backbone}_${dataset}.log") 2>&1
    reference="$(reference_path "$backbone" "$dataset")"
    if [[ "$backbone" == patchtst ]]; then
        prepare_patch_reference "$dataset" "$gpu" "$reference"
    fi
    test -s "$reference"
    sha256sum "$reference"
    for gate in token feature; do
        for weight in 0 0.05 0.1 0.2; do
            run_setting "$backbone" "$dataset" "$gpu" "$gate" "$weight" "$reference"
        done
    done
)

run_queue() {
    local gpu="$1"; shift
    local item
    for item in "$@"; do
        run_item "${item%%:*}" "${item#*:}" "$gpu"
    done
}

run_queue 0 patchtst:Traffic cyclenet:Traffic itransformer:Traffic & p0=$!
run_queue 1 patchtst:Electricity cyclenet:Electricity & p1=$!
run_queue 2 itransformer:Electricity patchtst:ETTh2 & p2=$!
run_queue 3 cyclenet:ETTh2 itransformer:ETTh2 & p3=$!
status=0
for pid in "$p0" "$p1" "$p2" "$p3"; do wait "$pid" || status=1; done
[[ "$status" -eq 0 ]] || { echo "one or more gain sweep queues failed" >&2; exit "$status"; }
"$PYTHON" tools/summarize_predictive_env_var_gain.py \
  --root "$OUTPUT_ROOT" --feature-base-root "$FEATURE_BASE_ROOT" \
  --backbone-base-root "$BACKBONE_BASE_ROOT"
echo "selected-three gain sweep complete $(date --iso-8601=seconds)"
