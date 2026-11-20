#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
SOURCE_ROOT="${SOURCE_ROOT:-results/fpem_no_future_anchor_patchtst_search}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/fpem_etth1_reliability_gate_ablation}"
GPU_LIST="${GPU_LIST:-0 1 2}"
cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_ROOT/logs"

json_value() {
  "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"
}

run_off() {
  local pred_len="$1" gpu="$2"
  local source_case="$SOURCE_ROOT/patchtst/ETTh1/pred_${pred_len}"
  local best="$source_case/best_config.json"
  local output="$OUTPUT_ROOT/pred_${pred_len}/r_off"
  local metrics="$output/A2/metrics_and_diagnostics.json"
  if [[ -s "$metrics" ]]; then
    echo "reuse complete r_off pred_len=$pred_len: $metrics"
    return 0
  fi
  [[ -s "$best" ]] || { echo "missing best config: $best" >&2; return 2; }
  local dec_lr env_lr future_lr reference archive_hash
  dec_lr="$(json_value "$best" decomposer_lr)"
  env_lr="$(json_value "$best" environment_classifier_lr)"
  future_lr="$(json_value "$best" future_zvar_lr)"
  reference="$(json_value "$best" reference_checkpoint)"
  archive_hash="$(json_value "$best" dataset_archive_sha256)"
  echo "r_off pred_len=$pred_len gpu=$gpu dec=$dec_lr env=$env_lr future=$future_lr"
  env \
    GPU="$gpu" BACKBONE=patchtst DATASET_NAME=ETTh1 PRED_LEN="$pred_len" \
    DATA_ROOT=./dataset/all_datasets/ETT-small DATA_CLASS=ett_hour \
    DATA_PATH=ETTh1.csv TARGET=OT FREQ=h CYCLENET_CYCLE_LEN=24 \
    DATASET_ARCHIVE_SHA256="$archive_hash" EXPERIMENTS=A2 \
    EPOCHS=10 WARMUP_EPOCHS=3 BATCH_SIZE=32 NUM_WORKERS=0 \
    ENV_NUM=3 EIIL_STEPS=50 STAGE_EPOCHS=2 LR=1e-4 \
    LR_BACKBONE=1e-4 LR_INV_HEAD=1e-4 LR_DECOMPOSER="$dec_lr" \
    LR_ENV_HEAD="$env_lr" LR_VARIANT="$future_lr" \
    LR_GAMMA_BETA_BASE=1e-4 LR_RELIABILITY=3e-4 \
    RELIABILITY_OBJECTIVE=mse REPRESENTATION_CONSTRAINT=classification \
    DECOMPOSITION_TYPE=complementary_gate GATE_NEAR_ZERO_THRESHOLD=0.1 \
    LAMBDA_GATE_ACTIVITY=0.0001 LAMBDA_COMPLEMENTARY_GATE_ACTIVITY=0.0 \
    LAMBDA_INVPRED=1.0 LAMBDA_FUTURE_VAR=0.0 LAMBDA_FUTURE_H=0.0 \
    LAMBDA_VARIANT_ANCHOR=0.0 LAMBDA_HORIZON_RELIABILITY=0.0 \
    LAMBDA_VAR_PREDICTIVE=0.0 LAMBDA_VAR_UTILITY=0.0 LAMBDA_VAR_GAIN=0.0 \
    LAMBDA_VAR_CONDITIONAL_GAIN=0.0 LAMBDA_H_ANCHOR=0.0 LAMBDA_DOMAIN=0.1 \
    VARIANT_FUSION_MODE=horizon_future_var VARIANT_FUSION_GATE_TYPE=feature \
    HORIZON_RELIABILITY_GATE=off PREDICTIVE_ENV_REFACTOR_MODE=current \
    ENVIRONMENT_MATCHING=overlap ENVIRONMENT_SUPERVISION=predictive \
    DIFFERENTIAL_LR=1 FIXED_GAMMA_BETA_LR=1 \
    ENVIRONMENT_QUALITY_DIAGNOSTICS=1 ENVIRONMENT_QUALITY_FINAL_ONLY=1 \
    RANDOM_PARTITION_REPEATS=1000 REFERENCE_CHECKPOINT="$reference" \
    REQUIRE_REFERENCE_CHECKPOINT=1 OUTPUT="$output" \
    bash scripts/run_predictive_env_iv_patchtst.sh
}

read -r -a gpus <<<"$GPU_LIST"
if (( ${#gpus[@]} < 3 )); then
  echo "GPU_LIST must contain at least three GPU ids" >&2
  exit 2
fi

pids=()
index=0
for pred_len in 96 336 720; do
  run_off "$pred_len" "${gpus[$index]}" \
    > >(tee -a "$OUTPUT_ROOT/logs/pred_${pred_len}_r_off.log") 2>&1 &
  pids+=("$!")
  index=$((index + 1))
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
(( status == 0 )) || exit "$status"

"$PYTHON" - "$SOURCE_ROOT" "$OUTPUT_ROOT" <<'PY'
import csv
import json
import sys
from pathlib import Path

source_root, output_root = map(Path, sys.argv[1:])
rows = []
for pred_len in (96, 336, 720):
    best = json.loads((source_root / "patchtst" / "ETTh1" /
                       f"pred_{pred_len}" / "best_config.json").read_text())
    on_metrics = json.loads(Path(best["metrics_path"]).read_text())
    off_metrics = json.loads((output_root / f"pred_{pred_len}" / "r_off" /
                              "A2" / "metrics_and_diagnostics.json").read_text())
    rows.append({
        "pred_len": pred_len,
        "decomposer_lr": best["decomposer_lr"],
        "environment_lr": best["environment_classifier_lr"],
        "future_zvar_lr": best["future_zvar_lr"],
        "r_on_MSE": on_metrics["MSE"],
        "r_on_MAE": on_metrics["MAE"],
        "r_off_MSE": off_metrics["MSE"],
        "r_off_MAE": off_metrics["MAE"],
        "MSE_off_minus_on": off_metrics["MSE"] - on_metrics["MSE"],
        "r_pred_mean": off_metrics.get("r_pred_mean"),
    })

csv_path = output_root / "reliability_gate_ablation.csv"
with csv_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
(output_root / "reliability_gate_ablation.txt").write_text(
    "\n".join(
        f"ETTh1-{row['pred_len']}: r_on MSE={row['r_on_MSE']:.6f}, "
        f"r_off MSE={row['r_off_MSE']:.6f}, "
        f"off-on={row['MSE_off_minus_on']:+.6f}"
        for row in rows
    ) + "\n"
)
PY
