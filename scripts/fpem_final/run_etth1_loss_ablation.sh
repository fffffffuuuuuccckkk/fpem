#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
PRED_LEN="${PRED_LEN:?set PRED_LEN to 96, 336, or 720}"
GPU="${GPU:-0}"
REFERENCE_CHECKPOINT="${REFERENCE_CHECKPOINT:?set the matching shared reference checkpoint}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/fpem_loss_ablation_etth1}"
ARCHIVE="${ARCHIVE:-dataset/all_datasets.zip}"

case "$PRED_LEN" in
  96)
    LR_DECOMPOSER=1e-4; LR_ENV_HEAD=1e-4; LR_VARIANT=1e-4 ;;
  336)
    LR_DECOMPOSER=1e-4; LR_ENV_HEAD=2e-5; LR_VARIANT=5e-4 ;;
  720)
    LR_DECOMPOSER=5e-5; LR_ENV_HEAD=2e-5; LR_VARIANT=5e-4 ;;
  *)
    echo "unsupported PRED_LEN=$PRED_LEN" >&2; exit 2 ;;
esac

cd "$PROJECT_DIR"
test -s "$REFERENCE_CHECKPOINT"
ARCHIVE_SHA256="${ARCHIVE_SHA256:-$(sha256sum "$ARCHIVE" | awk '{print $1}')}"
CASE_ROOT="$OUTPUT_ROOT/pred_$PRED_LEN"
mkdir -p "$CASE_ROOT"

run_one() {
  local tag="$1" future_weight="$2" anchor_weight="$3"
  local destination="$CASE_ROOT/$tag"
  if [[ -s "$destination/A2/metrics_and_diagnostics.json" && -e "$destination/run_complete" ]]; then
    echo "reuse complete: $destination"
    return
  fi
  if [[ -e "$destination" ]]; then
    mv "$destination" "${destination}.incomplete.$(date +%Y%m%d_%H%M%S)"
  fi
  mkdir -p "$destination"
  cat > "$destination/protocol.txt" <<EOF
experiment=ETTh1_loss_ablation
dataset=ETTh1
pred_len=$PRED_LEN
seed=2021
K=3
environment_mode=current
decomposer_lr=$LR_DECOMPOSER
environment_classifier_lr=$LR_ENV_HEAD
future_zvar_lr=$LR_VARIANT
gamma_beta_lr=1e-4
reliability_lr=3e-4
lambda_domain=0.1
lambda_future_h=$future_weight
lambda_variant_anchor=$anchor_weight
reference_checkpoint=$REFERENCE_CHECKPOINT
dataset_archive_sha256=$ARCHIVE_SHA256
EOF
  local -a cmd=(env
    GPU="$GPU" BACKBONE=patchtst SEED=2021 ENV_NUM=3 EXPERIMENTS=A2
    DATASET_NAME=ETTh1 DATASET_ARCHIVE_SHA256="$ARCHIVE_SHA256"
    DATA_ROOT=./dataset/all_datasets/ETT-small DATA_CLASS=ett_hour
    DATA_PATH=ETTh1.csv TARGET=OT FREQ=h CYCLENET_CYCLE_LEN=24
    SEQ_LEN=96 PRED_LEN="$PRED_LEN" BATCH_SIZE=32 NUM_WORKERS=0
    REPRESENTATION_CONSTRAINT=classification DECOMPOSITION_TYPE=complementary_gate
    VARIANT_FUSION_MODE=horizon_future_var VARIANT_FUSION_GATE_TYPE=feature
    EPOCHS=10 WARMUP_EPOCHS=3 STAGE_EPOCHS=2
    LAMBDA_INVPRED=1.0 LAMBDA_FUTURE_H="$future_weight"
    LAMBDA_VARIANT_ANCHOR="$anchor_weight" LAMBDA_HORIZON_RELIABILITY=0.0
    FUTURE_PATCH_LEN=16 FUTURE_TEACHER_PATCHES_PER_BATCH=2
    FUTURE_TEACHER_EVAL_PATCH_COUNT=3 LAMBDA_VAR_PREDICTIVE=0.0
    LAMBDA_VAR_UTILITY=0.0 LAMBDA_VAR_GAIN=0.0
    LAMBDA_VAR_CONDITIONAL_GAIN=0.0 LAMBDA_FUTURE_VAR=0.0
    LAMBDA_H_ANCHOR=0.0 LAMBDA_DOMAIN=0.1 PREDICTIVE_ENV_REFACTOR_MODE=current
    REFERENCE_CHECKPOINT="$REFERENCE_CHECKPOINT" REQUIRE_REFERENCE_CHECKPOINT=1
    DIFFERENTIAL_LR=1 FIXED_GAMMA_BETA_LR=1 LR=1e-4 LR_BACKBONE=1e-4
    LR_INV_HEAD=1e-4 LR_DECOMPOSER="$LR_DECOMPOSER" LR_ENV_HEAD="$LR_ENV_HEAD"
    LR_VARIANT="$LR_VARIANT" LR_GAMMA_BETA_BASE=1e-4 LR_RELIABILITY=3e-4
    SAVE_FINAL_CHECKPOINT=0 ENVIRONMENT_QUALITY_DIAGNOSTICS=1
    ENVIRONMENT_QUALITY_FINAL_ONLY=1 OUTPUT="$destination"
    bash scripts/run_predictive_env_iv_patchtst.sh)
  printf '%q ' "${cmd[@]}" > "$destination/exact_command.sh"
  printf '\n' >> "$destination/exact_command.sh"
  echo "run pred=$PRED_LEN tag=$tag future=$future_weight anchor=$anchor_weight GPU=$GPU"
  "${cmd[@]}" 2>&1 | tee "$destination/run.log"
  test -s "$destination/A2/metrics_and_diagnostics.json"
  touch "$destination/run_complete"
}

# The full/full point already exists as the selected best run. Only the three
# missing corners of the 2x2 loss ablation are trained here.
run_one no_future_patch 0.0 1.0
run_one no_anchor 0.1 0.0
run_one no_future_patch_no_anchor 0.0 0.0

"$PYTHON" - "$CASE_ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
lines = ["tag MSE MAE inv_MSE raw_full_MSE full_MSE"]
for path in sorted(root.glob("*/A2/metrics_and_diagnostics.json")):
    data = json.loads(path.read_text())
    lines.append(
        f"{path.parents[1].name} {data['MSE']:.9f} {data['MAE']:.9f} "
        f"{data['inv_MSE']:.9f} {data['raw_full_MSE']:.9f} {data['full_MSE']:.9f}"
    )
(root / "loss_ablation_summary.txt").write_text("\n".join(lines) + "\n")
PY
