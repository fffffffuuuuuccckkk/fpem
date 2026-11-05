#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
BEST_CONFIG="${BEST_CONFIG:?set BEST_CONFIG to a completed best_config.json}"
GPU="${GPU:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/fpem_final_environment_controls}"
DRY_RUN="${DRY_RUN:-0}"
cd "$PROJECT_DIR"

eval "$("$PYTHON" - "$BEST_CONFIG" <<'PY'
import json,shlex,sys
d=json.load(open(sys.argv[1]))
keys={'BACKBONE':'backbone','DATASET':'dataset','PRED_LEN':'pred_len','ENV_NUM':'K',
      'MODE':'environment_mode','LR_DEC':'decomposer_lr','LR_ENV':'environment_classifier_lr',
      'LR_FUTURE':'future_zvar_lr','LR_GAMMA':'gamma_beta_lr','LR_R':'reliability_lr'}
for shell,key in keys.items(): print(f'{shell}={shlex.quote(str(d[key]))}')
PY
)"
read -r DATA_ROOT DATA_CLASS DATA_PATH FREQ BATCH_SIZE CYCLE_LEN CHANNELS <<<"$(dataset_config "$DATASET")"
CASE_ROOT="$(dirname "$BEST_CONFIG")"
REFERENCE="$CASE_ROOT/shared_reference.pt"
ARCHIVE_SHA256="${ARCHIVE_SHA256:-$(sha256sum "$ARCHIVE" | awk '{print $1}')}"

for control in predictive random shuffled uniform; do
  out="$OUTPUT_ROOT/$BACKBONE/$DATASET/pred_$PRED_LEN/$control"
  echo "control=$control backbone=$BACKBONE dataset=$DATASET pred=$PRED_LEN GPU=$GPU output=$out"
  [[ "$DRY_RUN" == 1 ]] && continue
  if result_complete "$out"; then echo "reuse $out"; continue; fi
  [[ -e "$out" ]] && mv "$out" "${out}.incomplete.$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$out"
  cat > "$out/protocol.txt" <<EOF
control=$control
source_best_config=$BEST_CONFIG
selection_source=fixed_from_best_predictive_config
EOF
  GPU="$GPU" BACKBONE="$BACKBONE" SEED=2021 ENV_NUM="$ENV_NUM" EXPERIMENTS=A2 \
  DATASET_NAME="$DATASET" DATASET_ARCHIVE_SHA256="$ARCHIVE_SHA256" DATA_ROOT="$DATA_ROOT" \
  DATA_CLASS="$DATA_CLASS" DATA_PATH="$DATA_PATH" TARGET=OT FREQ="$FREQ" \
  CYCLENET_CYCLE_LEN="$CYCLE_LEN" SEQ_LEN=96 PRED_LEN="$PRED_LEN" BATCH_SIZE="$BATCH_SIZE" \
  NUM_WORKERS=0 REPRESENTATION_CONSTRAINT=classification DECOMPOSITION_TYPE=complementary_gate \
  VARIANT_FUSION_MODE=horizon_future_var VARIANT_FUSION_GATE_TYPE=feature EPOCHS=10 \
  WARMUP_EPOCHS=3 STAGE_EPOCHS=2 LAMBDA_INVPRED=1.0 LAMBDA_FUTURE_H=0.1 \
  LAMBDA_HORIZON_RELIABILITY=0.0 FUTURE_PATCH_LEN=16 FUTURE_TEACHER_PATCHES_PER_BATCH=2 \
  FUTURE_TEACHER_EVAL_PATCH_COUNT=3 PREDICTIVE_ENV_REFACTOR_MODE="$MODE" \
  ENVIRONMENT_SUPERVISION="$control" REFERENCE_CHECKPOINT="$REFERENCE" \
  REQUIRE_REFERENCE_CHECKPOINT=1 DIFFERENTIAL_LR=1 FIXED_GAMMA_BETA_LR=1 \
  LR="$(backbone_lr "$BACKBONE")" LR_BACKBONE="$(backbone_lr "$BACKBONE")" \
  LR_INV_HEAD="$(backbone_lr "$BACKBONE")" LR_DECOMPOSER="$LR_DEC" LR_ENV_HEAD="$LR_ENV" \
  LR_VARIANT="$LR_FUTURE" LR_GAMMA_BETA_BASE="$LR_GAMMA" LR_RELIABILITY="$LR_R" \
  SAVE_FINAL_CHECKPOINT=0 ENVIRONMENT_QUALITY_DIAGNOSTICS=1 ENVIRONMENT_QUALITY_FINAL_ONLY=1 \
  OUTPUT="$out" bash scripts/run_predictive_env_iv_patchtst.sh 2>&1 | tee "$out/run.log"
  touch "$out/run_complete"
done
