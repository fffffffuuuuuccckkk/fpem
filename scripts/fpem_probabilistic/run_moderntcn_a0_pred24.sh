#!/usr/bin/env bash
# Evaluate staged ModernTCN A0 references at pred=24; train only missing A0.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/fpem_best_a0_code_20261013}"
DATA_PROJECT="${DATA_PROJECT:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/tslib/bin/python}"
ROOT="${ROOT:-$DATA_PROJECT/results/fpem_probabilistic/test_best_backbones_a0_20261013}"
source "$PROJECT_DIR/scripts/fpem_final/common.sh"
dataset="${1:?dataset}" gpu="${2:?gpu}"
read -r DATA_ROOT DATA_CLASS DATA_PATH FREQ BATCH_SIZE CYCLE_LEN CHANNELS \
  <<<"$(dataset_config "$dataset")"
read -r PATCH_SIZE PATCH_STRIDE NUM_STAGES FFN_RATIO DROPOUT HEAD_DROPOUT \
  <<<"$(modern_tcn_config "$dataset")"
out="$ROOT/moderntcn_a0_pred24/$dataset"
reference="$ROOT/references/moderntcn/$dataset/pred_24.pt"
legacy="$ROOT/baseline_references/moderntcn/$dataset/pred_24.pt"
if [[ -s "$out/A0/metrics_and_diagnostics.json" && -s "$reference" ]]; then
  echo "SKIP complete $dataset"
  exit 0
fi
if [[ -e "$out/run_config.json" ]]; then
  echo "Incomplete existing A0 requires inspection: $out" >&2
  exit 1
fi
mkdir -p "$out" "$(dirname "$reference")"
require=1
if [[ ! -s "$reference" ]]; then
  [[ "$dataset" == ETTm1 ]] && require=0
fi
archive_sha="$(sha256sum "$DATA_PROJECT/dataset/all_datasets.zip" | awk '{print $1}')"
if [[ ! -s "$reference" && -s "$legacy" ]]; then
  "$PYTHON" "$PROJECT_DIR/tools/upgrade_legacy_moderntcn_reference.py" \
    --legacy "$legacy" --output "$reference" --dataset "$dataset" --horizon 24 \
    --archive-sha256 "$archive_sha" --data-class "$DATA_CLASS" \
    --data-path "$DATA_PATH" --freq "$FREQ" --enc-in "$CHANNELS" \
    --cycle-len "$CYCLE_LEN" --patch-size "$PATCH_SIZE" \
    --patch-stride "$PATCH_STRIDE" --num-stages "$NUM_STAGES" \
    --ffn-ratio "$FFN_RATIO" --dropout "$DROPOUT" \
    --head-dropout "$HEAD_DROPOUT" >"$out/reference_upgrade.log"
fi
if [[ "$require" == 1 && ! -s "$reference" ]]; then
  echo "Missing staged ModernTCN A0 reference: $reference" >&2
  exit 1
fi
cd "$PROJECT_DIR"
env PROJECT_DIR="$PROJECT_DIR" PYTHON="$PYTHON" BACKBONE=moderntcn \
  DATASET_NAME="$dataset" DATASET_ARCHIVE_SHA256="$archive_sha" \
  DATA_ROOT="$DATA_ROOT" DATA_CLASS="$DATA_CLASS" DATA_PATH="$DATA_PATH" FREQ="$FREQ" \
  CYCLENET_CYCLE_LEN="$CYCLE_LEN" SEQ_LEN=96 PRED_LEN=24 BATCH_SIZE="$BATCH_SIZE" \
  D_MODEL=64 D_FF="$((64*FFN_RATIO))" N_HEADS=2 E_LAYERS=1 DROPOUT="$DROPOUT" \
  MODERN_TCN_PATCH_SIZE="$PATCH_SIZE" MODERN_TCN_PATCH_STRIDE="$PATCH_STRIDE" \
  MODERN_TCN_NUM_STAGES="$NUM_STAGES" MODERN_TCN_FFN_RATIO="$FFN_RATIO" \
  MODERN_TCN_HEAD_DROPOUT="$HEAD_DROPOUT" \
  GPU="$gpu" EXPERIMENTS=A0 EPOCHS=0 WARMUP_EPOCHS=3 NUM_WORKERS=0 \
  REFERENCE_CHECKPOINT="$reference" REQUIRE_REFERENCE_CHECKPOINT="$require" \
  OUTPUT="$out" bash scripts/run_predictive_env_iv_patchtst.sh >"$out/run.log" 2>&1
if [[ ! -s "$reference" && -s "$out/shared_reference.pt" ]]; then
  cp "$out/shared_reference.pt" "$reference"
fi
[[ -s "$out/A0/metrics_and_diagnostics.json" && -s "$reference" ]]
echo "COMPLETE $dataset $(grep -m1 'A0 TEST' "$out/run.log")"
