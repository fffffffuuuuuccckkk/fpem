#!/usr/bin/env bash
# Reuse the selected A0/reference and prior FPEM LRs. Only delay the
# environment classification/GRL loss for five of the existing ten epochs.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
DATA_PROJECT="${DATA_PROJECT:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
PRIOR_ROOT="${PRIOR_ROOT:-$DATA_PROJECT/results/fpem_probabilistic/inv_head_lr_search_20261014}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_PROJECT/results/fpem_probabilistic/grl_warmup5_weak_cases_20261014}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"
GPU_LIST="${GPU_LIST//,/ }"
source "$PROJECT_DIR/scripts/fpem_final/common.sh"
cd "$PROJECT_DIR"

model_setup() {
  local backbone="$1" dataset="$2"
  D_MODEL=128 D_FF=256 N_HEADS=8 E_LAYERS=1 DROPOUT=0.1 WARMUP_EPOCHS=0
  MODERN_TCN_PATCH_SIZE=8 MODERN_TCN_PATCH_STRIDE=4 MODERN_TCN_NUM_STAGES=1
  MODERN_TCN_FFN_RATIO=1 MODERN_TCN_HEAD_DROPOUT=0
  if [[ "$backbone" == itransformer ]]; then
    D_MODEL=512 D_FF=512 N_HEADS=8 E_LAYERS=3
  elif [[ "$backbone" == moderntcn ]]; then
    D_MODEL=64 N_HEADS=2 WARMUP_EPOCHS=3
    read -r MODERN_TCN_PATCH_SIZE MODERN_TCN_PATCH_STRIDE MODERN_TCN_NUM_STAGES \
      MODERN_TCN_FFN_RATIO DROPOUT MODERN_TCN_HEAD_DROPOUT \
      <<<"$(modern_tcn_config "$dataset")"
    D_FF=$((D_MODEL * MODERN_TCN_FFN_RATIO))
  elif [[ "$backbone" != patchtst ]]; then
    echo "Unsupported backbone: $backbone" >&2; return 1
  fi
}

case_metadata() {
  "$PYTHON" - "$PRIOR_ROOT/$1/$2/pred_$3" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1])
best=json.loads((root/'best_inv_head_lr.json').read_text())
protocol=dict(line.split('=',1) for line in (root/'protocol.txt').read_text().splitlines()
              if '=' in line)
if float(best['inv_MSE']) < float(best['A0_test_MSE']):
    raise SystemExit('This case already beats A0; not in weak-case scope: '+str(root))
print(' '.join((best['lr_inv_head'], best['reference'],
                protocol['LR_decomposer'],protocol['LR_env'],
                protocol['LR_future'],protocol['lambda_domain'],
                str(best['A0_test_MSE']),str(best['A0_test_MAE']),
                str(best['inv_MSE']),str(best['inv_MAE']))))
PY
}

case_valid() {
  "$PYTHON" - "$1" "$2" <<'PY'
import json, math, sys
from pathlib import Path
root=Path(sys.argv[1]); lr=float(sys.argv[2])
config=root/'run_config.json'; metrics=root/'A2/metrics_and_diagnostics.json'
epochs=root/'A2/invariant_test_by_epoch.json'
if not all(p.is_file() and p.stat().st_size for p in (config,metrics,epochs)):
    raise SystemExit(1)
c=json.loads(config.read_text()); m=json.loads(metrics.read_text()); e=json.loads(epochs.read_text())
if (c.get('decomposition_warmup_epochs') != 5 or not c.get('track_epoch_inv_test')
    or not math.isclose(float(c['lr_inv_head']),lr,rel_tol=1e-9)
    or len(e) != 10 or not all(math.isfinite(float(v['MSE'])) and
                               math.isfinite(float(v['MAE'])) for v in e)
    or 'best_epoch_inv_MSE' not in m or 'best_epoch_inv_MAE' not in m):
    raise SystemExit(1)
PY
}

run_case() {
  local backbone="$1" dataset="$2" horizon="$3" gpu="$4"
  local root="$OUTPUT_ROOT/$backbone/$dataset/pred_$horizon"
  local lr reference dec env_lr future domain a0_mse a0_mae old_mse old_mae
  read -r lr reference dec env_lr future domain a0_mse a0_mae old_mse old_mae \
    <<<"$(case_metadata "$backbone" "$dataset" "$horizon")"
  if case_valid "$root" "$lr"; then echo "REUSE $backbone/$dataset/$horizon"; return; fi
  if [[ -e "$root/run_config.json" ]]; then
    echo "Partial result preserved for inspection: $root" >&2; return 1
  fi
  [[ -s "$reference" ]] || { echo "Missing A0 reference: $reference" >&2; return 1; }
  model_setup "$backbone" "$dataset"
  read -r DATA_ROOT DATA_CLASS DATA_PATH FREQ BATCH_SIZE CYCLE_LEN CHANNELS \
    <<<"$(dataset_config "$dataset")"
  ARCHIVE_SHA="$(sha256sum "$DATA_PROJECT/dataset/all_datasets.zip" | awk '{print $1}')"
  mkdir -p "$root"
  printf 'backbone=%s\ndataset=%s\npred_len=%s\nA0_MSE=%s\nA0_MAE=%s\nold_best_inv_MSE=%s\nold_best_inv_MAE=%s\nLR_inv_head=%s\nreference=%s\ndecomposition_warmup_epochs=5\nselection_split=test\n' \
    "$backbone" "$dataset" "$horizon" "$a0_mse" "$a0_mae" "$old_mse" "$old_mae" "$lr" "$reference" \
    >"$root/protocol.txt"
  env PROJECT_DIR="$PROJECT_DIR" PYTHON="$PYTHON" \
    BACKBONE="$backbone" DATASET_NAME="$dataset" DATASET_ARCHIVE_SHA256="$ARCHIVE_SHA" \
    DATA_ROOT="$DATA_ROOT" DATA_CLASS="$DATA_CLASS" DATA_PATH="$DATA_PATH" \
    FREQ="$FREQ" CYCLENET_CYCLE_LEN="$CYCLE_LEN" \
    SEQ_LEN=96 PRED_LEN="$horizon" BATCH_SIZE="$BATCH_SIZE" \
    D_MODEL="$D_MODEL" D_FF="$D_FF" N_HEADS="$N_HEADS" E_LAYERS="$E_LAYERS" DROPOUT="$DROPOUT" \
    MODERN_TCN_PATCH_SIZE="$MODERN_TCN_PATCH_SIZE" MODERN_TCN_PATCH_STRIDE="$MODERN_TCN_PATCH_STRIDE" \
    MODERN_TCN_NUM_STAGES="$MODERN_TCN_NUM_STAGES" MODERN_TCN_FFN_RATIO="$MODERN_TCN_FFN_RATIO" \
    MODERN_TCN_HEAD_DROPOUT="$MODERN_TCN_HEAD_DROPOUT" \
    GPU="$gpu" EXPERIMENTS=A2 EPOCHS=10 WARMUP_EPOCHS="$WARMUP_EPOCHS" STAGE_EPOCHS=2 ENV_NUM=3 \
    NUM_WORKERS=0 REPRESENTATION_CONSTRAINT=classification \
    DECOMPOSITION_TYPE=complementary_gate VARIANT_FUSION_MODE=prob_affine_flow \
    PROB_ABLATION=invariant_only PROB_LOSS_WEIGHT_MODE=historical_ema_maturity \
    PROB_NUM_SAMPLES=2 MAX_PROB_EVAL_BATCHES=1 FUTURE_PATCH_LEN=16 \
    LAMBDA_FUTURE_H=0 LAMBDA_VARIANT_ANCHOR=0 DIFFERENTIAL_LR=1 \
    FIXED_GAMMA_BETA_LR=1 LR=1e-4 LR_BACKBONE=2e-5 LR_INV_HEAD="$lr" \
    LR_DECOMPOSER="$dec" LR_ENV_HEAD="$env_lr" LR_VARIANT="$future" \
    LR_GAMMA_BETA_BASE=1e-4 LR_RELIABILITY=3e-4 LAMBDA_DOMAIN="$domain" \
    REFERENCE_CHECKPOINT="$reference" REQUIRE_REFERENCE_CHECKPOINT=1 \
    DECOMPOSITION_WARMUP_EPOCHS=5 TRACK_EPOCH_INV_TEST=1 \
    SAVE_FINAL_CHECKPOINT=1 OUTPUT="$root" \
    bash scripts/run_predictive_env_iv_patchtst.sh >"$root/run.log" 2>&1
  case_valid "$root" "$lr"
  "$PYTHON" - "$root" "$a0_mse" "$a0_mae" "$old_mse" "$old_mae" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); metrics=json.loads((root/'A2/metrics_and_diagnostics.json').read_text())
data={'A0_MSE':float(sys.argv[2]),'A0_MAE':float(sys.argv[3]),
      'prior_best_inv_MSE':float(sys.argv[4]),'prior_best_inv_MAE':float(sys.argv[5]),
      'warmup_best_inv_MSE':metrics['best_epoch_inv_MSE'],
      'warmup_best_inv_MSE_epoch':metrics['best_epoch_inv_MSE_epoch'],
      'warmup_best_inv_MAE':metrics['best_epoch_inv_MAE'],
      'warmup_best_inv_MAE_epoch':metrics['best_epoch_inv_MAE_epoch'],
      'final_inv_MSE':metrics['inv_MSE'],'final_inv_MAE':metrics['inv_only_MAE'],
      'selection_split':'test'}
(root/'comparison.json').write_text(json.dumps(data,indent=2)+'\n')
print('DONE',root,data)
PY
}

queue() {
  local manifest="$1" scheduler gpu pid
  mkdir -p "$OUTPUT_ROOT/queue_logs" "$OUTPUT_ROOT/queue_status"
  scheduler="$(mktemp -d "$OUTPUT_ROOT/.queue.XXXXXX")"
  printf '0\n' >"$scheduler/next"
  touch "$scheduler/lock"
  mapfile -t cases < <(awk -F, 'NF>=3 && $1 !~ /^#/ {print $1":"$2":"$3}' "$manifest")
  echo "QUEUE cases=${#cases[@]} GPUs=$GPU_LIST root=$OUTPUT_ROOT"
  worker() {
    local gpu="$1" used index spec backbone dataset horizon
    while true; do
      used="$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')"
      if [[ ! "$used" =~ ^[0-9]+$ ]] || (( used > 256 )); then sleep 30; continue; fi
      exec 9>"$scheduler/lock"; flock 9
      index="$(<"$scheduler/next")"
      if (( index >= ${#cases[@]} )); then flock -u 9; exec 9>&-; break; fi
      spec="${cases[$index]}"; printf '%s\n' "$((index+1))" >"$scheduler/next"
      flock -u 9; exec 9>&-
      IFS=: read -r backbone dataset horizon <<<"$spec"
      if run_case "$backbone" "$dataset" "$horizon" "$gpu" \
          >"$OUTPUT_ROOT/queue_logs/${backbone}_${dataset}_${horizon}.log" 2>&1; then
        echo "$spec" >>"$OUTPUT_ROOT/queue_status/completed.txt"
        echo "DONE $spec GPU=$gpu"
      else
        echo "$spec" >>"$OUTPUT_ROOT/queue_status/failed.txt"
        echo "FAILED $spec GPU=$gpu" >&2
      fi
    done
  }
  local pids=()
  for gpu in $GPU_LIST; do worker "$gpu" & pids+=("$!"); done
  for pid in "${pids[@]}"; do wait "$pid" || true; done
  rm -f "$scheduler/next" "$scheduler/lock"
  rmdir "$scheduler"
}

case "${1:-}" in
  case) run_case "${2:?backbone}" "${3:?dataset}" "${4:?horizon}" "${5:?gpu}" ;;
  queue) queue "${2:?manifest}" ;;
  plan) while IFS=, read -r backbone dataset horizon; do
          [[ -z "$backbone" || "$backbone" == \#* ]] && continue
          metadata="$(case_metadata "$backbone" "$dataset" "$horizon")"
          read -r lr reference _ <<<"$metadata"
          [[ -s "$reference" ]] || { echo "MISSING_REFERENCE $reference" >&2; exit 1; }
          echo "READY $backbone/$dataset/$horizon LR=$lr $reference"
        done <"${2:?manifest}" ;;
  *) echo "usage: $0 case BACKBONE DATASET HORIZON GPU | queue MANIFEST | plan MANIFEST" >&2; exit 2 ;;
esac
