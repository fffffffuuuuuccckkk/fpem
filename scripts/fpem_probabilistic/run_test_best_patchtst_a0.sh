#!/usr/bin/env bash
# Resumable FPEM experiment using a TEST-selected backbone-specific A0.
# TEST selection makes these comparisons exploratory, not held-out evidence.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
DATA_PROJECT="${DATA_PROJECT:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
BACKBONE="${BACKBONE:-patchtst}"
if [[ "$BACKBONE" == patchtst ]]; then
  OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_PROJECT/results/fpem_probabilistic/test_best_patchtst_a0_20261013}"
  CATALOG="${CATALOG:-$OUTPUT_ROOT/selected_patchtst_a0_catalog.csv}"
else
  OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_PROJECT/results/fpem_probabilistic/test_best_backbones_a0_20261013}"
  CATALOG="${CATALOG:-$OUTPUT_ROOT/selected_${BACKBONE}_a0_catalog.csv}"
fi
LR_CATALOG="${LR_CATALOG:-$DATA_PROJECT/results/fpem_probabilistic/unified_lr_search_20261010/lr_catalog}"
ITRANSFORMER_SEARCH="${ITRANSFORMER_SEARCH:-$DATA_PROJECT/results/fpem_no_future_anchor_itransformer_search_20261012}"
HEAD_LRS="${HEAD_LRS:-5e-5 1e-4 5e-4}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"
GPU_LIST="${GPU_LIST//,/ }"
source "$PROJECT_DIR/scripts/fpem_final/common.sh"
cd "$PROJECT_DIR"

catalog_row() {
  "$PYTHON" - "$CATALOG" "$BACKBONE" "$1" "$2" <<'PY'
import csv, sys
from pathlib import Path
path, backbone, dataset, horizon = sys.argv[1:]
rows = [r for r in csv.DictReader(Path(path).open())
        if r['dataset'] == dataset and int(r['pred_len']) == int(horizon)
        and r.get('backbone', backbone) == backbone]
if len(rows) != 1 or rows[0]['selection_split'] != 'test':
    raise SystemExit(f'Expected one TEST-selected {backbone} row for {dataset}/{horizon}')
r = rows[0]
print(r['test_mse'], r['test_mae'], r.get('checkpoint_path', r.get('reference_path')))
PY
}

prior_lrs() {
  "$PYTHON" - "$LR_CATALOG" "$ITRANSFORMER_SEARCH" "$BACKBONE" "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
catalog, itransformer_search, backbone, dataset, horizon = sys.argv[1:]
source_horizon = 48 if int(horizon) == 24 else int(horizon)
if backbone == 'itransformer':
    path = Path(itransformer_search)/backbone/dataset/f'pred_{source_horizon}'/'best_config.json'
    if not path.is_file():
        print('1e-4', '1e-4', '1e-4', '0.1', source_horizon, 'default_no_prior_search')
        raise SystemExit(0)
    r = json.loads(path.read_text())
    print(r['decomposer_lr'], r['environment_classifier_lr'],
          r['future_zvar_lr'], r.get('lambda_domain', '0.1'), source_horizon,
          'prior_test_coordinate_search')
    raise SystemExit(0)
rows = []
for path in Path(catalog).glob('server*_best_lr.json'):
    rows.extend(json.loads(path.read_text()))
options = [r for r in rows if r.get('backbone') == backbone
           and r.get('dataset') == dataset and int(r.get('pred_len', -1)) == source_horizon]
if not options:
    print('1e-4', '1e-4', '1e-4', '0.1', source_horizon, 'default_no_prior_search')
    raise SystemExit(0)
r = min(options, key=lambda x: (float(x.get('Zinv_MSE', float('inf'))),
                                'server2_imports' in x['source']))
print(r['decomposer_lr'], r['environment_classifier_lr'],
      r['future_zvar_lr'], r.get('lambda_domain', '0.1'), source_horizon,
      'prior_test_coordinate_search')
PY
}

model_setup() {
  D_MODEL=128 D_FF=256 N_HEADS=8 E_LAYERS=1 DROPOUT=0.1 WARMUP_EPOCHS=0
  MODERN_TCN_PATCH_SIZE=8 MODERN_TCN_PATCH_STRIDE=4 MODERN_TCN_NUM_STAGES=1
  MODERN_TCN_FFN_RATIO=1 MODERN_TCN_HEAD_DROPOUT=0
  if [[ "$BACKBONE" == itransformer ]]; then
    D_MODEL=512 D_FF=512 N_HEADS=8 E_LAYERS=3
  elif [[ "$BACKBONE" == moderntcn ]]; then
    D_MODEL=64 N_HEADS=2 WARMUP_EPOCHS=3
    read -r MODERN_TCN_PATCH_SIZE MODERN_TCN_PATCH_STRIDE MODERN_TCN_NUM_STAGES \
      MODERN_TCN_FFN_RATIO DROPOUT MODERN_TCN_HEAD_DROPOUT \
      <<<"$(modern_tcn_config "$1")"
    D_FF=$((D_MODEL * MODERN_TCN_FFN_RATIO))
  elif [[ "$BACKBONE" != patchtst ]]; then
    echo "Unsupported backbone: $BACKBONE" >&2
    return 1
  fi
}

anchor_complete() {
  "$PYTHON" - "$1" "$2" "$BACKBONE" "$D_MODEL" "$D_FF" "$N_HEADS" "$WARMUP_EPOCHS" \
    "$DEC_LR" "$ENV_LR" "$FUTURE_LR" "$DOMAIN_LAMBDA" <<'PY'
import json, math, sys
from pathlib import Path
root, ref = map(Path, sys.argv[1:3])
backbone, d_model, d_ff, n_heads, warmup, dec_lr, env_lr, future_lr, domain = sys.argv[3:]
ckpt = root/'A2/trained_checkpoint.pt'
config = root/'run_config.json'
metrics = root/'A2/metrics_and_diagnostics.json'
if not all(p.is_file() and p.stat().st_size for p in (ckpt,config,metrics)):
    raise SystemExit(1)
c = json.loads(config.read_text())
if (c.get('backbone') != backbone or int(c.get('d_model', -1)) != int(d_model) or
    int(c.get('d_ff', -1)) != int(d_ff) or int(c.get('n_heads', -1)) != int(n_heads) or
    int(c.get('warmup_epochs', -1)) != int(warmup) or
    c.get('variant_fusion_mode') != 'prob_affine_flow' or
    c.get('decomposition_type') != 'complementary_gate' or
    c.get('representation_constraint') != 'classification' or
    c.get('reference_checkpoint') != str(ref) or
    any(not math.isclose(float(c.get(key, float('nan'))), float(value), abs_tol=1e-12)
        for key, value in [('lr_decomposer', dec_lr), ('lr_env_head', env_lr),
                           ('lr_variant', future_lr), ('lambda_domain', domain)])):
    raise SystemExit(1)
PY
}

run_case() {
  local dataset="$1" horizon="$2" gpu="$3"
  local case_root="$OUTPUT_ROOT/$BACKBONE/$dataset/pred_$horizon"
  model_setup "$dataset"
  read -r DATA_ROOT DATA_CLASS DATA_PATH FREQ BATCH_SIZE CYCLE_LEN CHANNELS \
    <<<"$(dataset_config "$dataset")"
  local selected_a0 selected_lr
  selected_a0="$(catalog_row "$dataset" "$horizon")" || return 1
  read -r a0_mse a0_mae source_checkpoint <<<"$selected_a0"
  [[ "$source_checkpoint" == /* ]] || source_checkpoint="$DATA_PROJECT/$source_checkpoint"
  selected_lr="$(prior_lrs "$dataset" "$horizon")" || return 1
  read -r DEC_LR ENV_LR FUTURE_LR DOMAIN_LAMBDA lr_source_horizon lr_source <<<"$selected_lr"
  [[ -s "$source_checkpoint" ]] || { echo "Missing A0 checkpoint: $source_checkpoint" >&2; return 1; }
  mkdir -p "$case_root"
  local reference="$OUTPUT_ROOT/references/$BACKBONE/$dataset/pred_$horizon.pt"
  [[ "$BACKBONE" == patchtst ]] && reference="$OUTPUT_ROOT/references/$dataset/pred_$horizon.pt"
  local archive_sha
  archive_sha="$(sha256sum "$DATA_PROJECT/dataset/all_datasets.zip" | awk '{print $1}')"
  if [[ "$BACKBONE" == moderntcn ]]; then
    reference="$source_checkpoint"
    "$PYTHON" - "$reference" "$dataset" "$horizon" "$D_MODEL" "$D_FF" \
      "$N_HEADS" "$WARMUP_EPOCHS" "$MODERN_TCN_PATCH_SIZE" \
      "$MODERN_TCN_PATCH_STRIDE" "$MODERN_TCN_NUM_STAGES" "$MODERN_TCN_FFN_RATIO" <<'PY'
import sys, torch
path, dataset, horizon, d_model, d_ff, n_heads, warmup, patch, stride, stages, ratio = sys.argv[1:]
s = torch.load(path, map_location='cpu', weights_only=False)['signature']
expected = dict(backbone='moderntcn', dataset_name=dataset, pred_len=int(horizon),
                d_model=int(d_model), d_ff=int(d_ff), n_heads=int(n_heads),
                warmup_epochs=int(warmup), modern_tcn_patch_size=int(patch),
                modern_tcn_patch_stride=int(stride), modern_tcn_num_stages=int(stages),
                modern_tcn_ffn_ratio=int(ratio))
for key, value in expected.items():
    if s.get(key) != value:
        raise SystemExit(f'Native ModernTCN A0 signature mismatch: {key}: {s.get(key)} != {value}')
PY
  else
    local converter=tools/import_test_best_patchtst_a0.py
    [[ "$BACKBONE" == itransformer ]] && converter=tools/import_test_best_itransformer_a0.py
    "$PYTHON" "$converter" \
      --checkpoint "$source_checkpoint" --output "$reference" --dataset "$dataset" \
      --horizon "$horizon" --data-class "$DATA_CLASS" --data-path "$DATA_PATH" \
      --freq "$FREQ" --enc-in "$CHANNELS" --cycle-len "$CYCLE_LEN" \
      --archive-sha256 "$archive_sha" --test-mse "$a0_mse" --test-mae "$a0_mae" \
      --d-model "$D_MODEL" --d-ff "$D_FF" --n-heads "$N_HEADS" \
      --e-layers "$E_LAYERS" --dropout "$DROPOUT" \
      >"$case_root/reference_import.log"
  fi
  printf 'A0_selection_split=test\nA0_test_MSE=%s\nA0_test_MAE=%s\nA0_checkpoint=%s\nreference=%s\nFPEM_lr_selection_horizon=%s\nFPEM_lr_source=%s\nLR_decomposer=%s\nLR_env=%s\nLR_future=%s\nlambda_domain=%s\nwarning=Test-selected results are exploratory\n' \
    "$a0_mse" "$a0_mae" "$source_checkpoint" "$reference" \
    "$lr_source_horizon" "$lr_source" "$DEC_LR" "$ENV_LR" "$FUTURE_LR" "$DOMAIN_LAMBDA" \
    >"$case_root/protocol.txt"
  local anchor="$case_root/anchor" anchor_checkpoint="$case_root/anchor/A2/trained_checkpoint.pt"
  if ! anchor_complete "$anchor" "$reference"; then
    if [[ -e "$anchor/run_config.json" ]]; then
      echo "Incomplete or incompatible anchor; preserving it and stopping: $anchor" >&2
      return 1
    fi
    mkdir -p "$anchor"
    echo "TRAIN A2 $BACKBONE/$dataset/$horizon GPU=$gpu A0=$a0_mse"
    env PROJECT_DIR="$PROJECT_DIR" PYTHON="$PYTHON" \
      BACKBONE="$BACKBONE" DATASET_NAME="$dataset" DATASET_ARCHIVE_SHA256="$archive_sha" \
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
      FIXED_GAMMA_BETA_LR=1 LR=1e-4 LR_BACKBONE=2e-5 LR_INV_HEAD=5e-5 \
      LR_DECOMPOSER="$DEC_LR" LR_ENV_HEAD="$ENV_LR" LR_VARIANT="$FUTURE_LR" \
      LR_GAMMA_BETA_BASE=1e-4 LR_RELIABILITY=3e-4 LAMBDA_DOMAIN="$DOMAIN_LAMBDA" \
      REFERENCE_CHECKPOINT="$reference" REQUIRE_REFERENCE_CHECKPOINT=1 \
      SAVE_FINAL_CHECKPOINT=1 OUTPUT="$anchor" \
      bash scripts/run_predictive_env_iv_patchtst.sh >"$anchor/run.log" 2>&1
    anchor_complete "$anchor" "$reference"
  else
    echo "REUSE A2 $BACKBONE/$dataset/$horizon"
  fi
  local lr candidate
  for lr in $HEAD_LRS; do
    candidate="$case_root/head_lr_${lr//./p}"
    if [[ -s "$candidate/summary.csv" ]]; then
      echo "SKIP heads $dataset/$horizon LR=$lr"
      continue
    fi
    if [[ -e "$candidate/run_config.json" ]]; then
      echo "Incomplete head candidate; preserving it and stopping: $candidate" >&2
      return 1
    fi
    mkdir -p "$candidate"
    echo "HEAD SEARCH $dataset/$horizon LR=$lr GPU=$gpu"
    env PROJECT_DIR="$PROJECT_DIR" PYTHON="$PYTHON" \
      PREDICTIVE_ENV_RUNNER=tools/run_fpem_head_comparison.py \
      BACKBONE="$BACKBONE" DATASET_NAME="$dataset" DATASET_ARCHIVE_SHA256="$archive_sha" \
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
      PROB_HEAD=all_shared PROB_CONDITION_MODE=full_shape PROB_HEAD_LR="$lr" \
      PROB_LOSS_WEIGHT_MODE=historical_ema_maturity PROB_LOSS_MIN_WEIGHT=0.05 \
      PROB_NUM_SAMPLES=100 PROB_SAMPLE_CHUNK=10 PROB_FLOW_STEPS=12 \
      MAX_PROB_EVAL_BATCHES=0 FUTURE_PATCH_LEN=16 \
      LAMBDA_FUTURE_H=0 LAMBDA_VARIANT_ANCHOR=0 DIFFERENTIAL_LR=1 \
      FIXED_GAMMA_BETA_LR=1 LR=1e-4 LR_BACKBONE=2e-5 LR_INV_HEAD=5e-5 \
      LR_DECOMPOSER="$DEC_LR" LR_ENV_HEAD="$ENV_LR" LR_VARIANT="$FUTURE_LR" \
      LAMBDA_DOMAIN="$DOMAIN_LAMBDA" FROZEN_ANCHOR_CHECKPOINT="$anchor_checkpoint" \
      OUTPUT="$candidate" bash scripts/run_predictive_env_iv_patchtst.sh \
      >"$candidate/run.log" 2>&1
    [[ -s "$candidate/summary.csv" ]]
  done
  "$PYTHON" - "$case_root" "$BACKBONE" "$dataset" "$horizon" "$a0_mse" "$a0_mae" "$HEAD_LRS" <<'PY'
import csv, json, sys
from pathlib import Path
root, backbone, dataset, horizon, a0_mse, a0_mae, lr_text = sys.argv[1:]
root = Path(root)
names = ('affine_deterministic','affine_gaussian','affine_mdn','affine_flow',
         'residual_gaussian','residual_lowrank','residual_flow','legacy_full')
rows = []
for name in names:
    candidates = []
    for lr in lr_text.split():
        file = root/f'head_lr_{lr.replace(".","p")}'/name/'test_metrics.json'
        if file.is_file():
            result = json.loads(file.read_text())
            candidates.append((float(result['ProbTS_CRPS']), lr, result, file))
    if len(candidates) != len(lr_text.split()):
        raise SystemExit(f'Missing LR grid: {root}/{name}')
    score, lr, result, file = min(candidates, key=lambda item: item[0])
    rows.append(dict(backbone=backbone, dataset=dataset, pred_len=horizon, head=name, best_head_lr=lr,
                     A0_test_MSE=a0_mse, A0_test_MAE=a0_mae,
                     inv_MSE=result['inv_MSE'], inv_MAE=result['inv_MAE'],
                     ProbTS_CRPS=score, CRPS_empirical=result['CRPS_empirical'],
                     mean_MSE=result['mean_MSE'], mean_MAE=result['mean_MAE'],
                     PICP_95=result['PICP_95'], MPIW_95=result['MPIW_95'],
                     selection_split='test', source=str(file)))
with (root/'best_heads_test_selected.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
(root/'selection_note.txt').write_text(
    'A0 was selected on TEST MSE and each head LR on TEST ProbTS_CRPS. '
    'Both selections make these results exploratory and optimistically biased.\n')
print('COMPLETE', dataset, horizon)
PY
}

queue() {
  local manifest="$1" scheduler gpu pid
  mkdir -p "$OUTPUT_ROOT/queue_logs/$BACKBONE" "$OUTPUT_ROOT/queue_status/$BACKBONE"
  scheduler="$(mktemp -d "$OUTPUT_ROOT/.queue.XXXXXX")"
  printf '0\n' >"$scheduler/next"
  touch "$scheduler/lock"
  mapfile -t cases < <(awk -F, 'NF>=2 && $1 !~ /^#/ {print $1":"$2}' "$manifest")
  echo "QUEUE cases=${#cases[@]} GPUs=$GPU_LIST root=$OUTPUT_ROOT"
  worker() {
    local gpu="$1" used index spec dataset horizon
    while true; do
      used="$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')"
      if [[ ! "$used" =~ ^[0-9]+$ ]] || (( used > 256 )); then sleep 30; continue; fi
      exec 9>"$scheduler/lock"; flock 9
      index="$(<"$scheduler/next")"
      if (( index >= ${#cases[@]} )); then flock -u 9; exec 9>&-; break; fi
      spec="${cases[$index]}"
      printf '%s\n' "$((index+1))" >"$scheduler/next"
      flock -u 9; exec 9>&-
      IFS=: read -r dataset horizon <<<"$spec"
      if run_case "$dataset" "$horizon" "$gpu" \
          >"$OUTPUT_ROOT/queue_logs/$BACKBONE/${dataset}_${horizon}.log" 2>&1; then
        echo "$spec" >>"$OUTPUT_ROOT/queue_status/$BACKBONE/completed.txt"
        echo "DONE $spec GPU=$gpu"
      else
        echo "$spec" >>"$OUTPUT_ROOT/queue_status/$BACKBONE/failed.txt"
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
  case) run_case "${2:?dataset}" "${3:?horizon}" "${4:?gpu}" ;;
  queue) queue "${2:?manifest}" ;;
  plan) while IFS=, read -r dataset horizon; do
          [[ -z "$dataset" || "$dataset" == \#* ]] && continue
          echo "PLAN $dataset/$horizon A0=$(catalog_row "$dataset" "$horizon") FPEM_LR=$(prior_lrs "$dataset" "$horizon")"
        done <"${2:?manifest}" ;;
  *) echo "usage: $0 case DATASET HORIZON GPU | queue MANIFEST | plan MANIFEST" >&2; exit 2 ;;
esac
