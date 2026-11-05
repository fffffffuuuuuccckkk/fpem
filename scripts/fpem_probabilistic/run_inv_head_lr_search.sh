#!/usr/bin/env bash
# MSE-oriented search: keep A0/reference and every non-invariant LR fixed.
# Existing inv-head=5e-5 A2 checkpoints are reused; no probability heads train.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
DATA_PROJECT="${DATA_PROJECT:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
BACKBONE="${BACKBONE:-patchtst}"
INV_HEAD_LRS="${INV_HEAD_LRS:-2e-5 5e-5 1e-4}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"
GPU_LIST="${GPU_LIST//,/ }"
if [[ "$BACKBONE" == patchtst ]]; then
  OLD_ROOT="${OLD_ROOT:-$DATA_PROJECT/results/fpem_probabilistic/test_best_patchtst_a0_20261013}"
else
  OLD_ROOT="${OLD_ROOT:-$DATA_PROJECT/results/fpem_probabilistic/test_best_backbones_a0_20261013}"
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_PROJECT/results/fpem_probabilistic/inv_head_lr_search_20261014}"
CATALOG="${CATALOG:-$OLD_ROOT/selected_${BACKBONE}_a0_catalog.csv}"
LR_CATALOG="${LR_CATALOG:-$DATA_PROJECT/results/fpem_probabilistic/unified_lr_search_20261010/lr_catalog}"
ITRANSFORMER_SEARCH="${ITRANSFORMER_SEARCH:-$DATA_PROJECT/results/fpem_no_future_anchor_itransformer_search_20261012}"
source "$PROJECT_DIR/scripts/fpem_final/common.sh"
cd "$PROJECT_DIR"

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

case_metadata() {
  "$PYTHON" - "$OLD_ROOT" "$CATALOG" "$LR_CATALOG" \
    "$ITRANSFORMER_SEARCH" "$BACKBONE" "$1" "$2" <<'PY'
import csv, json, sys
from pathlib import Path
old, catalog, lr_catalog, itr_search = map(Path, sys.argv[1:5])
backbone, dataset, horizon = sys.argv[5:]
source_horizon = 48 if int(horizon) == 24 else int(horizon)
rows = [r for r in csv.DictReader(catalog.open())
        if r['dataset'] == dataset and int(r['pred_len']) == int(horizon)
        and r.get('backbone', backbone) == backbone]
if len(rows) != 1 or rows[0]['selection_split'] != 'test':
    raise SystemExit('Expected exactly one TEST-selected A0: '+str((backbone,dataset,horizon)))
r = rows[0]
checkpoint = Path(r.get('checkpoint_path') or r.get('reference_path'))
if not checkpoint.is_absolute():
    checkpoint = Path('/data/OuXiaoyu/Time-Series-Library-FPEM') / checkpoint
if backbone == 'moderntcn':
    reference = checkpoint
elif backbone == 'patchtst':
    reference = old/'references'/dataset/f'pred_{horizon}.pt'
else:
    reference = old/'references'/backbone/dataset/f'pred_{horizon}.pt'
if backbone == 'itransformer':
    prior = Path(itr_search)/backbone/dataset/f'pred_{source_horizon}'/'best_config.json'
    best = json.loads(prior.read_text()) if prior.is_file() else None
else:
    choices = []
    for path in lr_catalog.glob('server*_best_lr.json'):
        choices.extend(json.loads(path.read_text()))
    choices = [v for v in choices if v.get('backbone') == backbone
               and v.get('dataset') == dataset
               and int(v.get('pred_len', -1)) == source_horizon]
    best = min(choices, key=lambda v: (float(v.get('Zinv_MSE', float('inf'))),
                                    'server2_imports' in v['source'])) if choices else None
if best is None:
    dec = env = future = '1e-4'
    domain = '0.1'
    prior_source = 'default_no_prior_search'
else:
    dec = str(best['decomposer_lr'])
    env = str(best['environment_classifier_lr'])
    future = str(best['future_zvar_lr'])
    domain = str(best.get('lambda_domain', '0.1'))
    prior_source = 'prior_test_coordinate_search'
old_protocol = old/backbone/dataset/f'pred_{horizon}'/'protocol.txt'
if old_protocol.is_file():
    entries = dict(line.split('=', 1) for line in old_protocol.read_text().splitlines()
                   if '=' in line)
    # Reuse exactly what the older 5e-5 anchor actually trained with.
    reference = Path(entries['reference'])
    dec, env, future, domain = (entries['LR_decomposer'], entries['LR_env'],
                                entries['LR_future'], entries['lambda_domain'])
    prior_source = entries.get('FPEM_lr_source', prior_source)
if not checkpoint.is_file():
    raise SystemExit('Missing selected A0 checkpoint: '+str(checkpoint))
print(' '.join((str(r['test_mse']), str(r['test_mae']), str(reference),
                dec, env, future, domain, prior_source, str(checkpoint))))
PY
}

ensure_reference() {
  if [[ -s "$reference" ]]; then return; fi
  if [[ "$BACKBONE" == moderntcn ]]; then
    echo "Missing native ModernTCN A0 reference: $reference" >&2
    return 1
  fi
  local converter=tools/import_test_best_patchtst_a0.py
  [[ "$BACKBONE" == itransformer ]] && converter=tools/import_test_best_itransformer_a0.py
  mkdir -p "$(dirname "$reference")" "$case_root"
  "$PYTHON" "$converter" \
    --checkpoint "$source_checkpoint" --output "$reference" --dataset "$dataset" \
    --horizon "$horizon" --data-class "$DATA_CLASS" --data-path "$DATA_PATH" \
    --freq "$FREQ" --enc-in "$CHANNELS" --cycle-len "$CYCLE_LEN" \
    --archive-sha256 "$ARCHIVE_SHA" --test-mse "$a0_mse" --test-mae "$a0_mae" \
    --d-model "$D_MODEL" --d-ff "$D_FF" --n-heads "$N_HEADS" \
    --e-layers "$E_LAYERS" --dropout "$DROPOUT" \
    >"$case_root/reference_import.log"
  [[ -s "$reference" ]]
}

candidate_valid() {
  "$PYTHON" - "$1" "$2" "$3" "$BACKBONE" "$D_MODEL" \
    "$DEC_LR" "$ENV_LR" "$FUTURE_LR" "$DOMAIN_LAMBDA" <<'PY'
import json, math, sys
from pathlib import Path
root, reference = map(Path, sys.argv[1:3])
lr, backbone, dim, dec, env, future, domain = sys.argv[3:]
config = root/'run_config.json'
metrics = root/'A2/metrics_and_diagnostics.json'
checkpoint = root/'A2/trained_checkpoint.pt'
if not all(p.is_file() and p.stat().st_size > 0 for p in (config,metrics,checkpoint)):
    raise SystemExit(1)
c, m = json.loads(config.read_text()), json.loads(metrics.read_text())
expected = {'lr_inv_head':lr, 'lr_decomposer':dec, 'lr_env_head':env,
            'lr_variant':future, 'lambda_domain':domain}
if (c.get('backbone') != backbone or int(c.get('d_model',-1)) != int(dim)
    or c.get('reference_checkpoint') != str(reference)
    or c.get('prob_ablation') != 'invariant_only'
    or c.get('decomposition_type') != 'complementary_gate'
    or c.get('representation_constraint') != 'classification'
    or any(not math.isclose(float(c.get(k,float('nan'))), float(v),
                             rel_tol=1e-9, abs_tol=1e-12) for k,v in expected.items())
    or not math.isfinite(float(m.get('inv_MSE',float('nan'))))):
    raise SystemExit(1)
PY
}

run_candidate() {
  local root="$1" lr="$2" gpu="$3" dataset="$4" horizon="$5" reference="$6"
  if candidate_valid "$root" "$reference" "$lr"; then
    echo "REUSE A2 $BACKBONE/$dataset/$horizon LR=$lr"
    return
  fi
  if [[ -e "$root/run_config.json" ]]; then
    echo "Incomplete/incompatible candidate preserved; inspect before retry: $root" >&2
    return 1
  fi
  mkdir -p "$root"
  echo "TRAIN A2 $BACKBONE/$dataset/$horizon LR_inv_head=$lr GPU=$gpu"
  env PROJECT_DIR="$PROJECT_DIR" PYTHON="$PYTHON" \
    BACKBONE="$BACKBONE" DATASET_NAME="$dataset" DATASET_ARCHIVE_SHA256="$ARCHIVE_SHA" \
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
    LR_DECOMPOSER="$DEC_LR" LR_ENV_HEAD="$ENV_LR" LR_VARIANT="$FUTURE_LR" \
    LR_GAMMA_BETA_BASE=1e-4 LR_RELIABILITY=3e-4 LAMBDA_DOMAIN="$DOMAIN_LAMBDA" \
    REFERENCE_CHECKPOINT="$reference" REQUIRE_REFERENCE_CHECKPOINT=1 \
    SAVE_FINAL_CHECKPOINT=1 OUTPUT="$root" \
    bash scripts/run_predictive_env_iv_patchtst.sh >"$root/run.log" 2>&1
  candidate_valid "$root" "$reference" "$lr"
}

run_case() {
  local dataset="$1" horizon="$2" gpu="$3" lr root old_anchor case_root base_reused=0
  model_setup "$dataset"
  read -r DATA_ROOT DATA_CLASS DATA_PATH FREQ BATCH_SIZE CYCLE_LEN CHANNELS \
    <<<"$(dataset_config "$dataset")"
  read -r a0_mse a0_mae reference DEC_LR ENV_LR FUTURE_LR DOMAIN_LAMBDA prior_source source_checkpoint \
    <<<"$(case_metadata "$dataset" "$horizon")"
  ARCHIVE_SHA="$(sha256sum "$DATA_PROJECT/dataset/all_datasets.zip" | awk '{print $1}')"
  case_root="$OUTPUT_ROOT/$BACKBONE/$dataset/pred_$horizon"
  mkdir -p "$case_root"
  ensure_reference
  printf 'backbone=%s\ndataset=%s\npred_len=%s\nA0_test_MSE=%s\nA0_test_MAE=%s\nreference=%s\nLR_decomposer=%s\nLR_env=%s\nLR_future=%s\nlambda_domain=%s\nprior_source=%s\nselection_split=test\nwarning=TEST-based LR selection is exploratory\n' \
    "$BACKBONE" "$dataset" "$horizon" "$a0_mse" "$a0_mae" "$reference" \
    "$DEC_LR" "$ENV_LR" "$FUTURE_LR" "$DOMAIN_LAMBDA" "$prior_source" \
    >"$case_root/protocol.txt"
  old_anchor="$OLD_ROOT/$BACKBONE/$dataset/pred_$horizon/anchor"
  for lr in $INV_HEAD_LRS; do
    root="$case_root/inv_head_lr_${lr//./p}"
    if [[ "$lr" == 5e-5 ]] && candidate_valid "$old_anchor" "$reference" "$lr"; then
      echo "REUSE previous A2 $BACKBONE/$dataset/$horizon LR=$lr"
      base_reused=1
      continue
    fi
    run_candidate "$root" "$lr" "$gpu" "$dataset" "$horizon" "$reference"
  done
  "$PYTHON" - "$case_root" "$old_anchor" "$reference" "$BACKBONE" "$dataset" \
    "$horizon" "$a0_mse" "$a0_mae" "$INV_HEAD_LRS" "$base_reused" <<'PY'
import csv, json, math, sys
from pathlib import Path
root, old, reference = map(Path,sys.argv[1:4])
backbone,dataset,horizon,a0_mse,a0_mae,lr_text,base_reused=sys.argv[4:]
rows=[]
for lr in lr_text.split():
    candidate=(old if lr=='5e-5' and base_reused=='1'
               else root/f'inv_head_lr_{lr.replace(".","p")}')
    metric=candidate/'A2/metrics_and_diagnostics.json'
    if not metric.is_file():
        raise SystemExit('Missing full TEST metrics: '+str(metric))
    score=json.loads(metric.read_text())
    mse,mae=float(score['inv_MSE']),float(score['inv_only_MAE'])
    if not math.isfinite(mse) or not math.isfinite(mae):
        raise SystemExit('Non-finite inv metrics: '+str(metric))
    rows.append(dict(backbone=backbone,dataset=dataset,pred_len=int(horizon),
                     lr_inv_head=lr,A0_test_MSE=float(a0_mse),A0_test_MAE=float(a0_mae),
                     inv_MSE=mse,inv_MAE=mae,gain_vs_A0=float(a0_mse)-mse,
                     checkpoint=str(candidate/'A2/trained_checkpoint.pt'),
                     source_metrics=str(metric),reference=str(reference),
                     selection_split='test'))
with (root/'candidate_metrics.csv').open('w',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows(rows)
best=min(rows,key=lambda item:item['inv_MSE'])
(root/'best_inv_head_lr.json').write_text(json.dumps(best,indent=2)+'\n')
print('BEST',backbone,dataset,horizon,best['lr_inv_head'],best['inv_MSE'])
PY
}

stage_reference_case() {
  dataset="$1" horizon="$2"
  model_setup "$dataset"
  read -r DATA_ROOT DATA_CLASS DATA_PATH FREQ BATCH_SIZE CYCLE_LEN CHANNELS \
    <<<"$(dataset_config "$dataset")"
  read -r a0_mse a0_mae reference DEC_LR ENV_LR FUTURE_LR DOMAIN_LAMBDA prior_source source_checkpoint \
    <<<"$(case_metadata "$dataset" "$horizon")"
  ARCHIVE_SHA="$(sha256sum "$DATA_PROJECT/dataset/all_datasets.zip" | awk '{print $1}')"
  case_root="$OUTPUT_ROOT/$BACKBONE/$dataset/pred_$horizon"
  ensure_reference
  echo "REFERENCE_READY $BACKBONE/$dataset/$horizon $reference"
}

queue() {
  local manifest="$1" scheduler gpu pid
  mkdir -p "$OUTPUT_ROOT/queue_logs/$BACKBONE" "$OUTPUT_ROOT/queue_status/$BACKBONE"
  scheduler="$(mktemp -d "$OUTPUT_ROOT/.queue.XXXXXX")"
  printf '0\n' >"$scheduler/next"
  touch "$scheduler/lock"
  mapfile -t cases < <(awk -F, 'NF>=2 && $1 !~ /^#/ {print $1":"$2}' "$manifest")
  echo "QUEUE backbone=$BACKBONE cases=${#cases[@]} GPUs=$GPU_LIST root=$OUTPUT_ROOT"
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
        echo "DONE $BACKBONE $spec GPU=$gpu"
      else
        echo "$spec" >>"$OUTPUT_ROOT/queue_status/$BACKBONE/failed.txt"
        echo "FAILED $BACKBONE $spec GPU=$gpu" >&2
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
  reference) stage_reference_case "${2:?dataset}" "${3:?horizon}" ;;
  queue) queue "${2:?manifest}" ;;
  plan) while IFS=, read -r dataset horizon; do
          [[ -z "$dataset" || "$dataset" == \#* ]] && continue
          echo "PLAN $BACKBONE/$dataset/$horizon $(case_metadata "$dataset" "$horizon")"
        done <"${2:?manifest}" ;;
  *) echo "usage: $0 case DATASET HORIZON GPU | queue MANIFEST | plan MANIFEST" >&2; exit 2 ;;
esac
