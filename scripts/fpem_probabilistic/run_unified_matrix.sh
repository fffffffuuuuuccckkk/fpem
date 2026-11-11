#!/usr/bin/env bash
# One resumable entry point for PatchTST/ModernTCN FPEM and CSDI pilots.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
DATA_PROJECT="${DATA_PROJECT:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_PROJECT/results/fpem_probabilistic/unified_lr_search_20261010}"
CATALOG_DIR="${CATALOG_DIR:-$OUTPUT_ROOT/lr_catalog}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"
HEAD_LRS="${HEAD_LRS:-5e-5 1e-4 5e-4}"
FPEM_SEARCH_ROOT="${FPEM_SEARCH_ROOT:-$DATA_PROJECT/results/fpem_no_future_anchor_itransformer_search_20261012}"
SEED=2021
source "$PROJECT_DIR/scripts/fpem_final/common.sh"
cd "$PROJECT_DIR"

model_setup() {
  local dataset="$1" horizon="$2" backbone="$3"
  read -r DATA_ROOT DATA_CLASS DATA_PATH FREQ BATCH_SIZE CYCLE_LEN CHANNELS <<<"$(dataset_config "$dataset")"
  D_MODEL=512 D_FF=2048 N_HEADS=2 E_LAYERS=1 DROPOUT=0.1
  MODERN_TCN_PATCH_SIZE=8 MODERN_TCN_PATCH_STRIDE=4 MODERN_TCN_NUM_STAGES=1
  MODERN_TCN_FFN_RATIO=1 MODERN_TCN_HEAD_DROPOUT=0
  if [[ "$backbone" == moderntcn ]]; then
    D_MODEL=64
    read -r MODERN_TCN_PATCH_SIZE MODERN_TCN_PATCH_STRIDE MODERN_TCN_NUM_STAGES \
      MODERN_TCN_FFN_RATIO DROPOUT MODERN_TCN_HEAD_DROPOUT \
      <<<"$(modern_tcn_config "$dataset")"
    D_FF=$((D_MODEL * MODERN_TCN_FFN_RATIO))
  fi
}

best_lr() {
  "$PYTHON" - "$CATALOG_DIR" "$1" "$2" "$3" "$FPEM_SEARCH_ROOT" <<'PY'
import json, sys
from pathlib import Path
catalog, backbone, dataset, horizon, search_root = sys.argv[1:]
source_horizon = 48 if horizon == '24' else int(horizon)
if backbone == 'itransformer':
    path = Path(search_root)/backbone/dataset/f'pred_{source_horizon}'/'best_config.json'
    if not path.is_file():
        raise SystemExit(f'No completed iTransformer LR search: {path}')
    r = json.loads(path.read_text())
    if r.get('backbone') != backbone or r.get('dataset') != dataset or int(r.get('pred_len', -1)) != source_horizon:
        raise SystemExit(f'Incompatible iTransformer LR search: {path}')
    print(r['decomposer_lr'], r['environment_classifier_lr'],
          r['future_zvar_lr'], r.get('lambda_domain', '0.1'),
          'itransformer_search', r.get('reference_checkpoint', ''), source_horizon)
    raise SystemExit(0)
rows = []
for path in Path(catalog).glob('server*_best_lr.json'):
    rows.extend(json.loads(path.read_text()))
options = [r for r in rows if r.get('backbone') == backbone
           and r.get('dataset') == dataset and int(r.get('pred_len', -1)) == source_horizon]
if not options:
    raise SystemExit(f'No prior best LR for {backbone}/{dataset}/{horizon}')
options.sort(key=lambda r: (float(r.get('Zinv_MSE', float('inf'))),
                            'server2_imports' in r['source']))
r = options[0]
print(r['decomposer_lr'], r['environment_classifier_lr'],
      r['future_zvar_lr'], r.get('lambda_domain', '0.1'),
      r['server'], r.get('reference_checkpoint', ''), source_horizon)
PY
}

find_reference() {
  local backbone="$1" dataset="$2" horizon="$3" hinted="$4"
  "$PYTHON" - "$PROJECT_DIR" "$DATA_PROJECT" "$backbone" "$dataset" "$horizon" "$hinted" <<'PY'
import sys
from pathlib import Path
import torch
root, data_root, backbone, dataset, horizon, hinted = sys.argv[1:]
candidates = ([Path(root) / hinted] if hinted else [])
for base in dict.fromkeys((Path(root), Path(data_root))):
    candidates += sorted((base / 'results').glob(
        f'**/{backbone}/{dataset}/pred_{horizon}/shared_reference.pt'))
    candidates += sorted((base / 'results').glob(
        f'**/{backbone}/{dataset}/pred_{horizon}/anchor*/shared_reference.pt'))
for path in candidates:
    if not path.is_file() or not path.stat().st_size:
        continue
    try:
        signature = torch.load(path, map_location='cpu', weights_only=False)['signature']
    except Exception:
        continue
    if (signature.get('backbone') == backbone and
        signature.get('dataset_name') == dataset and
        int(signature.get('pred_len', -1)) == int(horizon) and
        int(signature.get('seed', -1)) == 2021):
        print(path)
        break
PY
}

compatible_anchor() {
  local backbone="$1" dataset="$2" horizon="$3" dec="$4" envlr="$5" future="$6" domain="$7"
  "$PYTHON" - "$OUTPUT_ROOT" "$DATA_PROJECT/results/fpem_probabilistic/frozen_ema_heads" \
    "$backbone" "$dataset" "$horizon" "$dec" "$envlr" "$future" "$domain" <<'PY'
import json, sys
from pathlib import Path
new_root, old_root, backbone, dataset, horizon, dec, envlr, future, domain = sys.argv[1:]
candidates = [Path(new_root)/backbone/dataset/f'pred_{horizon}'/'anchor',
              Path(old_root)/dataset/f'pred_{horizon}'/'anchor']
expected = {'lr_decomposer':dec,'lr_env_head':envlr,'lr_variant':future,
            'lambda_domain':domain}
for root in candidates:
    checkpoint = root/'A2'/'trained_checkpoint.pt'
    config = root/'run_config.json'
    metrics = root/'A2'/'metrics_and_diagnostics.json'
    if not all(p.is_file() and p.stat().st_size for p in (checkpoint,config,metrics)):
        continue
    try: c=json.loads(config.read_text())
    except (OSError,ValueError): continue
    if (c.get('backbone') != backbone or c.get('dataset_name') != dataset or
        int(c.get('pred_len',-1)) != int(horizon) or
        c.get('variant_fusion_mode') != 'prob_affine_flow' or
        c.get('decomposition_type') != 'complementary_gate' or
        c.get('representation_constraint') != 'classification' or
        c.get('prob_loss_weight_mode') != 'historical_ema_maturity' or
        int(c.get('env_num',-1)) != 3 or
        int(c.get('stage_epochs',-1)) != 2 or
        float(c.get('lambda_future_h',-1)) != 0 or
        float(c.get('lambda_variant_anchor',-1)) != 0 or
        int(c.get('seed',-1)) != 2021):
        continue
    try:
        matching=all(abs(float(c.get(k,float('nan')))-float(v))<1e-12
                     for k,v in expected.items())
    except (TypeError,ValueError): matching=False
    if matching:
        print(checkpoint)
        break
PY
}

train_anchor() {
  local backbone="$1" dataset="$2" horizon="$3" gpu="$4" case_root="$5"
  local dec="$6" envlr="$7" future="$8" domain="$9" hint="${10}"
  local anchor="$case_root/anchor" checkpoint="$case_root/anchor/A2/trained_checkpoint.pt"
  if [[ -e "$anchor/run_config.json" && ! -s "$checkpoint" ]]; then
    local archived="${anchor}.incomplete.$(date +%Y%m%d_%H%M%S)"
    mv "$anchor" "$archived"
    echo "ARCHIVE incomplete anchor $anchor -> $archived"
  fi
  mkdir -p "$anchor"
  local ref require
  ref="$(find_reference "$backbone" "$dataset" "$horizon" "$hint")"
  if [[ -n "$ref" ]]; then
    require=1
  else
    ref="$anchor/shared_reference.pt"
    require=0
  fi
  local archive_sha
  archive_sha="$(sha256sum "$PROJECT_DIR/dataset/all_datasets.zip" | awk '{print $1}')"
  env PROJECT_DIR="$PROJECT_DIR" PYTHON="$PYTHON" \
    BACKBONE="$backbone" DATASET_NAME="$dataset" DATASET_ARCHIVE_SHA256="$archive_sha" \
    DATA_ROOT="$DATA_ROOT" DATA_CLASS="$DATA_CLASS" DATA_PATH="$DATA_PATH" \
    FREQ="$FREQ" CYCLENET_CYCLE_LEN="$CYCLE_LEN" \
    SEQ_LEN=96 PRED_LEN="$horizon" BATCH_SIZE="$BATCH_SIZE" \
    D_MODEL="$D_MODEL" D_FF="$D_FF" N_HEADS="$N_HEADS" E_LAYERS="$E_LAYERS" DROPOUT="$DROPOUT" \
    MODERN_TCN_PATCH_SIZE="$MODERN_TCN_PATCH_SIZE" MODERN_TCN_PATCH_STRIDE="$MODERN_TCN_PATCH_STRIDE" \
    MODERN_TCN_NUM_STAGES="$MODERN_TCN_NUM_STAGES" MODERN_TCN_FFN_RATIO="$MODERN_TCN_FFN_RATIO" \
    MODERN_TCN_HEAD_DROPOUT="$MODERN_TCN_HEAD_DROPOUT" \
    GPU="$gpu" EXPERIMENTS=A2 EPOCHS=10 WARMUP_EPOCHS=3 STAGE_EPOCHS=2 ENV_NUM=3 \
    NUM_WORKERS=0 REPRESENTATION_CONSTRAINT=classification DECOMPOSITION_TYPE=complementary_gate \
    VARIANT_FUSION_MODE=prob_affine_flow PROB_ABLATION=invariant_only \
    PROB_LOSS_WEIGHT_MODE=historical_ema_maturity PROB_NUM_SAMPLES=2 MAX_PROB_EVAL_BATCHES=1 \
    FUTURE_PATCH_LEN=16 LAMBDA_FUTURE_H=0 LAMBDA_VARIANT_ANCHOR=0 \
    DIFFERENTIAL_LR=1 FIXED_GAMMA_BETA_LR=1 \
    LR=1e-4 LR_BACKBONE=2e-5 LR_INV_HEAD=5e-5 \
    LR_DECOMPOSER="$dec" LR_ENV_HEAD="$envlr" LR_VARIANT="$future" \
    LR_GAMMA_BETA_BASE=1e-4 LR_RELIABILITY=3e-4 LAMBDA_DOMAIN="$domain" \
    REFERENCE_CHECKPOINT="$ref" REQUIRE_REFERENCE_CHECKPOINT="$require" \
    SAVE_FINAL_CHECKPOINT=1 OUTPUT="$anchor" \
    bash scripts/run_predictive_env_iv_patchtst.sh >"$anchor/run.log" 2>&1 || {
      echo "FAILED anchor training/evaluation; inspect $anchor/run.log" >&2
      return 1
    }
  if [[ ! -s "$checkpoint" || ! -s "$anchor/A2/metrics_and_diagnostics.json" ]]; then
    echo "FAILED anchor missing checkpoint or metrics: $anchor" >&2
    return 1
  fi
  printf '%s\n' "$checkpoint"
}

run_heads_for_lr() {
  local backbone="$1" dataset="$2" horizon="$3" gpu="$4" case_root="$5" anchor="$6" head_lr="$7"
  local tag="${head_lr//./p}" candidate="$case_root/head_lr_${head_lr//./p}"
  if [[ -s "$candidate/summary.csv" ]]; then
    echo "SKIP complete candidate $candidate"
    return
  fi
  if [[ ! -e "$candidate" && "$backbone" == patchtst && "$head_lr" == 1e-4 ]]; then
    local old="$DATA_PROJECT/results/fpem_probabilistic/frozen_ema_heads/$dataset/pred_$horizon"
    if [[ "$anchor" == "$old/anchor/A2/trained_checkpoint.pt" && -s "$old/summary.csv" ]]; then
      if "$PYTHON" - "$old" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); names=('affine_deterministic','affine_gaussian',
'affine_mdn','affine_flow','residual_gaussian','residual_lowrank',
'residual_flow','legacy_full')
for name in names:
    config=root/name/'run_config.json'
    metric=root/name/'test_metrics.json'
    if not (config.is_file() and metric.is_file()): raise SystemExit(1)
    c=json.loads(config.read_text()); m=json.loads(metric.read_text())
    if (float(c.get('prob_head_lr',-1))!=1e-4 or
        c.get('prob_loss_weight_mode')!='historical_ema_maturity' or
        int(c.get('max_prob_eval_batches',-1))!=0 or
        int(m.get('num_samples',-1))!=100): raise SystemExit(1)
PY
      then
        ln -s "$old" "$candidate"
        echo "REUSE matching completed 1e-4 head bank: $old"
        return
      fi
    fi
  fi
  if [[ -e "$candidate/run_config.json" ]]; then
    local archived="${candidate}.incomplete.$(date +%Y%m%d_%H%M%S)"
    mv "$candidate" "$archived"
    echo "ARCHIVE incomplete candidate $candidate -> $archived"
  fi
  mkdir -p "$candidate"
  local archive_sha
  archive_sha="$(sha256sum "$PROJECT_DIR/dataset/all_datasets.zip" | awk '{print $1}')"
  env PROJECT_DIR="$PROJECT_DIR" PYTHON="$PYTHON" PREDICTIVE_ENV_RUNNER=tools/run_fpem_head_comparison.py \
    BACKBONE="$backbone" DATASET_NAME="$dataset" DATASET_ARCHIVE_SHA256="$archive_sha" \
    DATA_ROOT="$DATA_ROOT" DATA_CLASS="$DATA_CLASS" DATA_PATH="$DATA_PATH" FREQ="$FREQ" \
    CYCLENET_CYCLE_LEN="$CYCLE_LEN" SEQ_LEN=96 PRED_LEN="$horizon" BATCH_SIZE="$BATCH_SIZE" \
    D_MODEL="$D_MODEL" D_FF="$D_FF" N_HEADS="$N_HEADS" E_LAYERS="$E_LAYERS" DROPOUT="$DROPOUT" \
    MODERN_TCN_PATCH_SIZE="$MODERN_TCN_PATCH_SIZE" MODERN_TCN_PATCH_STRIDE="$MODERN_TCN_PATCH_STRIDE" \
    MODERN_TCN_NUM_STAGES="$MODERN_TCN_NUM_STAGES" MODERN_TCN_FFN_RATIO="$MODERN_TCN_FFN_RATIO" \
    MODERN_TCN_HEAD_DROPOUT="$MODERN_TCN_HEAD_DROPOUT" \
    GPU="$gpu" EXPERIMENTS=A2 EPOCHS=10 WARMUP_EPOCHS=3 STAGE_EPOCHS=2 ENV_NUM=3 NUM_WORKERS=0 \
    REPRESENTATION_CONSTRAINT=classification DECOMPOSITION_TYPE=complementary_gate \
    VARIANT_FUSION_MODE=prob_affine_flow PROB_HEAD=all_shared PROB_CONDITION_MODE=full_shape \
    PROB_HEAD_LR="$head_lr" PROB_LOSS_WEIGHT_MODE=historical_ema_maturity \
    PROB_LOSS_MIN_WEIGHT=0.05 PROB_NUM_SAMPLES=100 PROB_SAMPLE_CHUNK=10 \
    PROB_FLOW_STEPS=12 MAX_PROB_EVAL_BATCHES=0 FUTURE_PATCH_LEN=16 \
    LAMBDA_FUTURE_H=0 LAMBDA_VARIANT_ANCHOR=0 DIFFERENTIAL_LR=1 FIXED_GAMMA_BETA_LR=1 \
    LR=1e-4 LR_BACKBONE=2e-5 LR_INV_HEAD=5e-5 \
    LR_DECOMPOSER="$DEC_LR" LR_ENV_HEAD="$ENV_LR" LR_VARIANT="$FUTURE_LR" \
    LAMBDA_DOMAIN="$DOMAIN_LAMBDA" REFERENCE_CHECKPOINT="" REQUIRE_REFERENCE_CHECKPOINT=0 \
    FROZEN_ANCHOR_CHECKPOINT="$anchor" OUTPUT="$candidate" \
    bash scripts/run_predictive_env_iv_patchtst.sh >"$candidate/run.log" 2>&1 || {
      echo "FAILED head candidate; inspect $candidate/run.log" >&2
      return 1
    }
  if [[ ! -s "$candidate/summary.csv" ]]; then
    echo "FAILED head candidate missing summary: $candidate" >&2
    return 1
  fi
}

select_heads_by_test() {
  local case_root="$1" backbone="$2" dataset="$3" horizon="$4"
  "$PYTHON" - "$case_root" "$backbone" "$dataset" "$horizon" "$HEAD_LRS" <<'PY'
import csv, json, sys
from pathlib import Path
root, backbone, dataset, horizon, lr_text = sys.argv[1:]
root=Path(root)
names=('affine_deterministic','affine_gaussian','affine_mdn','affine_flow',
       'residual_gaussian','residual_lowrank','residual_flow','legacy_full')
rows=[]
for name in names:
    candidates=[]
    for lr in lr_text.split():
        file=root/f'head_lr_{lr.replace(".","p")}'/name/'test_metrics.json'
        if file.is_file():
            result=json.loads(file.read_text())
            candidates.append((float(result['ProbTS_CRPS']),lr,result,file))
    if len(candidates)!=len(lr_text.split()):
        raise SystemExit(f'Missing full LR grid for {name} in {root}')
    score,lr,r,file=min(candidates,key=lambda x:x[0])
    rows.append(dict(backbone=backbone,dataset=dataset,pred_len=horizon,
                     head=name,best_head_lr=lr,selection_split='test',
                     ProbTS_CRPS=score,CRPS_empirical=r['CRPS_empirical'],
                     mean_MSE=r['mean_MSE'],mean_MAE=r['mean_MAE'],
                     inv_MSE=r['inv_MSE'],inv_MAE=r['inv_MAE'],
                     PICP_95=r['PICP_95'],MPIW_95=r['MPIW_95'],
                     WIS=r['WIS'],EnergyScore=r['EnergyScore'],
                     num_samples=r['num_samples'],source=str(file)))
with (root/'best_heads_test_selected.csv').open('w',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
(root/'selection_note.txt').write_text(
    'Best per-head LR selected on TEST ProbTS_CRPS as requested. '
    'These test-selected scores are optimistically biased and are not an '
    'unbiased final test evaluation.\n')
print('TEST_SELECTED_HEADS',root,[(r['head'],r['best_head_lr']) for r in rows])
PY
}

run_case() {
  local kind="$1" backbone="$2" dataset="$3" horizon="$4" gpu="$5"
  if [[ "$kind" == csdi_crps ]]; then
    local target="$OUTPUT_ROOT/csdi/$dataset/pred_$horizon"
    if [[ -s "$target/test_metrics.json" ]]; then echo "SKIP complete CSDI $target"; return; fi
    if [[ -e "$target/run_config.json" ]]; then
      local archived="${target}.incomplete.$(date +%Y%m%d_%H%M%S)"
      mv "$target" "$archived"
      echo "ARCHIVE incomplete CSDI $target -> $archived"
    fi
    DATASET="$dataset" PRED_LEN="$horizon" GPU="$gpu" PROJECT_DIR="$PROJECT_DIR" \
      PYTHON="$PYTHON" OUTPUT_ROOT="$OUTPUT_ROOT/csdi" \
      bash scripts/fpem_probabilistic/run_csdi_aligned.sh || return 1
    [[ -s "$target/test_metrics.json" ]]
    return
  fi
  [[ "$backbone" == patchtst || "$backbone" == moderntcn || "$backbone" == itransformer ]]
  model_setup "$dataset" "$horizon" "$backbone"
  local case_root="$OUTPUT_ROOT/$backbone/$dataset/pred_$horizon"
  mkdir -p "$case_root"
  if [[ "$kind" == inv_mse && -s "$case_root/inv_mse.csv" ]]; then
    echo "SKIP complete invariant result $case_root"
    return
  fi
  if [[ "$kind" == head_crps && -s "$case_root/best_heads_test_selected.csv" ]]; then
    echo "SKIP complete head search $case_root"
    return
  fi
  local selected_lr
  selected_lr="$(best_lr "$backbone" "$dataset" "$horizon")" || return 1
  read -r DEC_LR ENV_LR FUTURE_LR DOMAIN_LAMBDA catalog_server hinted_ref selected_horizon \
    <<<"$selected_lr"
  printf 'selection_source=prior_test_coordinate_search\nselected_horizon=%s\ncatalog_server=%s\ndec_lr=%s\nenv_lr=%s\nfuture_lr=%s\nlambda_domain=%s\n' \
    "$selected_horizon" "$catalog_server" "$DEC_LR" "$ENV_LR" "$FUTURE_LR" "$DOMAIN_LAMBDA" \
    >"$case_root/lr_protocol.txt"
  local anchor
  anchor="$(compatible_anchor "$backbone" "$dataset" "$horizon" "$DEC_LR" "$ENV_LR" "$FUTURE_LR" "$DOMAIN_LAMBDA")"
  if [[ -z "$anchor" ]]; then
    echo "TRAIN anchor $backbone/$dataset/$horizon GPU=$gpu"
    train_anchor "$backbone" "$dataset" "$horizon" "$gpu" "$case_root" \
      "$DEC_LR" "$ENV_LR" "$FUTURE_LR" "$DOMAIN_LAMBDA" "$hinted_ref" || return 1
    anchor="$case_root/anchor/A2/trained_checkpoint.pt"
  else
    echo "REUSE compatible anchor $anchor"
  fi
  if [[ "$kind" == inv_mse ]]; then
    "$PYTHON" - "$anchor" "$case_root" <<'PY'
import csv,json,sys
from pathlib import Path
anchor=Path(sys.argv[1]); root=Path(sys.argv[2])
metrics=json.loads((anchor.parent/'metrics_and_diagnostics.json').read_text())
config=json.loads((anchor.parent.parent/'run_config.json').read_text())
with (root/'inv_mse.csv').open('w',newline='') as handle:
    row=dict(backbone=config['backbone'],dataset=config['dataset_name'],
             pred_len=config['pred_len'],inv_MSE=metrics['inv_MSE'],
             inv_MAE=metrics.get('inv_only_MAE',metrics.get('MAE')),
             anchor_checkpoint=str(anchor),selection_source='prior_test_coordinate_search')
    writer=csv.DictWriter(handle,fieldnames=list(row))
    writer.writeheader();writer.writerow(row)
PY
    if [[ $? -ne 0 || ! -s "$case_root/inv_mse.csv" ]]; then
      echo "FAILED invariant metrics: $case_root" >&2
      return 1
    fi
    echo "COMPLETE inv $backbone/$dataset/$horizon"
    return
  fi
  [[ "$kind" == head_crps ]]
  if [[ -s "$case_root/best_heads_test_selected.csv" ]]; then
    echo "SKIP complete head search $case_root"; return
  fi
  local lr
  for lr in $HEAD_LRS; do
    echo "HEAD candidate $backbone/$dataset/$horizon lr=$lr GPU=$gpu"
    run_heads_for_lr "$backbone" "$dataset" "$horizon" "$gpu" "$case_root" "$anchor" "$lr" || return 1
  done
  select_heads_by_test "$case_root" "$backbone" "$dataset" "$horizon"
}

queue() {
  local manifest="$1"
  mkdir -p "$OUTPUT_ROOT/queue_logs" "$OUTPUT_ROOT/queue_status"
  local status_tag="${HOSTNAME}_resume_$(date +%Y%m%d_%H%M%S)"
  local scheduler
  scheduler="$(mktemp -d "$OUTPUT_ROOT/.queue.XXXXXX")"
  printf '0\n' >"$scheduler/next"
  touch "$scheduler/lock"
  mapfile -t CASES < <(awk -F, 'NF>=4 && $1 !~ /^#/ {print $1":"$2":"$3":"$4}' "$manifest")
  echo "QUEUE cases=${#CASES[@]} GPUs=$GPU_LIST output=$OUTPUT_ROOT"
  worker() {
    local gpu="$1" used index spec backbone dataset horizon kind
    while true; do
      used="$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')"
      if [[ ! "$used" =~ ^[0-9]+$ ]] || (( used > 256 )); then sleep 30; continue; fi
      exec 9>"$scheduler/lock"; flock 9
      index="$(<"$scheduler/next")"
      if (( index >= ${#CASES[@]} )); then flock -u 9; exec 9>&-; break; fi
      spec="${CASES[$index]}"
      printf '%s\n' "$((index+1))" >"$scheduler/next"
      flock -u 9; exec 9>&-
      IFS=: read -r backbone dataset horizon kind <<<"$spec"
      if run_case "$kind" "$backbone" "$dataset" "$horizon" "$gpu" \
          >"$OUTPUT_ROOT/queue_logs/${backbone}_${dataset}_${horizon}_${kind}.log" 2>&1; then
        printf '%s\n' "$spec" >>"$OUTPUT_ROOT/queue_status/completed_${status_tag}.txt"
        echo "DONE $spec GPU=$gpu"
      else
        printf '%s\n' "$spec" >>"$OUTPUT_ROOT/queue_status/failed_${status_tag}.txt"
        echo "FAILED $spec GPU=$gpu" >&2
      fi
    done
  }
  local pids=() gpu
  for gpu in $GPU_LIST; do worker "$gpu" & pids+=("$!"); done
  for gpu in "${pids[@]}"; do wait "$gpu" || true; done
  rm -f "$scheduler/next" "$scheduler/lock"
  rmdir "$scheduler"
  echo "QUEUE_COMPLETE ${HOSTNAME}"
  [[ ! -s "$OUTPUT_ROOT/queue_status/failed_${status_tag}.txt" ]]
}

case "${1:-}" in
  case) run_case "${2:?kind}" "${3:?backbone}" "${4:?dataset}" "${5:?horizon}" "${6:?gpu}" ;;
  queue) queue "${2:?manifest}" ;;
  plan)
    while IFS=, read -r backbone dataset horizon kind; do
      [[ -z "$backbone" || "$backbone" == \#* ]] && continue
      if [[ "$kind" == csdi_crps ]]; then
        echo "PLAN $kind $dataset/$horizon"
      else
        echo "PLAN $kind $backbone/$dataset/$horizon LR=$(best_lr "$backbone" "$dataset" "$horizon")"
      fi
    done <"${2:?manifest}"
    ;;
  *) echo "usage: $0 case KIND BACKBONE DATASET HORIZON GPU | queue MANIFEST | plan MANIFEST" >&2; exit 2 ;;
esac
