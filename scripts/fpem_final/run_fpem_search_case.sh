#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

BACKBONE="${BACKBONE:?set BACKBONE}"
DATASET="${DATASET:?set DATASET}"
PRED_LEN="${PRED_LEN:?set PRED_LEN}"
GPU="${GPU:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/fpem_final_patchtst_search}"
DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-1}"
SEARCH_PROFILE="${SEARCH_PROFILE:-compact}"
FIXED_K=3
FIXED_MODE=current
FIXED_GAMMA_BETA_LR=1e-4
FIXED_RELIABILITY_LR=3e-4
FIXED_LAMBDA_FUTURE_H="${FIXED_LAMBDA_FUTURE_H:-0.0}"
FIXED_LAMBDA_VARIANT_ANCHOR="${FIXED_LAMBDA_VARIANT_ANCHOR:-0.0}"
LAMBDA_DOMAIN="${LAMBDA_DOMAIN:-0.1}"
SEARCH_LAMBDA_DOMAIN="${SEARCH_LAMBDA_DOMAIN:-0}"
LAMBDA_DOMAIN_CANDIDATES="${LAMBDA_DOMAIN_CANDIDATES:-0.05 0.1 0.2}"
EXPAND_HIGH_LR_ON_FAILURE="${EXPAND_HIGH_LR_ON_FAILURE:-1}"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_ROOT/$BACKBONE/$DATASET/pred_$PRED_LEN"
CASE_ROOT="$OUTPUT_ROOT/$BACKBONE/$DATASET/pred_$PRED_LEN"
read -r DATA_ROOT DATA_CLASS DATA_PATH FREQ BATCH_SIZE CYCLE_LEN CHANNELS <<<"$(dataset_config "$DATASET")"
BASE_LR="$(backbone_lr "$BACKBONE")"
ARCHIVE_SHA256="${ARCHIVE_SHA256:-$(sha256sum "$ARCHIVE" | awk '{print $1}')}"
REFERENCE="$CASE_ROOT/shared_reference.pt"

declare -A RESULT_DIRS

existing_best_beats_a0() {
  local config="$CASE_ROOT/best_config.json"
  [[ -s "$config" ]] || return 1
  "$PYTHON" - "$config" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
try:
    improved=float(d['Zinv_MSE']) < float(d['A0_MSE'])
except (KeyError, TypeError, ValueError):
    improved=False
raise SystemExit(0 if improved else 1)
PY
}

find_matching_result() {
  local k="$1" dec_lr="$2" env_lr="$3" future_lr="$4" gamma_lr="$5" mode="$6" domain="$7" future_h="$8" anchor="$9"
  "$PYTHON" - "$CASE_ROOT" "$k" "$dec_lr" "$env_lr" "$future_lr" "$gamma_lr" "$mode" "$domain" "$future_h" "$anchor" <<'PY'
import decimal,sys
from pathlib import Path
root=Path(sys.argv[1])
expected=dict(zip(
    ('K','decomposer_lr','environment_classifier_lr','future_zvar_lr',
     'gamma_beta_lr','environment_mode','lambda_domain','lambda_future_h',
     'lambda_variant_anchor'), sys.argv[2:]
))
def number(value):
    try: return decimal.Decimal(str(value)).normalize()
    except (decimal.InvalidOperation, TypeError): return None
def equal(key,left,right):
    if key in {'K','decomposer_lr','environment_classifier_lr','future_zvar_lr',
               'gamma_beta_lr','lambda_domain','lambda_future_h',
               'lambda_variant_anchor'}:
        return number(left)==number(right)
    return str(left)==str(right)
for protocol in sorted(root.glob('*/protocol.txt')):
    values={}
    for line in protocol.read_text(errors='ignore').splitlines():
        if '=' in line:
            key,value=line.split('=',1); values[key]=value
    values.setdefault('lambda_domain','0.1')
    values.setdefault('lambda_future_h','0.1')
    values.setdefault('lambda_variant_anchor','1.0')
    candidate=protocol.parent
    required=(candidate/'A2/metrics_and_diagnostics.json',
              candidate/'run_config.json')
    if (all(equal(k,values.get(k),v) for k,v in expected.items())
            and all(path.is_file() and path.stat().st_size > 0 for path in required)
            and (candidate/'run_complete').exists()):
        print(candidate)
        break
PY
}

print_case() {
  echo "dataset=$DATASET pred_len=$PRED_LEN backbone=$BACKBONE GPU=$GPU"
  echo "reference=$REFERENCE K=$1 dec_lr=$2 env_lr=$3 future_lr=$4 gamma_lr=$5 mode=$6"
  echo "output=$7"
}

find_reusable_reference() {
  "$PYTHON" - "$PROJECT_DIR/results" "$REFERENCE" "$BACKBONE" "$DATASET" \
    "$ARCHIVE_SHA256" "$DATA_CLASS" "$DATA_PATH" "$FREQ" "$CYCLE_LEN" \
    "$PRED_LEN" "$BATCH_SIZE" "$SEED" "$BASE_LR" <<'PY'
import json, sys
from pathlib import Path

(root, target, backbone, dataset, archive_sha, data_class, data_path, freq,
 cycle_len, pred_len, batch_size, seed, learning_rate) = sys.argv[1:]
target = Path(target).resolve()
expected = {
    "backbone": backbone, "dataset_name": dataset,
    "dataset_archive_sha256": archive_sha, "data_class": data_class,
    "data_path": data_path, "target": "OT", "freq": freq,
    "cyclenet_cycle_len": cycle_len, "seq_len": "96",
    "pred_len": pred_len, "batch_size": batch_size, "seed": seed,
    "warmup_epochs": "3", "d_model": "512", "d_ff": "2048",
    "n_heads": "2", "e_layers": "1", "dropout": "0.1",
    "lr": learning_rate,
}

def same(left, right):
    try:
        return abs(float(left) - float(right)) <= 1e-12
    except (TypeError, ValueError):
        return str(left) == str(right)

candidates = sorted(Path(root).rglob("shared_reference.pt"),
                    key=lambda path: path.stat().st_mtime, reverse=True)
for checkpoint in candidates:
    if checkpoint.resolve() == target or not checkpoint.stat().st_size:
        continue
    if checkpoint.parent.name != f"pred_{pred_len}":
        continue
    if checkpoint.parent.parent.name != dataset:
        continue
    if checkpoint.parent.parent.parent.name != backbone:
        continue
    configs = (checkpoint.parent / "reference_prepare" / "run_config.json",
               checkpoint.parent / "A0" / "run_config.json")
    for config in configs:
        if not config.is_file():
            continue
        try:
            values = json.loads(config.read_text())
        except (OSError, ValueError):
            continue
        if all(key in values and same(values[key], value)
               for key, value in expected.items()):
            print(checkpoint)
            raise SystemExit(0)
PY
}

reuse_reference_if_available() {
  [[ -s "$REFERENCE" ]] && return 0
  local source=""
  source="$(find_reusable_reference)"
  [[ -n "$source" ]] || return 0
  mkdir -p "$(dirname "$REFERENCE")"
  cp --reflink=auto -p "$source" "$REFERENCE"
  printf '%s\n' "$source" > "$CASE_ROOT/reference_reuse_source.txt"
  echo "reused compatible reference: $source -> $REFERENCE"
}

find_reusable_a0() {
  "$PYTHON" - "$PROJECT_DIR/results" "$A0_ROOT" "$BACKBONE" "$DATASET" \
    "$ARCHIVE_SHA256" "$DATA_CLASS" "$DATA_PATH" "$FREQ" "$CYCLE_LEN" \
    "$PRED_LEN" "$BATCH_SIZE" "$SEED" "$BASE_LR" <<'PY'
import json, sys
from pathlib import Path

(root, target, backbone, dataset, archive_sha, data_class, data_path, freq,
 cycle_len, pred_len, batch_size, seed, learning_rate) = sys.argv[1:]
target = Path(target).resolve()
expected = {
    "backbone": backbone, "dataset_name": dataset,
    "dataset_archive_sha256": archive_sha, "data_class": data_class,
    "data_path": data_path, "target": "OT", "freq": freq,
    "cyclenet_cycle_len": cycle_len, "seq_len": "96",
    "pred_len": pred_len, "batch_size": batch_size, "seed": seed,
    "warmup_epochs": "3", "epochs": "10", "d_model": "512",
    "d_ff": "2048", "n_heads": "2", "e_layers": "1",
    "dropout": "0.1", "lr": learning_rate, "experiments": "A0",
}

def same(left, right):
    try:
        return abs(float(left) - float(right)) <= 1e-12
    except (TypeError, ValueError):
        return str(left) == str(right)

candidates = sorted(Path(root).rglob("A0/A0/metrics_and_diagnostics.json"),
                    key=lambda path: path.stat().st_mtime, reverse=True)
for metrics in candidates:
    experiment_root = metrics.parents[1]
    if experiment_root.resolve() == target:
        continue
    case_root = experiment_root.parent
    if case_root.name != f"pred_{pred_len}":
        continue
    if case_root.parent.name != dataset or case_root.parent.parent.name != backbone:
        continue
    config = experiment_root / "run_config.json"
    if not config.is_file():
        continue
    try:
        values = json.loads(config.read_text())
    except (OSError, ValueError):
        continue
    if all(key in values and same(values[key], value)
           for key, value in expected.items()):
        print(experiment_root)
        raise SystemExit(0)
PY
}

reuse_a0_if_available() {
  [[ -s "$A0_ROOT/A0/metrics_and_diagnostics.json" ]] && return 0
  local source=""
  source="$(find_reusable_a0)"
  [[ -n "$source" ]] || return 0
  mkdir -p "$A0_ROOT"
  cp -a "$source/." "$A0_ROOT/"
  printf '%s\n' "$source" > "$A0_ROOT/reuse_source.txt"
  echo "reused compatible A0: $source -> $A0_ROOT"
}

prepare_reference() {
  reuse_reference_if_available
  [[ -s "$REFERENCE" ]] && return 0
  local out="$CASE_ROOT/reference_prepare"
  local -a cmd=(env GPU="$GPU" BACKBONE="$BACKBONE" SEED="$SEED" ENV_NUM=2
    EXPERIMENTS=A2 DATASET_NAME="$DATASET" DATASET_ARCHIVE_SHA256="$ARCHIVE_SHA256"
    DATA_ROOT="$DATA_ROOT" DATA_CLASS="$DATA_CLASS" DATA_PATH="$DATA_PATH" TARGET=OT
    FREQ="$FREQ" CYCLENET_CYCLE_LEN="$CYCLE_LEN" SEQ_LEN=96 PRED_LEN="$PRED_LEN"
    BATCH_SIZE="$BATCH_SIZE" NUM_WORKERS=0 EPOCHS=10 WARMUP_EPOCHS=3 STAGE_EPOCHS=2
    PREPARE_REFERENCE_ONLY=1 REFERENCE_CHECKPOINT="$REFERENCE" OUTPUT="$out"
    bash scripts/run_predictive_env_iv_patchtst.sh)
  echo "prepare reference: $(shell_quote_command "${cmd[@]}")"
  [[ "$DRY_RUN" == 1 ]] || "${cmd[@]}"
}

run_candidate() {
  local tag="$1" k="$2" dec_lr="$3" env_lr="$4" future_lr="$5" gamma_lr="$6" mode="$7" domain="${8:-$LAMBDA_DOMAIN}"
  local destination="$CASE_ROOT/$tag"
  RESULT_DIRS["$tag"]="$destination"
  print_case "$k" "$dec_lr" "$env_lr" "$future_lr" "$gamma_lr" "$mode" "$destination"
  if result_complete "$destination"; then
    echo "reuse complete: $destination"
    return 0
  fi
  local matching=""
  matching="$(find_matching_result "$k" "$dec_lr" "$env_lr" "$future_lr" "$gamma_lr" "$mode" "$domain" "$FIXED_LAMBDA_FUTURE_H" "$FIXED_LAMBDA_VARIANT_ANCHOR")"
  if [[ -n "$matching" ]]; then
    RESULT_DIRS["$tag"]="$matching"
    echo "reuse exact historical configuration: $matching"
    return 0
  fi
  if [[ "$DRY_RUN" == 1 ]]; then
    echo "exact command: BACKBONE=$BACKBONE DATASET_NAME=$DATASET PRED_LEN=$PRED_LEN ENV_NUM=$k LR_DECOMPOSER=$dec_lr LR_ENV_HEAD=$env_lr LR_VARIANT=$future_lr LR_GAMMA_BETA_BASE=$gamma_lr LAMBDA_FUTURE_H=$FIXED_LAMBDA_FUTURE_H LAMBDA_VARIANT_ANCHOR=$FIXED_LAMBDA_VARIANT_ANCHOR PREDICTIVE_ENV_REFACTOR_MODE=$mode OUTPUT=$destination bash scripts/run_predictive_env_iv_patchtst.sh"
    return 0
  fi
  [[ "$RESUME" == 1 ]] && archive_incomplete_result "$destination"
  mkdir -p "$destination"
  cat > "$destination/protocol.txt" <<EOF
selection_source=test
backbone=$BACKBONE
dataset=$DATASET
pred_len=$PRED_LEN
seed=$SEED
K=$k
environment_mode=$mode
backbone_lr=$BASE_LR
decomposer_lr=$dec_lr
environment_classifier_lr=$env_lr
future_zvar_lr=$future_lr
gamma_beta_lr=$gamma_lr
reliability_lr=$FIXED_RELIABILITY_LR
lambda_domain=$domain
lambda_future_h=$FIXED_LAMBDA_FUTURE_H
lambda_variant_anchor=$FIXED_LAMBDA_VARIANT_ANCHOR
search_policy=compact_coordinate_v3_no_future_patch_no_anchor
reference_checkpoint=$REFERENCE
dataset_archive_sha256=$ARCHIVE_SHA256
EOF
  local -a cmd=(env GPU="$GPU" BACKBONE="$BACKBONE" SEED="$SEED" ENV_NUM="$k"
    EXPERIMENTS=A2 DATASET_NAME="$DATASET" DATASET_ARCHIVE_SHA256="$ARCHIVE_SHA256"
    DATA_ROOT="$DATA_ROOT" DATA_CLASS="$DATA_CLASS" DATA_PATH="$DATA_PATH" TARGET=OT
    FREQ="$FREQ" CYCLENET_CYCLE_LEN="$CYCLE_LEN" SEQ_LEN=96 PRED_LEN="$PRED_LEN"
    BATCH_SIZE="$BATCH_SIZE" NUM_WORKERS=0 REPRESENTATION_CONSTRAINT=classification
    DECOMPOSITION_TYPE=complementary_gate VARIANT_FUSION_MODE=horizon_future_var
    VARIANT_FUSION_GATE_TYPE=feature EPOCHS=10 WARMUP_EPOCHS=3 STAGE_EPOCHS=2
    LAMBDA_INVPRED=1.0 LAMBDA_FUTURE_H="$FIXED_LAMBDA_FUTURE_H"
    LAMBDA_VARIANT_ANCHOR="$FIXED_LAMBDA_VARIANT_ANCHOR"
    LAMBDA_HORIZON_RELIABILITY=0.0
    FUTURE_PATCH_LEN=16 FUTURE_TEACHER_PATCHES_PER_BATCH=2 FUTURE_TEACHER_EVAL_PATCH_COUNT=3
    LAMBDA_VAR_PREDICTIVE=0.0 LAMBDA_VAR_UTILITY=0.0 LAMBDA_VAR_GAIN=0.0
    LAMBDA_VAR_CONDITIONAL_GAIN=0.0 LAMBDA_FUTURE_VAR=0.0 LAMBDA_H_ANCHOR=0.0
    LAMBDA_DOMAIN="$domain"
    PREDICTIVE_ENV_REFACTOR_MODE="$mode" REFERENCE_CHECKPOINT="$REFERENCE"
    REQUIRE_REFERENCE_CHECKPOINT=1 DIFFERENTIAL_LR=1 FIXED_GAMMA_BETA_LR=1
    LR="$BASE_LR" LR_BACKBONE="$BASE_LR" LR_INV_HEAD="$BASE_LR"
    LR_DECOMPOSER="$dec_lr" LR_ENV_HEAD="$env_lr" LR_VARIANT="$future_lr"
    LR_GAMMA_BETA_BASE="$gamma_lr" LR_RELIABILITY="$FIXED_RELIABILITY_LR"
    SAVE_FINAL_CHECKPOINT=0 ENVIRONMENT_QUALITY_DIAGNOSTICS=1
    ENVIRONMENT_QUALITY_FINAL_ONLY=1 OUTPUT="$destination"
    bash scripts/run_predictive_env_iv_patchtst.sh)
  shell_quote_command "${cmd[@]}" > "$destination/exact_command.sh"
  "${cmd[@]}" 2>&1 | tee "$destination/run.log"
  [[ -s "$destination/A2/metrics_and_diagnostics.json" ]] || return 1
  touch "$destination/run_complete"
}

select_best() {
  local metric="$1"; shift
  "$PYTHON" - "$metric" "$@" <<'PY'
import json, sys
metric, *items = sys.argv[1:]
rows=[]
for item in items:
    tag,path=item.split('|',1)
    d=json.load(open(path+'/A2/metrics_and_diagnostics.json'))
    rows.append((float(d[metric]),tag))
print(min(rows)[1])
PY
}

candidate_items() {
  local tag
  for tag in "$@"; do printf '%s|%s\n' "$tag" "${RESULT_DIRS[$tag]}"; done
}

protocol_value() {
  local result_dir="$1" key="$2"
  awk -F= -v key="$key" '$1 == key {print substr($0, index($0, "=") + 1); exit}' \
    "$result_dir/protocol.txt"
}

not_better_than_a0() {
  local zinv="$1" a0="$2"
  "$PYTHON" - "$zinv" "$a0" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)
PY
}

if existing_best_beats_a0; then
  echo "early stop: reuse existing best_config with Zinv < A0: $CASE_ROOT/best_config.json"
  exit 0
fi

prepare_reference

# A0 is created once per backbone/dataset/horizon and kept outside the search.
A0_ROOT="$CASE_ROOT/A0"
reuse_a0_if_available
if [[ "$DRY_RUN" == 1 ]]; then
  echo "A0 exact command: EXPERIMENTS=A0 BACKBONE=$BACKBONE DATASET_NAME=$DATASET PRED_LEN=$PRED_LEN OUTPUT=$A0_ROOT bash scripts/run_predictive_env_iv_patchtst.sh"
elif [[ ! -s "$A0_ROOT/A0/metrics_and_diagnostics.json" ]]; then
  [[ -e "$A0_ROOT" ]] && mv "$A0_ROOT" "${A0_ROOT}.incomplete.$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$A0_ROOT"
  env GPU="$GPU" BACKBONE="$BACKBONE" SEED="$SEED" ENV_NUM=2 EXPERIMENTS=A0 \
    DATASET_NAME="$DATASET" DATASET_ARCHIVE_SHA256="$ARCHIVE_SHA256" DATA_ROOT="$DATA_ROOT" \
    DATA_CLASS="$DATA_CLASS" DATA_PATH="$DATA_PATH" TARGET=OT FREQ="$FREQ" \
    CYCLENET_CYCLE_LEN="$CYCLE_LEN" SEQ_LEN=96 PRED_LEN="$PRED_LEN" BATCH_SIZE="$BATCH_SIZE" \
    NUM_WORKERS=0 EPOCHS=10 WARMUP_EPOCHS=3 LR="$BASE_LR" \
    REFERENCE_CHECKPOINT="$REFERENCE" REQUIRE_REFERENCE_CHECKPOINT=1 OUTPUT="$A0_ROOT" \
    bash scripts/run_predictive_env_iv_patchtst.sh 2>&1 | tee "$A0_ROOT/run.log"
fi

if [[ "$DRY_RUN" == 1 ]]; then
  echo "compact coordinate policy v3: K=3 mode=current gamma_beta_lr=1e-4 reliability_lr=3e-4 future_h=$FIXED_LAMBDA_FUTURE_H anchor=$FIXED_LAMBDA_VARIANT_ANCHOR"
  echo "dry-run uses dec=1e-4/env=1e-4 as placeholders for later coordinate winners"
  for lr in 5e-5 1e-4; do
    run_candidate "compact_dec_${lr}" "$FIXED_K" "$lr" 1e-4 1e-4 "$FIXED_GAMMA_BETA_LR" "$FIXED_MODE"
  done
  run_candidate "compact_env_2e-5" "$FIXED_K" 1e-4 2e-5 1e-4 "$FIXED_GAMMA_BETA_LR" "$FIXED_MODE"
  run_candidate "compact_future_5e-4" "$FIXED_K" 1e-4 1e-4 5e-4 "$FIXED_GAMMA_BETA_LR" "$FIXED_MODE"
  echo "failure-only high-LR fallback (only when best Zinv MSE >= A0 MSE):"
  run_candidate "compact_fallback_dec_2e-4" "$FIXED_K" 2e-4 1e-4 1e-4 "$FIXED_GAMMA_BETA_LR" "$FIXED_MODE"
  run_candidate "compact_fallback_env_2e-4" "$FIXED_K" 1e-4 2e-4 1e-4 "$FIXED_GAMMA_BETA_LR" "$FIXED_MODE"
  run_candidate "compact_fallback_future_5e-4" "$FIXED_K" 1e-4 1e-4 5e-4 "$FIXED_GAMMA_BETA_LR" "$FIXED_MODE"
  exit 0
fi

# Stage 1: optimize Zinv in the empirically supported low/mid LR range.
dec_tags=()
for lr in 5e-5 1e-4; do
  tag="compact_dec_${lr}"; dec_tags+=("$tag")
  run_candidate "$tag" "$FIXED_K" "$lr" 1e-4 1e-4 "$FIXED_GAMMA_BETA_LR" "$FIXED_MODE"
done
mapfile -t items < <(candidate_items "${dec_tags[@]}")
best_dec_tag="$(select_best inv_MSE "${items[@]}")"; best_dec="${best_dec_tag#compact_dec_}"

env_tags=("$best_dec_tag")
tag="compact_env_2e-5"; env_tags+=("$tag")
run_candidate "$tag" "$FIXED_K" "$best_dec" 2e-5 1e-4 "$FIXED_GAMMA_BETA_LR" "$FIXED_MODE"
mapfile -t items < <(candidate_items "${env_tags[@]}")
best_env_tag="$(select_best inv_MSE "${items[@]}")"
if [[ "$best_env_tag" == "$best_dec_tag" ]]; then
  best_env=1e-4
else
  best_env="${best_env_tag#compact_env_}"
fi

future_tags=("$best_env_tag")
tag=compact_future_5e-4
future_tags+=("$tag")
run_candidate "$tag" "$FIXED_K" "$best_dec" "$best_env" 5e-4 "$FIXED_GAMMA_BETA_LR" "$FIXED_MODE"
mapfile -t items < <(candidate_items "${future_tags[@]}")
best_final_tag="$(select_best inv_MSE "${items[@]}")"
best_dir="${RESULT_DIRS[$best_final_tag]}"

# Stage 4 is failure-only. The 11 completed PatchTST cases selected dec=2e-4
# zero times and env=2e-4 only once, so the high end is no longer paid for by
# default. When the reduced search still fails to beat A0, add one dec and one
# env high-LR probe, then re-coordinate the future LR once. This keeps the
# normal path at four unique FPEM runs and the failure path at at most seven.
a0_mse="$(json_metric "$A0_ROOT/A0/metrics_and_diagnostics.json" MSE)"
best_zinv="$(json_metric "$best_dir/A2/metrics_and_diagnostics.json" inv_MSE)"
if [[ "$EXPAND_HIGH_LR_ON_FAILURE" == 1 ]] && not_better_than_a0 "$best_zinv" "$a0_mse"; then
  echo "reduced search did not beat A0; enabling failure-only 2e-4 probes"
  fallback_tags=("${future_tags[@]}")

  tag=compact_fallback_dec_2e-4
  fallback_tags+=("$tag")
  run_candidate "$tag" "$FIXED_K" 2e-4 1e-4 1e-4 "$FIXED_GAMMA_BETA_LR" "$FIXED_MODE"
  mapfile -t items < <(candidate_items "${dec_tags[@]}" "$tag")
  fallback_best_dec_tag="$(select_best inv_MSE "${items[@]}")"
  fallback_best_dec="$(protocol_value "${RESULT_DIRS[$fallback_best_dec_tag]}" decomposer_lr)"

  tag=compact_fallback_env_2e-4
  fallback_tags+=("$tag")
  run_candidate "$tag" "$FIXED_K" "$fallback_best_dec" 2e-4 1e-4 "$FIXED_GAMMA_BETA_LR" "$FIXED_MODE"

  mapfile -t items < <(candidate_items "${fallback_tags[@]}")
  fallback_base_tag="$(select_best inv_MSE "${items[@]}")"
  fallback_base_dir="${RESULT_DIRS[$fallback_base_tag]}"
  fallback_dec="$(protocol_value "$fallback_base_dir" decomposer_lr)"
  fallback_env="$(protocol_value "$fallback_base_dir" environment_classifier_lr)"

  tag=compact_fallback_future_5e-4
  fallback_tags+=("$tag")
  run_candidate "$tag" "$FIXED_K" "$fallback_dec" "$fallback_env" 5e-4 "$FIXED_GAMMA_BETA_LR" "$FIXED_MODE"
  mapfile -t items < <(candidate_items "${fallback_tags[@]}")
  best_final_tag="$(select_best inv_MSE "${items[@]}")"
  best_dir="${RESULT_DIRS[$best_final_tag]}"
fi

best_dec="$(protocol_value "$best_dir" decomposer_lr)"
best_env="$(protocol_value "$best_dir" environment_classifier_lr)"
best_future="$(protocol_value "$best_dir" future_zvar_lr)"

# Optional failure-only fallback. It is disabled by default and therefore
# does not enlarge the normal search.
best_zinv="$(json_metric "$best_dir/A2/metrics_and_diagnostics.json" inv_MSE)"
if [[ "$SEARCH_LAMBDA_DOMAIN" == 1 ]] && not_better_than_a0 "$best_zinv" "$a0_mse"
then
  domain_tags=("$best_final_tag")
  for domain in $LAMBDA_DOMAIN_CANDIDATES; do
    [[ "$domain" == 0.1 ]] && continue
    tag="compact_domain_${domain}"; domain_tags+=("$tag")
    run_candidate "$tag" "$FIXED_K" "$best_dec" "$best_env" "$best_future" "$FIXED_GAMMA_BETA_LR" "$FIXED_MODE" "$domain"
  done
  mapfile -t items < <(candidate_items "${domain_tags[@]}")
  best_final_tag="$(select_best inv_MSE "${items[@]}")"
  best_dir="${RESULT_DIRS[$best_final_tag]}"
fi

"$PYTHON" - "$A0_ROOT/A0/metrics_and_diagnostics.json" "$best_dir/A2/metrics_and_diagnostics.json" \
  "$CASE_ROOT/best_config.json" "$CASE_ROOT/best_config.txt" "$best_dir/exact_command.sh" <<'PY'
import json,sys,shutil
from pathlib import Path
a0=json.loads(Path(sys.argv[1]).read_text())
m=json.loads(Path(sys.argv[2]).read_text())
protocol={}
for line in (Path(sys.argv[2]).parents[1]/'protocol.txt').read_text().splitlines():
    if '=' in line:
        k,v=line.split('=',1); protocol[k]=v
scores={'Zinv':m['inv_MSE'],'raw':m['raw_full_MSE'],'gated':m['full_MSE']}
source,best=min(scores.items(),key=lambda kv:kv[1])
out={**protocol,'A0_MSE':a0['MSE'],'Zinv_MSE':m['inv_MSE'],'raw_MSE':m['raw_full_MSE'],
     'gated_MSE':m['full_MSE'],'best_final_MSE':best,'best_prediction_source':source,
     'selection_source':'test','metrics_path':sys.argv[2]}
Path(sys.argv[3]).write_text(json.dumps(out,indent=2)+'\n')
Path(sys.argv[4]).write_text('\n'.join(f'{k}={v}' for k,v in out.items())+'\n')
command=Path(sys.argv[5])
destination=Path(sys.argv[3]).with_name('best_command.sh')
if command.is_file():
    shutil.copy2(command, destination)
else:
    destination.write_text('# Exact historical command unavailable; use parameters in best_config.json.\n')
PY
echo "complete case: $BACKBONE $DATASET $PRED_LEN"
