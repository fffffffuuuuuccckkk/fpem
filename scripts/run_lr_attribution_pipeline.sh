#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
ROOT1="results/predictive_env_lr_attribution_etth1_720_l2"
ROOT2="results/predictive_env_feedback_diagnosis_etth1_720_l2"
ROOT3="results/predictive_env_lr_rule_backtest_l2"
REF_ETTH1="results/server2_imports/gpu-47/predictive_env_patchtst_multihorizon_96/pred_720/ETTh1/shared_reference.pt"
REF_ETTM2="results/server2_imports/gpu-47/predictive_env_patchtst_multihorizon_96/pred_336/ETTm2/shared_reference.pt"
REF_EXCHANGE="results/server2_imports/gpu-47/predictive_env_patchtst_multihorizon_96/pred_720/ExchangeRate/shared_reference.pt"

cd "$PROJECT_DIR"

wait_result() {
  local result_file="$1"
  local label="$2"
  local attempt=0
  while [[ ! -s "$result_file" ]]; do
    sleep 30
    attempt=$((attempt + 1))
    if (( attempt >= 240 )); then
      echo "timeout waiting for $label: $result_file" >&2
      exit 10
    fi
  done
}

run_case() {
  local dataset="$1" pred_len="$2" env_num="$3" a0="$4" tag="$5" gpu="$6"
  local backbone_lr="$7" decomposer_lr="$8" head_lr="$9" mode="${10}" root="${11}" reference="${12}"
  DATASET="$dataset" PRED_LEN="$pred_len" ENV_NUM="$env_num" A0_MSE="$a0" \
    RUN_TAG="$tag" GPU="$gpu" LR_BACKBONE="$backbone_lr" \
    LR_DECOMPOSER="$decomposer_lr" LR_INV_HEAD="$head_lr" \
    REFACTOR_MODE="$mode" REFERENCE_CHECKPOINT="$reference" OUTPUT_ROOT="$root" \
    bash scripts/run_lr_attribution_case.sh
}

best_phase1() {
  "$PYTHON" - "$ROOT1" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
candidates = [
    ("reused_diffLR_L2_current", 2e-5, 1e-4, 5e-5),
    ("ETTh1_pred720_k2_decomp2e-5", 2e-5, 2e-5, 5e-5),
    ("ETTh1_pred720_k2_decomp5e-5", 2e-5, 5e-5, 5e-5),
    ("ETTh1_pred720_k2_decomp1e-4_head1e-4", 2e-5, 1e-4, 1e-4),
    ("ETTh1_pred720_k2_decomp1e-4_head2e-4", 2e-5, 1e-4, 2e-4),
    ("ETTh1_pred720_k2_best_head_backbone5e-5", 5e-5, None, None),
    ("ETTh1_pred720_k2_best_head_backbone1e-4", 1e-4, None, None),
]
rows = []
for tag, b, d, h in candidates:
    path = root / tag / "A2" / "metrics_and_diagnostics.json"
    if not path.exists():
        continue
    m = json.loads(path.read_text())
    rows.append((float(m["inv_MSE"]), tag, float(m.get("lr/backbone", b)),
                 float(m.get("lr/decomposer", d or 0)), float(m.get("lr/inv_head", h or 0)), path))
if not rows:
    raise SystemExit("no phase-1 metrics")
inv, tag, b, d, h, path = min(rows)
print(f"{tag}|{b:.12g}|{d:.12g}|{h:.12g}|{inv:.12g}|{path}")
PY
}

write_summary() {
  local phase="$1" output="$2" purpose="$3"
  shift 3
  "$PYTHON" - "$phase" "$output" "$purpose" "$@" <<'PY'
import json, sys
from pathlib import Path

phase, output, purpose, *specs = sys.argv[1:]
rows = []
for spec in specs:
    label, path = spec.split("=", 1)
    p = Path(path)
    if not p.exists():
        continue
    m = json.loads(p.read_text())
    rows.append((label, m))
header = "| config | backbone LR | decomposer LR | head LR | mode | Zinv | raw | gated | gate abs | inv energy | H drift | ARI | NMI | env sim |"
sep = "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
table = [header, sep]
for label, m in rows:
    vals = [
        label, m.get("lr/backbone"), m.get("lr/decomposer"), m.get("lr/inv_head"),
        m.get("predictive_env_refactor_mode"), m.get("inv_MSE"), m.get("raw_full_MSE"),
        m.get("MSE"), m.get("gate/abs_mean"), m.get("inv_energy_ratio"),
        m.get("h_prediction_drift_from_pretrained"), m.get("stage_ARI_to_previous"),
        m.get("stage_NMI_to_previous"), m.get("final_environment_similarity_correlation"),
    ]
    fmt = lambda x: "" if x is None else (f"{x:.6g}" if isinstance(x, (int, float)) else str(x))
    table.append("| " + " | ".join(map(fmt, vals)) + " |")
best = min(rows, key=lambda item: float(item[1]["inv_MSE"])) if rows else None
best_text = "none" if best is None else f"{best[0]} (Zinv={best[1]['inv_MSE']:.6f})"
md = f"""# {phase}

## Experiment purpose

{purpose}

## Modified item

Only the explicitly listed LR or environment-source variable changes between rows.

## Fixed items

Single-stage training, no freeze/unfreeze, L2-gradient maturity, shared pretrained
reference, seed=2021, data split, losses, environment count and model structure.

## Results

{chr(10).join(table)}

## Current conclusion

Best diagnostic row: **{best_text}**. These are development/test diagnostics and
must not be presented as a per-dataset test-selected benchmark.
"""
Path(output).write_text(md)
Path(output).with_name(Path(output).stem + "_comparison.txt").write_text("\n".join(table) + "\n")
PY
}

# Phase 1.1 runs are launched by the caller so this pipeline can be safely resumed.
wait_result "$ROOT1/ETTh1_pred720_k2_decomp2e-5/comparison.txt" "decomposer=2e-5"
wait_result "$ROOT1/ETTh1_pred720_k2_decomp5e-5/comparison.txt" "decomposer=5e-5"

# Phase 1.2: the best Phase 1.1 decomposer is 1e-4 unless the completed metrics say otherwise.
phase11_best="$(best_phase1)"
IFS='|' read -r tag b d h inv path <<< "$phase11_best"
head1="$ROOT1/ETTh1_pred720_k2_decomp1e-4_head1e-4/comparison.txt"
wait_result "$head1" "head=1e-4"
phase12_best="$(best_phase1)"
IFS='|' read -r tag b d h inv path <<< "$phase12_best"
if "$PYTHON" -c "import sys; sys.exit(0 if float(sys.argv[1]) > 0.53 else 1)" "$inv"; then
  run_case ETTh1 720 2 0.521851 decomp1e-4_head2e-4 0 2e-5 1e-4 2e-4 current "$ROOT1" "$REF_ETTH1"
fi

# Phase 1.3: only if the best head/decomposer combination still misses 0.52x.
phase12_best="$(best_phase1)"
IFS='|' read -r tag b d h inv path <<< "$phase12_best"
if "$PYTHON" -c "import sys; sys.exit(0 if float(sys.argv[1]) > 0.53 else 1)" "$inv"; then
  run_case ETTh1 720 2 0.521851 best_head_backbone5e-5 0 5e-5 "$d" "$h" current "$ROOT1" "$REF_ETTH1" &
  pid1=$!
  run_case ETTh1 720 2 0.521851 best_head_backbone1e-4 1 1e-4 "$d" "$h" current "$ROOT1" "$REF_ETTH1" &
  pid2=$!
  wait "$pid1" "$pid2"
fi

best="$(best_phase1)"
IFS='|' read -r best_tag best_b best_d best_h best_inv best_path <<< "$best"
cat > "$ROOT1/phase1_best.env" <<EOF
BEST_TAG=$best_tag
LR_BACKBONE=$best_b
LR_DECOMPOSER=$best_d
LR_INV_HEAD=$best_h
BEST_INV_MSE=$best_inv
BEST_METRICS=$best_path
EOF

specs=()
while IFS= read -r p; do specs+=("$(basename "$(dirname "$(dirname "$p")")")=$p"); done < <(find "$ROOT1" -path '*/A2/metrics_and_diagnostics.json' | sort)
write_summary "Phase 1: ETTh1 LR attribution" "$ROOT1/phase1_summary.md" \
  "Attribute ETTh1-720 Zinv degradation to decomposer, forecast-head, or backbone relative learning rate." "${specs[@]}"

# Phase 2: isolate prediction/environment feedback at the selected LR tuple.
run_case ETTh1 720 2 0.521851 best_lr_h_reference 0 "$best_b" "$best_d" "$best_h" h_reference "$ROOT2" "$REF_ETTH1" &
pid1=$!
run_case ETTh1 720 2 0.521851 best_lr_h_reference_grad_isolated 1 "$best_b" "$best_d" "$best_h" h_reference_grad_isolated "$ROOT2" "$REF_ETTH1" &
pid2=$!
wait "$pid1" "$pid2"
write_summary "Phase 2: ETTh1 environment feedback" "$ROOT2/phase2_summary.md" \
  "Compare current prediction feedback against fixed H-reference environment discovery at the Phase-1 LR tuple." \
  "current=$best_path" \
  "h_reference=$ROOT2/ETTh1_pred720_k2_best_lr_h_reference/A2/metrics_and_diagnostics.json" \
  "h_reference_grad_isolated=$ROOT2/ETTh1_pred720_k2_best_lr_h_reference_grad_isolated/A2/metrics_and_diagnostics.json"

# Phase 3: apply the same ETTh1-derived LR rule without per-dataset retuning.
run_case ETTm2 336 3 0.326629 etth1_rule 0 "$best_b" "$best_d" "$best_h" current "$ROOT3" "$REF_ETTM2" &
pid1=$!
run_case ExchangeRate 720 2 0.876098 etth1_rule 1 "$best_b" "$best_d" "$best_h" current "$ROOT3" "$REF_EXCHANGE" &
pid2=$!
wait "$pid1" "$pid2"
write_summary "Phase 3: unified LR rule backtest" "$ROOT3/phase3_summary.md" \
  "Backtest the ETTh1-derived LR tuple on ETTm2-336 and ExchangeRate-720 without dataset-specific retuning." \
  "ETTm2_current=$ROOT3/reused_current_ETTm2/A2/metrics_and_diagnostics.json" \
  "ETTm2_etth1_rule=$ROOT3/ETTm2_pred336_k3_etth1_rule/A2/metrics_and_diagnostics.json" \
  "Exchange_current=$ROOT3/reused_current_ExchangeRate/A2/metrics_and_diagnostics.json" \
  "Exchange_etth1_rule=$ROOT3/ExchangeRate_pred720_k2_etth1_rule/A2/metrics_and_diagnostics.json"

touch "$ROOT3/phases_1_to_3_complete"
echo "Phases 1-3 complete. Best LR: backbone=$best_b decomposer=$best_d head=$best_h ETTh1_Zinv=$best_inv"
