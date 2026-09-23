#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
WAIT_FOR="results/predictive_env_correction_diagnostics_l2/phase4_phase6_complete"
ROOT="results/predictive_env_variant_lr_coupling_etth1_720"
REFERENCE="results/server2_imports/gpu-47/predictive_env_patchtst_multihorizon_96/pred_720/ETTh1/shared_reference.pt"
BASELINE="results/predictive_env_feedback_diagnosis_etth1_720_l2/ETTh1_pred720_k2_best_lr_h_reference/A2/metrics_and_diagnostics.json"

cd "$PROJECT_DIR"
for _ in $(seq 1 240); do
  [[ -e "$WAIT_FOR" ]] && break
  sleep 30
done
[[ -e "$WAIT_FOR" ]] || { echo "timeout waiting for correction diagnostics" >&2; exit 10; }

# B: slow the upstream future-Zvar representation to the gamma/beta range.
DATASET=ETTh1 PRED_LEN=720 ENV_NUM=2 A0_MSE=0.521851 \
  RUN_TAG=variant_fixed5e-5 GPU=0 LR_BACKBONE=1e-4 \
  LR_DECOMPOSER=1e-4 LR_INV_HEAD=2e-4 LR_VARIANT=5e-5 \
  REFACTOR_MODE=h_reference REFERENCE_CHECKPOINT="$REFERENCE" \
  OUTPUT_ROOT="$ROOT" bash scripts/run_lr_attribution_case.sh &
pid1=$!

# C: future-Zvar and gamma/beta both follow the same detached maturity factor.
ADAPTIVE_VARIANT_LR=1 DATASET=ETTh1 PRED_LEN=720 ENV_NUM=2 A0_MSE=0.521851 \
  RUN_TAG=variant_adaptive GPU=1 LR_BACKBONE=1e-4 \
  LR_DECOMPOSER=1e-4 LR_INV_HEAD=2e-4 LR_VARIANT=1e-4 \
  REFACTOR_MODE=h_reference REFERENCE_CHECKPOINT="$REFERENCE" \
  OUTPUT_ROOT="$ROOT" bash scripts/run_lr_attribution_case.sh &
pid2=$!
wait "$pid1" "$pid2"

"$PYTHON" - "$ROOT" "$BASELINE" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
specs = [
    ("A current variant LR=1e-4", Path(sys.argv[2])),
    ("B fixed variant LR=5e-5", root / "ETTh1_pred720_k2_variant_fixed5e-5/A2/metrics_and_diagnostics.json"),
    ("C maturity-coupled variant LR", root / "ETTh1_pred720_k2_variant_adaptive/A2/metrics_and_diagnostics.json"),
]
columns = [
    ("Zinv", "inv_MSE"), ("raw", "raw_full_MSE"), ("gated", "MSE"),
    ("raw gain", "raw_variant_gain"), ("gated gain", "gated_variant_gain"),
    ("train cosine", "train/correction_cosine_mean"),
    ("test cosine", "test/correction_cosine_mean"),
    ("beneficial", "test/correction_beneficial_ratio"),
    ("variant LR", "lr/variant"), ("variant LR min", "lr/variant_min"),
    ("variant LR max", "lr/variant_max"),
    ("gamma LR", "lr/gamma_beta_mean"),
]
header = "| config | " + " | ".join(label for label, _ in columns) + " |"
separator = "|---|" + "---:|" * len(columns)
table = [header, separator]
loaded = []
for label, path in specs:
    metrics = json.loads(path.read_text())
    loaded.append((label, metrics))
    values = []
    for _, key in columns:
        value = metrics.get(key, float("nan"))
        values.append(f"{value:.6g}" if isinstance(value, (int, float)) else str(value))
    table.append("| " + label + " | " + " | ".join(values) + " |")
best = min(loaded, key=lambda item: float(item[1]["raw_full_MSE"]))
summary = f"""# Phase 5: future-Zvar / gamma-beta LR coupling

## Purpose

Test whether the future-Zvar representation moves too quickly for the slower
gamma/beta generator. No loss, architecture, environment discovery, epoch, seed,
or checkpoint setting changes.

## Results

{chr(10).join(table)}

## Conclusion

Best raw correction row: **{best[0]}**. This remains a diagnostic ETTh1 result;
it is not a per-dataset benchmark selection rule.
"""
(root / "phase5_summary.md").write_text(summary)
(root / "phase5_comparison.txt").write_text("\n".join(table) + "\n")
PY

touch "$ROOT/phase5_complete"
echo "Phase 5 complete"
