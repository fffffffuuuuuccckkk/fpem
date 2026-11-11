#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
ROOT="results/predictive_env_reliability_environment_disagreement_etth1_720"
REFERENCE="results/server2_imports/gpu-47/predictive_env_patchtst_multihorizon_96/pred_720/ETTh1/shared_reference.pt"
BASELINE="results/predictive_env_reliability_objectives_etth1_720/ETTh1_pred720_k2_reliability_mse_usefulness/A2/metrics_and_diagnostics.json"

cd "$PROJECT_DIR"
mkdir -p "$ROOT"

RELIABILITY_ENVIRONMENT_DISAGREEMENT=1 \
RELIABILITY_OBJECTIVE=mse_usefulness \
DATASET=ETTh1 PRED_LEN=720 ENV_NUM=2 A0_MSE=0.521851 \
RUN_TAG=reliability_env_disagreement GPU=0 LR_BACKBONE=1e-4 \
LR_DECOMPOSER=1e-4 LR_INV_HEAD=2e-4 LR_VARIANT=5e-5 \
REFACTOR_MODE=h_reference REFERENCE_CHECKPOINT="$REFERENCE" \
OUTPUT_ROOT="$ROOT" bash scripts/run_lr_attribution_case.sh

"$PYTHON" - "$ROOT" "$BASELINE" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
specs = [
    ("A usefulness supervision", Path(sys.argv[2])),
    (
        "B usefulness + environment disagreement",
        root / "ETTh1_pred720_k2_reliability_env_disagreement/A2/metrics_and_diagnostics.json",
    ),
]
columns = [
    ("Zinv", "inv_MSE"), ("raw", "raw_full_MSE"), ("gated", "MSE"),
    ("raw gain", "raw_variant_gain"), ("gated gain", "gated_variant_gain"),
    ("corr(r,r*)", "corr_r_pred_r_target"),
    ("useful acc", "reliability_usefulness_accuracy"),
    ("D mean", "reliability_environment_disagreement_mean"),
    ("D p90", "reliability_environment_disagreement_p90"),
]
table = [
    "| config | " + " | ".join(label for label, _ in columns) + " |",
    "|---|" + "---:|" * len(columns),
]
loaded = []
for label, path in specs:
    metrics = json.loads(path.read_text())
    loaded.append((label, metrics))
    table.append("| " + label + " | " + " | ".join(
        f"{metrics.get(key, float('nan')):.6g}" for _, key in columns
    ) + " |")
best_corr = max(loaded, key=lambda item: item[1].get("corr_r_pred_r_target", -999))
best_gated = min(loaded, key=lambda item: item[1]["MSE"])
summary = f"""# Phase 8: environment-disagreement reliability feature

## Purpose

Test the FPEM-specific hypothesis that disagreement among TRAIN-only,
environment-conditioned correction estimates predicts unreliable refinement.
The estimates receive detached Future-Zvar inputs, so their auxiliary training
cannot reshape the forecasting representation or gamma/beta correction.

## Results

{chr(10).join(table)}

## Conclusion

- Best reliability rank correlation: **{best_corr[0]}**.
- Best gated forecast: **{best_gated[0]}**.
- Full benchmark sweep remains blocked unless correction or fallback becomes
  consistently non-harmful under a fixed TRAIN-only rule.
"""
(root / "phase8_summary.md").write_text(summary)
(root / "phase8_comparison.txt").write_text("\n".join(table) + "\n")
PY

touch "$ROOT/phase8_complete"
echo "Phase 8 complete"
