#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
ROOT="results/predictive_env_h_reference_cross_dataset"
ETTM2_REF="results/server2_imports/gpu-47/predictive_env_patchtst_multihorizon_96/pred_336/ETTm2/shared_reference.pt"
EXCHANGE_REF="results/server2_imports/gpu-47/predictive_env_patchtst_multihorizon_96/pred_720/ExchangeRate/shared_reference.pt"

cd "$PROJECT_DIR"
mkdir -p "$ROOT"

run_case() {
  local dataset="$1" pred="$2" k="$3" a0="$4" mode="$5" gpu="$6" ref="$7"
  DATASET="$dataset" PRED_LEN="$pred" ENV_NUM="$k" A0_MSE="$a0" \
    RUN_TAG="$mode" GPU="$gpu" LR_BACKBONE=1e-4 LR_INV_HEAD=1e-4 \
    LR_DECOMPOSER=1e-4 LR_ENV_HEAD=1e-4 LR_VARIANT=1e-4 \
    REFACTOR_MODE="$mode" REFERENCE_CHECKPOINT="$ref" \
    OUTPUT_ROOT="$ROOT" bash scripts/run_lr_attribution_case.sh
}

run_case ETTm2 336 3 0.326629 current 0 "$ETTM2_REF" & p0=$!
run_case ETTm2 336 3 0.326629 h_reference 1 "$ETTM2_REF" & p1=$!
run_case ExchangeRate 720 2 0.876098 current 2 "$EXCHANGE_REF" & p2=$!
run_case ExchangeRate 720 2 0.876098 h_reference 3 "$EXCHANGE_REF" & p3=$!
wait "$p0" "$p1" "$p2" "$p3"

"$PYTHON" - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
specs = [
    ("ETTm2", 336, 3, 0.326629, "current"),
    ("ETTm2", 336, 3, 0.326629, "h_reference"),
    ("ExchangeRate", 720, 2, 0.876098, "current"),
    ("ExchangeRate", 720, 2, 0.876098, "h_reference"),
]
columns = [
    ("Zinv", "inv_MSE"), ("raw", "raw_full_MSE"), ("gated", "MSE"),
    ("ARI", "stage_ARI_to_previous"), ("NMI", "stage_NMI_to_previous"),
    ("env sim", "final_environment_similarity_correlation"),
    ("H drift", "h_prediction_drift_from_pretrained"),
    ("inv acc", "inv_acc"), ("var acc", "var_acc"),
    ("gate abs", "gate/abs_mean"), ("inv energy", "inv_energy_ratio"),
]
table = [
    "| dataset | pred | K | mode | A0 | " + " | ".join(x[0] for x in columns) + " |",
    "|---|---:|---:|---|---:|" + "---:|" * len(columns),
]
rows = []
for dataset, pred, k, a0, mode in specs:
    path = root / f"{dataset}_pred{pred}_k{k}_{mode}/A2/metrics_and_diagnostics.json"
    metrics = json.loads(path.read_text())
    rows.append((dataset, pred, k, a0, mode, metrics))
    table.append(
        f"| {dataset} | {pred} | {k} | {mode} | {a0:.6f} | "
        + " | ".join(f"{metrics.get(key, float('nan')):.6g}" for _, key in columns)
        + " |"
    )
comparisons = []
for dataset in ("ETTm2", "ExchangeRate"):
    current = next(x[-1] for x in rows if x[0] == dataset and x[4] == "current")
    href = next(x[-1] for x in rows if x[0] == dataset and x[4] == "h_reference")
    comparisons.append(
        f"- {dataset}: h_reference - current Zinv = "
        f"{href['inv_MSE'] - current['inv_MSE']:+.6f}; final = "
        f"{href['MSE'] - current['MSE']:+.6f}."
    )
summary = f"""# Phase A: h_reference cross-dataset validation

## Controlled variable

Only environment discovery source changes. Backbone and original forecast-head
LR are both `1e-4`, matching the current A0 reproduction LR; all FPEM module
settings, shared references, splits, seeds and epochs are paired.

## Results

{chr(10).join(table)}

## Paired differences

{chr(10).join(comparisons)}

The result is diagnostic. `h_reference` becomes the default only if it does not
materially harm either dataset's invariant branch.
"""
(root / "phaseA_summary.md").write_text(summary)
(root / "comparison.txt").write_text("\n".join(table) + "\n")
PY

touch "$ROOT/phaseA_complete"
echo "Phase A complete"
