#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
WAIT_FOR="results/predictive_env_variant_lr_coupling_etth1_720/phase5_complete"
ROOT5="results/predictive_env_variant_lr_coupling_etth1_720"
ROOT="results/predictive_env_reliability_objectives_etth1_720"
REFERENCE="results/server2_imports/gpu-47/predictive_env_patchtst_multihorizon_96/pred_720/ETTh1/shared_reference.pt"
BASE_A="results/predictive_env_feedback_diagnosis_etth1_720_l2/ETTh1_pred720_k2_best_lr_h_reference/A2/metrics_and_diagnostics.json"

cd "$PROJECT_DIR"
for _ in $(seq 1 240); do
  [[ -e "$WAIT_FOR" ]] && break
  sleep 30
done
[[ -e "$WAIT_FOR" ]] || { echo "timeout waiting for Phase 5" >&2; exit 10; }

selection="$($PYTHON - "$BASE_A" "$ROOT5" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[2])
rows = [
    ("baseline", Path(sys.argv[1]), "1e-4", "0"),
    ("fixed5e-5", root / "ETTh1_pred720_k2_variant_fixed5e-5/A2/metrics_and_diagnostics.json", "5e-5", "0"),
    ("adaptive", root / "ETTh1_pred720_k2_variant_adaptive/A2/metrics_and_diagnostics.json", "1e-4", "1"),
]
best = min(rows, key=lambda row: json.loads(row[1].read_text())["raw_full_MSE"])
print("|".join((best[0], str(best[1]), best[2], best[3])))
PY
)"
IFS='|' read -r selected_tag baseline_metrics variant_lr adaptive_variant <<< "$selection"

ADAPTIVE_VARIANT_LR="$adaptive_variant" RELIABILITY_OBJECTIVE=mse_ranking \
  DATASET=ETTh1 PRED_LEN=720 ENV_NUM=2 A0_MSE=0.521851 \
  RUN_TAG=reliability_mse_ranking GPU=0 LR_BACKBONE=1e-4 \
  LR_DECOMPOSER=1e-4 LR_INV_HEAD=2e-4 LR_VARIANT="$variant_lr" \
  REFACTOR_MODE=h_reference REFERENCE_CHECKPOINT="$REFERENCE" \
  OUTPUT_ROOT="$ROOT" bash scripts/run_lr_attribution_case.sh &
pid1=$!

ADAPTIVE_VARIANT_LR="$adaptive_variant" RELIABILITY_OBJECTIVE=mse_usefulness \
  DATASET=ETTh1 PRED_LEN=720 ENV_NUM=2 A0_MSE=0.521851 \
  RUN_TAG=reliability_mse_usefulness GPU=1 LR_BACKBONE=1e-4 \
  LR_DECOMPOSER=1e-4 LR_INV_HEAD=2e-4 LR_VARIANT="$variant_lr" \
  REFACTOR_MODE=h_reference REFERENCE_CHECKPOINT="$REFERENCE" \
  OUTPUT_ROOT="$ROOT" bash scripts/run_lr_attribution_case.sh &
pid2=$!
wait "$pid1" "$pid2"

"$PYTHON" - "$ROOT" "$baseline_metrics" "$selected_tag" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
specs = [
    (f"A MSE ({sys.argv[3]} variant LR)", Path(sys.argv[2])),
    ("B MSE + ranking", root / "ETTh1_pred720_k2_reliability_mse_ranking/A2/metrics_and_diagnostics.json"),
    ("C MSE + usefulness", root / "ETTh1_pred720_k2_reliability_mse_usefulness/A2/metrics_and_diagnostics.json"),
]
columns = [
    ("Zinv", "inv_MSE"), ("raw", "raw_full_MSE"), ("gated", "MSE"),
    ("raw gain", "raw_variant_gain"), ("gated gain", "gated_variant_gain"),
    ("corr(r,r*)", "corr_r_pred_r_target"),
    ("fraction MSE", "reliability_fraction_mse"),
    ("rank loss", "reliability_ranking_loss"),
    ("useful acc", "reliability_usefulness_accuracy"),
    ("useful rate", "reliability_usefulness_positive_ratio"),
    ("r mean", "r_pred_mean"), ("target mean", "r_target_mean"),
]
table = [
    "| objective | " + " | ".join(x[0] for x in columns) + " |",
    "|---|" + "---:|" * len(columns),
]
loaded = []
for label, path in specs:
    metrics = json.loads(path.read_text())
    loaded.append((label, metrics))
    table.append("| " + label + " | " + " | ".join(
        f"{metrics.get(key, float('nan')):.6g}" for _, key in columns
    ) + " |")
best_corr = max(loaded, key=lambda item: item[1].get("corr_r_pred_r_target", float("-inf")))
best_gated = min(loaded, key=lambda item: item[1]["MSE"])
summary = f"""# Phase 7: reliability objective ablation

## Purpose

Compare fraction regression, deterministic pairwise ranking, and correction
usefulness supervision after diagnosing target motion/clipping. Reliability inputs,
network, LR, forecasting loss and environment discovery remain unchanged.

## Results

{chr(10).join(table)}

## Conclusion

- Best rank correlation: **{best_corr[0]}**.
- Best gated forecast: **{best_gated[0]}**.
- This is a diagnostic ETTh1 result, not a benchmark hyperparameter selection.
"""
(root / "phase7_summary.md").write_text(summary)
(root / "phase7_comparison.txt").write_text("\n".join(table) + "\n")
PY

touch "$ROOT/phase7_complete"
echo "Phase 7 complete"
