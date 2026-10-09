#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
WAIT_FOR="results/predictive_env_lr_rule_backtest_l2/phases_1_to_3_complete"
ROOT="results/predictive_env_correction_diagnostics_l2"
REF_ETTM2="results/server2_imports/gpu-47/predictive_env_patchtst_multihorizon_96/pred_336/ETTm2/shared_reference.pt"
REF_EXCHANGE="results/server2_imports/gpu-47/predictive_env_patchtst_multihorizon_96/pred_720/ExchangeRate/shared_reference.pt"
ETTH1_METRICS="results/predictive_env_feedback_diagnosis_etth1_720_l2/ETTh1_pred720_k2_best_lr_h_reference/A2/metrics_and_diagnostics.json"

cd "$PROJECT_DIR"
for _ in $(seq 1 240); do
  [[ -e "$WAIT_FOR" ]] && break
  sleep 30
done
[[ -e "$WAIT_FOR" ]] || { echo "timeout waiting for phases 1-3" >&2; exit 10; }

DATASET=ETTm2 PRED_LEN=336 ENV_NUM=3 A0_MSE=0.326629 \
  RUN_TAG=stable_l2_diagnostics GPU=2 LR_BACKBONE=2e-5 \
  LR_DECOMPOSER=1e-4 LR_INV_HEAD=5e-5 REFACTOR_MODE=current \
  REFERENCE_CHECKPOINT="$REF_ETTM2" OUTPUT_ROOT="$ROOT" \
  bash scripts/run_lr_attribution_case.sh &
pid1=$!
DATASET=ExchangeRate PRED_LEN=720 ENV_NUM=2 A0_MSE=0.876098 \
  RUN_TAG=stable_l2_diagnostics GPU=3 LR_BACKBONE=2e-5 \
  LR_DECOMPOSER=1e-4 LR_INV_HEAD=5e-5 REFACTOR_MODE=current \
  REFERENCE_CHECKPOINT="$REF_EXCHANGE" OUTPUT_ROOT="$ROOT" \
  bash scripts/run_lr_attribution_case.sh &
pid2=$!
wait "$pid1" "$pid2"

"$PYTHON" - "$ROOT" "$ETTH1_METRICS" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
specs = [
    ("ETTh1-720 h_reference", Path(sys.argv[2])),
    ("ETTm2-336 stable L2", root / "ETTm2_pred336_k3_stable_l2_diagnostics/A2/metrics_and_diagnostics.json"),
    ("ExchangeRate-720 stable L2", root / "ExchangeRate_pred720_k2_stable_l2_diagnostics/A2/metrics_and_diagnostics.json"),
]
rows = []
for name, path in specs:
    metrics = json.loads(path.read_text())
    rows.append((name, metrics))

alignment = [
    "| config | split | cosine | optimal scalar | beneficial ratio | magnitude/residual |",
    "|---|---|---:|---:|---:|---:|",
]
reliability = [
    "| config | epoch corr | epoch rank | stage corr | stage rank | target std | clip 0 | clip 1 | corr(r,r*) |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
conclusions = []
for name, m in rows:
    for split in ("train", "test"):
        prefix = f"{split}/correction_"
        alignment.append(
            f"| {name} | {split} | {m.get(prefix+'cosine_mean', float('nan')):.6f} | "
            f"{m.get(prefix+'optimal_scalar_mean', float('nan')):.6f} | "
            f"{m.get(prefix+'beneficial_ratio', float('nan')):.6f} | "
            f"{m.get(prefix+'magnitude_ratio_mean', float('nan')):.6f} |"
        )
    reliability.append(
        f"| {name} | {m.get('r_target_epoch_correlation', float('nan')):.6f} | "
        f"{m.get('r_target_epoch_rank_stability', float('nan')):.6f} | "
        f"{m.get('r_target_stage_correlation', float('nan')):.6f} | "
        f"{m.get('r_target_stage_rank_stability', float('nan')):.6f} | "
        f"{m.get('r_target_std', float('nan')):.6f} | "
        f"{m.get('r_target_clip_zero_ratio', float('nan')):.6f} | "
        f"{m.get('r_target_clip_one_ratio', float('nan')):.6f} | "
        f"{m.get('corr_r_pred_r_target', float('nan')):.6f} |"
    )
    train_cos = m.get("train/correction_cosine_mean", float("nan"))
    test_cos = m.get("test/correction_cosine_mean", float("nan"))
    test_mag = m.get("test/correction_magnitude_ratio_mean", float("nan"))
    if test_cos < 0:
        reason = "direction error (negative test alignment)"
    elif test_mag > 1:
        reason = "magnitude/calibration error"
    elif train_cos > 0 and test_cos <= 0:
        reason = "train/test variant generalization failure"
    else:
        reason = "weak direction signal; magnitude is already small"
    conclusions.append(f"- **{name}**: {reason}.")

summary = f"""# Phase 4 and Phase 6 diagnostics

## Purpose

Diagnose raw gamma/beta correction alignment without changing the loss, then test
whether the analytic reliability target is stable and non-collapsed.

## Correction alignment

{chr(10).join(alignment)}

## Reliability target motion

{chr(10).join(reliability)}

## Interpretation

{chr(10).join(conclusions)}

The target clipping ratios and epoch/stage correlations determine whether the
reliability failure is primarily a moving/bimodal target rather than optimizer LR.
"""
(root / "phase4_phase6_summary.md").write_text(summary)
(root / "phase4_phase6_comparison.txt").write_text(
    "\n".join(alignment + [""] + reliability) + "\n"
)
PY

touch "$ROOT/phase4_phase6_complete"
echo "Phase 4/6 diagnostics complete"
