#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
WAIT_FOR="results/predictive_env_h_reference_cross_dataset/phaseA_complete"
ROOT="results/predictive_env_zvar_residual_probe"
CHECKPOINT="results/predictive_env_variant_lr_coupling_etth1_720/ETTh1_pred720_k2_variant_fixed5e-5/A2/trained_checkpoint.pt"

cd "$PROJECT_DIR"
mkdir -p "$ROOT"
for _ in $(seq 1 720); do
  [[ -e "$WAIT_FOR" ]] && break
  sleep 30
done
[[ -e "$WAIT_FOR" ]] || { echo "timeout waiting for Phase A" >&2; exit 10; }

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u tools/probe_zvar_residual.py \
  --checkpoint "$CHECKPOINT" \
  --output "$ROOT/ETTh1_pred720_k2_linear" \
  --gpu 0 --probe_epochs 5 --probe_lr 1e-3 --probe_hidden 0 \
  --batch_size 32 2>&1 | tee "$ROOT/ETTh1_pred720_k2_linear.log"

"$PYTHON" - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
path = root / "ETTh1_pred720_k2_linear/probe_metrics.json"
metrics = json.loads(path.read_text())
rows = []
for source in ("current_zvar", "predicted_future_zvar", "teacher_future_zvar"):
    for split, values in metrics[source].items():
        if values is None:
            continue
        overall = values.get("overall", values)
        rows.append((
            source, split, overall["baseline_residual_MSE"],
            overall["residual_MSE"], overall["explained_residual_ratio"],
            overall["alignment_cosine"], overall["beneficial_ratio"],
        ))
table = [
    "| source | split | zero baseline | probe MSE | explained | cosine | beneficial |",
    "|---|---|---:|---:|---:|---:|---:|",
] + [
    f"| {source} | {split} | {base:.6g} | {mse:.6g} | {explained:.6g} | {cosine:.6g} | {beneficial:.6g} |"
    for source, split, base, mse, explained, cosine, beneficial in rows
]
teacher = metrics["teacher_future_zvar"]["heldout_train"]
predicted = metrics["predicted_future_zvar"]["test"]["overall"]
summary = f"""# Phase B: frozen Zvar-to-residual linear probes

## Fixed protocol

The trained forecasting model is frozen. Only simple linear probes are trained.
Current and predicted Future-Zvar are evaluated on TEST. Oracle teacher
Future-Zvar is strictly TRAIN-only and evaluated on a chronological held-out
TRAIN tail; no teacher future is read from TEST.

## Results

{chr(10).join(table)}

## Routing evidence

- Teacher held-out TRAIN explained ratio: `{teacher['overall']['explained_residual_ratio']:.6g}`.
- Teacher held-out TRAIN cosine: `{teacher['overall']['alignment_cosine']:.6g}`.
- Predicted Future-Zvar TEST explained ratio: `{predicted['explained_residual_ratio']:.6g}`.
- Predicted Future-Zvar TEST cosine: `{predicted['alignment_cosine']:.6g}`.

No Phase C/D/E branch is launched automatically from an arbitrary numerical
threshold; the continuous diagnostics must first be inspected.
"""
(root / "phaseB_summary.md").write_text(summary)
(root / "comparison.txt").write_text("\n".join(table) + "\n")
PY

touch "$ROOT/phaseB_complete"
echo "Phase B complete"
