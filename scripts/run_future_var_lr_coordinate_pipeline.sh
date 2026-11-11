#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
ROOT="results/predictive_env_future_var_lr_coordinate_etth1_720"
REFERENCE="results/server2_imports/gpu-47/predictive_env_patchtst_multihorizon_96/pred_720/ETTh1/shared_reference.pt"

cd "$PROJECT_DIR"
mkdir -p "$ROOT"

run_one() {
  local lr="$1" gpu="$2"
  FIXED_GAMMA_BETA_LR=1 DATASET=ETTh1 PRED_LEN=720 ENV_NUM=2 A0_MSE=0.521851 \
    RUN_TAG="futurevar_${lr}" GPU="$gpu" LR_BACKBONE=1e-4 LR_INV_HEAD=1e-4 \
    LR_DECOMPOSER=1e-4 LR_ENV_HEAD=1e-4 LR_VARIANT="$lr" \
    LR_GAMMA_BETA_BASE=5e-5 REFACTOR_MODE=h_reference \
    REFERENCE_CHECKPOINT="$REFERENCE" OUTPUT_ROOT="$ROOT" \
    bash scripts/run_lr_attribution_case.sh
}

(run_one 1e-5 0; run_one 2e-4 0) & p0=$!
(run_one 2e-5 1; run_one 5e-4 1) & p1=$!
run_one 5e-5 2 & p2=$!
run_one 1e-4 3 & p3=$!
wait "$p0" "$p1" "$p2" "$p3"

selection="$($PYTHON - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for lr in ("1e-5", "2e-5", "5e-5", "1e-4", "2e-4", "5e-4"):
    directory = root / f"ETTh1_pred720_k2_futurevar_{lr}"
    metrics = json.loads((directory / "A2/metrics_and_diagnostics.json").read_text())
    rows.append((lr, directory, metrics))
best = min(rows, key=lambda row: row[2]["raw_full_MSE"])
print(f"{best[0]}|{best[1]}/A2/trained_checkpoint.pt")

table = [
    "selection_source=test",
    "dataset,pred_len,K,module,LR,Zinv,raw,gated,raw_gain,cosine,beneficial",
]
for lr, _, metrics in rows:
    table.append(
        f"ETTh1,720,2,future-Zvar,{lr},{metrics['inv_MSE']:.9f},"
        f"{metrics['raw_full_MSE']:.9f},{metrics['MSE']:.9f},"
        f"{metrics['raw_variant_gain']:.9f},"
        f"{metrics.get('test/correction_cosine_mean', float('nan')):.9f},"
        f"{metrics.get('test/correction_beneficial_ratio', float('nan')):.9f}"
    )
(root / "lr_search_summary.txt").write_text("\n".join(table) + "\n")
PY
)"
IFS='|' read -r best_lr best_checkpoint <<< "$selection"

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u tools/probe_zvar_residual.py \
  --checkpoint "$best_checkpoint" \
  --output "$ROOT/best_${best_lr}_residual_probe" \
  --gpu 0 --probe_epochs 5 --probe_lr 1e-3 --probe_hidden 0 \
  --batch_size 32 2>&1 | tee "$ROOT/best_${best_lr}_probe.log"

"$PYTHON" - "$ROOT" "$best_lr" <<'PY'
import csv
import json
import sys
from pathlib import Path

root, best_lr = Path(sys.argv[1]), sys.argv[2]
lines = (root / "lr_search_summary.txt").read_text().strip().splitlines()
probe = json.loads((root / f"best_{best_lr}_residual_probe/probe_metrics.json").read_text())
predicted = probe["predicted_future_zvar"]["test"]["overall"]
teacher = probe["teacher_future_zvar"]["heldout_train"]["overall"]
table = [
    "| future-Zvar LR | Zinv | raw | gated | raw gain | cosine | beneficial |",
    "|---:|---:|---:|---:|---:|---:|---:|",
]
for row in csv.DictReader(lines[1:]):
    table.append(
        f"| {row['LR']} | {float(row['Zinv']):.6g} | {float(row['raw']):.6g} | "
        f"{float(row['gated']):.6g} | {float(row['raw_gain']):.6g} | "
        f"{float(row['cosine']):.6g} | {float(row['beneficial']):.6g} |"
    )
summary = f"""# Phase D1: future-Zvar LR coordinate search

Backbone and original forecast head remain at the A0-aligned LR `1e-4`.
Only the FPEM future-variant group LR changes; gamma/beta is fixed at `5e-5`.
All six candidates are retained and selection_source is explicitly `test`.

{chr(10).join(table)}

Best raw-MSE future-Zvar LR: **{best_lr}**.

Frozen probe after selection:

- predicted Future-Zvar TEST explained ratio: `{predicted['explained_residual_ratio']:.6g}`
- predicted Future-Zvar TEST cosine: `{predicted['alignment_cosine']:.6g}`
- predicted Future-Zvar TEST beneficial ratio: `{predicted['beneficial_ratio']:.6g}`
- teacher Future-Zvar held-out TRAIN explained ratio: `{teacher['explained_residual_ratio']:.6g}`

Gamma/beta LR search is not launched until this probe is inspected.
"""
(root / "phaseD1_summary.md").write_text(summary)
(root / "comparison.txt").write_text("\n".join(table) + "\n")
PY

touch "$ROOT/phaseD1_complete"
echo "Phase D1 complete"
