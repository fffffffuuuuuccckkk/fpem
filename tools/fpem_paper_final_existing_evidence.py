"""Consolidate pre-existing ABC evidence without changing old experiments.

In particular A2 is a strict forward-in-time G1 map test; A3 (leave-block-
out with future training) is never relabelled as forward transfer.  This
script computes the missing relative-A0 denominator from the exact purged
target windows used by A2 and emits honest partial-status documentation.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.fpem_abc_mechanism import chronological_blocks


CASES = (
    ("patchtst", "ETTh1", 96), ("patchtst", "ETTh1", 720),
    ("itransformer", "ETTm1", 96), ("itransformer", "ETTm1", 720),
    ("moderntcn", "ETTm2", 96), ("moderntcn", "ETTm2", 720),
)


def case_path(backbone, dataset, horizon):
    if backbone == "patchtst" and horizon == 96:
        return ROOT / "results/fpem_paper_completion_20261101/abc_patchtst_etth1_96"
    return (ROOT / "results/fpem_abc_mechanism_search_20261031" /
            backbone / dataset / f"pred_{horizon}")


def read_rows(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def write_rows(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, stem):
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix("." + suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)


def transfer_rows(case, a_rows):
    selected = [r for r in a_rows if r["experiment"] == "A2" and r["split"] == "TRAIN_forward"]
    if not selected:
        return []
    payload = torch.load(case / "records.pt", map_location="cpu", weights_only=False)
    train = payload["data"]["train"]
    span = int(payload["meta"]["span"])
    blocks = chronological_blocks(train["ids"], 5)
    rows = []
    for row in selected:
        source = int(row["source_block"])
        target = int(row["target_block"])
        src = blocks[source]
        hold = blocks[target]
        hold = hold[train["ids"][hold] >= train["ids"][src].max() + span]
        if len(hold) < 4:
            raise AssertionError("Published A2 row lacks its purged holdout")
        error = ((train["base"][hold] - train["truth"][hold]) *
                 train["scale"][hold].transpose(1, 2))
        a0_mse = float(error.square().mean())
        gain = float(row["gain_mse"])
        future_end = int(row["source_future_end"])
        target_start = int(row["target_history_start"])
        if future_end > target_start:
            raise AssertionError("Future overlap in temporal transfer")
        rows.append({"backbone": row["backbone"], "dataset": row["dataset"],
                     "pred_len": row["pred_len"], "source_block": source,
                     "target_block": target, "source_future_end": future_end,
                     "target_history_start": target_start, "target_windows": len(hold),
                     "a0_target_mse": a0_mse, "g1_absolute_gain_mse": gain,
                     "relative_a0_mse_improvement_pct": 100 * gain / a0_mse,
                     "protocol": "TRAIN strict forward A2; TEST not used for this matrix"})
    return rows


def transfer_plot(rows, stem):
    matrix = np.full((5, 5), np.nan)
    for row in rows:
        matrix[int(row["source_block"]), int(row["target_block"])] = \
            row["relative_a0_mse_improvement_pct"]
    fig, ax = plt.subplots(figsize=(5.4, 4.5))
    vmax = max(1., float(np.nanmax(np.abs(matrix))))
    image = ax.imshow(matrix, cmap="RdBu", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(5), [f"{i+1}" for i in range(5)])
    ax.set_yticks(range(5), [f"{i+1}" for i in range(5)])
    ax.set_xlabel("Later target TRAIN time block")
    ax.set_ylabel("Earlier source TRAIN time block")
    ax.set_title("G1 strict forward transfer (% vs A0)")
    for i in range(5):
        for j in range(5):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i,j]:+.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="A0 relative MSE gain (%)")
    save_figure(fig, stem)
    np.savetxt(stem.with_suffix(".csv"), matrix, delimiter=",")


def r3_plot(rows, stem):
    quintiles = [r for r in rows if r["experiment"] == "B1" and r.get("density_quintile")]
    experts = [r for r in rows if r["experiment"] == "B4" and r.get("routing")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    if quintiles:
        x = [int(r["density_quintile"]) for r in quintiles]
        y = [float(r["gain_mse"]) for r in quintiles]
        axes[0].plot(x, y, marker="o")
    axes[0].axhline(0, color="gray", linewidth=.8)
    axes[0].set(xlabel="TRAIN-frequency quintile", ylabel="G1 minus R3 MSE",
                title="R3 gain by history frequency")
    if experts:
        labels = [r["routing"] for r in experts]
        y = [float(r["gain_mse"]) for r in experts]
        axes[1].bar(range(len(y)), y)
        axes[1].set_xticks(range(len(y)), labels, rotation=50, ha="right")
    axes[1].axhline(0, color="gray", linewidth=.8)
    axes[1].set(title="Correct vs wrong expert routing", ylabel="G1 minus variant MSE")
    fig.tight_layout()
    save_figure(fig, stem)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=ROOT / "results/fpem_paper_final_evidence_20261101")
    a = p.parse_args()
    out = a.output
    out.mkdir(parents=True, exist_ok=True)
    transfer, r3, risk, controls, synthetic = [], [], [], [], []
    done = []
    for backbone, dataset, horizon in CASES:
        case = case_path(backbone, dataset, horizon)
        a_path, b_path, c_path = (case / "A_all_results.csv", case / "B_all_results.csv",
                                  case / "C_risk_results.csv")
        if not all(path.is_file() for path in (a_path, b_path, c_path)):
            continue
        a_rows, b_rows, c_rows = map(read_rows, (a_path, b_path, c_path))
        current = transfer_rows(case, a_rows)
        transfer += current
        if current:
            transfer_plot(current, out / "figures" / f"fig1_g1_transfer_{backbone}_{dataset}_{horizon}")
        for row in b_rows:
            if row["experiment"] != "B11":
                r3.append(row)
            else:
                synthetic.append(row)
        r3_plot(b_rows, out / "figures" / f"fig2_3_r3_{backbone}_{dataset}_{horizon}")
        risk += c_rows
        controls += [r for r in a_rows if r["experiment"] == "A9"]
        synthetic += [r for r in a_rows if r["experiment"] == "A10"]
        done.append(f"{backbone}/{dataset}-{horizon}")
    write_rows(out / "g1_temporal_transfer.csv", transfer)
    write_rows(out / "r3_mechanism_results.csv", r3)
    write_rows(out / "condition_risk_diagnostics.csv", risk)
    write_rows(out / "g1_simple_controls.csv", controls)
    write_rows(out / "synthetic_existing_smoke_only.csv", synthetic)
    status = [
        "# Final-evidence status (partial, resumable)", "",
        f"Existing ABC cases consolidated: {len(done)}/6 ({', '.join(done)})", "",
        "- G1 temporal matrices use only strict forward A2 TRAIN blocks; A3 offline LOBO is excluded.",
        "- R3 and risk CSVs preserve negative results and TEST-exploratory labels.",
        "- Existing A10/B11 synthetic rows are smoke checks, not the controlled-factor P6 experiment.",
        "- P1 probability pilots, P2 matched Flow anchors, P3 counterfactuals are separate running queues.",
        "- P4 selective gating, P7 advanced calibration/multi-seed/efficiency remain pending.",
        "- Do not interpret these plots alone as proof of causal environment mechanisms.",
    ]
    (out / "paper_ready_status.md").write_text("\n".join(status) + "\n")
    print(json.dumps({"cases": len(done), "g1_transfer_rows": len(transfer),
                      "r3_rows": len(r3), "risk_rows": len(risk)}, indent=2))


if __name__ == "__main__":
    main()
