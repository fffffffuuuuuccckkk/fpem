#!/usr/bin/env python3
"""Source-linked, non-causal exploratory mechanism figures for the paper."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def rows(path):
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def number(row, key):
    try:
        x = float(row[key])
        return x if np.isfinite(x) else None
    except (KeyError, TypeError, ValueError):
        return None


def finish(fig, out, stem):
    fig.tight_layout()
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, default=Path("."))
    p.add_argument("--result_root", type=Path, required=True)
    a = p.parse_args()
    repo, root = a.repo.resolve(), a.result_root.resolve()
    out = root / "paper_ready/figures"
    out.mkdir(parents=True, exist_ok=True)
    abc = repo / "results/fpem_abc_mechanism_search_20261031"
    pilots = [("patchtst", "ETTh1", "720"), ("itransformer", "ETTm1", "720"),
              ("moderntcn", "ETTm2", "720")]
    labels = ["PatchTST / ETTh1", "iTransformer / ETTm1", "ModernTCN / ETTm2"]

    # A1 block-wise selected G1 gain; visualizes temporal heterogeneity, not invariance proof.
    arows = rows(abc / "A_all_results.csv")
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    data_rows = []
    for pilot, label in zip(pilots, labels):
        matched = [r for r in arows if (r.get("backbone"), r.get("dataset"), r.get("pred_len")) == pilot
                   and r.get("experiment") == "A1" and r.get("block") not in ("", None)
                   and number(r, "gain_mse") is not None]
        block_gain = defaultdict(list)
        for r in matched:
            block_gain[r["block"]].append(float(r["gain_mse"]))
        pairs = sorted(((int(k), float(np.mean(v))) for k, v in block_gain.items()), key=lambda x: x[0])
        if pairs:
            ax.plot([x for x, _ in pairs], [y for _, y in pairs], marker="o", label=label)
            data_rows += [{"backbone": pilot[0], "dataset": pilot[1], "pred_len": pilot[2],
                           "block": x, "gain_mse": y, "source": "A_all_results.csv/A1"} for x, y in pairs]
    ax.axhline(0, color="black", linewidth=.8)
    ax.set(xlabel="Chronological block", ylabel="A0 MSE − G1 MSE",
           title="Shared-correction gain varies across time blocks")
    ax.legend(fontsize=8)
    finish(fig, out, "g1_temporal_transfer")
    if data_rows:
        with (out / "g1_temporal_transfer.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data_rows[0])); w.writeheader(); w.writerows(data_rows)

    brows = rows(abc / "B_all_results.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    for j, pilot in enumerate(pilots):
        current = [r for r in brows if (r.get("backbone"), r.get("dataset"), r.get("pred_len")) == pilot
                   and r.get("split") == "TEST"]
        b0 = next((number(r, "gain_mse") for r in current if r.get("experiment") == "B0"
                   and number(r, "gain_mse") is not None), np.nan)
        positive = next((number(r, "positive_ratio") for r in current if r.get("experiment") == "B0"
                         and number(r, "positive_ratio") is not None), np.nan)
        axes[0].bar(j, b0, color="#3567a0")
        axes[1].bar(j, positive, color="#ca7441")
    axes[0].axhline(0, color="black", linewidth=.8)
    axes[0].set(ylabel="G1 MSE − full MSE", title="R3 incremental MSE gain")
    axes[1].axhline(.5, color="black", linestyle="--", linewidth=.8)
    axes[1].set(ylabel="Fraction of helpful samples", ylim=(0, 1), title="R3 sample-level benefit")
    for ax in axes:
        ax.set_xticks(range(3), labels, rotation=20, ha="right", fontsize=8)
    finish(fig, out, "r3_local_adaptation")

    crows = rows(abc / "C_risk_results.csv")
    feature_values = defaultdict(list)
    for row in crows:
        value = number(row, "test_corr_squared_error")
        if value is not None and row.get("feature"):
            feature_values[row["feature"]].append(value)
    ordered = sorted(((key, float(np.mean(value))) for key, value in feature_values.items()),
                     key=lambda x: abs(x[1]), reverse=True)[:12]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    if ordered:
        ax.barh(range(len(ordered)), [value for _, value in reversed(ordered)], color="#6b8e73")
        ax.set_yticks(range(len(ordered)), [name.replace("C5_neighbour_", "retrieved_").replace("C6_neighbour_", "OOF_")
                                         for name, _ in reversed(ordered)], fontsize=8)
    ax.axvline(0, color="black", linewidth=.8)
    ax.set(xlabel="Correlation with squared point error", title="Retrieved future risk association (not causality)")
    finish(fig, out, "retrieved_future_risk")

    coverage = []
    for path in root.glob("**/coverage_by_horizon.csv"):
        for row in rows(path):
            if row.get("mode") == "selected_scale":
                try:
                    dataset = path.parts[path.parts.index("patchtst") + 1]
                except (ValueError, IndexError):
                    continue
                coverage.append((dataset, int(row["horizon"]), float(row["PICP_95"])))
    if coverage:
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        for dataset in sorted(set(x[0] for x in coverage)):
            pairs = sorted((h, c) for d, h, c in coverage if d == dataset)
            ax.plot([x for x, _ in pairs], [y for _, y in pairs], label=dataset)
        ax.axhline(.95, color="black", linestyle="--", linewidth=.8, label="Nominal 95%")
        ax.set(xlabel="Forecast horizon", ylabel="Empirical pointwise coverage", ylim=(0, 1),
               title="95% prediction interval coverage by horizon")
        ax.legend(fontsize=8, ncol=2)
        finish(fig, out, "interval_coverage")
    print(f"Generated mechanism figures in {out}; coverage cases={len(set(x[0] for x in coverage))}")


if __name__ == "__main__":
    main()
