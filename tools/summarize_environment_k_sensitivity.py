#!/usr/bin/env python3
"""Summarize existing K2/K3 TRAIN-only environment diagnostics without selection."""

import csv
import json
from pathlib import Path

from scipy.stats import spearmanr


ROOTS = {
    2: Path("results/predictive_env_future_patch_gradrel_patchtst_96_k2"),
    3: Path("results/predictive_env_future_patch_gradrel_patchtst_96_k3"),
}
OUTPUT = Path("results/predictive_env_k_sensitivity_train_only")
KEYS = (
    "gradient_separation",
    "random_partition_z_score",
    "random_partition_p_value",
    "min_environment_mass",
    "normalized_assignment_entropy",
    "max_q_mean",
    "stage_ARI_to_previous",
    "stage_NMI_to_previous",
    "final_environment_similarity_correlation",
    "env_acc_gap_var_minus_inv",
)


def finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value == value and abs(value) != float("inf") else None


def main():
    rows = []
    for environment_count, root in ROOTS.items():
        for path in sorted(root.glob("*/horizon_future_var/A2/metrics_and_diagnostics.json")):
            metrics = json.loads(path.read_text())
            row = {
                "dataset": metrics.get("dataset_name", path.parents[2].name),
                "K": environment_count,
                "Zinv_MSE": metrics.get("inv_MSE"),
                "final_MSE": metrics.get("MSE"),
            }
            row.update({key: metrics.get(key) for key in KEYS})
            rows.append(row)
    if not rows:
        raise SystemExit("no K2/K3 metrics found")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fieldnames = tuple(rows[0])
    with (OUTPUT / "k2_k3_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    table = [
        "| dataset | K | Zinv | final | min mass | entropy | separation | z-score | ARI | NMI | env sim | acc gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        value = lambda key: finite(row.get(key))
        fmt = lambda x: "nan" if x is None else f"{x:.6g}"
        table.append(
            "| {dataset} | {K} | {zinv} | {final} | {mass} | {entropy} | "
            "{separation} | {z} | {ari} | {nmi} | {sim} | {gap} |".format(
                dataset=row["dataset"], K=row["K"],
                zinv=fmt(value("Zinv_MSE")), final=fmt(value("final_MSE")),
                mass=fmt(value("min_environment_mass")),
                entropy=fmt(value("normalized_assignment_entropy")),
                separation=fmt(value("gradient_separation")),
                z=fmt(value("random_partition_z_score")),
                ari=fmt(value("stage_ARI_to_previous")),
                nmi=fmt(value("stage_NMI_to_previous")),
                sim=fmt(value("final_environment_similarity_correlation")),
                gap=fmt(value("env_acc_gap_var_minus_inv")),
            )
        )

    correlations = []
    zinv = [finite(row["Zinv_MSE"]) for row in rows]
    for key in KEYS:
        pairs = [
            (x, finite(row.get(key))) for x, row in zip(zinv, rows)
            if x is not None and finite(row.get(key)) is not None
        ]
        if len(pairs) < 3:
            continue
        score, pvalue = spearmanr(
            [item[1] for item in pairs], [item[0] for item in pairs]
        )
        correlations.append((key, score, pvalue, len(pairs)))
    corr_table = [
        "| TRAIN-only criterion | Spearman with test Zinv MSE | p-value | n |",
        "|---|---:|---:|---:|",
    ]
    corr_table.extend(
        f"| {key} | {score:.6g} | {pvalue:.6g} | {count} |"
        for key, score, pvalue, count in correlations
    )

    by_dataset = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], []).append(row)
    winners = []
    for dataset, items in sorted(by_dataset.items()):
        if len(items) != 2:
            continue
        best = min(items, key=lambda row: float(row["Zinv_MSE"]))
        winners.append(f"- {dataset}: lower diagnostic Zinv is K={best['K']}")

    summary = f"""# Phase 9: K2/K3 sensitivity from existing runs

This is a retrospective diagnostic only. Test MSE is used to check whether a
TRAIN-only environment-quality criterion is promising; it is **not** used as an
automatic K selector or final benchmark rule.

## Existing comparable 96-horizon runs

{chr(10).join(table)}

## Criterion association

{chr(10).join(corr_table)}

## Per-dataset diagnostic winner

{chr(10).join(winners)}

No automatic K rule should be introduced unless one TRAIN-only criterion remains
consistent under the 336/720 protocols and chronological validation.
"""
    (OUTPUT / "phase9_summary.md").write_text(summary)


if __name__ == "__main__":
    main()
