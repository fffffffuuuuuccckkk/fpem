#!/usr/bin/env python
"""Summarize K=3 contrastive versus classification constraints."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    for constraint in ("contrastive", "classification"):
        for experiment in ("A2", "A4"):
            path = root / constraint / experiment / "metrics_and_diagnostics.json"
            rows.append(json.loads(path.read_text()))
    hashes = {row["reference_checkpoint_sha256"] for row in rows}
    paths = {row["reference_checkpoint"] for row in rows}
    if len(hashes) != 1 or len(paths) != 1:
        raise RuntimeError("Comparison did not use one shared reference checkpoint")
    if any(row["reference_source"] != "loaded" for row in rows):
        raise RuntimeError("Every comparison run must load the shared reference")
    if any(row["environment_count"] != 3 or row["seed"] != 2021 for row in rows):
        raise RuntimeError("Comparison requires K=3 and seed=2021")
    lines = [
        "K=3 representation-constraint comparison",
        f"shared_reference_sha256: {next(iter(hashes))}",
        f"shared_reference_path: {next(iter(paths))}",
        "name MSE MAE corr_inv_env corr_var_env var_soft_ce inv_soft_ce "
        "var_acc inv_acc var_entropy inv_entropy overlap_before overlap_after "
        "final_env_sim_corr hungarian_permutation",
    ]
    for row in rows:
        name = f"{row['experiment']}_{row['representation_constraint']}"
        lines.append(
            f"{name} {row['MSE']:.9f} {row['MAE']:.9f} "
            f"{row['corr_inv_env']:.6f} {row['corr_var_env']:.6f} "
            f"{row['var_env_soft_ce']:.6f} {row['inv_env_soft_ce']:.6f} "
            f"{row['var_env_accuracy_argmax']:.6f} "
            f"{row['inv_env_accuracy_argmax']:.6f} "
            f"{row['var_env_entropy']:.6f} {row['inv_env_entropy']:.6f} "
            f"{row['alignment_overlap_before']:.6f} "
            f"{row['alignment_overlap_after']:.6f} "
            f"{row['final_environment_similarity_correlation']:.6f} "
            f"{row['hungarian_permutation']}"
        )
    for experiment in ("A2", "A4"):
        candidates = [row for row in rows if row["experiment"] == experiment]
        best = min(candidates, key=lambda row: row["MSE"])
        lines.append(
            f"best_{experiment}: {best['representation_constraint']} "
            f"MSE={best['MSE']:.9f} MAE={best['MAE']:.9f}"
        )
    (root / "representation_constraint_summary.txt").write_text(
        "\n".join(lines) + "\n"
    )


if __name__ == "__main__":
    main()
