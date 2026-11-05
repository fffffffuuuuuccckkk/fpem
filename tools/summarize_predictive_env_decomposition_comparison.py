#!/usr/bin/env python
"""Summarize projection versus signed-gate decomposition at K=3."""

import argparse
import json
from pathlib import Path


def read_metric(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()
    root = Path(args.root)
    rows = [read_metric(root / "baseline" / "A0" / "metrics_and_diagnostics.json")]
    for constraint in ("contrastive", "classification"):
        for decomposition in ("projection", "signed_gate"):
            for experiment in ("A2", "A4"):
                rows.append(
                    read_metric(
                        root
                        / constraint
                        / decomposition
                        / experiment
                        / "metrics_and_diagnostics.json"
                    )
                )
    hashes = {row["reference_checkpoint_sha256"] for row in rows}
    paths = {row["reference_checkpoint"] for row in rows}
    if len(hashes) != 1 or len(paths) != 1:
        raise RuntimeError("All decomposition runs must share one reference")
    if any(row["reference_source"] != "loaded" for row in rows):
        raise RuntimeError("Every run must load, never recreate, the reference")
    if any(row["seed"] != 2021 or row["environment_count"] != 3 for row in rows):
        raise RuntimeError("Decomposition comparison requires seed=2021 and K=3")
    lines = [
        "K=3 projection versus signed-gate decomposition",
        f"shared_reference_sha256: {next(iter(hashes))}",
        f"shared_reference_path: {next(iter(paths))}",
        "name MSE MAE corr_inv_env corr_var_env var_soft_ce inv_soft_ce "
        "var_acc inv_acc gate_mean gate_std gate_pos gate_neg gate_abs "
        "gate_near_zero env_separation min_env_mass final_env_sim_corr",
    ]
    for row in rows:
        if row["experiment"] == "A0":
            name = "A0_baseline"
        else:
            name = (
                f"{row['experiment']}_{row['representation_constraint']}_"
                f"{row['decomposition_type']}"
            )
        lines.append(
            f"{name} {row['MSE']:.9f} {row['MAE']:.9f} "
            f"{row['corr_inv_env']:.6f} {row['corr_var_env']:.6f} "
            f"{row.get('var_env_soft_ce', float('nan')):.6f} "
            f"{row.get('inv_env_soft_ce', float('nan')):.6f} "
            f"{row.get('var_env_accuracy_argmax', float('nan')):.6f} "
            f"{row.get('inv_env_accuracy_argmax', float('nan')):.6f} "
            f"{row['gate/mean']:.6f} {row['gate/std']:.6f} "
            f"{row['gate/positive_ratio']:.6f} "
            f"{row['gate/negative_ratio']:.6f} "
            f"{row['gate/abs_mean']:.6f} "
            f"{row['gate/near_zero_ratio']:.6f} "
            f"{row['gradient_separation']:.6f} "
            f"{row['min_environment_mass']:.6f} "
            f"{row['final_environment_similarity_correlation']:.6f}"
        )
    for constraint in ("contrastive", "classification"):
        for experiment in ("A2", "A4"):
            candidates = [
                row
                for row in rows
                if row["experiment"] == experiment
                and row["representation_constraint"] == constraint
            ]
            best = min(candidates, key=lambda row: row["MSE"])
            lines.append(
                f"best_{experiment}_{constraint}: {best['decomposition_type']} "
                f"MSE={best['MSE']:.9f} MAE={best['MAE']:.9f}"
            )
    (root / "decomposition_comparison_summary.txt").write_text(
        "\n".join(lines) + "\n"
    )


if __name__ == "__main__":
    main()
