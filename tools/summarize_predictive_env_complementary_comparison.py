#!/usr/bin/env python
"""Summarize the information-preserving complementary gate experiment."""

import argparse
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def previous_result(root, constraint, decomposition, experiment):
    return load(
        root
        / constraint
        / decomposition
        / experiment
        / "metrics_and_diagnostics.json"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--previous_root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    previous_root = Path(args.previous_root)
    rows = [load(root / "baseline" / "A0" / "metrics_and_diagnostics.json")]
    for constraint in ("contrastive", "classification"):
        for decomposition in (
            "projection",
            "signed_gate",
            "complementary_gate",
        ):
            rows.append(
                load(
                    root
                    / constraint
                    / decomposition
                    / "A2"
                    / "metrics_and_diagnostics.json"
                )
            )
    hashes = {row["reference_checkpoint_sha256"] for row in rows}
    paths = {row["reference_checkpoint"] for row in rows}
    if len(hashes) != 1 or len(paths) != 1:
        raise RuntimeError("Every decomposition must use the same reference")
    if any(row["reference_source"] != "loaded" for row in rows):
        raise RuntimeError("Every run must load the existing shared reference")
    if any(row["seed"] != 2021 or row["environment_count"] != 3 for row in rows):
        raise RuntimeError("Comparison requires fixed seed=2021 and K=3")

    old = {}
    for constraint in ("contrastive", "classification"):
        for experiment in ("A2", "A4"):
            projection = previous_result(
                previous_root, constraint, "projection", experiment
            )
            signed = previous_result(
                previous_root, constraint, "signed_gate", experiment
            )
            old[(constraint, experiment)] = (
                100.0 * (signed["MSE"] / projection["MSE"] - 1.0)
            )
    old_projection_cls = previous_result(
        previous_root, "classification", "projection", "A2"
    )
    old_signed_cls = previous_result(
        previous_root, "classification", "signed_gate", "A2"
    )

    lines = [
        "K=3 information-preserving complementary-gate comparison",
        f"shared_reference_sha256: {next(iter(hashes))}",
        f"shared_reference_path: {next(iter(paths))}",
        "activity_regularization: signed_gate=0.0001 complementary_gate=0.0",
        "",
        "Previous signed-gate conclusion (preserved):",
        "signed assignment improved environment predictability: "
        f"A2 classification var_acc {old_projection_cls['var_env_accuracy_argmax']:.6f} "
        f"-> {old_signed_cls['var_env_accuracy_argmax']:.6f}.",
        "Forecasting MSE degradation versus projection: "
        f"A2 contrastive {old[('contrastive', 'A2')]:.2f}%, "
        f"A2 classification {old[('classification', 'A2')]:.2f}%, "
        f"A4 contrastive {old[('contrastive', 'A4')]:.2f}%, "
        f"A4 classification {old[('classification', 'A4')]:.2f}%.",
        "Working hypothesis: relu(g)/relu(-g) attenuated or deleted too much H; "
        "the complementary gate tests information-preserving allocation.",
        "",
        "name MSE MAE corr_inv_env corr_var_env var_soft_ce inv_soft_ce "
        "var_acc inv_acc gate_mean gate_std gate_pos gate_neg gate_abs "
        "gate_near_zero reconstruction_error inv_energy var_energy "
        "env_separation min_env_mass final_env_sim_corr",
    ]
    for row in rows:
        name = (
            "A0_baseline"
            if row["experiment"] == "A0"
            else f"A2_{row['representation_constraint']}_{row['decomposition_type']}"
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
            f"{row['reconstruction_error']:.9g} "
            f"{row['inv_energy_ratio']:.6f} "
            f"{row['var_energy_ratio']:.6f} "
            f"{row['gradient_separation']:.6f} "
            f"{row['min_environment_mass']:.6f} "
            f"{row['final_environment_similarity_correlation']:.6f}"
        )
    for constraint in ("contrastive", "classification"):
        candidates = [
            row
            for row in rows
            if row["experiment"] == "A2"
            and row["representation_constraint"] == constraint
        ]
        best = min(candidates, key=lambda row: row["MSE"])
        lines.append(
            f"best_A2_{constraint}: {best['decomposition_type']} "
            f"MSE={best['MSE']:.9f} MAE={best['MAE']:.9f}"
        )
    (root / "complementary_gate_summary.txt").write_text(
        "\n".join(lines) + "\n"
    )


if __name__ == "__main__":
    main()
