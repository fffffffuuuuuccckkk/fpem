#!/usr/bin/env python
"""Create a readable summary for the environment-count sweep."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    for path in sorted(root.glob("env_k*/A*/metrics_and_diagnostics.json")):
        metric = json.loads(path.read_text())
        rows.append(metric)
    rows.sort(key=lambda item: (item["experiment"], item["environment_count"]))
    hashes = sorted({row["reference_checkpoint_sha256"] for row in rows})
    if len(hashes) != 1:
        raise RuntimeError(f"Expected one shared reference hash, found: {hashes}")
    checkpoint_paths = sorted({row["reference_checkpoint"] for row in rows})
    if len(checkpoint_paths) != 1:
        raise RuntimeError(f"Expected one shared reference path, found: {checkpoint_paths}")
    seeds = sorted({row["seed"] for row in rows})
    if seeds != [2021]:
        raise RuntimeError(f"Environment sweep requires fixed seed=2021, found: {seeds}")
    non_loaded = [row for row in rows if row.get("reference_source") != "loaded"]
    if non_loaded:
        raise RuntimeError("Every K experiment must load the shared checkpoint")
    lines = [
        "EIIL environment-count sweep (Mapping Variation removed)",
        f"shared_reference_sha256: {hashes[0]}",
        f"shared_reference_path: {checkpoint_paths[0]}",
        "reference_protocol: seed=2021 warm-up once; every K loaded the same checkpoint",
        "experiment env_count MSE MAE inv_MSE g_z DeltaZ/Zinv "
        "corr_inv_env corr_var_env conflict Dwithin_grad Dbetween_grad "
        "separation random_z min_mass final_env_sim_corr",
    ]
    for row in rows:
        lines.append(
            "{experiment} {environment_count} {MSE:.9f} {MAE:.9f} "
            "{inv_only_MSE:.9f} {g_z_mean:.6f} "
            "{feature_variation_to_Zinv:.6f} {corr_inv_env:.6f} "
            "{corr_var_env:.6f} {gradient_disagreement:.8g} "
            "{D_within_grad:.8g} {D_between_grad:.8g} "
            "{gradient_separation:.6f} {random_partition_z_score:.6f} "
            "{min_environment_mass:.6f} "
            "{final_environment_similarity_correlation:.6f}".format(**row)
        )
    lines.append("")
    for experiment in sorted({row["experiment"] for row in rows}):
        candidates = [row for row in rows if row["experiment"] == experiment]
        best = min(candidates, key=lambda item: item["MSE"])
        lines.append(
            f"best_{experiment}: K={best['environment_count']} "
            f"MSE={best['MSE']:.9f} MAE={best['MAE']:.9f}"
        )
    (root / "environment_count_summary.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
