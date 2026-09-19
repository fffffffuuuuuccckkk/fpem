#!/usr/bin/env python
"""Summarize sample-wise Zvar prediction-gain experiments."""

import argparse
import json
from pathlib import Path


BACKBONES = ("patchtst", "cyclenet", "itransformer")
DATASETS = ("ETTh2", "Electricity", "Traffic")
GATES = ("token", "feature")
WEIGHTS = (0.0, 0.05, 0.1, 0.2)


def label(weight):
    return str(weight).replace(".", "")


def load(path):
    return json.loads(path.read_text())


def locate(args, backbone, dataset, gate, weight):
    local = args.root / backbone / dataset / f"{gate}_gate" / f"lambda_{label(weight)}" / "A2" / "metrics_and_diagnostics.json"
    if local.exists():
        return local
    if weight != 0:
        raise FileNotFoundError(local)
    if backbone == "patchtst" or gate == "feature":
        return args.feature_base_root / backbone / dataset / f"{gate}_gate" / "A2" / "metrics_and_diagnostics.json"
    return args.backbone_base_root / backbone / dataset / "A2_predictive_env" / "A2" / "metrics_and_diagnostics.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--feature-base-root", type=Path, required=True)
    parser.add_argument("--backbone-base-root", type=Path, required=True)
    args = parser.parse_args()
    lines = [
        "Sample-wise Zvar prediction-gain sweep",
        "fixed: selected ETTh2/Electricity/Traffic, K=3, classification, complementary_gate, direct_gated, temperature=0.01",
        "backbone dataset gate lambda full_MSE inv_MSE mean_gain median_gain positive_gain_ratio gain_p10 gain_p50 gain_p90 mask_mean mask_std DeltaZ_over_Zinv var_gain_loss MAE",
    ]
    for backbone in BACKBONES:
        for dataset in DATASETS:
            for gate in GATES:
                for weight in WEIGHTS:
                    row = load(locate(args, backbone, dataset, gate, weight))
                    lines.append(
                        f"{backbone} {dataset} {gate} {weight:g} "
                        f"{row.get('full_MSE', row['MSE']):.9f} {row['inv_MSE']:.9f} "
                        f"{row['mean_gain']:.9f} {row['median_gain']:.9f} "
                        f"{row['positive_gain_ratio']:.6f} {row['gain_p10']:.9f} "
                        f"{row['gain_p50']:.9f} {row['gain_p90']:.9f} "
                        f"{row.get('mask_mean', row['g_mean']):.6f} "
                        f"{row.get('mask_std', float('nan')):.6f} "
                        f"{row.get('DeltaZ/Zinv', row['feature_variation_to_Zinv']):.6f} "
                        f"{row.get('var_gain_loss', float('nan')):.9f} {row['MAE']:.9f}"
                    )
    output = args.root / "var_gain_summary.txt"
    output.write_text("\n".join(lines) + "\n")
    print(output)


if __name__ == "__main__":
    main()
