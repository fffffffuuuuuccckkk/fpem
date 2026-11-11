#!/usr/bin/env python
"""Summarize token-wise versus feature-wise Zvar fusion gates."""

import argparse
import json
from pathlib import Path


DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Electricity", "ExchangeRate", "Weather", "Traffic")
BACKBONES = ("patchtst", "cyclenet", "itransformer")


def load(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--existing-backbone-root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for backbone in BACKBONES:
        for dataset in DATASETS:
            feature_path = args.feature_root / backbone / dataset / "feature_gate" / "A2" / "metrics_and_diagnostics.json"
            if backbone == "patchtst":
                token_path = args.feature_root / backbone / dataset / "token_gate" / "A2" / "metrics_and_diagnostics.json"
            else:
                token_path = args.existing_backbone_root / backbone / dataset / "A2_predictive_env" / "A2" / "metrics_and_diagnostics.json"
            token, feature = load(token_path), load(feature_path)
            rows.append((backbone, dataset, token, feature))

    lines = [
        "PredictiveEnvIV token-wise [B,C,P,1] vs feature-wise [B,C,P,D] Zvar fusion gate",
        "fixed: A2 classification complementary_gate direct_gated K=3 seed=2021 conditional_gain=0.2",
        "backbone dataset token_MSE feature_MSE delta_MSE feature_improvement_percent token_MAE feature_MAE delta_MAE token_gate_mean feature_gate_mean",
    ]
    for backbone, dataset, token, feature in rows:
        delta_mse = feature["MSE"] - token["MSE"]
        delta_mae = feature["MAE"] - token["MAE"]
        improvement = -100.0 * delta_mse / token["MSE"]
        lines.append(
            f"{backbone} {dataset} {token['MSE']:.9f} {feature['MSE']:.9f} "
            f"{delta_mse:+.9f} {improvement:+.6f} {token['MAE']:.9f} "
            f"{feature['MAE']:.9f} {delta_mae:+.9f} "
            f"{token.get('g_mean', float('nan')):.6f} {feature.get('g_mean', float('nan')):.6f}"
        )
    lines.append("")
    lines.append("overall")
    for backbone in BACKBONES:
        chosen = [(token, feature) for name, _, token, feature in rows if name == backbone]
        wins = sum(feature["MSE"] < token["MSE"] for token, feature in chosen)
        mean_improvement = sum(
            100.0 * (token["MSE"] - feature["MSE"]) / token["MSE"]
            for token, feature in chosen
        ) / len(chosen)
        lines.append(
            f"{backbone} feature_MSE_wins={wins}/{len(chosen)} "
            f"mean_relative_MSE_improvement={mean_improvement:+.6f}%"
        )
    output = args.feature_root / "featurewise_gate_summary.txt"
    output.write_text("\n".join(lines) + "\n")
    print(output)


if __name__ == "__main__":
    main()
