#!/usr/bin/env python3
"""Create fixed A0 configs for the full periodic-token backbone matrix."""

import argparse
import json
from pathlib import Path


DATASETS = {
    "ETTh1": ("./dataset/all_datasets/ETT-small", "ett_hour", "ETTh1.csv", "h", 32, 24, 7),
    "ETTh2": ("./dataset/all_datasets/ETT-small", "ett_hour", "ETTh2.csv", "h", 32, 24, 7),
    "ETTm1": ("./dataset/all_datasets/ETT-small", "ett_minute", "ETTm1.csv", "15min", 32, 96, 7),
    "ETTm2": ("./dataset/all_datasets/ETT-small", "ett_minute", "ETTm2.csv", "15min", 32, 96, 7),
    "Weather": ("./dataset/all_datasets/weather", "custom", "weather.csv", "10min", 16, 144, 21),
    "Electricity": ("./dataset/all_datasets/electricity", "custom", "electricity.csv", "h", 4, 168, 321),
    "ExchangeRate": ("./dataset/all_datasets/exchange_rate", "custom", "exchange_rate.csv", "d", 32, 7, 8),
}
HORIZONS = (96, 336, 720)
BACKBONES = ("patchtst", "moderntcn", "itransformer")
MODERN = {
    "ETTh1": (1, 1, 0.3, 0.0),
    "ETTh2": (1, 1, 0.3, 0.0),
    "ETTm1": (3, 8, 0.3, 0.0),
    "ETTm2": (3, 8, 0.3, 0.1),
    "Weather": (1, 8, 0.4, 0.0),
    "Electricity": (1, 8, 0.3, 0.0),
    "ExchangeRate": (1, 1, 0.2, 0.6),
}


def config(backbone, dataset, horizon):
    root, data_class, data_path, freq, batch, cycle, channels = DATASETS[dataset]
    item = {
        "backbone": backbone,
        "dataset_name": dataset,
        "root_path": root,
        "data_class": data_class,
        "data_path": data_path,
        "target": "OT",
        "freq": freq,
        "batch_size": batch,
        "cyclenet_cycle_len": cycle,
        "enc_in": channels,
        "seq_len": 96,
        "pred_len": horizon,
        "seed": 2021,
        "warmup_epochs": 3,
        "lr": 1e-4,
        "num_workers": 0,
        "d_model": 512,
        "d_ff": 2048,
        "n_heads": 2,
        "e_layers": 1,
        "dropout": 0.1,
        "patch_len": 16,
        "predictive_env_stride": 8,
        "exog_gate_init": 0.0,
    }
    if backbone == "moderntcn":
        stages, ratio, dropout, head_dropout = MODERN[dataset]
        item.update({
            "d_model": 64,
            "d_ff": 512,
            "dropout": dropout,
            "modern_tcn_patch_size": 8,
            "modern_tcn_patch_stride": 4,
            "modern_tcn_num_stages": stages,
            "modern_tcn_ffn_ratio": ratio,
            "modern_tcn_large_size": 51,
            "modern_tcn_small_size": 5,
            "modern_tcn_downsample_ratio": 2,
            "modern_tcn_head_dropout": head_dropout,
        })
    return item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    config_root = args.root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    cases = []
    for backbone in BACKBONES:
        for dataset in DATASETS:
            for horizon in HORIZONS:
                case = f"{backbone}_{dataset}_{horizon}"
                target = config_root / f"{case}.json"
                target.write_text(
                    json.dumps(config(backbone, dataset, horizon), indent=2) + "\n",
                    encoding="utf-8",
                )
                cases.append(case)
    (args.root / "cases.txt").write_text("\n".join(cases) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(cases), "root": str(args.root)}))


if __name__ == "__main__":
    main()
