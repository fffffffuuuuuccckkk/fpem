#!/usr/bin/env python3
"""Disentangle ABC train-condition shuffle from TEST-inference shuffle."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.fpem_abc_probability import ConditionedResidualFlow, make_data, score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "verified.complete").exists():
        print("SKIP", args.output, flush=True)
        return
    torch.set_num_threads(2)
    data_cache = torch.load(args.case / "records.pt", map_location="cpu", weights_only=False)
    risk_cache = torch.load(args.case / "C_risk_cache.pt", map_location="cpu", weights_only=False)
    data = data_cache["data"]["test"]
    raw_n = len(data["ids"])
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    rows = []
    for trained_mode in ("all_fused", "shuffled_fused"):
        checkpoint = torch.load(args.case / f"C12_{trained_mode}.pt", map_location="cpu", weights_only=False)
        info = json.loads((args.case / f"C12_{trained_mode}.json").read_text())
        model = ConditionedResidualFlow(data_cache["meta"]["pred_len"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        testing = make_data(data, risk_cache["compact_test"]["full"],
                            risk_cache["features_test"], checkpoint["feature_mean"],
                            checkpoint["feature_std"])
        scale = float(info["calibrated_scale"])
        for inference_condition, shift in (("true_test_condition", 0),
                                           ("test_condition_roll_1", 1),
                                           ("test_condition_roll_7", 7)):
            packet = {key: value for key, value in testing.items()}
            if shift:
                packet["features"] = testing["features"].roll(shift, 0)
            metrics = score(model, packet, device, "all_fused", seed=2021,
                            sample_scale=scale, stride=1)
            if metrics["evaluated_windows"] != raw_n:
                raise AssertionError("incomplete ABC TEST evaluation")
            if not all(math.isfinite(float(metrics[key])) for key in
                       ("CRPS", "ProbTS_CRPS", "PICP_95", "MPIW_95", "WIS")):
                raise FloatingPointError("nonfinite ABC metric")
            row = {"backbone": data_cache["meta"]["backbone"],
                   "dataset": data_cache["meta"]["dataset"],
                   "pred_len": data_cache["meta"]["pred_len"],
                   "trained_mode": trained_mode,
                   "inference_condition": inference_condition,
                   "train_condition_shuffled": trained_mode == "shuffled_fused",
                   "test_condition_shuffled": bool(shift),
                   "shuffle_shift": shift, "sample_scale": scale,
                   "test_stride": 1, "num_samples": 100, **metrics}
            rows.append(row)
            print("ABC", trained_mode, inference_condition, metrics["CRPS"], flush=True)
    with (args.output / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "run_config.json").write_text(json.dumps(vars(args), default=str, indent=2))
    (args.output / "verified.complete").write_text("verified\n")


if __name__ == "__main__":
    main()
