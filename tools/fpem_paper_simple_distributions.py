#!/usr/bin/env python3
"""Frozen-anchor Gaussian/Student-t residual baselines on full TEST.

These are deliberately simple, horizon/channel-wise TRAIN-fitted distribution
controls. They are not independent neural forecasting models. TEST targets are
used only for scoring; no distribution parameter is selected on TEST.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.predictive_env.dual_pattern_memory.rare_learning import RarePatternLearner
from tools.evaluate_fpem_probabilistic import probabilistic_metrics, probts_quantile_crps
from tools.run_fpem_dppm_g1_r3_full import collect, record, ridge_query, subset
from tools.run_fpem_g1r3_probability_search import point_record
from tools.run_fpem_shared_factorization_g1r3 import load_experiment


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--case_dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--train_stride", type=int, default=32)
    p.add_argument("--seed", type=int, default=2021)
    return p.parse_args()


def anchors(raw, floor, point):
    data = record(raw, floor)
    full = point_record(raw, floor,
                        {"weight": point["g1_weight"], "scale": point["g1_scale"]},
                        RarePatternLearner.from_export(point["rare"]),
                        point["reliability"], point["offset"])
    g1_norm = data["base"] + point["g1_scale"] * ridge_query(data["key"], point["g1_weight"])
    g1_raw = raw["base"].float() + ((g1_norm - data["base"]) * data["scale"].transpose(1, 2)).transpose(1, 2)
    return {"a0": raw["base"].float(), "g1": g1_raw, "g1_r3": full["y_inv"]}, raw["target"].float()


def score_samples(samples, truth, anchor):
    out = probabilistic_metrics(samples, truth, anchor)
    out["ProbTS_CRPS_Sum"] = probts_quantile_crps(samples.sum(-1, keepdim=True),
                                                   truth.sum(-1, keepdim=True)).mean()
    return {key: float(value) for key, value in out.items() if torch.as_tensor(value).ndim == 0}


def main():
    cli = parse()
    cli.output.mkdir(parents=True, exist_ok=True)
    marker = cli.output / "verified.complete"
    if marker.is_file():
        print("SKIP", cli.output, flush=True)
        return
    if cli.samples < 100:
        raise ValueError("at least 100 samples required")
    torch.set_num_threads(2)
    torch.manual_seed(cli.seed)
    config = SimpleNamespace(base_run_config=str(cli.case_dir / "base_run_config.json"),
                             gpu=cli.gpu, seed=cli.seed,
                             allow_legacy_reference_signature=True)
    args, model, loaders, device, sha = load_experiment(config)
    point = torch.load(cli.case_dir / "search/model.pt", map_location="cpu", weights_only=False)
    if point["reference_sha256"] != sha:
        raise AssertionError("A0 checkpoint SHA mismatch")
    train_raw = collect(model, loaders["train"].dataset.dataset, args, device, cli.train_stride)
    starts = train_raw["ids"].long()
    train_limit = len(loaders["train"].dataset.dataset.data_x) - args.seq_len
    train_raw = subset(train_raw, starts + args.seq_len + args.pred_len <= train_limit)
    history_std = train_raw["x"].float().var(1, unbiased=False).add(1e-5).sqrt()
    floor = (history_std.median(0).values * .25).clamp_min(.01)
    train_anchor, train_truth = anchors(train_raw, floor, point)
    fitted = {}
    for anchor_name, anchor in train_anchor.items():
        residual = train_truth - anchor
        fitted[anchor_name] = (residual.mean(0), residual.std(0, unbiased=False).clamp_min(.01))
    del train_raw, train_anchor, train_truth
    test_raw = collect(model, loaders["test"].dataset.dataset, args, device, 1)
    test_count = len(loaders["test"].dataset.dataset)
    if len(test_raw["target"]) != test_count:
        raise AssertionError("TEST is not full stride-1")
    test_anchor, test_truth = anchors(test_raw, floor, point)
    del model, test_raw
    rows = []
    for anchor_name, anchor in test_anchor.items():
        loc, std = fitted[anchor_name]
        for distribution in ("gaussian", "student_t_df5"):
            totals = {}
            rng = np.random.default_rng(cli.seed + (0 if distribution == "gaussian" else 107))
            for start in range(0, test_count, cli.batch_size):
                y = test_truth[start:start + cli.batch_size].to(device)
                a = anchor[start:start + cli.batch_size].to(device)
                shape = (len(y), cli.samples, y.shape[1], y.shape[2])
                if distribution == "gaussian":
                    noise = rng.standard_normal(shape, dtype=np.float32)
                else:
                    # Student-t scaled to the same variance as the Gaussian.
                    noise = (rng.standard_t(5, shape) * math.sqrt(3/5)).astype("float32")
                samples = a[:, None] + loc.to(device)[None, None] + \
                    std.to(device)[None, None] * torch.from_numpy(noise).to(device)
                metrics = score_samples(samples, y, a)
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.) + value * len(y)
            row = {key: value/test_count for key, value in totals.items()}
            row.update({"dataset": args.dataset_name, "backbone": args.backbone,
                        "pred_len": args.pred_len, "anchor": anchor_name,
                        "distribution": distribution, "num_samples": cli.samples,
                        "evaluated_windows": test_count, "total_test_windows": test_count,
                        "test_stride": 1, "reference_sha256": sha,
                        "training_residual": "TRAIN in-sample; not OOF"})
            if not all(math.isfinite(float(row[key])) for key in ("CRPS", "ProbTS_CRPS", "WIS", "PICP_95", "MPIW_95")):
                raise FloatingPointError("nonfinite probability baseline metric")
            rows.append(row)
            print("BASELINE", anchor_name, distribution, "CRPS", row["CRPS"], flush=True)
    with (cli.output / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (cli.output / "run_config.json").write_text(json.dumps(vars(cli), default=str, indent=2))
    marker.write_text("verified\n")


if __name__ == "__main__":
    main()
