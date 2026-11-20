"""VAL-only interval calibration of an existing frozen G1/R3 residual Flow.

One set of TEST trajectories is reused across methods in each batch.  The
selected raw Flow checkpoint/scale are historical TEST-selected exploratory
choices; none of the new calibrator parameters sees TEST labels.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.predictive_env.dual_pattern_memory.rare_learning import RarePatternLearner
from models.predictive_env.probability_heads import ProbabilityHeadBank
from tools.evaluate_fpem_probabilistic import probabilistic_metrics
from tools.fpem_paper_full_eval import batch_iter, samples_for
from tools.run_fpem_dppm_g1_r3_full import collect, subset
from tools.run_fpem_g1r3_probability_search import point_record
from tools.run_fpem_shared_factorization_g1r3 import load_experiment


def finite_quantile(values, coverage):
    values = torch.cat(values).flatten().float()
    values = values[torch.isfinite(values)]
    if not len(values):
        raise ValueError("Empty calibration scores")
    rank = min(len(values), max(1, math.ceil((len(values) + 1) * coverage)))
    return float(values.kthvalue(rank).values)


@torch.no_grad()
def fit_calibrators(bank, val, device, cli, raw_scale):
    generator = torch.Generator(device=device).manual_seed(cli.seed + 1401)
    collected = {"global": [], "low": [], "high": [], "conformal": []}
    group_scale = []
    group_score = []
    for batch in batch_iter(val, cli.batch_size):
        samples, point, truth, scale = samples_for(bank, batch, device, cli, generator)
        samples = point[:, None] + raw_scale * (samples - point[:, None])
        lo, hi = samples.quantile(.025, 1), samples.quantile(.975, 1)
        left = (point - lo).abs().clamp_min(1e-5)
        right = (hi - point).abs().clamp_min(1e-5)
        global_score = torch.where(truth < point,
                                   (point - truth).abs() / left,
                                   (truth - point).abs() / right)
        collected["global"].append(global_score.flatten().cpu())
        collected["low"].append(((point - truth) / left).clamp_min(0).flatten().cpu())
        collected["high"].append(((truth - point) / right).clamp_min(0).flatten().cpu())
        collected["conformal"].append(torch.maximum(lo - truth, truth - hi).clamp_min(0).flatten().cpu())
        group_scale.append(scale.float().mean((1, 2)).cpu())
        group_score.append(global_score.flatten(1).cpu())
    scalars = torch.cat(group_scale)
    scores = torch.cat(group_score)
    thresholds = torch.quantile(scalars, torch.tensor([.25, .5, .75]))
    global_factor = max(.01, finite_quantile(collected["global"], .95))
    group_factors = []
    groups = torch.bucketize(scalars, thresholds.contiguous())
    for group in range(4):
        local = scores[groups == group].flatten()
        group_factors.append(global_factor if len(local) < 100
                             else max(.01, finite_quantile([local], .95)))
    return {"global_factor": global_factor,
            "low_factor": max(.01, finite_quantile(collected["low"], .975)),
            "high_factor": max(.01, finite_quantile(collected["high"], .975)),
            "conformal_additive": finite_quantile(collected["conformal"], .95),
            "history_scale_thresholds": thresholds.tolist(),
            "conditional_factors": group_factors,
            "val_windows": len(scalars),
            "protocol": "All factors use VAL labels only; no TEST fit"}


@torch.no_grad()
def evaluate(bank, test, device, cli, raw_scale, calibration):
    generator = torch.Generator(device=device).manual_seed(cli.seed + 401)
    methods = ("raw", "global_scale", "asymmetric", "conditional_scale",
               "split_conformal_interval")
    sums = {name: {} for name in methods}
    count = 0
    for batch in batch_iter(test, cli.batch_size):
        samples, point, truth, scale = samples_for(bank, batch, device, cli, generator)
        raw = point[:, None] + raw_scale * (samples - point[:, None])
        residual = raw - point[:, None]
        scale_group = torch.bucketize(scale.float().mean((1, 2)).cpu(),
                                       torch.tensor(calibration["history_scale_thresholds"]))
        local_factor = torch.tensor(calibration["conditional_factors"], device=device)[scale_group.to(device)]
        transformed = {
            "raw": raw,
            "global_scale": point[:, None] + calibration["global_factor"] * residual,
            "asymmetric": point[:, None] + residual * torch.where(
                residual < 0, calibration["low_factor"], calibration["high_factor"]),
            "conditional_scale": point[:, None] + local_factor[:, None, None, None] * residual,
            "split_conformal_interval": raw,
        }
        for name, prediction in transformed.items():
            metric = probabilistic_metrics(prediction, truth, point)
            lo, hi = prediction.quantile(.025, 1), prediction.quantile(.975, 1)
            if name == "split_conformal_interval":
                lo = lo - calibration["conformal_additive"]
                hi = hi + calibration["conformal_additive"]
            coverage = ((truth >= lo) & (truth <= hi)).float()
            width = hi - lo
            alpha = .05
            wis95 = width + (2 / alpha) * ((lo - truth).clamp_min(0) +
                                          (truth - hi).clamp_min(0))
            scalars = {"CRPS": metric["CRPS"], "WIS": metric["WIS"],
                       "WIS95": wis95.mean(), "PICP_95": coverage.mean(),
                       "MPIW_95": width.mean(), "mean_MSE": metric["mean_MSE"],
                       "mean_MAE": metric["mean_MAE"]}
            for j, positions in enumerate(torch.tensor_split(
                    torch.arange(truth.shape[1], device=device), 4)):
                scalars[f"PICP_95_bin{j+1}"] = coverage[:, positions].mean()
                scalars[f"MPIW_95_bin{j+1}"] = width[:, positions].mean()
            for key, value in scalars.items():
                sums[name][key] = sums[name].get(key, 0.) + float(value) * len(truth)
        count += len(truth)
        if count % 512 < cli.batch_size:
            print("CAL_TEST", count, "/", len(test["truth"]), flush=True)
    if count != len(test["truth"]):
        raise AssertionError("Not all TEST windows evaluated")
    return [{"method": name, "evaluated_windows": count, "test_stride": 1,
             "num_samples": 100, "WIS_note": (
                 "Sample WIS unchanged; 95% interval-only score is WIS95"
                 if name == "split_conformal_interval" else "Full sample WIS"),
             **{key: value/count for key, value in sums[name].items()}}
            for name in methods]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case_dir", type=Path, required=True)
    p.add_argument("--prob_dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--flow_steps", type=int, default=12)
    p.add_argument("--val_stride", type=int, default=4)
    a = p.parse_args()
    if a.samples < 100:
        raise ValueError("Paper calibration requires at least 100 trajectories")
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output / "verified.complete").exists():
        print("SKIP verified", a.output, flush=True)
        return
    torch.set_num_threads(2)
    config = SimpleNamespace(base_run_config=str(a.case_dir / "base_run_config.json"),
                             gpu=a.gpu, seed=a.seed,
                             allow_legacy_reference_signature=True)
    args, backbone, loaders, device, sha = load_experiment(config)
    if any(param.requires_grad for param in backbone.parameters()):
        raise AssertionError("Backbone not frozen")
    point = torch.load(a.case_dir / "search/model.pt", map_location="cpu", weights_only=False)
    if point["reference_sha256"] != sha:
        raise AssertionError("Point/A0 mismatch")
    best = json.loads((a.prob_dir / "metrics.json").read_text())["best"]
    checkpoint = torch.load(best["checkpoint"], map_location="cpu", weights_only=False)
    if Path(checkpoint["point_checkpoint"]).resolve() != (a.case_dir / "search/model.pt").resolve():
        raise AssertionError("Flow/point mismatch")
    bank = ProbabilityHeadBank(64, args.pred_len, patch_len=16,
                               heads=("residual_flow",)).to(device)
    bank.load_state_dict(checkpoint["state_dict"])
    bank.eval()
    train_raw = collect(backbone, loaders["train"].dataset.dataset, args, device, 32)
    starts = train_raw["ids"].long()
    limit = len(loaders["train"].dataset.dataset.data_x) - args.seq_len
    train_raw = subset(train_raw, starts + args.seq_len + args.pred_len <= limit)
    std = train_raw["x"].float().var(1, unbiased=False).add(1e-5).sqrt()
    floor = (std.median(0).values * .25).clamp_min(.01)
    del train_raw
    learner = RarePatternLearner.from_export(point["rare"])
    g1 = {"weight": point["g1_weight"], "scale": point["g1_scale"]}
    val_raw = collect(backbone, loaders["val"].dataset.dataset, args, device, a.val_stride)
    test_raw = collect(backbone, loaders["test"].dataset.dataset, args, device, 1)
    if len(test_raw["target"]) != len(loaders["test"].dataset.dataset):
        raise AssertionError("Incomplete TEST collection")
    val = point_record(val_raw, floor, g1, learner, point["reliability"], point["offset"])
    test = point_record(test_raw, floor, g1, learner, point["reliability"], point["offset"])
    del backbone, val_raw, test_raw
    raw_scale = float(best["sample_scale"])
    calibration = fit_calibrators(bank, val, device, a, raw_scale)
    rows = evaluate(bank, test, device, a, raw_scale, calibration)
    for row in rows:
        row.update({"backbone": args.backbone, "dataset": args.dataset_name,
                    "pred_len": args.pred_len, "reference_sha256": sha,
                    "total_test_windows": len(test["truth"]),
                    "selection_caveat": "Original Flow checkpoint/scale TEST-selected; new calibration VAL-only"})
        if not all(math.isfinite(float(row[key])) for key in
                   ("CRPS", "PICP_95", "MPIW_95", "WIS95")):
            raise FloatingPointError("Nonfinite calibration result")
    with (a.output / "calibration_results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (a.output / "calibration_factors.json").write_text(json.dumps(calibration, indent=2) + "\n")
    (a.output / "run_config.json").write_text(json.dumps(vars(a), default=str, indent=2) + "\n")
    (a.output / "verified.complete").write_text("verified\n")
    print("CALIBRATION COMPLETE", args.dataset_name, args.pred_len, flush=True)


if __name__ == "__main__":
    main()
