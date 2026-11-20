"""Full-stride TEST re-evaluation of a selected frozen G1/R3 residual Flow.

Old search outputs stay untouched. Calibration uses VAL only; TEST targets are
read solely inside scoring. This script does not retrain the point or Flow net.
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
from tools.evaluate_fpem_probabilistic import probabilistic_metrics, probts_quantile_crps
from tools.run_fpem_dppm_g1_r3_full import collect, subset
from tools.run_fpem_g1r3_probability_search import point_record, pack
from tools.run_fpem_shared_factorization_g1r3 import load_experiment


SCALARS = ("CRPS", "ProbTS_CRPS", "WIS", "PICP_95", "MPIW_95",
           "inv_MSE", "inv_MAE", "mean_MSE", "mean_MAE", "sample_std_mean",
           "sample_diversity")


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--case_dir", type=Path, required=True)
    p.add_argument("--prob_dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--flow_steps", type=int, default=12)
    p.add_argument("--val_stride", type=int, default=4)
    p.add_argument("--seed", type=int, default=2021)
    return p.parse_args()


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def batch_iter(data, size):
    for start in range(0, len(data["truth"]), size):
        yield {key: value[start:start + size] for key, value in data.items()}


@torch.no_grad()
def samples_for(bank, batch, device, cli, generator):
    future, point, truth, scale = pack(batch, device)
    condition = bank.condition(future, point, scale)
    samples = bank.sample_head("residual_flow", condition, future, point, scale,
                               cli.samples, generator, chunk_size=8,
                               steps=cli.flow_steps)["y_samples"]
    return samples, point, truth, scale


def quantile_scale(bank, val, device, cli):
    """VAL-only scale for symmetric 95% coverage around the frozen anchor."""
    bank.eval()
    gen = torch.Generator(device=device).manual_seed(cli.seed + 1401)
    ratios = []
    for batch in batch_iter(val, cli.batch_size):
        samples, point, truth, _ = samples_for(bank, batch, device, cli, gen)
        lo = torch.quantile(samples, .025, dim=1)
        hi = torch.quantile(samples, .975, dim=1)
        denom = torch.where(truth >= point, (hi-point).abs(), (point-lo).abs())
        ratios.append(((truth-point).abs() / denom.clamp_min(1e-5)).flatten().cpu())
    scores = torch.cat(ratios)
    # Finite-sample split-quantile; no TEST label participates.
    rank = min(len(scores), math.ceil((len(scores)+1)*.95))
    return float(scores.kthvalue(rank).values.clamp_min(.01))


@torch.no_grad()
def evaluate(bank, data, device, cli, scale_factor):
    bank.eval()
    generator = torch.Generator(device=device).manual_seed(cli.seed + 401)
    totals = {}
    near_hits = near_count = far_hits = far_count = 0.
    horizon_hits = horizon_count = None
    count = 0
    for batch in batch_iter(data, cli.batch_size):
        samples, point, truth, _ = samples_for(bank, batch, device, cli, generator)
        samples = point[:, None] + scale_factor * (samples - point[:, None])
        metrics = probabilistic_metrics(samples, truth, point)
        for name in SCALARS:
            totals[name] = totals.get(name, 0.) + float(metrics[name]) * len(truth)
        sum_crps = probts_quantile_crps(samples.sum(-1, keepdim=True),
                                       truth.sum(-1, keepdim=True)).mean()
        totals["ProbTS_CRPS_Sum"] = totals.get("ProbTS_CRPS_Sum", 0.) + float(sum_crps) * len(truth)
        cover = ((truth >= samples.quantile(.025, dim=1)) &
                 (truth <= samples.quantile(.975, dim=1)))
        midpoint = truth.shape[1] // 2
        near_hits += float(cover[:, :midpoint].sum())
        far_hits += float(cover[:, midpoint:].sum())
        near_count += cover[:, :midpoint].numel()
        far_count += cover[:, midpoint:].numel()
        hits = cover.float().sum((0, 2)).cpu()
        horizon_hits = hits if horizon_hits is None else horizon_hits + hits
        horizon_count = len(truth) * truth.shape[-1] if horizon_count is None else horizon_count + len(truth) * truth.shape[-1]
        count += len(truth)
        if count % 256 < cli.batch_size:
            print("FULL_TEST", count, "/", len(data["truth"]), flush=True)
    if count != len(data["truth"]):
        raise AssertionError("TEST stride or window count mismatch")
    result = {name: value/count for name, value in totals.items()}
    result.update({"evaluated_windows": count, "total_test_windows": len(data["truth"]),
                   "test_stride": 1, "num_samples": cli.samples,
                   "PICP_95_near": near_hits/near_count,
                   "PICP_95_far": far_hits/far_count,
                   "calibration_scale": scale_factor})
    coverage = [{"horizon": j+1, "PICP_95": float(value/horizon_count)}
                for j, value in enumerate(horizon_hits)]
    return result, coverage


def main():
    cli = args()
    cli.output.mkdir(parents=True, exist_ok=True)
    if (cli.output / "verified.complete").exists():
        print("SKIP verified", cli.output, flush=True)
        return
    if cli.samples < 100:
        raise ValueError("Paper evaluation requires at least 100 trajectories")
    torch.set_num_threads(2)
    torch.manual_seed(cli.seed)
    config = SimpleNamespace(base_run_config=str(cli.case_dir / "base_run_config.json"),
                             gpu=cli.gpu, seed=cli.seed,
                             allow_legacy_reference_signature=True)
    args0, model, loaders, device, sha = load_experiment(config)
    if any(p.requires_grad for p in model.parameters()):
        raise AssertionError("A0 must be frozen")
    point = torch.load(cli.case_dir / "search/model.pt", map_location="cpu",
                       weights_only=False)
    if point["reference_sha256"] != sha:
        raise AssertionError("A0 SHA mismatch")
    best = json.loads((cli.prob_dir / "metrics.json").read_text())["best"]
    ckpt = torch.load(best["checkpoint"], map_location="cpu", weights_only=False)
    if Path(ckpt["point_checkpoint"]).resolve() != (cli.case_dir / "search/model.pt").resolve():
        raise AssertionError("Flow/point checkpoint mismatch")
    bank = ProbabilityHeadBank(64, args0.pred_len, patch_len=16,
                               heads=("residual_flow",)).to(device)
    bank.load_state_dict(ckpt["state_dict"])
    learner = RarePatternLearner.from_export(point["rare"])
    g1 = {"weight": point["g1_weight"], "scale": point["g1_scale"]}
    train_raw = collect(model, loaders["train"].dataset.dataset, args0, device, 32)
    starts = train_raw["ids"].long()
    limit = len(loaders["train"].dataset.dataset.data_x) - args0.seq_len
    train_raw = subset(train_raw, starts + args0.seq_len + args0.pred_len <= limit)
    std = train_raw["x"].float().var(1, unbiased=False).add(1e-5).sqrt()
    floor = (std.median(0).values * .25).clamp_min(.01)
    del train_raw
    val_raw = collect(model, loaders["val"].dataset.dataset, args0, device,
                      cli.val_stride)
    test_raw = collect(model, loaders["test"].dataset.dataset, args0, device, 1)
    total_test = len(loaders["test"].dataset.dataset)
    if len(test_raw["target"]) != total_test:
        raise AssertionError("Incomplete TEST collection")
    del model
    val = point_record(val_raw, floor, g1, learner, point["reliability"],
                       point["offset"])
    test = point_record(test_raw, floor, g1, learner, point["reliability"],
                        point["offset"])
    del val_raw, test_raw
    calibrated_scale = quantile_scale(bank, val, device, cli)
    selected_scale = float(best["sample_scale"])
    results = {}
    coverage_rows = []
    for label, factor in (("selected_scale", selected_scale),
                          ("val_global_scale", calibrated_scale)):
        metrics, coverage = evaluate(bank, test, device, cli, factor)
        if not all(math.isfinite(float(metrics[key])) for key in SCALARS):
            raise FloatingPointError("Nonfinite probability metric")
        results[label] = metrics
        coverage_rows.extend({"mode": label, **row} for row in coverage)
    payload = {
        "backbone": args0.backbone, "dataset": args0.dataset_name,
        "seq_len": args0.seq_len, "pred_len": args0.pred_len,
        "reference_sha256": sha, "point_checkpoint": str(cli.case_dir / "search/model.pt"),
        "flow_checkpoint": str(best["checkpoint"]),
        "point_anchor_frozen": True,
        "test_selection": "Original Flow LR/scale selected on sparse TEST; exploratory",
        "calibration": "VAL-only split quantile scale; no TEST label fitting",
        "flow_training_residual": "TRAIN in-sample point residual; not full-pipeline OOF",
        "val_stride": cli.val_stride, "total_test_windows": total_test,
        "results": results,
    }
    (cli.output / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_csv(cli.output / "coverage_by_horizon.csv", coverage_rows)
    (cli.output / "verified.complete").write_text("verified\n")
    print("COMPLETE", cli.output, "windows", total_test, flush=True)


if __name__ == "__main__":
    main()
