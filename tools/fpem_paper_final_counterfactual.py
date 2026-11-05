"""Strong, paired TEST counterfactuals for the existing ABC probability Flow.

Only the *extra retrieved condition* changes at inference.  The historical
key, point anchor, scale, checkpoint, test windows, and Monte Carlo seed stay
fixed.  A separately re-trained wrong-pair control uses the same architecture
and training budget.  TEST labels are read only for scoring/analysis.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.evaluate_fpem_probabilistic import probabilistic_metrics
from tools.fpem_abc_mechanism import block_bootstrap_gain
from tools.fpem_abc_probability import (
    ConditionedResidualFlow, make_data, packet, score, train_flow,
)

CONDITIONS = ("correct", "zero", "far_time", "variable", "scale_stratum")
SEEDS = (2021, 2022, 2023)


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=True) + "\n")


def wrong_index(record, kind: str, seed=2021):
    n = len(record["ids"])
    if n < 2:
        raise ValueError("Counterfactual needs at least two TEST windows")
    if kind == "far_time":
        return torch.arange(n).roll(n // 2)
    if kind == "scale_stratum":
        scale = record["scale"].float().mean((1, 2))
        boundaries = torch.quantile(scale, torch.arange(1, 10) / 10)
        bins = torch.bucketize(scale, boundaries.contiguous())
        perm = torch.arange(n)
        for group in range(10):
            indices = (bins == group).nonzero().flatten()
            if len(indices) > 1:
                perm[indices] = indices.roll(max(1, len(indices) // 2))
            elif len(indices) == 1:
                perm[indices] = (indices + n // 2) % n
        return perm
    raise ValueError(kind)


def condition_record(record, kind: str):
    out = dict(record)
    features = record["features"]
    if kind == "correct":
        return out
    if kind == "zero":
        out["features"] = torch.zeros_like(features)
    elif kind in ("far_time", "scale_stratum"):
        out["features"] = features[wrong_index(record, kind)]
    elif kind == "variable":
        out["features"] = (features.roll(1, 1) if features.shape[1] > 1
                           else features[wrong_index(record, "far_time")])
    else:
        raise ValueError(kind)
    return out


@torch.no_grad()
def evaluate(model, record, device, seed, scale, batch_size=4):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed + 401)
    n = len(record["ids"])
    totals = {}
    crps = []
    point_mse = []
    for ids in torch.arange(n).split(batch_size):
        future, point, truth, hist_scale, extra = packet(record, ids, device, "all_fused")
        condition = model.condition(future, point, hist_scale, extra)
        samples = model.bank.sample_head(
            "residual_flow", condition, future, point, hist_scale, 100,
            generator, chunk_size=8, steps=12,
        )["y_samples"]
        samples = point[:, None] + scale * (samples - point[:, None])
        result = probabilistic_metrics(samples, truth, point)
        for key in ("CRPS", "ProbTS_CRPS", "WIS", "PICP_95", "MPIW_95",
                    "inv_MSE", "inv_MAE", "mean_MSE", "mean_MAE"):
            totals[key] = totals.get(key, 0.0) + float(result[key]) * len(ids)
        crps.append(result["crps_per_sample"].cpu().numpy())
        point_mse.append((truth - point).square().mean((1, 2)).cpu().numpy())
    if sum(map(len, crps)) != n:
        raise AssertionError("Incomplete stride-1 TEST")
    return ({key: value / n for key, value in totals.items()}
            | {"evaluated_windows": n, "num_samples": 100, "test_stride": 1},
            np.concatenate(crps), np.concatenate(point_mse))


def load_case(case, device):
    data_cache = torch.load(case / "records.pt", map_location="cpu", weights_only=False)
    risk_cache = torch.load(case / "C_risk_cache.pt", map_location="cpu", weights_only=False)
    oof = risk_cache["oof"]
    columns = __import__("tools.fpem_abc_probability", fromlist=["FEATURES"]).FEATURES["all_fused"]
    values = np.stack([risk_cache["features_oof"][name] for name in columns], -1)
    mean = np.nan_to_num(values.mean((0, 1)), nan=0)
    std = np.nan_to_num(values.std((0, 1)), nan=1).clip(.01)
    training = make_data(oof["data"], oof["prediction"], risk_cache["features_oof"], mean, std)
    testing = make_data(data_cache["data"]["test"], risk_cache["compact_test"]["full"],
                        risk_cache["features_test"], mean, std)
    split = max(1, int(len(training["ids"]) * .75))
    fit = {key: value[:split] for key, value in training.items()}
    cal = {key: value[split:] for key, value in training.items()}
    return data_cache["meta"], fit, cal, testing


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=4)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output / "verified.complete").exists():
        print("SKIP verified", a.output, flush=True)
        return
    torch.set_num_threads(2)
    device = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    meta, fit, cal, test = load_case(a.case, device)
    n = len(test["ids"])
    if n <= 0:
        raise AssertionError("Empty TEST cache")
    original_info = json.loads((a.case / "C12_all_fused.json").read_text())
    original_state = torch.load(a.case / "C12_all_fused.pt", map_location="cpu", weights_only=False)
    original = ConditionedResidualFlow(meta["pred_len"]).to(device)
    original.load_state_dict(original_state["state_dict"])
    fitted = [("trained_correct", original, float(original_info["calibrated_scale"]))]

    # Stronger re-training control: keep all labels and optimizer/budget, but
    # pair every TRAIN history with a distant TRAIN retrieved-condition vector.
    bad_path = a.output / "trained_far_time.pt"
    bad_info = a.output / "trained_far_time.json"
    if not bad_path.exists():
        fit_bad = condition_record(fit, "far_time")
        bad, trace = train_flow(fit_bad, "all_fused", device, 2021,
                                meta["pred_len"], epochs=8)
        # Same VAL calibration rule as the original C12 training protocol.
        scored = [(scale, score(bad, cal, device, "all_fused", seed=2021,
                                sample_scale=scale, stride=2))
                  for scale in (.8, 1., 1.2)]
        selected_scale = min(scored, key=lambda pair: pair[1]["CRPS"])[0]
        torch.save({"state_dict": bad.state_dict(), "feature_mean": original_state["feature_mean"],
                    "feature_std": original_state["feature_std"]}, bad_path)
        write_json(bad_info, {"calibrated_scale": selected_scale, "trace": trace,
                              "calibration": scored, "fit_condition": "far_time",
                              "budget": "8 epochs, Adam 1e-4, identical C12 capacity"})
    else:
        bad = ConditionedResidualFlow(meta["pred_len"]).to(device)
        bad.load_state_dict(torch.load(bad_path, map_location="cpu", weights_only=False)["state_dict"])
    fitted.append(("trained_wrong", bad, float(json.loads(bad_info.read_text())["calibrated_scale"])))

    output_rows = []
    for trained_name, model, scale in fitted:
        for seed in SEEDS:
            correct_path = a.output / f"{trained_name}_{seed}_correct.npz"
            if not correct_path.exists():
                metrics, values, point_error = evaluate(model, test, device, seed, scale, a.batch_size)
                np.savez_compressed(correct_path, crps=values, point_mse=point_error)
                write_json(correct_path.with_suffix(".json"), metrics)
            correct = np.load(correct_path)
            threshold = np.quantile(correct["point_mse"], .8)
            for condition in CONDITIONS:
                path = a.output / f"{trained_name}_{seed}_{condition}.npz"
                if not path.exists():
                    metrics, values, _ = evaluate(model, condition_record(test, condition),
                                                  device, seed, scale, a.batch_size)
                    np.savez_compressed(path, crps=values)
                    write_json(path.with_suffix(".json"), metrics)
                metrics = json.loads(path.with_suffix(".json").read_text())
                wrong = np.load(path)["crps"]
                gain = wrong - correct["crps"]
                bootstrap = block_bootstrap_gain(gain, blocks=min(20, max(2, n // 20)),
                                                 repeats=1000, seed=seed)
                output_rows.append({"backbone": meta["backbone"], "dataset": meta["dataset"],
                                    "pred_len": meta["pred_len"], "trained_mode": trained_name,
                                    "condition": condition, "seed": seed,
                                    "correct_minus_wrong_crps": float(-gain.mean()),
                                    "correct_advantage_crps": float(gain.mean()),
                                    "median_correct_advantage": float(np.median(gain)),
                                    "positive_window_ratio": float((gain > 0).mean()),
                                    "high_risk_advantage": float(gain[correct["point_mse"] >= threshold].mean()),
                                    **bootstrap, **metrics})
                print("P3", trained_name, seed, condition, metrics["CRPS"], flush=True)
    with (a.output / "condition_counterfactual.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    write_json(a.output / "run_config.json", {"case": str(a.case), "gpu": a.gpu,
              "conditions": CONDITIONS, "seeds": SEEDS, "test_windows": n,
              "protocol": "fixed point/Flow, paired TEST condition changes; three MC seeds; wrong-pair retrain",
              "caveat": "Original C12 point OOF; TEST is exploratory. No TEST label conditions."})
    (a.output / "verified.complete").write_text("verified\n")


if __name__ == "__main__":
    main()
