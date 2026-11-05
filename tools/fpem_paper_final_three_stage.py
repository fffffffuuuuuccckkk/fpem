"""Matched A0/G1/G1+R3 point and residual-Flow layer ablation.

Uses the already audited strict forward point OOF records of an ABC case.
Every Flow has identical ConditionedResidualFlow capacity and eight-epoch
training budget.  The three existing full-anchor C12 checkpoints are reused
without retraining.  TEST is full stride-1 and never used to train or calibrate.
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
from tools.fpem_abc_probability import (
    FEATURES, ConditionedResidualFlow, make_data, score, train_flow,
)
from tools.fpem_paper_final_counterfactual import condition_record


def load_data(case):
    saved = torch.load(case / "records.pt", map_location="cpu", weights_only=False)
    risk = torch.load(case / "C_risk_cache.pt", map_location="cpu", weights_only=False)
    oof = risk["oof"]
    cols = FEATURES["all_fused"]
    values = np.stack([risk["features_oof"][name] for name in cols], -1)
    mean = np.nan_to_num(values.mean((0, 1)), nan=0)
    std = np.nan_to_num(values.std((0, 1)), nan=1).clip(.01)
    point_oof = {"a0": oof["data"]["base"],
                 "g1": oof["compact"]["anchor"],
                 "g1_r3": oof["prediction"]}
    point_test = {"a0": saved["data"]["test"]["base"],
                  "g1": risk["compact_test"]["anchor"],
                  "g1_r3": risk["compact_test"]["full"]}
    train = {name: make_data(oof["data"], point, risk["features_oof"], mean, std)
             for name, point in point_oof.items()}
    test = {name: make_data(saved["data"]["test"], point,
                           risk["features_test"], mean, std)
            for name, point in point_test.items()}
    return saved["meta"], train, test


def point_metrics(record):
    error = record["y_inv"] - record["truth"]
    return {"point_MSE": float(error.square().mean()),
            "point_MAE": float(error.abs().mean()),
            "evaluated_windows": len(record["ids"])}


def load_or_train(name, case, out, train, horizon, device):
    anchor, mode = name.split("__", 1)
    state_path = out / f"{name}.pt"
    info_path = out / f"{name}.json"
    old_mode = {"history_key": "history_key", "retrieved_future": "retrieved_future",
                "all_fused": "all_fused"}.get(mode)
    old_path = case / f"C12_{old_mode}.pt" if anchor == "g1_r3" and old_mode else None
    if old_path is not None and old_path.exists():
        state_path = old_path
        info_path = case / f"C12_{old_mode}.json"
    if state_path.exists() and info_path.exists():
        model = ConditionedResidualFlow(horizon).to(device)
        model.load_state_dict(torch.load(state_path, map_location="cpu",
                                         weights_only=False)["state_dict"])
        return model, float(json.loads(info_path.read_text())["calibrated_scale"])
    split = max(1, int(len(train["ids"]) * .75))
    fit = {key: value[:split] for key, value in train.items()}
    cal = {key: value[split:] for key, value in train.items()}
    train_mode = mode
    if mode == "wrong_pair_train":
        fit = condition_record(fit, "far_time")
        train_mode = "all_fused"
    model, trace = train_flow(fit, train_mode, device, 2021, horizon, epochs=8)
    options = [(scale, score(model, cal, device, train_mode, seed=2021,
                             sample_scale=scale, stride=2))
               for scale in (.8, 1., 1.2)]
    selected = min(options, key=lambda item: item[1]["CRPS"])[0]
    torch.save({"state_dict": model.state_dict(), "anchor": anchor,
                "mode": mode}, state_path)
    info_path.write_text(json.dumps({"calibrated_scale": selected,
                                     "calibration": options, "trace": trace,
                                     "training_budget": "8 epochs, Adam 1e-4"}, indent=2))
    return model, selected


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--smoke_only", action="store_true")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output / "verified.complete").exists():
        print("SKIP verified", a.output, flush=True)
        return
    torch.set_num_threads(2)
    device = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    meta, train, test = load_data(a.case)
    test_count = len(test["a0"]["ids"])
    if not all(len(value["ids"]) == test_count for value in test.values()):
        raise AssertionError("Point anchors have unmatched TEST windows")
    if a.smoke_only:
        print(json.dumps({name: point_metrics(value) for name, value in test.items()},
                         indent=2), flush=True)
        return
    plan = [
        ("a0", "point_only"), ("a0", "history_key"),
        ("g1", "point_only"), ("g1", "history_key"),
        ("g1_r3", "point_only"), ("g1_r3", "history_key"),
        ("g1_r3", "retrieved_future"), ("g1_r3", "all_fused"),
        ("g1_r3", "wrong_pair_train"),
    ]
    rows = []
    for anchor, mode in plan:
        name = f"{anchor}__{mode}"
        metrics_path = a.output / f"{name}_metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
        elif mode == "point_only":
            metrics = point_metrics(test[anchor])
            metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
        else:
            model, scale = load_or_train(name, a.case, a.output, train[anchor],
                                         meta["pred_len"], device)
            score_mode = "all_fused" if mode == "wrong_pair_train" else mode
            metrics = score(model, test[anchor], device, score_mode, seed=2021,
                            sample_scale=scale, stride=1)
            metrics["calibrated_scale"] = scale
            if metrics["evaluated_windows"] != test_count or not math.isfinite(metrics["CRPS"]):
                raise AssertionError("Full TEST scoring failed")
            metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
        rows.append({"backbone": meta["backbone"], "dataset": meta["dataset"],
                     "pred_len": meta["pred_len"], "point_anchor": anchor,
                     "flow_condition": mode, "model_capacity": "same C12 Flow" if mode != "point_only" else "none",
                     "training_budget": "8 epochs" if mode != "point_only" else "none",
                     "test_selection": "exploratory", **metrics})
        print("P2", name, metrics.get("CRPS", metrics.get("point_MSE")), flush=True)
    with (a.output / "three_stage_ablation.csv").open("w", newline="") as file:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (a.output / "run_config.json").write_text(json.dumps({
        "case": str(a.case), "gpu": a.gpu, "plan": plan,
        "point_source": "strict forward OOF for Flow TRAIN; TEST labels only for scoring",
        "flow_capacity": "identical ConditionedResidualFlow for all Flow rows",
        "selection": "Flow calibration from held-out OOF CAL; original point hyperparameters TEST-selected",
    }, indent=2) + "\n")
    (a.output / "verified.complete").write_text("verified\n")


if __name__ == "__main__":
    main()
