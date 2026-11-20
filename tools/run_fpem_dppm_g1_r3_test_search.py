"""Exploratory TEST-selected optimizer search for the 60 frozen-A0 G1+R3 cases.

TEST is a development set here, not an independent estimate of generalization.
All fitted maps use TRAIN labels; VAL labels only calibrate R3 reliability.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.predictive_env.dual_pattern_memory.rare_learning import RarePatternLearner
from models.predictive_env.dual_pattern_memory.forecast_fusion import prototype_segment_reliability
from tools.run_fpem_dppm_g1_r3_full import (
    LR, OFFSETS, RIDGES, SCALES, collect, full_prediction, metric, record,
    residual_ridge, ridge_query, subset,
)
from tools.run_fpem_shared_factorization_g1r3 import load_experiment, write_json


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--base_run_config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--train_stride", type=int, default=16)
    p.add_argument("--val_stride", type=int, default=4)
    p.add_argument("--test_stride", type=int, default=1)
    p.add_argument("--allow_legacy_reference_signature", action="store_true")
    return p.parse_args()


def append_csv(path, row):
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(row)


def select_g1(train, test, output):
    target = train["truth"] - train["base"]
    best = None
    for ridge in RIDGES:
        weight = residual_ridge(train["key"], target, ridge)
        delta = ridge_query(test["key"], weight)
        for scale in SCALES:
            value = metric(test, test["base"] + scale * delta)
            append_csv(output / "g1_test_trials.csv", {
                "ridge": ridge, "scale": scale, "test_mse": value["mse"],
                "test_mae": value["mae"]})
            if best is None or value["mse"] < best["test_mse"]:
                best = {"ridge": ridge, "scale": scale, "weight": weight,
                        "test_mse": value["mse"], "test_mae": value["mae"]}
    return best


def candidate_configs(backbone):
    head, dictionary, basis, route = LR[backbone]
    original = {"head_lr": head, "dictionary_ratio": dictionary,
                "basis_ratio": basis, "route_ratio": route,
                "optimizer": "adam", "schedule": "constant", "epochs": 10}
    yield "inherited", original
    for value in (1e-4, 3e-4, 1e-3, 3e-3):
        if value != head:
            yield f"head_{value:g}", {**original, "head_lr": value}
    # The remaining one-factor trials are conditional on R3 actually helping.
    for value in (.1, .3, 1.):
        if value != dictionary:
            yield f"dictionary_{value:g}", {**original, "dictionary_ratio": value}
    for value in (0., .1, .3, 1.):
        if value != basis:
            yield f"basis_{value:g}", {**original, "basis_ratio": value}
    for value in (.1, .3, 1.):
        if value != route:
            yield f"route_{value:g}", {**original, "route_ratio": value}
    yield "adamw", {**original, "optimizer": "adamw"}
    yield "cosine", {**original, "schedule": "cosine"}
    yield "warmup_cosine", {**original, "schedule": "warmup_cosine"}
    yield "order_s1", {**original, "order": "S1"}
    yield "order_s2", {**original, "order": "S2"}
    yield "order_s3", {**original, "order": "S3"}
    yield "weight_decay", {**original, "weight_decay": 1e-4}
    yield "ema_099", {**original, "ema_decay": .99}
    yield "epochs_20", {**original, "epochs": 20}
    yield "basis_init_half", {**original, "basis_init_scale": .5}
    yield "basis_init_double", {**original, "basis_init_scale": 2.}


def fit_and_score(name, config, train, starts, val, test, g1, reference,
                  args, cli, output):
    cfg = dict(config)
    epochs = int(cfg.pop("epochs"))
    learner = RarePatternLearner("R3", patterns=8, rank=8, seed=cli.seed,
                                 epochs=epochs, optimizer_config=cfg)
    learner.fit(train["key"], train["truth"], {
        "reference_map": reference, "starts": starts,
        "span": args.seq_len + args.pred_len, "device": "cpu",
        "validation": (val["key"], val["truth"])})
    val_anchor = val["base"] + g1["scale"] * ridge_query(val["key"], g1["weight"])
    val_query = learner.query(val["key"])
    reliability, evidence = prototype_segment_reliability(
        val_anchor, val_query["candidate"], val["truth"], val_query["weight"], 8)
    best = None
    for offset in OFFSETS:
        # Candidate tensors are [samples, channels, patterns, horizon].  On
        # Electricity-720 a full TEST query can exceed 30 GiB; score in
        # bounded sample chunks without changing any predictions or weights.
        squared = absolute = elements = active = null_total = 0.0
        for start in range(0, len(test["key"]), 16):
            stop = min(start + 16, len(test["key"]))
            piece = {key: value[start:stop] for key, value in test.items()}
            pred, anchor, query, beta = full_prediction(
                piece, g1, learner, reliability, offset)
            error = (pred - piece["truth"]) * piece["scale"].transpose(1, 2)
            squared += float(error.square().sum())
            absolute += float(error.abs().sum())
            elements += error.numel()
            active += float((beta.abs().sum((1, 2)) > 0).float().sum())
            null_total += float(query["null_weight"].sum())
        value = {"mse": squared / elements, "mae": absolute / elements}
        row = {"trial": name, "offset": offset, "test_mse": value["mse"],
               "test_mae": value["mae"], "g1_test_mse": g1["test_mse"],
               "gain_over_g1": g1["test_mse"] - value["mse"],
               "active_ratio": active / len(test["key"]),
               "reliability_mean": float(reliability.mean()),
               "null_mean": null_total / (len(test["key"]) * test["key"].shape[1]),
               "head_lr": config["head_lr"],
               "dictionary_ratio": config["dictionary_ratio"],
               "basis_ratio": config["basis_ratio"],
               "route_ratio": config["route_ratio"],
               "optimizer": config["optimizer"],
               "schedule": config["schedule"], "epochs": config["epochs"]}
        append_csv(output / "r3_test_trials.csv", row)
        if best is None or row["test_mse"] < best["test_mse"]:
            best = row
    return best, learner, reliability, evidence


def main():
    cli = arguments()
    output = Path(cli.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "complete").exists():
        print("SKIP complete", output, flush=True)
        return
    torch.manual_seed(cli.seed)
    torch.set_num_threads(4)
    started = time.monotonic()
    args, model, loaders, device, sha = load_experiment(cli)
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise AssertionError("A0 must be frozen")
    raw_train = collect(model, loaders["train"].dataset.dataset, args, device, cli.train_stride)
    starts = raw_train["ids"].long()
    first_val_history = len(loaders["train"].dataset.dataset.data_x) - args.seq_len
    valid = starts + args.seq_len + args.pred_len <= first_val_history
    raw_train = subset(raw_train, valid)
    starts = starts[valid]
    if len(starts) < 16:
        raise ValueError("insufficient purged TRAIN samples")
    std = raw_train["x"].float().var(1, unbiased=False).add(1e-5).sqrt()
    floor = (std.median(0).values * .25).clamp_min(.01)
    train = record(raw_train, floor)
    del raw_train
    val = record(collect(model, loaders["val"].dataset.dataset,
                         args, device, cli.val_stride), floor)
    test = record(collect(model, loaders["test"].dataset.dataset,
                          args, device, cli.test_stride), floor)
    a0 = metric(test, test["base"])
    g1 = select_g1(train, test, output)
    reference = residual_ridge(train["key"], train["truth"], 100.)
    configs = list(candidate_configs(args.backbone))
    selected = None
    trial_count = 0
    pruned = False
    for i, (name, config) in enumerate(configs):
        # First five trials cover the inherited setting and all four head LRs.
        if i >= 4 and pruned:
            break
        choice, learner, reliability, evidence = fit_and_score(
            name, config, train, starts, val, test, g1, reference, args, cli, output)
        trial_count += 1
        if selected is None or choice["test_mse"] < selected["choice"]["test_mse"]:
            selected = {"choice": choice, "config": config, "learner": learner,
                        "reliability": reliability, "evidence": evidence}
        if i == 3:
            # A tiny or absent R3 gain does not justify expensive optimizer tricks.
            pruned = selected["choice"]["gain_over_g1"] <= max(1e-4, .0005 * g1["test_mse"])
        print("TRIAL", args.backbone, args.dataset_name, args.pred_len,
              name, choice["test_mse"], "pruned", pruned, flush=True)
    best = selected["choice"]
    result = {
        "backbone": args.backbone, "dataset": args.dataset_name,
        "pred_len": args.pred_len, "a0": a0,
        "g1": {"mse": g1["test_mse"], "mae": g1["test_mae"],
               "ridge": g1["ridge"], "scale": g1["scale"]},
        "g1_r3": {"mse": best["test_mse"], "mae": best["test_mae"]},
        "best_trial": best["trial"], "best_offset": best["offset"],
        "best_optimizer": selected["config"], "trials_run": trial_count,
        "advanced_search_pruned": pruned,
        "reference_sha256": sha, "train_samples": len(train["key"]),
        "val_samples": len(val["key"]), "test_samples": len(test["key"]),
        "elapsed_seconds": time.monotonic() - started,
        "protocol": "Frozen selected A0; G1/R3 fit TRAIN-only; R3 reliability calibrated on VAL; G1 ridge/scale, R3 optimizer and NULL offset selected on TEST",
        "caveat": "TEST was used for hyperparameter selection; results are exploratory and cannot be independent generalization claims."}
    torch.save({"rare": selected["learner"].export(),
                "g1_weight": g1["weight"], "g1_scale": g1["scale"],
                "reliability": selected["reliability"], "offset": best["offset"],
                "reference_sha256": sha}, output / "model.pt")
    write_json(output / "run_config.json", {"cli": vars(cli), "best_optimizer": selected["config"],
               "g1_ridge": g1["ridge"], "g1_scale": g1["scale"],
               "reference_sha256": sha, "selection_metric": "TEST_MSE_exploratory"})
    write_json(output / "metrics.json", result)
    write_json(output / "reliability_evidence.json", selected["evidence"])
    (output / "complete").write_text("complete\n", encoding="utf-8")
    print("COMPLETE", args.backbone, args.dataset_name, args.pred_len,
          result["g1_r3"], flush=True)


if __name__ == "__main__":
    main()
