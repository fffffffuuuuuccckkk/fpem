#!/usr/bin/env python3
"""Reuse frozen EIIL3 F03/router artifacts; replace only residual Ridge by gated linear experts.

Offline TRAIN B1 fit, TRAIN B2 LR/epoch selection, VAL/TEST reporting. COSA-inspired
linear output correction, not the paper's online test-time adaptation protocol.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.predictive_env.eiil3_linear_correction import (
    EnvironmentLinearCorrection, correction_design, weighted_environment_loss,
)
from models.predictive_env.relation_prototypes import route
from tools.run_fpem_eiil3 import file_hash, dump, SPACES
from tools.run_fpem_mechanism_fusion import score


def router_from_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    network = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.Tanh(), torch.nn.Linear(32, 3))
    network.load_state_dict(state)
    network.eval()
    return network


def stable_forecast(state_path, record):
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    weight = state["weight"].numpy()
    bias = state["bias"].numpy()
    result = []
    for start in range(0, len(record["design"]), 512):
        end = start + 512
        result.append(((record["design"][start:end] @ weight.T + bias)
                       * record["ratio"][start:end]).astype(np.float32))
    return np.concatenate(result)


@torch.no_grad()
def predict(model, features, router_q, device, batch_size=256):
    corrections, uncertainties = [], []
    model.eval()
    for start in range(0, len(features), batch_size):
        end = start + batch_size
        x = torch.from_numpy(features[start:end]).to(device)
        q = torch.from_numpy(router_q[start:end]).to(device)
        _, correction, u = model(x, q)
        corrections.append(correction.cpu().numpy())
        uncertainties.append(u.cpu().numpy())
    return np.concatenate(corrections), np.concatenate(uncertainties)


def fit(features, stable, truth, q, select_features, select_stable, select_truth,
        select_q, lr, epochs, batch_size, seed, device):
    torch.manual_seed(seed)
    model = EnvironmentLinearCorrection(features.shape[1], truth.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    inputs = [torch.from_numpy(array) for array in (features, stable, truth, q)]
    best = None
    trace = []
    for epoch in range(epochs + 1):
        correction, _ = predict(model, select_features, select_q, device)
        valid = score(select_stable + correction, select_truth)["mse"]
        trace.append({"epoch": epoch, "B2_mse": valid,
                      "g": model.g.detach().cpu().tolist(),
                      "tanh_g": torch.tanh(model.g).detach().cpu().tolist()})
        if best is None or valid < best[0]:
            best = (valid, epoch, copy.deepcopy({key: value.cpu() for key, value in model.state_dict().items()}))
        if epoch == epochs:
            break
        model.train()
        for index in torch.randperm(len(features)).split(batch_size):
            x, base, y, weights = (item[index].to(device) for item in inputs)
            candidates, _, _ = model(x, weights)
            loss = weighted_environment_loss(candidates, base, y, weights)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite gated linear correction loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.load_state_dict(best[2]); model.eval()
    return model, {"lr": lr, "best_epoch": best[1], "B2_mse": best[0], "trace": trace}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-case", type=Path, required=True)
    p.add_argument("--output-case", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lrs", default="0.0001,0.0003,0.001")
    p.add_argument("--skip-good-ridge", action="store_true")
    p.add_argument("--resume", action="store_true")
    cli = p.parse_args(); torch.set_num_threads(4)
    source, dest = cli.source_case, cli.output_case
    if not (source / "point_complete").exists():
        raise FileNotFoundError("source F03/Ridge point stage incomplete")
    dest.mkdir(parents=True, exist_ok=True)
    original = json.loads((source / "point_metrics.json").read_text())
    source_sha = file_hash(source / "point_metrics.json")
    marker = dest / "point_complete"
    if cli.resume and marker.exists():
        previous = json.loads((dest / "point_metrics.json").read_text())
        if previous["source_point_sha256"] == source_sha:
            print(json.dumps({"case": original["case"], "status": "REUSED"}), flush=True)
            return
    ridge = original["scores"]["B2"]["I3_R3_EIIL3"]["mse"]
    anchor = original["scores"]["B2"]["I3_SoftIRM_EIIL3"]["mse"]
    if cli.skip_good_ridge and ridge < anchor:
        dump(dest / "linear_status.json", {"status": "SKIPPED_GOOD_RIDGE", "case": original["case"],
                                          "B2_ridge_mse": ridge, "B2_anchor_mse": anchor,
                                          "source_point_sha256": source_sha})
        print(json.dumps({"case": original["case"], "status": "SKIPPED_GOOD_RIDGE"}), flush=True)
        return
    data = {}
    with np.load(source / "features.npz") as cache:
        for space in ("B1", "B2", "val", "test"):
            data[space] = {field: cache[f"{space}_{field}"] for field in
                           ("design", "key", "y", "a0", "ratio", "start")}
    router = router_from_checkpoint(source / "router.pt")
    shuffled_router = router_from_checkpoint(source / "router_shuffled.pt")
    for space in data:
        record = data[space]
        record["stable"] = stable_forecast(source / "I3_SoftIRM_EIIL3_head.pt", record)
        record["features"] = correction_design(record["stable"], record["key"])
        record["q"] = route(router, record["key"])
        record["q_shuffled"] = route(shuffled_router, record["key"])
    device = torch.device(f"cuda:{cli.gpu}" if torch.cuda.is_available() else "cpu")
    train, select = data["B1"], data["B2"]
    lrs = [float(value) for value in cli.lrs.split(",")]
    if not lrs or any(value <= 0 for value in lrs):
        raise ValueError("positive learning rates required")
    trials = []
    best = None
    for lr in lrs:
        model, info = fit(train["features"], train["stable"], train["y"], train["q"],
                          select["features"], select["stable"], select["y"], select["q"],
                          lr, cli.epochs, cli.batch_size, cli.seed, device)
        trials.append(info)
        if best is None or info["B2_mse"] < best[0]:
            best = (info["B2_mse"], model.cpu(), info)
    _, model, chosen = best
    model.to(device)
    negative, negative_info = fit(train["features"], train["stable"], train["y"],
                                  train["q_shuffled"], select["features"], select["stable"],
                                  select["y"], select["q_shuffled"], chosen["lr"], cli.epochs,
                                  cli.batch_size, cli.seed, device)
    torch.save(model.cpu().state_dict(), dest / "linear_eiil3.pt")
    torch.save(negative.cpu().state_dict(), dest / "linear_shuffleq.pt")
    model.to(device); negative.to(device)
    shutil.copyfile(source / "features.json", dest / "features.json")
    result = {"case": original["case"], "status": "complete", "method": "COSA_inspired_offline_3_linear",
              "source_point_sha256": source_sha, "A0_sha256": original["A0_sha256"],
              "router_sha256": original["router_sha256"], "I3_sha256": original["I3_sha256"],
              "R3_sha256": file_hash(dest / "linear_eiil3.pt"),
              "selected_lr": chosen["lr"], "selected_epoch": chosen["best_epoch"],
              "gate_tanh": torch.tanh(model.g).detach().cpu().tolist(),
              "negative_gate_tanh": torch.tanh(negative.g).detach().cpu().tolist(),
              "lr_trials": trials, "negative_info": negative_info,
              "skip_rule": "only cases where Ridge failed to improve TRAIN B2 anchor",
              "scores": {}}
    flow = {}
    for space in ("B2", "val", "test"):
        record = data[space]
        correction, u = predict(model, record["features"], record["q"], device)
        shuffled_correction, _ = predict(negative, record["features"], record["q_shuffled"], device)
        full = record["stable"] + correction
        result["scores"][space] = {
            "A0": score(record["a0"], record["y"]),
            "I3_SoftIRM_EIIL3": score(record["stable"], record["y"]),
            "I3_R3_Ridge_EIIL3": original["scores"][space]["I3_R3_EIIL3"],
            "I3_R3_Linear_EIIL3": score(full, record["y"]),
            "I3_R3_Linear_ShuffleQ": score(record["stable"] + shuffled_correction, record["y"]),
        }
        if space in ("val", "test"):
            for name, value in (("key", record["key"]), ("y", record["y"]),
                                ("start", record["start"]), ("point", full), ("u", u)):
                flow[f"{space}_{name}"] = value
        if space == "test":
            result["u_test"] = {"mean": float(u.mean()), "std": float(u.std()),
                                "candidate_spread": float(np.sqrt(u).mean())}
    np.savez_compressed(dest / "flow_input.npz", **flow)
    dump(dest / "point_metrics.json", result)
    dump(dest / "run_config.json", {"source_case": str(source), "output_case": str(dest),
                                    "epochs": cli.epochs, "lrs": lrs, "seed": cli.seed,
                                    "input": "[F03_full_forecast,64_history_key,11_recent_observed_stats]",
                                    "fit": "TRAIN B1", "selection": "TRAIN B2", "test": "report_only"})
    marker.write_text("complete\n")
    print(json.dumps({"case": original["case"], "status": "complete",
                      "B2": result["scores"]["B2"], "test": result["scores"]["test"]}), flush=True)


if __name__ == "__main__":
    main()
