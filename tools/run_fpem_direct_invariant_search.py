"""Independent, TRAIN-supervised direct H->future pilot (no G1 residual)."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.predictive_env.direct_invariant_forecaster import (
    DirectInvariantForecaster, direct_ridge, group_risk)
from tools.run_fpem_shared_factorization_g1r3 import load_experiment
from tools.run_predictive_env_iv_patchtst import IndexedDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_run_config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--train_stride", type=int, default=16)
    p.add_argument("--val_stride", type=int, default=4)
    p.add_argument("--test_stride", type=int, default=1)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--methods", default="F00,F01,F02,F03,F08,E01_VREx,E01_DRO,E02_VREx,P01,P02")
    p.add_argument("--allow_legacy_reference_signature", action="store_true")
    return p.parse_args()


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@torch.no_grad()
def collect(model, dataset, args, device, stride):
    ds = IndexedDataset(dataset, cycle_len=args.cyclenet_cycle_len)
    loader = DataLoader(Subset(ds, range(0, len(dataset), stride)),
                        batch_size=args.batch_size, num_workers=0, shuffle=False)
    chunks = {name: [] for name in ("hidden", "mean", "std", "truth", "base", "ids")}
    model.eval()
    for x, y, ids, cycle in loader:
        x, cycle = x.float().to(device), cycle.to(device)
        h, mean, std = model.encode(x, cycle)
        base = model.frozen_h_head(h, cycle).transpose(1, 2) * std + mean
        vals = {"hidden": h, "mean": mean, "std": std,
                "truth": y[:, -args.pred_len:], "base": base, "ids": ids}
        for name, value in vals.items():
            chunks[name].append(value.detach().cpu())
    return {name: torch.cat(items) for name, items in chunks.items()}


def metrics(data, pred):
    err = pred - data["truth"]
    return {"mse": float(err.square().mean()), "mae": float(err.abs().mean()),
            "pred_std": float(pred.std(unbiased=False))}


@torch.no_grad()
def predict(net, data, device, batch_size=64):
    out = []
    net.eval()
    for start in range(0, len(data["hidden"]), batch_size):
        stop = start + batch_size
        h = data["hidden"][start:stop].float().to(device)
        y = net(h).transpose(1, 2).cpu()
        out.append(y * data["std"][start:stop] + data["mean"][start:stop])
    return torch.cat(out)


def time_groups(data, args, count=4):
    # Use only historical sample start; no future label enters assignment.
    starts = data["ids"].long()
    boundaries = torch.quantile(starts.float(), torch.linspace(0, 1, count + 1)[1:-1])
    return torch.bucketize(starts.float(), boundaries)


def cluster_groups(data, count=4, seed=2021):
    # Simple TRAIN-only H clustering. Validation/test never infer a label.
    pooled = data["hidden"].float().mean((1, 2))
    pooled = (pooled - pooled.mean(0)) / pooled.std(0).clamp_min(1e-5)
    g = torch.Generator().manual_seed(seed)
    center = pooled[torch.randperm(len(pooled), generator=g)[:count]].clone()
    for _ in range(12):
        label = torch.cdist(pooled, center).argmin(1)
        for k in range(count):
            if (label == k).any():
                center[k] = pooled[label == k].mean(0)
    return torch.cdist(pooled, center).argmin(1)


def train_direct(method, train, val, test, model, args, device, cli, sha):
    h = train["hidden"]
    p, d = h.shape[-2:]
    rank = 16 if method in ("P01", "P02") else 0
    pretrain = model.h_forecast_head if method in ("F03", "F08", "E01_VREx", "E01_DRO", "E02_VREx") else None
    net = DirectInvariantForecaster(p, d, args.pred_len, rank, pretrain).to(device)
    if method == "F03":
        initial = metrics(val, predict(net, val, device))["mse"]
        anchor = metrics(val, val["base"])["mse"]
        if abs(initial - anchor) > 1e-6:
            raise AssertionError(f"A0 head initialization mismatch: {initial} vs {anchor}")
    opt = torch.optim.Adam(net.parameters(), lr=3e-4 if method == "F03" else 1e-3)
    groups = (time_groups(train, args) if method.startswith("E01") else
              cluster_groups(train, seed=cli.seed) if method.startswith("E02") else None)
    generator = torch.Generator().manual_seed(cli.seed)
    best = None
    t0 = time.monotonic()
    history = []
    for epoch in range(cli.epochs):
        net.train()
        order = torch.randperm(len(h), generator=generator)
        losses = []
        for ids in order.split(args.batch_size):
            x = h[ids].float().to(device)
            truth = train["truth"][ids].float().to(device)
            mean = train["mean"][ids].float().to(device)
            std = train["std"][ids].float().to(device)
            pred = net(x).transpose(1, 2) * std + mean
            if groups is not None:
                kind = "dro" if method.endswith("DRO") else "vrex"
                loss = group_risk(pred, truth, groups[ids].to(device), kind, 0.1)
            elif method == "F08":
                sample_risk = (pred - truth).square().mean((1, 2))
                loss = 0.1 * (torch.logsumexp(sample_risk / 0.1, dim=0)
                              - math.log(len(sample_risk)))
            else:
                loss = (pred - truth).square().mean()
            if method == "P02":
                loss = loss + 0.01 * net.orthogonality_loss()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite direct loss in {method}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        vm = metrics(val, predict(net, val, device))
        history.append({"epoch": epoch + 1, "train_loss": sum(losses) / len(losses), **vm})
        if best is None or vm["mse"] < best["val"]["mse"]:
            best = {"epoch": epoch + 1, "val": vm,
                    "state": {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}}
        print(method, "epoch", epoch + 1, "val", round(vm["mse"], 6), flush=True)
    net.load_state_dict(best.pop("state"))
    result = {"method": method, "best_epoch": best["epoch"],
              "val": best["val"], "test": metrics(test, predict(net, test, device)),
              "train_rows": len(train["hidden"]), "rank": rank,
              "trainable_parameters": sum(v.numel() for v in net.parameters()),
              "elapsed_seconds": time.monotonic() - t0,
              "reference_sha256": sha,
              "selection_protocol": "VAL epoch; exploratory TEST method ranking",
              "loss_target": "complete true future Y; never A0 residual",
              "history": history}
    return result, net.state_dict()


def fit_ridge(train, val, test, model, args, cli, sha):
    h = train["hidden"].float()
    x = h.permute(0, 1, 3, 2).flatten(-2).reshape(-1, h.shape[-1] * h.shape[-2])
    y = ((train["truth"] - train["mean"]) / train["std"]).transpose(1, 2).reshape(-1, args.pred_len)
    best = None
    for ridge in (0.01, 0.1, 1.0):
        w, b, rows = direct_ridge(x, y, ridge, max_rows=1024, seed=cli.seed)
        def apply(data):
            z = data["hidden"].float().permute(0, 1, 3, 2).flatten(-2)
            norm = (z @ w + b).transpose(1, 2)
            return norm * data["std"] + data["mean"]
        vm = metrics(val, apply(val))
        if best is None or vm["mse"] < best["val"]["mse"]:
            best = {"ridge": ridge, "val": vm, "test": metrics(test, apply(test)),
                    "weight": w, "bias": b, "train_rows": rows}
    weight = {"weight": best.pop("weight"), "bias": best.pop("bias")}
    best.update({"method": "F02", "reference_sha256": sha,
                 "selection_protocol": "VAL Ridge; exploratory TEST method ranking",
                 "loss_target": "complete true future Y; never A0 residual"})
    return best, weight


def main():
    cli = parse_args()
    out = Path(cli.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(cli.seed)
    args, model, loaders, device, sha = load_experiment(cli)
    if any(p.requires_grad for p in model.parameters()):
        raise AssertionError("frozen pilot must not update A0")
    datasets = {name: loader.dataset.dataset for name, loader in loaders.items()}
    data = {name: collect(model, datasets[name], args, device,
                          {"train": cli.train_stride, "val": cli.val_stride,
                           "test": cli.test_stride}[name])
            for name in ("train", "val", "test")}
    h = data["train"]["hidden"]
    if h.ndim != 4 or h.shape[1] != args.enc_in:
        raise AssertionError("unexpected backbone H shape")
    methods = cli.methods.split(",")
    registry = []
    for method in methods:
        target = out / method
        if (target / "complete").is_file():
            registry.append(json.loads((target / "metrics.json").read_text()))
            continue
        target.mkdir(parents=True, exist_ok=True)
        print("BEGIN", args.backbone, args.dataset_name, args.pred_len, method, flush=True)
        if method == "F00":
            result = {"method": "F00", "val": metrics(data["val"], data["val"]["base"]),
                      "test": metrics(data["test"], data["test"]["base"]),
                      "reference_sha256": sha, "trainable_parameters": 0}
            state = None
        elif method == "F02":
            result, state = fit_ridge(data["train"], data["val"], data["test"], model, args, cli, sha)
        else:
            result, state = train_direct(method, data["train"], data["val"],
                                         data["test"], model, args, device, cli, sha)
        result.update({"backbone": args.backbone, "dataset": args.dataset_name,
                       "pred_len": args.pred_len, "seed": cli.seed,
                       "hidden_shape": list(h.shape), "train_stride": cli.train_stride,
                       "val_stride": cli.val_stride, "test_stride": cli.test_stride})
        if state is not None:
            torch.save(state, target / "model.pt")
        save_json(target / "metrics.json", result)
        (target / "complete").write_text("complete\n", encoding="utf-8")
        registry.append(result)
    save_json(out / "run_config.json", {"cli": vars(cli), "reference_sha256": sha,
                "backbone": args.backbone, "dataset": args.dataset_name,
                "pred_len": args.pred_len, "hidden_shape": list(h.shape)})
    with (out / "pilot_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "val_mse", "val_mae",
                     "test_mse", "test_mae", "best_epoch", "reference_sha256"])
        writer.writeheader()
        for row in registry:
            writer.writerow({"method": row["method"], "val_mse": row["val"]["mse"],
                        "val_mae": row["val"]["mae"], "test_mse": row["test"]["mse"],
                        "test_mae": row["test"]["mae"], "best_epoch": row.get("best_epoch"),
                        "reference_sha256": sha})
    print("PILOT COMPLETE", out, flush=True)


if __name__ == "__main__":
    main()
