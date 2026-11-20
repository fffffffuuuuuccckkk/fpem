"""Bounded TRAIN-only F02 Ridge audit; VAL selects the deployed candidate.

The chronological TRAIN holdout is diagnostic, not a tuning target. TEST is
evaluated once for the VAL-selected candidate and is never used to select it.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from tools.run_fpem_direct_invariant_search import collect, metrics
from tools.run_fpem_shared_factorization_g1r3 import load_experiment


RIDGES = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0)


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--base_run_config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--train_stride", type=int, default=16)
    p.add_argument("--val_stride", type=int, default=4)
    p.add_argument("--test_stride", type=int, default=1)
    p.add_argument("--max_rows", type=int, default=1024)
    p.add_argument("--allow_legacy_reference_signature", action="store_true")
    return p.parse_args()


def features(data):
    h = data["hidden"].float()
    return h.permute(0, 1, 3, 2).flatten(-2).reshape(-1, h.shape[-1] * h.shape[-2]).double()


def targets(data):
    return (((data["truth"] - data["mean"]) / data["std"])
            .transpose(1, 2).reshape(-1, data["truth"].shape[1]).double())


def subset(data, sample_ids):
    return {key: value[sample_ids] for key, value in data.items()}


def ridge_spectrum(x, y, mode, max_rows, seed):
    if len(x) > max_rows:
        indices = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))[:max_rows]
        x, y = x[indices], y[indices]
    x_mean = x.mean(0)
    x_scale = torch.ones_like(x_mean)
    if mode == "standardized":
        raw_std = x.std(0, unbiased=False)
        x_scale = torch.where(raw_std > 1e-6, raw_std, torch.ones_like(raw_std))
    x = (x - x_mean) / x_scale
    y_mean = y.mean(0)
    y = y - y_mean
    feature_count = x.shape[1]
    gram = x @ x.T / feature_count
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    eigenvalues = eigenvalues.clamp_min(0)
    projected_y = eigenvectors.T @ y
    return x, x_mean, x_scale, y_mean, eigenvalues, eigenvectors, projected_y


def predict_norm(fit, x_eval, ridge):
    x, x_mean, x_scale, y_mean, eigenvalues, eigenvectors, projected_y = fit
    alpha = eigenvectors @ (projected_y / (eigenvalues[:, None] + ridge))
    chunks = []
    for start in range(0, len(x_eval), 512):
        z = (x_eval[start:start + 512] - x_mean) / x_scale
        kernel = z @ x.T / x.shape[1]
        chunks.append(y_mean + kernel @ alpha)
    return torch.cat(chunks).float()


def actual_metrics(data, norm):
    n, horizon, channels = data["truth"].shape
    pred = norm.reshape(n, channels, horizon).transpose(1, 2)
    pred = pred * data["std"] + data["mean"]
    return metrics(data, pred)


def main():
    cli = arguments()
    out = Path(cli.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(cli.seed)
    args, model, loaders, device, sha = load_experiment(cli)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("A0 must remain frozen")
    datasets = {name: loader.dataset.dataset for name, loader in loaders.items()}
    data = {name: collect(model, datasets[name], args, device,
                          {"train": cli.train_stride, "val": cli.val_stride,
                           "test": cli.test_stride}[name])
            for name in ("train", "val", "test")}
    train, val, test = data["train"], data["val"], data["test"]
    x_train, y_train = features(train), targets(train)
    x_val = features(val)
    channels = train["truth"].shape[-1]
    starts = train["ids"].long()
    split_at = int(torch.quantile(starts.float(), 0.75).item())
    fit_samples = torch.nonzero(starts + args.seq_len + args.pred_len <= split_at).flatten()
    hold_samples = torch.nonzero(starts >= split_at).flatten()
    if len(fit_samples) < 16 or len(hold_samples) < 16:
        raise ValueError("not enough strictly embargoed TRAIN samples for chronological audit")
    hold_data = subset(train, hold_samples)
    fit_rows = (fit_samples[:, None] * channels + torch.arange(channels)[None, :]).flatten()
    hold_rows = (hold_samples[:, None] * channels + torch.arange(channels)[None, :]).flatten()
    rows = []
    candidates = {}
    for mode in ("raw", "standardized"):
        fit_oof = ridge_spectrum(x_train[fit_rows], y_train[fit_rows], mode, cli.max_rows, cli.seed)
        fit_full = ridge_spectrum(x_train, y_train, mode, cli.max_rows, cli.seed)
        for ridge in RIDGES:
            oof = actual_metrics(hold_data, predict_norm(fit_oof, x_train[hold_rows], ridge))
            val_metrics = actual_metrics(val, predict_norm(fit_full, x_val, ridge))
            row = {"mode": mode, "ridge": ridge, "train_fit_rows": len(fit_full[0]),
                   "oof_fit_rows": len(fit_oof[0]), "oof_samples": len(hold_samples),
                   "train_forward_oof_mse": oof["mse"], "train_forward_oof_mae": oof["mae"],
                   "val_mse": val_metrics["mse"], "val_mae": val_metrics["mae"]}
            rows.append(row)
            candidates[(mode, ridge)] = fit_full
            print(mode, ridge, "OOF", round(oof["mse"], 6),
                  "VAL", round(val_metrics["mse"], 6), flush=True)
    best = min(rows, key=lambda row: row["val_mse"])
    key = (best["mode"], best["ridge"])
    chosen_test = actual_metrics(test, predict_norm(candidates[key], features(test), best["ridge"]))
    a0_val = metrics(val, val["base"])
    a0_test = metrics(test, test["base"])
    result = {"backbone": args.backbone, "dataset": args.dataset_name,
              "pred_len": args.pred_len, "reference_sha256": sha,
              "selection": "VAL MSE only; TEST evaluated once after selection",
              "oof": "chronological TRAIN holdout with future-end embargo",
              "max_rows": cli.max_rows, "best": best, "selected_test": chosen_test,
              "a0_val": a0_val, "a0_test": a0_test}
    with (out / "ridge_grid.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "selection.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (out / "complete").write_text("complete\n", encoding="utf-8")
    print("RIDGE AUDIT COMPLETE", out, flush=True)


if __name__ == "__main__":
    main()
