"""VAL-selected F03 head-only LR pilot using one frozen, reusable A0."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from models.predictive_env.direct_invariant_forecaster import DirectInvariantForecaster
from tools.run_fpem_direct_invariant_search import collect, metrics, predict
from tools.run_fpem_shared_factorization_g1r3 import load_experiment


LRS = (2e-5, 5e-5, 1e-4)


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--base_run_config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--train_stride", type=int, default=16)
    p.add_argument("--val_stride", type=int, default=4)
    p.add_argument("--test_stride", type=int, default=1)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--allow_legacy_reference_signature", action="store_true")
    return p.parse_args()


def source_baseline(args, sha):
    root = Path("results")
    candidates = (
        root / "fpem_performance_optimization_20261103/direct_shared_pilot"
        / args.backbone / args.dataset_name / f"pred_{args.pred_len}" / "F03",
        root / "fpem_direct_invariant_20261102/pilot"
        / args.backbone / args.dataset_name / f"pred_{args.pred_len}" / "F03",
    )
    for folder in candidates:
        file = folder / "metrics.json"
        if not (folder / "complete").is_file() or not file.is_file():
            continue
        data = json.loads(file.read_text(encoding="utf-8"))
        if (data.get("reference_sha256") == sha and data.get("seed") == 2021
                and data.get("train_stride") == 16 and data.get("val_stride") == 4
                and data.get("test_stride") == 1):
            return {"lr": 3e-4, "val": data["val"], "test": data["test"],
                    "best_epoch": data["best_epoch"], "source": str(file),
                    "state_file": str(folder / "model.pt")}
    raise FileNotFoundError("matching completed 3e-4 F03 baseline is required; do not retrain A0")


def train_one(lr, data, model, args, cli, device, folder, anchor_lambda=0.0):
    file = folder / "metrics.json"
    state_file = folder / "model.pt"
    if (folder / "complete").is_file() and file.is_file() and state_file.is_file():
        result = json.loads(file.read_text(encoding="utf-8"))
        result["state_file"] = str(state_file)
        return result
    folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cli.seed)
    hidden = data["train"]["hidden"]
    token_count, dim = hidden.shape[-2:]
    net = DirectInvariantForecaster(token_count, dim, args.pred_len,
                                   pretrained_head=model.h_forecast_head).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(cli.seed)
    best = None
    history = []
    for epoch in range(cli.epochs):
        net.train()
        losses = []
        for ids in torch.randperm(len(hidden), generator=generator).split(args.batch_size):
            x = hidden[ids].float().to(device)
            truth = data["train"]["truth"][ids].float().to(device)
            mean = data["train"]["mean"][ids].float().to(device)
            std = data["train"]["std"][ids].float().to(device)
            prediction = net(x).transpose(1, 2) * std + mean
            loss = (prediction - truth).square().mean()
            if anchor_lambda:
                a0 = data["train"]["base"][ids].float().to(device)
                loss = loss + anchor_lambda * (prediction - a0).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite F03 loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        val = metrics(data["val"], predict(net, data["val"], device))
        history.append({"epoch": epoch + 1, "train_loss": sum(losses) / len(losses),
                        "val_mse": val["mse"], "val_mae": val["mae"]})
        if best is None or val["mse"] < best["val"]["mse"]:
            best = {"lr": lr, "best_epoch": epoch + 1, "val": val,
                    "state": {key: value.detach().cpu().clone()
                              for key, value in net.state_dict().items()}}
        print(f"F03 lr={lr} epoch={epoch+1} val_mse={val['mse']:.6f}", flush=True)
    torch.save(best.pop("state"), state_file)
    best["history"] = history
    best["anchor_lambda"] = anchor_lambda
    best["state_file"] = str(state_file)
    file.write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")
    (folder / "complete").write_text("complete\n", encoding="utf-8")
    return best


def main():
    cli = arguments()
    out = Path(cli.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    args, model, loaders, device, sha = load_experiment(cli)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("A0 must be frozen")
    datasets = {name: loader.dataset.dataset for name, loader in loaders.items()}
    data = {name: collect(model, datasets[name], args, device,
                          {"train": cli.train_stride, "val": cli.val_stride,
                           "test": cli.test_stride}[name])
            for name in ("train", "val", "test")}
    baseline = source_baseline(args, sha)
    results = [baseline]
    for lr in LRS:
        folder = out / f"lr_{lr:g}"
        result = train_one(lr, data, model, args, cli, device, folder)
        results.append(result)
    chosen = min(results, key=lambda item: item["val"]["mse"])
    if "test" not in chosen:
        net = DirectInvariantForecaster(data["train"]["hidden"].shape[-2],
                                       data["train"]["hidden"].shape[-1],
                                       args.pred_len, pretrained_head=model.h_forecast_head).to(device)
        net.load_state_dict(torch.load(chosen["state_file"], map_location=device,
                                       weights_only=False))
        chosen["test"] = metrics(data["test"], predict(net, data["test"], device))
    report = {"backbone": args.backbone, "dataset": args.dataset_name,
              "pred_len": args.pred_len, "reference_sha256": sha,
              "a0_val": metrics(data["val"], data["val"]["base"]),
              "a0_test": metrics(data["test"], data["test"]["base"]),
              "candidates": [{k: v for k, v in item.items() if k != "history"} for item in results],
              "selected": {k: v for k, v in chosen.items() if k != "history"},
              "selection": "VAL MSE selects both epoch and LR; TEST is report-only",
              "backbone_training": "frozen A0; only original forecast head finetuned"}
    (out / "selection.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "complete").write_text("complete\n", encoding="utf-8")
    print("F03 LR SEARCH COMPLETE", out, flush=True)


if __name__ == "__main__":
    main()
