"""Independent PTM pilot with a signature-verified, existing A0 checkpoint."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.predictive_env.ptm import PrototypeAdapter, PrototypeBank
from tools.run_fpem_shared_factorization import load_experiment, prediction, train_loader, write_json


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--base_run_config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--bank_checkpoint", required=True)
    p.add_argument("--mode", choices=("A0", "T0", "T1", "T2", "T3", "T4"), required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr_head", type=float, default=5e-5)
    p.add_argument("--lr_adapter", type=float, default=1e-3)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--history_size", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--ema_beta", type=float, default=0.99)
    p.add_argument("--eta_max", type=float, default=0.2)
    p.add_argument("--gate_max", type=float, default=0.2)
    p.add_argument("--max_batches", type=int, default=0, help="smoke only")
    p.add_argument("--allow_legacy_reference_signature", action="store_true")
    return p.parse_args()


def read_or_fit_bank(bank, dataset, path, args, cli, reference_hash):
    path = Path(path)
    identity = {"data_path": args.data_path, "seq_len": args.seq_len,
                "pred_len": args.pred_len, "k": cli.k, "history_size": cli.history_size,
                "seed": cli.seed, "reference_sha256": reference_hash}
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["identity"] != identity:
            raise ValueError("existing prototype bank identity mismatch")
        bank.load_state_dict(payload["bank"])
        return payload["fit_diagnostics"]
    fit = bank.fit_train(dataset, seed=cli.seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"identity": identity, "bank": bank.state_dict(),
                "fit_diagnostics": fit}, path)
    return fit


def base_forecast(model, x, cycle, active=False):
    hidden, mean, std = model.encode(x, cycle)
    return prediction(model, hidden, mean, std, cycle,
                      "active" if active else "reference")


def forward(model, bank, adapter, mode, x, cycle, schedule=1.0):
    query = bank.query(x)
    if mode in ("T1", "T2"):
        enhanced, gate = adapter.enhance_input(x, query, schedule)
        y = base_forecast(model, enhanced, cycle)
        change = (enhanced-x).abs().mean()
    else:
        y0 = base_forecast(model, x, cycle, active=(mode == "T0"))
        if mode in ("T3", "T4"):
            y, gate = adapter.calibrate_shape(y0, query)
            change = (y-y0).abs().mean()
        else:
            y, gate = y0, x.new_zeros((x.shape[0], x.shape[2]))
            change = x.new_zeros(())
    return y, query, gate, change


@torch.no_grad()
def evaluate(model, bank, adapter, mode, loader, args, device, max_batches=0):
    model.eval(); bank.eval()
    if adapter is not None:
        adapter.eval()
    totals = {k: 0. for k in ("mse", "mae", "a0_mse", "a0_mae",
                              "prototype_mse", "prototype_mae")}
    diagnostics = {k: 0. for k in ("match_max_weight", "match_entropy",
                                    "match_min_distance", "gate_mean",
                                    "gate_abs_mean", "input_or_output_change")}
    elements = series = batches = 0
    usage = torch.zeros(bank.k)
    start = time.perf_counter()
    for step, (x, y, _, cycle) in enumerate(loader):
        if max_batches and step >= max_batches:
            break
        x, y, cycle = x.float().to(device), y[:, -args.pred_len:].float().to(device), cycle.to(device)
        forecast, query, gate, change = forward(model, bank, adapter, mode, x, cycle)
        a0 = base_forecast(model, x, cycle)
        for prefix, pred in (("", forecast), ("a0_", a0), ("prototype_", query["y_proto"])):
            totals[prefix+"mse"] += float((pred-y).square().sum())
            totals[prefix+"mae"] += float((pred-y).abs().sum())
        count = x.shape[0]*x.shape[2]
        for key, value in (("match_max_weight", query["max_weight"]),
                           ("match_entropy", query["entropy"]),
                           ("match_min_distance", query["min_distance"]),
                           ("gate_mean", gate), ("gate_abs_mean", gate.abs())):
            diagnostics[key] += float(value.sum())
        diagnostics["input_or_output_change"] += float(change)*count
        usage += query["weight"].detach().sum((0, 1)).cpu()
        elements += y.numel(); series += count; batches += 1
    if not elements:
        raise RuntimeError("empty evaluation")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter()-start
    return {**{k: v/elements for k, v in totals.items()},
            **{k: v/series for k, v in diagnostics.items()},
            "prototype_usage_fraction": (usage/usage.sum()).tolist(),
            "prototype_coverage_counts": bank.counts.detach().cpu().tolist(),
            "elapsed_seconds": elapsed, "milliseconds_per_batch": 1000*elapsed/batches}


def main():
    cli = arguments()
    outdir = Path(cli.output); outdir.mkdir(parents=True, exist_ok=True)
    args, model, loaders, device, reference_hash = load_experiment(cli)
    dataset = loaders["train"].dataset.dataset
    bank = PrototypeBank(args.pred_len, cli.k, cli.history_size,
                         cli.temperature, cli.ema_beta)
    fit = read_or_fit_bank(bank, dataset, cli.bank_checkpoint, args, cli, reference_hash)
    bank = bank.to(device)
    adapter = None
    if cli.mode in ("T1", "T2", "T3", "T4"):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cli.seed+4221)
            adapter = PrototypeAdapter("input" if cli.mode in ("T1", "T2") else "shape",
                                       cli.eta_max, cli.gate_max).to(device)
    if cli.mode == "T0":
        model.head_linear.requires_grad_(True)
        params = [{"params": model.head_linear.parameters(), "lr": cli.lr_head}]
    elif adapter is not None:
        params = [{"params": adapter.parameters(), "lr": cli.lr_adapter}]
    else:
        params = []
    optimizer = torch.optim.Adam(params) if params else None
    x0, _, _, cycle0 = next(iter(loaders["train"]))
    x0, cycle0 = x0.float().to(device), cycle0.to(device)
    model.eval()
    with torch.no_grad():
        initial, _, _, _ = forward(model, bank, adapter, cli.mode, x0, cycle0,
                                   schedule=0.0 if cli.mode in ("T1", "T2") else 1.0)
        a0_initial = base_forecast(model, x0, cycle0)
        init_error = float((initial-a0_initial).abs().max())
    if init_error > 1e-4:
        raise AssertionError(f"A0 function preservation failed: {init_error}")
    write_json(outdir/"run_config.json", {
        "cli": vars(cli), "reference_sha256": reference_hash, "bank_fit": fit,
        "checkpoint_selection": "VAL MSE", "initial_a0_max_abs_difference": init_error,
        "backbone_trainable": False, "head_trainable": cli.mode == "T0",
        "trainable_parameters": sum(p.numel() for group in params for p in group["params"]),
        "bank_float_count": bank.U.numel()+bank.V.numel(), "train_only_bank": True})
    if cli.mode == "A0":
        val = evaluate(model, bank, adapter, cli.mode, loaders["val"], args, device, cli.max_batches)
        test = evaluate(model, bank, adapter, cli.mode, loaders["test"], args, device, cli.max_batches)
        write_json(outdir/"result.json", {"mode": cli.mode, "best_epoch": 0, "val": val,
                   "test": test, "history": [], "checkpoint": args.reference_checkpoint,
                   "training_seconds": 0})
        (outdir/"run_complete").touch()
        return
    best = math.inf; history = []; start = time.perf_counter(); step_count = 0
    training = train_loader(dataset, args, cli.seed)
    warm_steps = max(1, math.ceil(0.2*cli.epochs*len(training)))
    for epoch in range(cli.epochs):
        model.eval()
        if adapter is not None:
            adapter.train()
        total = seen = 0
        for step, (x, y, _, cycle) in enumerate(training):
            if cli.max_batches and step >= cli.max_batches:
                break
            x, y, cycle = x.float().to(device), y[:, -args.pred_len:].float().to(device), cycle.to(device)
            schedule = min(1.0, step_count/warm_steps)
            forecast, _, _, _ = forward(model, bank, adapter, cli.mode, x, cycle, schedule)
            loss = (forecast-y).square().mean()
            if not bool(torch.isfinite(loss.detach())):
                raise FloatingPointError("nonfinite PTM loss")
            if cli.mode == "T0" or cli.mode in ("T3", "T4") or schedule > 0:
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            total += float(loss.detach())*len(x); seen += len(x); step_count += 1
        ema = bank.update_train_epoch(dataset) if cli.mode in ("T2", "T4") else None
        val = evaluate(model, bank, adapter, cli.mode, loaders["val"], args, device, cli.max_batches)
        record = {"epoch": epoch+1, "train_mse": total/max(seen, 1),
                  "val_mse": val["mse"], "val_mae": val["mae"],
                  "gate_mean": val["gate_mean"], "ema": ema}
        history.append(record); write_json(outdir/"history.json", history)
        print(json.dumps(record), flush=True)
        if val["mse"] < best:
            best = val["mse"]
            torch.save({"model": model.state_dict(), "adapter": adapter.state_dict() if adapter else None,
                        "bank": bank.state_dict(), "epoch": epoch+1,
                        "reference_sha256": reference_hash}, outdir/"best.pt")
    selected = torch.load(outdir/"best.pt", map_location=device, weights_only=False)
    model.load_state_dict(selected["model"]); bank.load_state_dict(selected["bank"])
    if adapter is not None:
        adapter.load_state_dict(selected["adapter"])
    val = evaluate(model, bank, adapter, cli.mode, loaders["val"], args, device, cli.max_batches)
    test = evaluate(model, bank, adapter, cli.mode, loaders["test"], args, device, cli.max_batches)
    write_json(outdir/"result.json", {"mode": cli.mode, "best_epoch": selected["epoch"],
               "val": val, "test": test, "history": history,
               "checkpoint": str(outdir/"best.pt"),
               "training_seconds": time.perf_counter()-start})
    (outdir/"run_complete").touch()


if __name__ == "__main__":
    main()
