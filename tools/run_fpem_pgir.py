"""Frozen-A0 PGIR point-forecast pilot. TEST is never used for selection."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.predictive_env.pgir import PGIRMask
from models.predictive_env.ptm import PrototypeBank
from tools.run_fpem_ptm import base_forecast, read_or_fit_bank
from tools.run_fpem_shared_factorization import load_experiment, train_loader, write_json


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--base_run_config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--bank_checkpoint", required=True)
    p.add_argument("--mode", choices=("A0", "P1", "P2", "P3"), required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr_mask", type=float, default=1e-3)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--history_size", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--ema_beta", type=float, default=0.99)
    p.add_argument("--segment_len", type=int, default=16)
    p.add_argument("--lambda_sparse", type=float, default=0.005)
    p.add_argument("--lambda_temporal", type=float, default=0.001)
    p.add_argument("--lambda_contrast", type=float, default=0.01)
    p.add_argument("--max_batches", type=int, default=0)
    p.add_argument("--allow_legacy_reference_signature", action="store_true")
    return p.parse_args()


def forecast(model, bank, selector, x, cycle, mode, replacement="prototype", generator=None):
    selected, info = selector(x, bank, mode=mode, replacement=replacement, generator=generator)
    return base_forecast(model, selected, cycle), selected, info


def counterfactual_loss(model, x, selected, info, cycle, y, segment_len):
    # Remove the most retained, non-invariant observed variation segment.
    eligible = (1-info["invariant"]) * info["probability"].detach() * info["residual_energy"]
    idx = eligible.argmax(-1)
    b, length, channels = x.shape
    patch = torch.arange((length+segment_len-1)//segment_len, device=x.device)
    removed = (patch[None, None, :] == idx[..., None]) & (eligible.max(-1).values[..., None] > 0)
    removed = removed.repeat_interleave(segment_len, -1)[..., :length].transpose(1, 2)
    x_removed = torch.where(removed, info["query"]["x_proto"], selected.detach())
    with torch.no_grad():
        y_removed = base_forecast(model, x_removed, cycle)
        removed_error = (y_removed-y).square().mean((1, 2))
    return removed_error


@torch.no_grad()
def evaluate(model, bank, selector, loader, args, device, mode, max_batches=0,
             replacement="prototype"):
    model.eval(); selector.eval()
    totals = {"mse": 0., "mae": 0., "a0_mse": 0., "a0_mae": 0.,
              "prototype_mse": 0., "prototype_mae": 0.}
    diags = {"replacement_fraction": 0., "variation_keep_fraction": 0.,
             "matching_entropy": 0., "matching_distance": 0., "mask_probability": 0.}
    elements = series = batches = 0
    start = time.perf_counter()
    gen = torch.Generator(device="cpu").manual_seed(99917)
    for step, (x, y, _, cycle) in enumerate(loader):
        if max_batches and step >= max_batches: break
        x = x.float().to(device)
        y = y[:, -args.pred_len:].float().to(device)
        cycle = cycle.to(device)
        pred, _, info = forecast(model, bank, selector, x, cycle, mode, replacement, gen)
        baseline = base_forecast(model, x, cycle)
        for prefix, value in (("", pred), ("a0_", baseline),
                              ("prototype_", info["query"]["y_proto"])):
            totals[prefix+"mse"] += float((value-y).square().sum())
            totals[prefix+"mae"] += float((value-y).abs().sum())
        count = len(x)*x.shape[-1]
        diags["replacement_fraction"] += float(info["replacement_fraction"])*count
        diags["variation_keep_fraction"] += float(info["variation_keep_fraction"])*count
        diags["matching_entropy"] += float(info["query"]["entropy"].sum())
        diags["matching_distance"] += float(info["query"]["min_distance"].sum())
        diags["mask_probability"] += float(info["probability"].sum()/info["probability"].numel())*count
        elements += y.numel(); series += count; batches += 1
    if not elements: raise RuntimeError("empty evaluation")
    if device.type == "cuda": torch.cuda.synchronize(device)
    elapsed = time.perf_counter()-start
    return {**{k:v/elements for k,v in totals.items()},
            **{k:v/series for k,v in diags.items()},
            "elapsed_seconds": elapsed, "milliseconds_per_batch": elapsed*1000/batches}


def main():
    cli = arguments()
    out = Path(cli.output); out.mkdir(parents=True, exist_ok=True)
    args, model, loaders, device, reference_hash = load_experiment(cli)
    dataset = loaders["train"].dataset.dataset
    bank = PrototypeBank(args.pred_len, cli.k, cli.history_size, cli.temperature, cli.ema_beta)
    fit = read_or_fit_bank(bank, dataset, cli.bank_checkpoint, args, cli, reference_hash)
    bank.to(device).eval()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cli.seed+5739)
        selector = PGIRMask(cli.segment_len).to(device)
    threshold = selector.fit_threshold(bank, dataset, device)
    model.eval()
    model.requires_grad_(False)
    x0, _, _, c0 = next(iter(loaders["train"]))
    x0, c0 = x0.float().to(device), c0.to(device)
    with torch.no_grad():
        initial, selected0, info0 = forecast(model, bank, selector, x0, c0, cli.mode)
        baseline = base_forecast(model, x0, c0)
        initial_difference = float((initial-baseline).abs().max())
    if not torch.equal(selected0, x0):
        raise AssertionError("PGIR initial input is not exactly the A0 input")
    # ModernTCN CUDA kernels can differ by about 1e-5 between repeated
    # identical evaluations. The input equality above is the exact guarantee.
    if initial_difference > 1e-4:
        raise AssertionError(f"PGIR initial forecast differs from A0: {initial_difference}")
    write_json(out/"run_config.json", {"cli": vars(cli), "reference_sha256": reference_hash,
               "bank_fit": fit, "train_only_invariant_threshold": threshold,
               "initial_a0_max_abs_difference": initial_difference,
               "checkpoint_selection": "VAL MSE", "backbone_frozen": True,
               "trainable_parameters": sum(p.numel() for p in selector.parameters())})
    if cli.mode in ("A0", "P1"):
        val = evaluate(model, bank, selector, loaders["val"], args, device, cli.mode, cli.max_batches)
        test = evaluate(model, bank, selector, loaders["test"], args, device, cli.mode, cli.max_batches)
        write_json(out/"result.json", {"mode": cli.mode, "best_epoch": 0, "val": val,
                   "test": test, "history": [], "checkpoint": args.reference_checkpoint,
                   "training_seconds": 0.})
        (out/"run_complete").touch()
        return
    optimizer = torch.optim.Adam(selector.parameters(), lr=cli.lr_mask)
    training = train_loader(dataset, args, cli.seed)
    history = []; best = math.inf; start = time.perf_counter()
    for epoch in range(cli.epochs):
        selector.train(); model.eval()
        total = count = 0
        for step, (x, y, _, cycle) in enumerate(training):
            if cli.max_batches and step >= cli.max_batches: break
            x = x.float().to(device)
            y = y[:, -args.pred_len:].float().to(device)
            cycle = cycle.to(device)
            pred, selected, info = forecast(model, bank, selector, x, cycle, cli.mode)
            pred_loss = (pred-y).square().mean()
            noninv = 1-info["invariant"]
            sparse = (noninv*info["probability"]).sum()/noninv.sum().clamp_min(1)
            p = info["probability"]
            temporal = (p[..., 1:]-p[..., :-1]).abs().mean() if p.shape[-1] > 1 else pred_loss*0
            loss = pred_loss+cli.lambda_sparse*sparse+cli.lambda_temporal*temporal
            contrast = pred_loss.new_zeros(())
            if cli.mode == "P3" and cli.lambda_contrast:
                removed = counterfactual_loss(model, x, selected, info, cycle, y, cli.segment_len)
                factual = (pred-y).square().mean((1, 2))
                # Detached counterfactual: the model cannot win by damaging it.
                contrast = F.softplus((factual-removed.detach())/0.05).mean()*0.05
                loss = loss+cli.lambda_contrast*contrast
            if not torch.isfinite(loss): raise FloatingPointError("non-finite PGIR loss")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.)
            optimizer.step()
            total += float(pred_loss.detach())*len(x); count += len(x)
        val = evaluate(model, bank, selector, loaders["val"], args, device, cli.mode, cli.max_batches)
        record = {"epoch": epoch+1, "train_mse": total/max(count, 1),
                  "val_mse": val["mse"], "val_mae": val["mae"],
                  "replacement_fraction": val["replacement_fraction"]}
        history.append(record); write_json(out/"history.json", history)
        print(json.dumps(record), flush=True)
        if val["mse"] < best:
            best = val["mse"]
            torch.save({"selector": selector.state_dict(), "bank": bank.state_dict(),
                        "epoch": epoch+1, "reference_sha256": reference_hash}, out/"best.pt")
    saved = torch.load(out/"best.pt", map_location=device, weights_only=False)
    selector.load_state_dict(saved["selector"]); bank.load_state_dict(saved["bank"])
    val = evaluate(model, bank, selector, loaders["val"], args, device, cli.mode, cli.max_batches)
    test = evaluate(model, bank, selector, loaders["test"], args, device, cli.mode, cli.max_batches)
    controls = {}
    if cli.mode == "P2":
        for name, rep in (("random", "prototype"), ("smooth", "smooth")):
            controls[name] = evaluate(model, bank, selector, loaders["test"], args, device,
                                      name if name == "random" else "P2", cli.max_batches, rep)
    write_json(out/"result.json", {"mode": cli.mode, "best_epoch": saved["epoch"],
               "val": val, "test": test, "controls": controls, "history": history,
               "checkpoint": str(out/"best.pt"),
               "training_seconds": time.perf_counter()-start})
    (out/"run_complete").touch()


if __name__ == "__main__":
    main()
