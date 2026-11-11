"""Independent CSIF B1--B3 point-forecast pilot; existing A0 is never retrained."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.predictive_env.csif import CommonSpecificFactors, ConsensusLatentFusion
from tools.run_fpem_shared_factorization import (
    load_experiment, prediction, train_loader, write_json,
)


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--base_run_config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=("B1", "B2", "B3"), required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr_head", type=float, default=5e-5)
    p.add_argument("--lr_factor", type=float, default=1e-4)
    p.add_argument("--lambda_common", type=float, default=0.05)
    p.add_argument("--lambda_sparse", type=float, default=0.0)
    p.add_argument("--activation_threshold", type=float, default=1.0)
    p.add_argument("--max_batches", type=int, default=0)
    p.add_argument("--allow_legacy_reference_signature", action="store_true")
    return p.parse_args()


def forward_parts(model, factor, fusion, x, cycle):
    hidden, mean, std = model.encode(x, cycle)
    common, specific, support, activations, diag = factor(hidden)
    fused, alpha = fusion(common, specific, support)
    return {"a0": prediction(model, hidden, mean, std, cycle, "reference"),
            "common": prediction(model, common, mean, std, cycle, "active"),
            "point": prediction(model, fused, mean, std, cycle, "active"),
            "hidden": hidden, "common_hidden": common, "specific_hidden": specific,
            "activations": activations, "alpha": alpha, "support": support,
            "diagnostics": diag}


@torch.no_grad()
def evaluate(model, factor, fusion, loader, args, device, max_batches=0):
    model.eval(); factor.eval(); fusion.eval()
    sums = {key: 0. for key in ("a0_mse", "a0_mae", "mse", "mae",
                                "common_mse", "common_mae")}
    stats = {key: 0. for key in ("alpha_mean", "support_mean", "reconstruction_mae",
                                 "common_energy_ratio", "specific_energy_ratio",
                                 "effective_common_factors", "factor_assignment_entropy")}
    elements = samples = 0
    for step, (x, y, ids, cycle) in enumerate(loader):
        if max_batches and step >= max_batches:
            break
        x, y, cycle = x.float().to(device), y[:, -args.pred_len:].float().to(device), cycle.to(device)
        out = forward_parts(model, factor, fusion, x, cycle)
        for label, pred in (("a0", out["a0"]), ("", out["point"]), ("common", out["common"])):
            prefix = label + "_" if label else ""
            sums[prefix+"mse"] += float((pred-y).square().sum())
            sums[prefix+"mae"] += float((pred-y).abs().sum())
        for key, value in {"alpha_mean": out["alpha"].mean(), **out["diagnostics"]}.items():
            if key in stats:
                stats[key] += float(value)*len(x)
        samples += len(x); elements += y.numel()
    if not samples:
        raise RuntimeError("empty evaluation")
    w = factor.basis().float()
    return {**{key: value/elements for key, value in sums.items()},
            **{key: value/samples for key, value in stats.items()},
            "orthogonality_max": float((w.T@w-torch.eye(w.shape[-1], device=w.device)).abs().max()),
            "factor_p_ema": factor.p_ema.tolist(),
            "factor_pi": factor.popularity().tolist()}


def main():
    cli = arguments()
    outdir = Path(cli.output); outdir.mkdir(parents=True, exist_ok=True)
    args, model, loaders, device, digest = load_experiment(cli)
    with torch.no_grad():
        x0, _, _, c0 = next(iter(loaders["train"]))
        hidden, _, _ = model.encode(x0.float().to(device), c0.to(device))
    width = hidden.shape[-1]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cli.seed+3401)
        factor = CommonSpecificFactors(
            width, "dictionary" if cli.mode == "B1" else "memory",
            activation_threshold=cli.activation_threshold,
        ).to(device)
        fusion = ConsensusLatentFusion(width, cli.mode == "B3").to(device)
    init = forward_parts(model, factor, fusion, x0.float().to(device), c0.to(device))
    initial_a0_error = float((init["point"]-init["a0"]).abs().max().detach())
    initial_a0_mse = float((init["point"]-init["a0"]).square().mean().detach())
    if initial_a0_error > 1e-3:
        raise RuntimeError(f"A0 function preservation failed: {initial_a0_error}")
    model.head_linear.requires_grad_(True)
    optimizer = torch.optim.Adam([
        {"params": model.head_linear.parameters(), "lr": cli.lr_head},
        {"params": factor.parameters(), "lr": cli.lr_factor},
        *([{"params": fusion.parameters(), "lr": cli.lr_factor}] if cli.mode == "B3" else []),
    ])
    write_json(outdir/"run_config.json", {"cli": vars(cli), "reference_sha256": digest,
        "backbone_frozen": True, "checkpoint_selection": "VAL point MSE",
        "anchor_id": "TRAIN start_index // seq_len", "a0_retrained": False,
        "initial_a0_max_abs_difference": initial_a0_error,
        "initial_a0_prediction_mse": initial_a0_mse})
    training = train_loader(loaders["train"].dataset.dataset, args, cli.seed)
    best = math.inf; history = []; start = time.perf_counter()
    for epoch in range(cli.epochs):
        model.eval(); model.head_dropout.train(); factor.train(); fusion.train()
        totals = {key: 0. for key in ("loss", "point", "common", "sparse")}; seen = 0
        for step, (x, y, ids, cycle) in enumerate(training):
            if cli.max_batches and step >= cli.max_batches:
                break
            x, y, cycle = x.float().to(device), y[:, -args.pred_len:].float().to(device), cycle.to(device)
            out = forward_parts(model, factor, fusion, x, cycle)
            point = (out["point"]-y).square().mean()
            common = (out["common"]-y).square().mean()
            sparse = out["activations"].abs().mean()
            loss = point + cli.lambda_common*common + cli.lambda_sparse*sparse
            if not bool(torch.isfinite(loss.detach())):
                raise FloatingPointError("nonfinite CSIF point loss")
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            # No diagnostic/evaluation forward may mutate factor memory.
            factor.observe_train_batch(out["activations"].detach(), ids, args.seq_len)
            for key, value in (("loss", loss), ("point", point), ("common", common), ("sparse", sparse)):
                totals[key] += float(value.detach())*len(x)
            seen += len(x)
        memory = factor.finish_train_epoch()
        val = evaluate(model, factor, fusion, loaders["val"], args, device, cli.max_batches)
        row = {"epoch": epoch+1, **{k: v/seen for k,v in totals.items()},
               "val_mse": val["mse"], "val_mae": val["mae"],
               "val_common_mse": val["common_mse"], **memory}
        history.append(row); write_json(outdir/"history.json", history)
        print(json.dumps(row), flush=True)
        if val["mse"] < best:
            best = val["mse"]
            torch.save({"model": model.state_dict(), "factor": factor.state_dict(),
                        "fusion": fusion.state_dict(), "best_epoch": epoch+1,
                        "val": val, "reference_sha256": digest}, outdir/"best.pt")
    selected = torch.load(outdir/"best.pt", map_location=device, weights_only=False)
    model.load_state_dict(selected["model"]); factor.load_state_dict(selected["factor"])
    fusion.load_state_dict(selected["fusion"])
    test = evaluate(model, factor, fusion, loaders["test"], args, device, cli.max_batches)
    write_json(outdir/"result.json", {"mode": cli.mode, "best_epoch": selected["best_epoch"],
        "reference_sha256": digest, "val": selected["val"], "test": test,
        "history": history, "training_seconds": time.perf_counter()-start})
    (outdir/"run_complete").touch()


if __name__ == "__main__":
    main()
