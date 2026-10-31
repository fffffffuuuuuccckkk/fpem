"""TRAIN-only GCID pilot using an existing A0 checkpoint and forecast head."""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.predictive_env.group_consensus import GroupConsensusSubspace, ConsensusAwareFusion
from models.predictive_env.swap_residual import VariantCorrection, ConditionalResidualScale
from tools.evaluate_fpem_probabilistic import probabilistic_metrics
from tools.run_fpem_shared_factorization import (
    load_experiment, ordered_training_loader, prediction, train_loader, write_json,
)


MODES = {
    "A1": ("random", "fixed"), "A2": ("static", "fixed"),
    "A3": ("ema", "fixed"), "A4": ("robust", "fixed"),
    "A5": ("robust", "ordinary"), "A6": ("robust", "consensus"),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_run_config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--rank_fraction", type=float, default=0.25)
    p.add_argument("--ema_beta", type=float, default=0.99)
    p.add_argument("--refresh_steps", type=int, default=64)
    p.add_argument("--lr_head", type=float, default=5e-5)
    p.add_argument("--lr_subspace", type=float, default=1e-4)
    p.add_argument("--lambda_inv", type=float, default=0.05)
    p.add_argument("--lambda_prob", type=float, default=0.1)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--max_batches", type=int, default=0)
    p.add_argument("--allow_legacy_reference_signature", action="store_true")
    return p.parse_args()


def components(model, subspace, gate, correction, residual, x, cycle):
    hidden, mean, std = model.encode(x, cycle)
    inv_tokens, dyn_tokens, sharedness, continuity, diagnostics = subspace(hidden)
    y_inv = prediction(model, inv_tokens, mean, std, cycle, head="active")
    delta = correction(dyn_tokens, std)
    alpha = gate(inv_tokens, dyn_tokens, sharedness, continuity, y_inv.shape[1])
    y_full = y_inv + alpha * delta
    sigma = residual(inv_tokens, dyn_tokens, detach_condition=True)
    return {"hidden": hidden, "inv_tokens": inv_tokens, "dyn_tokens": dyn_tokens,
            "y_inv": y_inv, "y_full": y_full, "delta": delta, "alpha": alpha,
            "sigma": sigma, "std": std, "mean": mean, "diagnostics": diagnostics}


@torch.no_grad()
def evaluate(model, subspace, gate, correction, residual, loader, args, device,
             *, samples=0, seed=2021, max_batches=0):
    for module in (model, subspace, gate, correction, residual):
        module.eval()
    generator = torch.Generator(device=device).manual_seed(seed+1701)
    sums = {key: 0. for key in ("mse", "mae", "inv_mse", "inv_mae",
                                 "a0_mse", "a0_mae", "picp_95_exact", "mpiw_95_exact",
                                 "alpha_mean", "delta_abs_mean", "sharedness_mean",
                                 "dynamic_energy_ratio", "orthogonality", "reconstruction_mae")}
    scores = {}; elements = sample_total = 0
    for index, (x, y, _, cycle) in enumerate(loader):
        if max_batches and index >= max_batches:
            break
        x, y, cycle = (x.float().to(device), y[:, -args.pred_len:].float().to(device),
                       cycle.to(device))
        out = components(model, subspace, gate, correction, residual, x, cycle)
        hidden, mean, std = model.encode(x, cycle)
        a0 = prediction(model, hidden, mean, std, cycle, head="reference")
        full, inv = out["y_full"], out["y_inv"]
        width = 1.96 * out["sigma"] * out["std"]
        totals = {"mse": (full-y).square().sum(), "mae": (full-y).abs().sum(),
                  "inv_mse": (inv-y).square().sum(), "inv_mae": (inv-y).abs().sum(),
                  "a0_mse": (a0-y).square().sum(), "a0_mae": (a0-y).abs().sum(),
                  "picp_95_exact": ((y >= full-width) & (y <= full+width)).sum(),
                  "mpiw_95_exact": (2*width).sum()}
        for key, value in totals.items():
            sums[key] += float(value)
        for key, value in {"alpha_mean": out["alpha"].mean(),
                           "delta_abs_mean": out["delta"].abs().mean(),
                           **out["diagnostics"]}.items():
            if key in sums and key not in totals:
                sums[key] += float(value) * y.numel()
        elements += y.numel()
        if samples:
            sampled = residual.sample(full, out["sigma"], out["std"], samples, generator)
            score = probabilistic_metrics(sampled, y, inv)
            for key in ("CRPS", "WIS", "sample_std_mean"):
                scores[key] = scores.get(key, 0.) + float(score[key])*len(x)
            sample_total += len(x)
    if not elements:
        raise RuntimeError("empty evaluation")
    return {**{key: value/elements for key, value in sums.items()},
            **({key: value/sample_total for key, value in scores.items()} if sample_total else {})}


def main():
    cli = parse_args()
    if not 0 < cli.rank_fraction < 1 or cli.lambda_inv < 0 or cli.lambda_prob < 0:
        raise ValueError("invalid rank or loss weight")
    outdir = Path(cli.output)
    outdir.mkdir(parents=True, exist_ok=True)
    args, model, loaders, device, digest = load_experiment(cli)
    ordered = ordered_training_loader(loaders["train"].dataset.dataset, args)
    x0, _, _, cycle0 = next(iter(ordered))
    with torch.no_grad():
        hidden0, _, _ = model.encode(x0.float().to(device), cycle0.to(device))
    width, patches = hidden0.shape[-1], hidden0.shape[2]
    rank = max(1, min(width-1, round(width*cli.rank_fraction)))
    submode, fusionmode = MODES[cli.mode]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cli.seed+2701)
        subspace = GroupConsensusSubspace(width, rank, submode,
                                          beta=cli.ema_beta,
                                          refresh_steps=cli.refresh_steps).to(device)
        gate = ConsensusAwareFusion(width, args.pred_len, fusionmode).to(device)
        correction = VariantCorrection(patches, width, args.pred_len).to(device)
        residual = ConditionalResidualScale(width, args.pred_len).to(device)
    if submode in ("ema", "robust"):
        @torch.no_grad()
        def train_hidden_batches():
            model.eval()
            for index, (x, _, _, cycle) in enumerate(ordered):
                if cli.max_batches and index >= cli.max_batches:
                    break
                hidden, _, _ = model.encode(x.float().to(device), cycle.to(device))
                yield hidden
        subspace.initialize_from_training(train_hidden_batches())
    model.head_linear.requires_grad_(True)
    groups = [{"name": "invariant_head", "params": list(model.head_linear.parameters()),
               "lr": cli.lr_head},
              {"name": "dynamic_correction", "params": list(correction.parameters()),
               "lr": cli.lr_head},
              {"name": "residual_interval", "params": list(residual.parameters()),
               "lr": cli.lr_head}]
    if fusionmode != "fixed":
        groups.append({"name": "fusion_gate", "params": list(gate.parameters()),
                       "lr": cli.lr_head})
    if submode == "static":
        groups.append({"name": "static_subspace", "params": list(subspace.parameters()),
                       "lr": cli.lr_subspace})
    optimizer = torch.optim.Adam(groups)
    training = train_loader(loaders["train"].dataset.dataset, args, cli.seed)
    config = {"cli": vars(cli), "reference_sha256": digest, "subspace_mode": submode,
              "fusion_mode": fusionmode, "rank": rank, "train_only_basis_updates": True,
              "bootstrap": "all TRAIN hidden, unweighted", "robust_delta": "batch median residual",
              "checkpoint_selection": "VAL full-prediction MSE", "a0_retrained": False,
              "backbone_frozen": True, "interval": "Gaussian 95% using previous ConditionalResidualScale"}
    write_json(outdir/"run_config.json", config)
    initial_val = evaluate(model, subspace, gate, correction, residual,
                           loaders["val"], args, device, max_batches=cli.max_batches)
    history, best_val = [], math.inf
    started = time.perf_counter()
    for epoch in range(cli.epochs):
        model.eval(); model.head_dropout.train()
        subspace.train(); gate.train(); correction.train(); residual.train()
        sums = {key: 0. for key in ("train_mse", "train_mae", "loss/inv", "loss/point",
                                    "loss/prob", "alpha_mean", "sharedness_mean",
                                    "dynamic_energy_ratio", "orthogonality",
                                    "basis_orthogonality", "reconstruction_mae")}
        count = 0
        for batch_index, (x, y, _, cycle) in enumerate(training):
            if cli.max_batches and batch_index >= cli.max_batches:
                break
            x, y, cycle = (x.float().to(device), y[:, -args.pred_len:].float().to(device),
                           cycle.to(device))
            out = components(model, subspace, gate, correction, residual, x, cycle)
            point = (out["y_full"]-y).square().mean()
            inv_loss = (out["y_inv"]-y).square().mean()
            prob = residual.nll(out["sigma"], (y-out["y_full"]).detach(), out["std"])
            loss = point + cli.lambda_inv*inv_loss + cli.lambda_prob*prob
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss, epoch {epoch+1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            # Current TRAIN history only, after the gradient update; no future target enters Q.
            subspace.update(out["hidden"].detach())
            terms = {"train_mse": (out["y_full"]-y).detach().square().mean(),
                     "train_mae": (out["y_full"]-y).detach().abs().mean(),
                     "loss/inv": inv_loss, "loss/point": point, "loss/prob": prob,
                     "alpha_mean": out["alpha"].mean(), **out["diagnostics"]}
            for key in sums:
                sums[key] += float(terms[key].detach())*len(x)
            count += len(x)
        val = evaluate(model, subspace, gate, correction, residual,
                       loaders["val"], args, device, max_batches=cli.max_batches)
        row = {"epoch": epoch+1, **{key: value/count for key, value in sums.items()},
               "val_mse": val["mse"], "val_mae": val["mae"],
               "val_inv_mse": val["inv_mse"], "val_picp_95": val["picp_95_exact"]}
        history.append(row)
        print(json.dumps(row), flush=True)
        write_json(outdir/"history.json", history)
        if val["mse"] < best_val:
            best_val = val["mse"]
            torch.save({"model": model.state_dict(), "subspace": subspace.state_dict(),
                        "gate": gate.state_dict(), "correction": correction.state_dict(),
                        "residual": residual.state_dict(), "best_epoch": epoch+1,
                        "val": val, "reference_sha256": digest}, outdir/"best.pt")
    checkpoint = torch.load(outdir/"best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    subspace.load_state_dict(checkpoint["subspace"])
    gate.load_state_dict(checkpoint["gate"])
    correction.load_state_dict(checkpoint["correction"])
    residual.load_state_dict(checkpoint["residual"])
    test = evaluate(model, subspace, gate, correction, residual, loaders["test"],
                    args, device, samples=cli.samples, seed=cli.seed,
                    max_batches=cli.max_batches)
    torch.save(subspace.basis().detach().cpu(), outdir/"final_consensus_basis.pt")
    result = {"mode": cli.mode, "reference_sha256": digest, "rank": rank,
              "best_epoch": checkpoint["best_epoch"], "initial_val": initial_val,
              "val": checkpoint["val"], "test": test, "history": history,
              "training_seconds": time.perf_counter()-started,
              "subspace_updates": int(subspace.updates),
              "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device)
              if device.type == "cuda" else 0,
              "cpu_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    write_json(outdir/"result.json", result)
    (outdir/"run_complete").touch()


if __name__ == "__main__":
    main()
