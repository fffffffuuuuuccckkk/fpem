"""O1/O2/O3/O4 S4-compatible pilot with no environment-q dependence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.predictive_env.orthogonal_spectral_decomposition import (
    OrthogonalComplementaryDecomposer, PredictiveSpectralDecomposer,
    StreamingPredictiveSpectrum, orthogonal_complement,
)
from tools.run_fpem_shared_factorization import (
    decomposition_latency_ms, evaluate, factorize, load_experiment,
    ordered_training_loader, prediction, train_loader, write_json,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_run_config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--method", choices=("O1", "O2", "O3", "O4"), required=True)
    p.add_argument("--variation_fraction", type=float, required=True)
    p.add_argument("--lr_head", type=float, required=True)
    p.add_argument("--lr_basis", type=float, default=1e-4)
    p.add_argument("--lr_backbone", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--train_backbone", action="store_true")
    p.add_argument("--spectrum_cache", default="")
    p.add_argument("--gradient_separation", type=float, default=1.0)
    p.add_argument("--lambda_shared", type=float, default=0.01)
    p.add_argument("--lambda_partial", type=float, default=0.01)
    p.add_argument("--lambda_variance", type=float, default=0.001)
    p.add_argument("--max_batches", type=int, default=0, help="smoke only")
    p.add_argument("--allow_legacy_reference_signature", action="store_true")
    return p.parse_args()


@torch.no_grad()
def fit_spectrum(model, loader, args, device, width, rank, digest, cache_path,
                 max_batches=0):
    if cache_path and cache_path.is_file():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cached["reference_sha256"] != digest or cached["shared_rank"] != rank:
            raise ValueError("spectrum cache checkpoint/rank mismatch")
        return cached["q"], cached["info"]
    stats = StreamingPredictiveSpectrum(width, args.pred_len)
    model.eval()
    for batch_index, (x, y, _, cycle) in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        x, cycle = x.float().to(device), cycle.to(device)
        hidden, _, _ = model.encode(x, cycle)
        stats.add(hidden, y[:, -args.pred_len:], source_split="train")
    q, info = stats.fit(rank)
    info["reference_sha256"] = digest
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"q": q, "info": info, "reference_sha256": digest,
                    "shared_rank": rank}, cache_path)
    return q, info


@torch.no_grad()
def train_representation_diagnostics(model, adapter, loader, device, batches=8):
    model.eval(); adapter.eval()
    inv_rows, var_rows = [], []
    for i, (x, _, ids, cycle) in enumerate(loader):
        if i >= batches:
            break
        hidden, _, _ = model.encode(x.float().to(device), cycle.to(device))
        out = factorize(adapter, hidden)
        inv_rows.append(out["z_inv_tokens"].mean((1, 2)).float().cpu())
        var_rows.append(out["z_var_tokens"].mean((1, 2)).float().cpu())
    def effective_rank(rows):
        matrix = torch.cat(rows).double()
        centered = matrix-matrix.mean(0)
        values = torch.linalg.svdvals(centered).square()
        prob = values/values.sum().clamp_min(1e-12)
        entropy = -(prob*prob.clamp_min(1e-12).log()).sum()
        return float(entropy.exp())
    return {"inv_effective_rank": effective_rank(inv_rows),
            "var_effective_rank": effective_rank(var_rows)}


def main():
    cli = parse_args()
    if not 0 < cli.variation_fraction < 1:
        raise ValueError("variation fraction must be in (0,1)")
    if not 0 <= cli.gradient_separation <= 1:
        raise ValueError("gradient separation must be in [0,1]")
    outdir = Path(cli.output)
    outdir.mkdir(parents=True, exist_ok=True)
    # A running primary queue may reach a case delegated to another server.
    # Wait for its completed result instead of training the same trial twice.
    if (outdir.parents[1] / "delegated_to_secondary.json").is_file():
        deadline = time.monotonic() + 12 * 60 * 60
        print(f"Waiting for delegated trial: {outdir}", flush=True)
        while time.monotonic() < deadline:
            if (outdir / "run_complete").is_file() and (outdir / "result.json").is_file():
                print(f"Delegated trial received: {outdir}", flush=True)
                return
            time.sleep(30)
        raise TimeoutError(f"Delegated trial did not arrive within 12 hours: {outdir}")
    args, model, loaders, device, digest = load_experiment(cli)
    ordered = ordered_training_loader(loaders["train"].dataset.dataset, args)
    x, y, _, cycle = next(iter(ordered))
    with torch.no_grad():
        hidden, _, _ = model.encode(x.float().to(device), cycle.to(device))
    width = hidden.shape[-1]
    variation_rank = max(1, min(width-1, round(width*cli.variation_fraction)))
    shared_rank = width-variation_rank
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cli.seed+811)
        if cli.method == "O3":
            adapter = PredictiveSpectralDecomposer(width, shared_rank).to(device)
        else:
            adapter = OrthogonalComplementaryDecomposer(
                width, variation_rank, coverage=(cli.method == "O2"),
                lambda_shared=cli.lambda_shared,
                lambda_partial=cli.lambda_partial,
                lambda_variance=cli.lambda_variance).to(device)
    spectral_info = None
    if cli.method in ("O3", "O4"):
        q, spectral_info = fit_spectrum(
            model, ordered, args, device, width, shared_rank, digest,
            Path(cli.spectrum_cache) if cli.spectrum_cache else None,
            cli.max_batches)
        if cli.method == "O3":
            adapter.fit(q.to(device), spectral_info, source_split="train")
        else:
            adapter.initialize_variation_basis(orthogonal_complement(q).to(device))
        write_json(outdir/"spectrum.json", spectral_info)
    initial_basis = adapter.basis().detach().cpu() if cli.method != "O3" else None
    if initial_basis is not None:
        torch.save(initial_basis, outdir/"initial_variation_basis.pt")
    model.head_linear.requires_grad_(True)
    groups = [{"params": list(model.head_linear.parameters()), "lr": cli.lr_head,
               "name": "head"}]
    if cli.method != "O3":
        groups.append({"params": list(adapter.parameters()), "lr": cli.lr_basis,
                       "name": "orthogonal_basis"})
    if cli.train_backbone:
        prefixes = ("patch_embedding", "inverted_embedding", "encoder", "modern_tcn_backbone")
        backbone = [p for name, p in model.named_parameters() if name.startswith(prefixes)]
        for parameter in backbone:
            parameter.requires_grad_(True)
        groups.append({"params": backbone, "lr": cli.lr_backbone, "name": "backbone"})
    optimizer = torch.optim.Adam(groups)
    training = train_loader(loaders["train"].dataset.dataset, args, cli.seed)
    config = {"cli": vars(cli), "base_run_config": vars(args),
              "reference_sha256": digest, "variation_rank": variation_rank,
              "shared_rank": shared_rank, "environment_q_used": False,
              "spectrum_source_split": "train" if spectral_info else None,
              "checkpoint_selection": "VAL MSE", "method_ranking": "TEST MSE exploratory",
              "parameters": sum(p.numel() for p in list(model.parameters())+
                                list(adapter.parameters()) if p.requires_grad)}
    write_json(outdir/"run_config.json", config)
    initial_val = evaluate(model, adapter, loaders["val"], args, device,
                           max_batches=cli.max_batches)
    history, best_val = [], math.inf
    started = time.perf_counter()
    for epoch in range(cli.epochs):
        model.train() if cli.train_backbone else model.eval()
        model.head_dropout.train(); adapter.train()
        mse_sum = mae_sum = count = conflict = grad_batches = 0
        for batch_index, (x, y, ids, cycle) in enumerate(training):
            if cli.max_batches and batch_index >= cli.max_batches:
                break
            x, y, cycle = x.float().to(device), y[:, -args.pred_len:].float().to(device), cycle.to(device)
            hidden, mean, std = model.encode(x, cycle)
            out = factorize(adapter, hidden)
            pred = prediction(model, out["z_inv_tokens"], mean, std, cycle, head="active")
            primary = (pred-y).square().mean()
            aux = sum(out["aux_losses"].values(), primary.new_zeros(()))
            total = primary+aux
            if not torch.isfinite(total):
                raise FloatingPointError(f"nonfinite loss epoch {epoch+1}")
            optimizer.zero_grad(set_to_none=True)
            if cli.method == "O2":
                basis = adapter.raw_basis
                pred_grad = torch.autograd.grad(primary, basis, retain_graph=True)[0]
                aux_grad = torch.autograd.grad(aux, basis, retain_graph=True)[0]
            total.backward()
            if cli.method == "O2":
                dot = (pred_grad*aux_grad).sum()
                if dot < 0:
                    projected = aux_grad - cli.gradient_separation * dot / (
                        pred_grad.square().sum()+1e-12)*pred_grad
                    basis.grad = pred_grad+projected
                    conflict += 1
                grad_batches += 1
            optimizer.step()
            mse_sum += float((pred-y).detach().square().sum())
            mae_sum += float((pred-y).detach().abs().sum())
            count += y.numel()
        val = evaluate(model, adapter, loaders["val"], args, device,
                       max_batches=cli.max_batches)
        row = {"epoch": epoch+1, "train_mse": mse_sum/count,
               "train_mae": mae_sum/count, "val_mse": val["mse"],
               "val_mae": val["mae"], "conflict_fraction": conflict/max(grad_batches, 1),
               "inv_energy_ratio": val["diagnostics"].get("inv_energy_ratio"),
               "var_energy_ratio": val["diagnostics"].get("var_energy_ratio"),
               "inv_participation": val["diagnostics"].get("inv_participation"),
               "basis_orthogonal_error": val["diagnostics"].get("basis_orthogonal_error")}
        history.append(row)
        print(json.dumps(row), flush=True)
        write_json(outdir/"history.json", history)
        if val["mse"] < best_val:
            best_val = val["mse"]
            torch.save({"model": model.state_dict(), "decomposer": adapter.state_dict(),
                        "best_epoch": epoch+1, "val": val,
                        "reference_sha256": digest}, outdir/"best.pt")
    best = torch.load(outdir/"best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    adapter.load_state_dict(best["decomposer"])
    test = evaluate(model, adapter, loaders["test"], args, device,
                    max_batches=cli.max_batches)
    train_diagnostics = train_representation_diagnostics(model, adapter, ordered, device,
                                                         batches=1 if cli.max_batches else 8)
    final_basis = adapter.basis().detach().cpu() if cli.method != "O3" else adapter.shared_basis.detach().cpu()
    torch.save(final_basis, outdir/"final_basis.pt")
    result = {"method": cli.method, "reference_sha256": digest,
              "best_epoch": best["best_epoch"], "initial_val": initial_val,
              "val": best["val"], "test": test, "train_history": history,
              "train_representation_diagnostics": train_diagnostics,
              "spectrum": spectral_info, "variation_rank": variation_rank,
              "shared_rank": shared_rank,
              "basis_change_norm": float((final_basis-initial_basis).norm())
              if initial_basis is not None else 0.0,
              "training_seconds": time.perf_counter()-started,
              "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device)
              if device.type == "cuda" else 0,
              "cpu_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
              "decomposition_milliseconds_per_batch": decomposition_latency_ms(
                  model, adapter, loaders["val"], device)}
    write_json(outdir/"result.json", result)
    (outdir/"run_complete").touch()


if __name__ == "__main__":
    main()
