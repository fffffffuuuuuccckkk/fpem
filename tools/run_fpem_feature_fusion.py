"""O3/O4 representation-space F1/F2 fusion; F0 is the unchanged original run."""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.predictive_env.orthogonal_spectral_decomposition import (
    OrthogonalComplementaryDecomposer, PredictiveSpectralDecomposer,
    StreamingPredictiveSpectrum, orthogonal_complement,
)
from models.predictive_env.feature_fusion import FeatureFusion
from models.predictive_env.swap_residual import select_swap_pairs
from tools.run_fpem_shared_factorization import (
    decomposition_latency_ms, factorize, load_experiment,
    ordered_training_loader, prediction, train_loader, write_json,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_run_config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--method", choices=("O3", "O4"), required=True)
    p.add_argument("--fusion", choices=("F1", "F2"), required=True)
    p.add_argument("--swap", action="store_true", help="F3: add Zinv swap supervision")
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
    p.add_argument("--lambda_swap", type=float, default=0.01)
    p.add_argument("--swap_k", type=int, default=2)
    p.add_argument("--swap_max_distance", type=float, default=0.5)
    p.add_argument("--swap_temperature", type=float, default=0.25)
    return p.parse_args()


@torch.no_grad()
def evaluate_fused(model, adapter, fusion, loader, args, device, max_batches=0):
    model.eval(); adapter.eval(); fusion.eval()
    totals = {key: 0. for key in ("mse", "mae", "inv_mse", "inv_mae",
                                   "fusion_to_inv_ratio", "var_energy_ratio")}
    count = samples = 0
    for index, (x, y, _, cycle) in enumerate(loader):
        if max_batches and index >= max_batches:
            break
        x = x.float().to(device)
        y = y[:, -args.pred_len:].float().to(device)
        cycle = cycle.to(device)
        hidden, mean, std = model.encode(x, cycle)
        out = factorize(adapter, hidden)
        inv, var = out["z_inv_tokens"], out["z_var_tokens"]
        fused = fusion(inv, var)
        y_inv = prediction(model, inv, mean, std, cycle, head="active")
        y_fused = prediction(model, fused, mean, std, cycle, head="active")
        totals["mse"] += float((y_fused-y).square().sum())
        totals["mae"] += float((y_fused-y).abs().sum())
        totals["inv_mse"] += float((y_inv-y).square().sum())
        totals["inv_mae"] += float((y_inv-y).abs().sum())
        totals["fusion_to_inv_ratio"] += float((fused-inv).norm(dim=-1).mean() /
                                                inv.norm(dim=-1).mean().clamp_min(1e-8)) * len(x)
        totals["var_energy_ratio"] += float(out["diagnostics"].get("var_energy_ratio", 0.)) * len(x)
        count += y.numel(); samples += len(x)
    if not count:
        raise RuntimeError("empty evaluation")
    return {key: value / (samples if key.endswith("ratio") else count)
            for key, value in totals.items()}


@torch.no_grad()
def evaluate_reference(model, loader, args, device, max_batches=0):
    """Evaluate the exact loaded A0 checkpoint on this experiment's TEST split."""
    model.eval()
    squared = absolute = count = 0
    for index, (x, y, _, cycle) in enumerate(loader):
        if max_batches and index >= max_batches:
            break
        x = x.float().to(device)
        y = y[:, -args.pred_len:].float().to(device)
        cycle = cycle.to(device)
        hidden, mean, std = model.encode(x, cycle)
        baseline = prediction(model, hidden, mean, std, cycle, head="reference")
        squared += float((baseline-y).square().sum())
        absolute += float((baseline-y).abs().sum())
        count += y.numel()
    if not count:
        raise RuntimeError("empty A0 evaluation")
    return {"mse": squared/count, "mae": absolute/count}


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


def fused_swap_loss(model, fusion, inv, var, y, mean, std, cycle, pairs,
                    distance, temperature):
    """Swap only Zinv; keep recipient Zvar and RevIN statistics."""
    if pairs.numel() == 0:
        return inv.new_zeros(()), {"pair_fraction": 0., "mean_distance": 0.}
    i, j = pairs[:, 0], pairs[:, 1]
    ij = prediction(model, fusion(inv[j], var[i]), mean[i], std[i], cycle[i], head="active")
    ji = prediction(model, fusion(inv[i], var[j]), mean[j], std[j], cycle[j], head="active")
    weights = torch.exp(-distance.detach().square() / temperature)
    errors = F.smooth_l1_loss(ij, y[i], reduction="none").mean((1, 2)) + \
             F.smooth_l1_loss(ji, y[j], reduction="none").mean((1, 2))
    value = (weights * errors).sum() / (2 * weights.sum().clamp_min(1e-8))
    return value, {"pair_fraction": float(torch.unique(pairs).numel()/len(inv)),
                   "mean_distance": float(distance.mean())}


def main():
    cli = parse_args()
    if not 0 < cli.variation_fraction < 1:
        raise ValueError("variation fraction must be in (0,1)")
    if cli.swap and (cli.swap_temperature <= 0 or cli.swap_k < 1):
        raise ValueError("invalid swap configuration")
    outdir = Path(cli.output)
    outdir.mkdir(parents=True, exist_ok=True)
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
                width, variation_rank, coverage=False,
                lambda_shared=cli.lambda_shared,
                lambda_partial=cli.lambda_partial,
                lambda_variance=cli.lambda_variance).to(device)
        fusion = FeatureFusion(width, cli.fusion).to(device)
    spectral_info = None
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
               "name": "head"},
              {"params": list(fusion.parameters()), "lr": cli.lr_head,
               "name": "feature_fusion"}]
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
                                list(adapter.parameters())+list(fusion.parameters())
                                if p.requires_grad)}
    write_json(outdir/"run_config.json", config)
    model.eval(); adapter.eval(); fusion.eval()
    with torch.no_grad():
        initial_hidden, initial_mean, initial_std = model.encode(
            x.float().to(device), cycle.to(device))
        initial_out = factorize(adapter, initial_hidden)
        initial_inv = initial_out["z_inv_tokens"]
        initial_fused = fusion(initial_inv, initial_out["z_var_tokens"])
        initial_inv_pred = prediction(model, initial_inv, initial_mean, initial_std,
                                      cycle.to(device), head="active")
        initial_fused_pred = prediction(model, initial_fused, initial_mean, initial_std,
                                        cycle.to(device), head="active")
        initial_equivalence = float((initial_fused_pred-initial_inv_pred).square().mean())
        initial_max_abs = float((initial_fused_pred-initial_inv_pred).abs().max())
    if initial_max_abs > 1e-5:
        raise AssertionError(f"fusion did not preserve Zinv-only prediction: {initial_max_abs}")
    initial_val = evaluate_fused(model, adapter, fusion, loaders["val"], args, device,
                                 max_batches=cli.max_batches)
    a0_test = evaluate_reference(model, loaders["test"], args, device,
                                 max_batches=cli.max_batches)
    history, best_val = [], math.inf
    started = time.perf_counter()
    for epoch in range(cli.epochs):
        model.train() if cli.train_backbone else model.eval()
        model.head_dropout.train(); adapter.train(); fusion.train()
        mse_sum = mae_sum = count = swap_total = swap_pairs = 0
        for batch_index, (x, y, ids, cycle) in enumerate(training):
            if cli.max_batches and batch_index >= cli.max_batches:
                break
            x, y, cycle = x.float().to(device), y[:, -args.pred_len:].float().to(device), cycle.to(device)
            hidden, mean, std = model.encode(x, cycle)
            out = factorize(adapter, hidden)
            inv, var = out["z_inv_tokens"], out["z_var_tokens"]
            pred = prediction(model, fusion(inv, var), mean, std, cycle, head="active")
            primary = (pred-y).square().mean()
            aux = sum(out["aux_losses"].values(), primary.new_zeros(()))
            swap = primary.new_zeros(())
            if cli.swap:
                pairs, distances = select_swap_pairs(
                    inv, ids.to(device), seq_len=args.seq_len, pred_len=args.pred_len,
                    top_k=cli.swap_k, max_distance=cli.swap_max_distance)
                swap, swap_info = fused_swap_loss(
                    model, fusion, inv, var, y, mean, std, cycle, pairs,
                    distances, cli.swap_temperature)
                swap_total += float(swap.detach()); swap_pairs += swap_info["pair_fraction"]
            total = primary+aux+(cli.lambda_swap*swap if cli.swap else 0.)
            if not torch.isfinite(total):
                raise FloatingPointError(f"nonfinite loss epoch {epoch+1}")
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            mse_sum += float((pred-y).detach().square().sum())
            mae_sum += float((pred-y).detach().abs().sum())
            count += y.numel()
        val = evaluate_fused(model, adapter, fusion, loaders["val"], args, device,
                             max_batches=cli.max_batches)
        row = {"epoch": epoch+1, "train_mse": mse_sum/count,
               "train_mae": mae_sum/count, "val_mse": val["mse"],
               "val_mae": val["mae"], "val_inv_mse": val["inv_mse"],
               "val_inv_mae": val["inv_mae"], "fusion_to_inv_ratio": val["fusion_to_inv_ratio"],
               "var_energy_ratio": val["var_energy_ratio"],
               "swap_loss_sum": swap_total, "swap_pair_fraction_sum": swap_pairs}
        history.append(row)
        print(json.dumps(row), flush=True)
        write_json(outdir/"history.json", history)
        if val["mse"] < best_val:
            best_val = val["mse"]
            torch.save({"model": model.state_dict(), "decomposer": adapter.state_dict(),
                        "fusion": fusion.state_dict(),
                        "best_epoch": epoch+1, "val": val,
                        "reference_sha256": digest}, outdir/"best.pt")
    best = torch.load(outdir/"best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    adapter.load_state_dict(best["decomposer"])
    fusion.load_state_dict(best["fusion"])
    test = evaluate_fused(model, adapter, fusion, loaders["test"], args, device,
                          max_batches=cli.max_batches)
    train_diagnostics = train_representation_diagnostics(model, adapter, ordered, device,
                                                         batches=1 if cli.max_batches else 8)
    final_basis = adapter.basis().detach().cpu() if cli.method != "O3" else adapter.shared_basis.detach().cpu()
    torch.save(final_basis, outdir/"final_basis.pt")
    result = {"method": cli.method, "fusion": cli.fusion, "swap": cli.swap,
              "reference_sha256": digest,
              "best_epoch": best["best_epoch"], "initial_val": initial_val,
              "a0_test": a0_test,
              "initial_zinv_prediction_mse": initial_equivalence,
              "initial_zinv_prediction_max_abs": initial_max_abs,
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
