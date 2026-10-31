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
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.predictive_env.orthogonal_spectral_decomposition import (
    OrthogonalComplementaryDecomposer, PredictiveSpectralDecomposer,
    StreamingPredictiveSpectrum, orthogonal_complement,
)
from models.predictive_env.swap_residual import (
    ConditionalResidualScale, VariantCorrection, select_swap_pairs, swap_loss,
    variant_difference_loss,
)
from tools.fpem_probabilistic_training import HistoricalEMAMaturity
from tools.evaluate_fpem_probabilistic import probabilistic_metrics
from tools.run_fpem_shared_factorization import (
    decomposition_latency_ms, evaluate, factorize, load_experiment,
    ordered_training_loader, prediction, train_loader, write_json,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_run_config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--method", choices=("O3", "O4"), required=True)
    p.add_argument("--ablation", choices=("E1", "E2", "E3", "E4", "E5", "E6"), required=True)
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
    p.add_argument("--swap_k", type=int, default=2)
    p.add_argument("--swap_max_distance", type=float, default=0.5)
    p.add_argument("--swap_temperature", type=float, default=0.25)
    p.add_argument("--swap_criterion", choices=("huber", "mse"), default="huber")
    p.add_argument("--lambda_swap", type=float, default=0.01)
    p.add_argument("--lambda_inv", type=float, default=1.0)
    p.add_argument("--lambda_var_diff", type=float, default=0.01)
    p.add_argument("--lambda_prob", type=float, default=0.1)
    p.add_argument("--prob_min_weight", type=float, default=0.05)
    p.add_argument("--prob_samples", type=int, default=8)
    p.add_argument("--prob_detach_condition", type=int, choices=(0, 1), default=1)
    return p.parse_args()


@torch.no_grad()
def evaluate_extended(model, adapter, correction, residual, loader, args, device,
                      *, probability=False, sample_count=8, max_batches=0, seed=2021):
    model.eval(); adapter.eval()
    if correction is not None: correction.eval()
    if residual is not None: residual.eval()
    generator = torch.Generator(device=device).manual_seed(seed + 701)
    totals = {"mse": 0., "mae": 0., "inv_mse": 0., "inv_mae": 0.,
              "correction_abs": 0., "point_gain": 0.,
              "inv_energy_ratio": 0., "var_energy_ratio": 0.,
              "basis_orthogonal_error": 0.}
    prob_sums = {}
    count = sample_count_total = 0
    for n, (x, y, _, cycle) in enumerate(loader):
        if max_batches and n >= max_batches: break
        x = x.float().to(device); y = y[:, -args.pred_len:].float().to(device)
        cycle = cycle.to(device)
        hidden, mean, std = model.encode(x, cycle)
        out = factorize(adapter, hidden)
        z_inv, z_var = out["z_inv_tokens"], out["z_var_tokens"]
        inv = prediction(model, z_inv, mean, std, cycle, head="active")
        delta = (correction(z_var, std) if correction is not None
                 else torch.zeros_like(inv))
        point = inv + delta
        totals["mse"] += float((point-y).square().sum())
        totals["mae"] += float((point-y).abs().sum())
        totals["inv_mse"] += float((inv-y).square().sum())
        totals["inv_mae"] += float((inv-y).abs().sum())
        totals["correction_abs"] += float(delta.abs().sum())
        totals["point_gain"] += float(((inv-y).square()-(point-y).square()).sum())
        for key in ("inv_energy_ratio", "var_energy_ratio", "basis_orthogonal_error"):
            totals[key] += float(out["diagnostics"].get(key, 0.)) * y.numel()
        count += y.numel()
        if probability and residual is not None:
            sigma = residual(z_inv, z_var)
            samples = residual.sample(point, sigma, std, sample_count, generator)
            score = probabilistic_metrics(samples, y, inv)
            for key in ("CRPS", "PICP_95", "MPIW_95", "WIS", "sample_std_mean"):
                prob_sums[key] = prob_sums.get(key, 0.) + float(score[key]) * len(x)
            sample_count_total += len(x)
    if not count: raise RuntimeError("empty evaluation")
    return {**{k: v/count for k, v in totals.items()},
            **({k: v/sample_count_total for k, v in prob_sums.items()}
               if sample_count_total else {})}


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
    if not 0 < cli.prob_min_weight <= 1 or cli.lambda_swap < 0 or cli.lambda_prob < 0:
        raise ValueError("invalid nonnegative loss weight or probability floor")
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
    use_correction = cli.ablation in ("E2", "E3", "E4", "E5", "E6")
    use_difference = cli.ablation in ("E3", "E6")
    use_probability = cli.ablation in ("E4", "E5", "E6")
    use_ema = cli.ablation in ("E5", "E6")
    # E1 point==invariant, so a second identical invariant term would change
    # the E0 forecast gradient independently of swap supervision.
    effective_lambda_inv = cli.lambda_inv if use_correction else 0.0
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cli.seed + 9901)
        correction = (VariantCorrection(hidden.shape[2], width, args.pred_len).to(device)
                      if use_correction else None)
        residual = (ConditionalResidualScale(width, args.pred_len).to(device)
                    if use_probability else None)
    ema = HistoricalEMAMaturity() if use_ema else None
    initial_basis = adapter.basis().detach().cpu() if cli.method != "O3" else None
    if initial_basis is not None:
        torch.save(initial_basis, outdir/"initial_variation_basis.pt")
    model.head_linear.requires_grad_(True)
    groups = [{"params": list(model.head_linear.parameters()), "lr": cli.lr_head,
               "name": "head"}]
    if cli.method != "O3":
        groups.append({"params": list(adapter.parameters()), "lr": cli.lr_basis,
                       "name": "orthogonal_basis"})
    if correction is not None:
        groups.append({"params": list(correction.parameters()), "lr": cli.lr_head,
                       "name": "variant_correction"})
    if residual is not None:
        groups.append({"params": list(residual.parameters()), "lr": cli.lr_head,
                       "name": "residual_distribution"})
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
              "checkpoint_selection": "VAL point MSE", "ablation": cli.ablation,
              "probability_target": "TRAIN-only detached online residual",
              "swap_excludes_overlapping_window_starts": True,
              "effective_lambda_inv": effective_lambda_inv,
              "ema_formula_source": "tools/fpem_probabilistic_training.py:HistoricalEMAMaturity",
              "parameters": sum(p.numel() for p in list(model.parameters())+
                                list(adapter.parameters())+
                                (list(correction.parameters()) if correction else [])+
                                (list(residual.parameters()) if residual else [])
                                if p.requires_grad)}
    write_json(outdir/"run_config.json", config)
    initial_val = evaluate_extended(model, adapter, correction, residual,
                                    loaders["val"], args, device,
                                    max_batches=cli.max_batches)
    history, best_val = [], math.inf
    started = time.perf_counter()
    for epoch in range(cli.epochs):
        model.train() if cli.train_backbone else model.eval()
        model.head_dropout.train(); adapter.train()
        if correction is not None: correction.train()
        if residual is not None: residual.train()
        mse_sum = mae_sum = count = seen = 0
        sums = {k: 0. for k in ("loss/inv", "loss/point", "loss/swap", "loss/difference",
                                    "loss/prob", "swap/pair_fraction", "swap/mean_distance",
                                    "swap/target_difference", "residual/abs", "residual/energy",
                                    "residual/bin1", "residual/bin2", "residual/bin3",
                                    "residual/bin4",
                                    "prob/maturity", "prob/weight", "prob/ema_loss",
                                    "prob/relative_change")}
        for batch_index, (x, y, ids, cycle) in enumerate(training):
            if cli.max_batches and batch_index >= cli.max_batches:
                break
            x, y, ids, cycle = (x.float().to(device), y[:, -args.pred_len:].float().to(device),
                                ids.to(device), cycle.to(device))
            hidden, mean, std = model.encode(x, cycle)
            out = factorize(adapter, hidden)
            z_inv, z_var = out["z_inv_tokens"], out["z_var_tokens"]
            inv = prediction(model, z_inv, mean, std, cycle, head="active")
            delta = correction(z_var, std) if correction is not None else torch.zeros_like(inv)
            pred = inv + delta
            inv_loss = (inv-y).square().mean()
            point_loss = (pred-y).square().mean()
            pairs, distance = select_swap_pairs(
                z_inv, ids, seq_len=args.seq_len, pred_len=args.pred_len,
                top_k=cli.swap_k, max_distance=cli.swap_max_distance)
            swap, swap_info = swap_loss(model, prediction, z_inv, y, mean, std,
                                        cycle, pairs, distance,
                                        temperature=cli.swap_temperature,
                                        criterion=cli.swap_criterion)
            diff = (variant_difference_loss(delta, y, inv, pairs) if use_difference
                    else inv_loss.new_zeros(()))
            online_target = (y-pred).detach()
            prob = inv_loss.new_zeros(())
            prob_weight = 0.
            ema_info = {"maturity": 0., "inv_loss_ema": 0.,
                        "inv_loss_relative_change": 0.}
            if residual is not None:
                sigma = residual(z_inv, z_var,
                                 detach_condition=bool(cli.prob_detach_condition))
                prob = residual.nll(sigma, online_target, std)
                if ema is not None:
                    ema_info = ema.update(inv_loss)
                    prob_weight = cli.lambda_prob * (cli.prob_min_weight +
                                                     (1-cli.prob_min_weight)*ema_info["maturity"])
                else:
                    prob_weight = cli.lambda_prob
            aux = sum(out["aux_losses"].values(), inv_loss.new_zeros(()))
            total = (point_loss + effective_lambda_inv*inv_loss + cli.lambda_swap*swap +
                     cli.lambda_var_diff*diff + prob_weight*prob + aux)
            if not torch.isfinite(total):
                raise FloatingPointError(f"nonfinite loss epoch {epoch+1}")
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            mse_sum += float((pred-y).detach().square().sum())
            mae_sum += float((pred-y).detach().abs().sum())
            count += y.numel()
            seen += len(x)
            terms = {"loss/inv": inv_loss, "loss/point": point_loss,
                     "loss/swap": swap, "loss/difference": diff, "loss/prob": prob,
                     "swap/pair_fraction": swap_info["pair_fraction"],
                     "swap/mean_distance": swap_info["mean_distance"],
                     "swap/target_difference": swap_info["target_difference"],
                     "residual/abs": online_target.abs().mean(),
                     "residual/energy": online_target.square().mean(),
                     "prob/maturity": ema_info["maturity"], "prob/weight": prob_weight,
                     "prob/ema_loss": ema_info["inv_loss_ema"],
                     "prob/relative_change": ema_info["inv_loss_relative_change"]}
            for section, horizon_indices in enumerate(
                    torch.tensor_split(torch.arange(args.pred_len), 4), 1):
                terms[f"residual/bin{section}"] = (
                    online_target[:, horizon_indices].square().mean())
            for key, value in terms.items():
                sums[key] += float(torch.as_tensor(value).detach()) * len(x)
        val = evaluate_extended(model, adapter, correction, residual,
                                loaders["val"], args, device,
                                max_batches=cli.max_batches)
        row = {"epoch": epoch+1, "train_mse": mse_sum/count,
               "train_mae": mae_sum/count, "val_mse": val["mse"],
               "val_mae": val["mae"], "val_inv_mse": val["inv_mse"],
               **{key: value / seen
                  for key, value in sums.items()}}
        history.append(row)
        print(json.dumps(row), flush=True)
        write_json(outdir/"history.json", history)
        if val["mse"] < best_val:
            best_val = val["mse"]
            torch.save({"model": model.state_dict(), "decomposer": adapter.state_dict(),
                        "correction": correction.state_dict() if correction else None,
                        "residual": residual.state_dict() if residual else None,
                        "ema": vars(ema) if ema else None,
                        "best_epoch": epoch+1, "val": val,
                        "reference_sha256": digest}, outdir/"best.pt")
    best = torch.load(outdir/"best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    adapter.load_state_dict(best["decomposer"])
    if correction: correction.load_state_dict(best["correction"])
    if residual: residual.load_state_dict(best["residual"])
    test = evaluate_extended(model, adapter, correction, residual, loaders["test"],
                             args, device, probability=use_probability,
                             sample_count=cli.prob_samples,
                             max_batches=cli.max_batches, seed=cli.seed)
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
              "prob_first_nonzero_epoch": 1 if use_probability else None,
              "prob_first_mature_epoch": next((row["epoch"] for row in history
                  if row["prob/maturity"] > 0.01), None) if use_ema else None,
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
