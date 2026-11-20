"""Isolated, VAL-selected FSLD pilot; never calls EIIL or uses TEST to select."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.PatchTST_PredictiveEnvIV import Model
from models.predictive_env.forecast_sufficient_decomposition import (
    ForecastSufficientDecomposer, flatten_forecast_hidden, random_basis,
    spectral_candidates, svd_reference_head,
)
from models.predictive_env.signed_gate_decomposition import ComplementaryGateDecomposer
from tools.run_predictive_env_iv_patchtst import (
    IndexedDataset, build_data, load_shared_reference_state, model_config,
    parse_args as legacy_parse_args, reference_signature,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_run_config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--phase", choices=("direct", "train", "probe", "stability"), default="direct")
    p.add_argument("--method", choices=("A1", "D0", "F1", "F2", "F3", "R1", "R2"))
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr_head", type=float, default=5e-5)
    p.add_argument("--lr_backbone", type=float, default=2e-5)
    p.add_argument("--lr_decomposer", type=float, default=1e-4)
    p.add_argument("--pca_rows", type=int, default=4096)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--max_batches", type=int, default=0, help="Smoke only; zero means full data")
    return p.parse_args()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_experiment(cli):
    saved = json.loads(Path(cli.base_run_config).read_text(encoding="utf-8"))
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]]
        defaults = vars(legacy_parse_args())
    finally:
        sys.argv = original_argv
    defaults.update(saved)
    saved = defaults
    saved.update({"num_workers": 0, "select_invariant_on_val": True,
                  "variant_fusion_mode": "inv_only", "seed": cli.seed})
    args = SimpleNamespace(**saved)
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)
    datasets, train, _, test = build_data(args)
    val = DataLoader(IndexedDataset(datasets["val"], cycle_len=args.cyclenet_cycle_len),
                     batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device(f"cuda:{cli.gpu}" if torch.cuda.is_available() else "cpu")
    model = Model(model_config(args, "A0")).to(device)
    reference = Path(args.reference_checkpoint)
    if not reference.is_file():
        raise FileNotFoundError(f"Existing A0 is required, not retrained: {reference}")
    payload = torch.load(reference, map_location="cpu", weights_only=False)
    if payload.get("signature") != reference_signature(args):
        raise ValueError(f"A0 checkpoint signature mismatch: {reference}")
    load_shared_reference_state(model, payload["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    digest = hashlib.sha256(reference.read_bytes()).hexdigest()
    return args, model, {"train": train, "val": val, "test": test}, device, digest


def prediction(model, hidden, mean, std, cycle_index, head="reference"):
    normalized = (model.frozen_h_head(hidden, cycle_index) if head == "reference"
                  else model.forecast_head(hidden, cycle_index))
    return normalized.permute(0, 2, 1) * std + mean


def mse_mae(pred, target):
    return (pred - target).square(), (pred - target).abs()


def evaluate(model, loader, args, device, decomposer=None, *, head="reference",
             max_batches=0, blocks=False, perturb_scale=0.0):
    model.eval()
    if decomposer is not None:
        decomposer.eval()
    sums = {key: 0.0 for key in ("mse", "mae", "a0_output_mse", "inv_energy", "var_energy",
                                  "cross_inner", "reconstruction")}
    count = 0
    block_sse = np.zeros(4, dtype=np.float64)
    block_n = np.zeros(4, dtype=np.float64)
    length = len(loader.dataset)
    perturb_generator = torch.Generator(device=device).manual_seed(9173)
    with torch.no_grad():
        for batch_number, (x, y, sample_id, cycle_index) in enumerate(loader):
            if max_batches and batch_number >= max_batches:
                break
            x, y, cycle_index = x.float().to(device), y[:, -args.pred_len:].float().to(device), cycle_index.to(device)
            if perturb_scale:
                historical_scale = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
                noise = torch.randn(x.shape, device=device, dtype=x.dtype, generator=perturb_generator)
                x = x + float(perturb_scale) * historical_scale * noise
            hidden, mean, std = model.encode(x, cycle_index)
            z_inv, z_var, _ = decomposer(hidden) if decomposer else (hidden, torch.zeros_like(hidden), None)
            output = prediction(model, z_inv, mean, std, cycle_index, head=head)
            ref = prediction(model, hidden, mean, std, cycle_index, head="reference")
            se, ae = mse_mae(output, y)
            n = se.numel()
            count += n
            sums["mse"] += se.sum().item()
            sums["mae"] += ae.sum().item()
            sums["a0_output_mse"] += (output - ref).square().sum().item()
            sums["inv_energy"] += z_inv.square().sum().item()
            sums["var_energy"] += z_var.square().sum().item()
            sums["cross_inner"] += (z_inv * z_var).sum().item()
            sums["reconstruction"] += (z_inv + z_var - hidden).abs().sum().item()
            if blocks:
                bid = torch.clamp(sample_id.long() * 4 // max(length, 1), 0, 3).cpu().numpy()
                per_sample = se.mean((1, 2)).cpu().numpy()
                for index in range(4):
                    selected = bid == index
                    block_sse[index] += per_sample[selected].sum()
                    block_n[index] += selected.sum()
    if not count:
        raise RuntimeError("Empty evaluation loader")
    result = {key: value / count for key, value in sums.items()}
    result["samples"] = len(loader.dataset) if not max_batches else count // (args.pred_len * args.enc_in)
    if blocks:
        result["block_mse"] = (block_sse / np.maximum(block_n, 1)).tolist()
        result["block_mean_mse"] = float(np.mean(result["block_mse"]))
        result["block_worst_mse"] = float(np.max(result["block_mse"]))
        result["block_variance_mse"] = float(np.var(result["block_mse"]))
    return result


def direct(cli, output, args, model, loaders, device, digest):
    u, singular, full_rank = svd_reference_head(model.h_forecast_head)
    features = u.shape[0]
    ranks = spectral_candidates(singular, full_rank)
    torch.save({"basis": u.cpu(), "singular": singular.cpu(), "full_rank": full_rank,
                "reference_sha256": digest}, output / "svd_basis.pt")
    spectrum = {"features": features, "horizon": args.pred_len, "numeric_rank": full_rank,
                "singular_values": singular.cpu().tolist(), "candidate_ranks": ranks,
                "full_projection_is_identity": full_rank == features}
    write_json(output / "spectrum.json", spectrum)
    rows = []
    for rank in [None] + ranks:
        name = "A0" if rank is None else ("FSLD-Full" if rank == full_rank else "FSLD-Truncated")
        decomposer = None if rank is None else ForecastSufficientDecomposer(u, rank).to(device)
        for split in ("train", "val", "test"):
            metrics = evaluate(model, loaders[split], args, device, decomposer,
                               max_batches=cli.max_batches, blocks=split == "val")
            row = {"method": name, "rank": features if rank is None else rank,
                   "split": split, **metrics}
            rows.append(row)
            print(json.dumps({key: row[key] for key in ("method", "rank", "split", "mse", "mae", "a0_output_mse")}), flush=True)
    with (output / "direct_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    full = next(row for row in rows if row["method"] == "FSLD-Full" and row["split"] == "val")
    a0 = next(row for row in rows if row["method"] == "A0" and row["split"] == "val")
    if full["a0_output_mse"] > 1e-7 or abs(full["mse"] - a0["mse"]) > 1e-5:
        raise AssertionError(f"Full projection did not preserve frozen A0: {full}")
    selected = min((row for row in rows if row["split"] == "val" and row["method"] == "FSLD-Truncated"),
                   key=lambda row: (row["mse"], row["rank"]))
    write_json(output / "validation_selection.json", {
        "selected_rank": selected["rank"], "validation_mse": selected["mse"],
        "selection_rule": "minimum VAL MSE; TEST never used", "reference_sha256": digest})
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        values = [row for row in rows if row["split"] == "val" and row["method"].startswith("FSLD")]
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
        axes[0].plot([row["rank"] for row in values], [row["mse"] for row in values], marker="o")
        axes[0].axhline(a0["mse"], color="gray", linestyle="--", label="A0")
        axes[0].set(ylabel="VAL MSE", xlabel="rank")
        axes[0].legend()
        axes[1].plot([row["rank"] for row in values], [row["a0_output_mse"] for row in values], marker="o")
        axes[1].set(ylabel="MSE vs frozen A0 output", xlabel="rank")
        axes[2].plot([row["rank"] for row in values],
                     [row["inv_energy"] / max(row["inv_energy"] + row["var_energy"], 1e-12) for row in values], marker="o")
        axes[2].set(ylabel="Zinv energy fraction", xlabel="rank")
        fig.tight_layout()
        fig.savefig(output / "rank_diagnostics.png", dpi=160)
        plt.close(fig)
    except ImportError:
        pass
    (output / "direct_complete").touch()


def train_loader(dataset, args, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(IndexedDataset(dataset, cycle_len=args.cyclenet_cycle_len),
                      batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0)


@torch.no_grad()
def pca_basis(model, train, args, device, rank, row_limit, seed):
    # TRAIN-only randomized PCA; bounds storage, never reads VAL/TEST targets.
    generator = torch.Generator().manual_seed(seed)
    rows = []
    stride = max(1, math.ceil(len(train.dataset) * args.enc_in / row_limit))
    offset = 0
    for x, _, _, cycle in train:
        hidden, _, _ = model.encode(x.float().to(device), cycle.to(device))
        flat = flatten_forecast_hidden(hidden).reshape(-1, model.head_linear.in_features).cpu()
        selected = torch.arange(flat.shape[0]) + offset
        rows.append(flat[selected.remainder(stride) == 0])
        offset += flat.shape[0]
    matrix = torch.cat(rows)[:row_limit]
    chosen = torch.randperm(matrix.shape[0], generator=generator)[:row_limit]
    matrix = matrix[chosen].to(device)
    matrix -= matrix.mean(0, keepdim=True)
    q = min(rank + 8, min(matrix.shape))
    _, _, basis = torch.pca_lowrank(matrix, q=q, center=False, niter=2)
    return basis[:, :rank].detach()


def configure_method(cli, output, args, model, loaders, device, rank, u, full_rank):
    method = cli.method
    if method in ("F1", "F2", "F3"):
        projection_rank = full_rank if method == "F1" else rank
        decomposer = ForecastSufficientDecomposer(u, projection_rank).to(device)
    elif method == "R1":
        decomposer = ForecastSufficientDecomposer(random_basis(u.shape[0], rank, cli.seed, device), rank).to(device)
    elif method == "R2":
        decomposer = ForecastSufficientDecomposer(pca_basis(model, loaders["train"], args, device,
                                                             rank, cli.pca_rows, cli.seed), rank).to(device)
    elif method == "D0":
        decomposer = ComplementaryGateDecomposer(args.d_model, args.bottleneck).to(device)
    else:
        decomposer = None
    model.head_linear.requires_grad_(True)
    groups = [{"params": list(model.head_linear.parameters()), "lr": cli.lr_head}]
    if method in ("A1", "F3"):
        backbone_names = ("patch_embedding", "inverted_embedding", "encoder", "modern_tcn_backbone")
        backbone = [p for n, p in model.named_parameters() if n.startswith(backbone_names)]
        for parameter in backbone:
            parameter.requires_grad_(True)
        groups.append({"params": backbone, "lr": cli.lr_backbone})
    if method == "D0":
        groups.append({"params": list(decomposer.parameters()), "lr": cli.lr_decomposer})
    return decomposer, torch.optim.Adam(groups)


def train(cli, output, args, model, loaders, device, digest):
    if not cli.method:
        raise ValueError("--method is required for train")
    direct_root = output.parent / "direct"
    selected = json.loads((direct_root / "validation_selection.json").read_text())
    basis_data = torch.load(direct_root / "svd_basis.pt", map_location=device, weights_only=False)
    if selected["reference_sha256"] != digest or basis_data["reference_sha256"] != digest:
        raise ValueError("Direct SVD and training A0 checkpoint hashes differ")
    rank = int(selected["selected_rank"])
    u = basis_data["basis"]
    decomposer, optimizer = configure_method(cli, output, args, model, loaders, device,
                                             rank, u, int(basis_data["full_rank"]))
    shuffled = train_loader(loaders["train"].dataset.dataset, args, cli.seed)
    best = math.inf
    history = []
    start = time.time()
    for epoch in range(1, cli.epochs + 1):
        # A frozen backbone must also keep BatchNorm statistics frozen.
        model.train() if cli.method in ("A1", "F3") else model.eval()
        model.head_dropout.train()
        if decomposer is not None:
            decomposer.train()
        losses = []
        for batch_number, (x, y, _, cycle) in enumerate(shuffled):
            if cli.max_batches and batch_number >= cli.max_batches:
                break
            x, y, cycle = x.float().to(device), y[:, -args.pred_len:].float().to(device), cycle.to(device)
            hidden, mean, std = model.encode(x, cycle)
            z_inv = decomposer(hidden)[0] if decomposer is not None else hidden
            output_y = prediction(model, z_inv, mean, std, cycle, head="active")
            loss = (output_y - y).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        val = evaluate(model, loaders["val"], args, device, decomposer, head="active", max_batches=cli.max_batches, blocks=True)
        row = {"epoch": epoch, "train_mse": float(np.mean(losses)), "val_mse": val["mse"], "val_mae": val["mae"],
               "seconds_elapsed": time.time() - start}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val["mse"] < best:
            best = val["mse"]
            torch.save({"model": model.state_dict(), "decomposer": decomposer.state_dict() if decomposer else None,
                        "epoch": epoch, "val": val}, output / "best.pt")
    best_data = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_data["model"])
    if decomposer is not None:
        decomposer.load_state_dict(best_data["decomposer"])
    test = evaluate(model, loaders["test"], args, device, decomposer, head="active", max_batches=cli.max_batches)
    perturbed = evaluate(model, loaders["test"], args, device, decomposer,
                         head="active", max_batches=cli.max_batches, blocks=True, perturb_scale=0.05)
    reported_rank = int(basis_data["full_rank"]) if cli.method == "F1" else (None if cli.method in ("A1", "D0") else rank)
    result = {"method": cli.method, "rank": reported_rank, "best_epoch": best_data["epoch"], "val": best_data["val"],
              "test": test, "test_history_noise_0p05": perturbed,
              "reference_sha256": digest, "history": history, "seconds": time.time() - start}
    write_json(output / "result.json", result)
    (output / "run_complete").touch()


def probe(cli, output, args, model, loaders, device, digest):
    """Independent matched-capacity heads on frozen H, Zinv, Zvar and pairs."""
    direct_root = output.parent / "direct"
    selected = json.loads((direct_root / "validation_selection.json").read_text())
    basis_data = torch.load(direct_root / "svd_basis.pt", map_location=device, weights_only=False)
    if selected["reference_sha256"] != digest:
        raise ValueError("Probe reference does not match the direct phase")
    basis = basis_data["basis"]
    projector = ForecastSufficientDecomposer(basis, int(selected["selected_rank"])).to(device)
    feature_count = model.head_linear.in_features
    modes = ("zinv_only", "zvar_only", "joint", "h_matched", "joint_shuffled")
    all_results = []
    for mode in modes:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cli.seed + 33)
            head = nn.Linear(2 * feature_count, args.pred_len).to(device)
        optimizer = torch.optim.Adam(head.parameters(), lr=cli.lr_head)
        shuffled = train_loader(loaders["train"].dataset.dataset, args, cli.seed)
        best = math.inf
        best_epoch = 0

        def features(hidden):
            inv, var, _ = projector(hidden)
            if mode == "zinv_only":
                left, right = inv, torch.zeros_like(var)
            elif mode == "zvar_only":
                left, right = torch.zeros_like(inv), var
            elif mode == "joint":
                left, right = inv, var
            elif mode == "h_matched":
                left, right = hidden, hidden
            else:
                left, right = inv, torch.roll(var, shifts=1, dims=0)
            return torch.cat((flatten_forecast_hidden(left), flatten_forecast_hidden(right)), dim=-1)

        def score(loader):
            head.eval()
            se_sum = ae_sum = count = 0
            with torch.no_grad():
                for batch_number, (x, y, _, cycle) in enumerate(loader):
                    if cli.max_batches and batch_number >= cli.max_batches:
                        break
                    x, y, cycle = x.float().to(device), y[:, -args.pred_len:].float().to(device), cycle.to(device)
                    hidden, mean, std = model.encode(x, cycle)
                    estimate = head(features(hidden)).permute(0, 2, 1) * std + mean
                    se, ae = mse_mae(estimate, y)
                    se_sum += se.sum().item()
                    ae_sum += ae.sum().item()
                    count += se.numel()
            return {"mse": se_sum / count, "mae": ae_sum / count}

        for epoch in range(1, cli.epochs + 1):
            head.train()
            for batch_number, (x, y, _, cycle) in enumerate(shuffled):
                if cli.max_batches and batch_number >= cli.max_batches:
                    break
                x, y, cycle = x.float().to(device), y[:, -args.pred_len:].float().to(device), cycle.to(device)
                with torch.no_grad():
                    hidden, mean, std = model.encode(x, cycle)
                    predictor_input = features(hidden)
                estimate = head(predictor_input).permute(0, 2, 1) * std + mean
                loss = (estimate - y).square().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            val = score(loaders["val"])
            print(json.dumps({"mode": mode, "epoch": epoch, "val": val}), flush=True)
            if val["mse"] < best:
                best, best_epoch = val["mse"], epoch
                torch.save(head.state_dict(), output / f"{mode}_best.pt")
        head.load_state_dict(torch.load(output / f"{mode}_best.pt", map_location=device, weights_only=True))
        result = {"mode": mode, "head_parameters": sum(p.numel() for p in head.parameters()),
                  "best_epoch": best_epoch, "val_mse": best, "test": score(loaders["test"])}
        all_results.append(result)
        write_json(output / "probe_results.json", {"results": all_results, "reference_sha256": digest})
    (output / "run_complete").touch()


def main():
    cli = parse_args()
    output = Path(cli.output)
    output.mkdir(parents=True, exist_ok=True)
    args, model, loaders, device, digest = load_experiment(cli)
    write_json(output / "run_config.json", {"pilot": vars(cli), "base": vars(args), "reference_sha256": digest,
                                                  "test_selection": False, "environment_used": False})
    if cli.phase == "direct":
        direct(cli, output, args, model, loaders, device, digest)
    elif cli.phase == "train":
        train(cli, output, args, model, loaders, device, digest)
    elif cli.phase == "probe":
        probe(cli, output, args, model, loaders, device, digest)
    else:
        raise NotImplementedError(f"{cli.phase} is reserved for the later Zvar/stability stage")


if __name__ == "__main__":
    main()
