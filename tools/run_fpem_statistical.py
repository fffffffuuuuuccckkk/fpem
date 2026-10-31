"""One shared, VAL-selected point-forecast pilot for H-space statistics.

No EIIL, GRL, environment labels, residual bypass or TEST-based selection.
The existing A0 checkpoint and unchanged ForecastHead are reused.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.predictive_env.signed_gate_decomposition import ComplementaryGateDecomposer
from models.predictive_env.statistical_decomposition import make_statistical_decomposer
from tools.run_fpem_fsld import load_experiment, prediction, train_loader, write_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_run_config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--decomposition_type", choices=("latent_statistics", "baseline"), default="latent_statistics")
    parser.add_argument("--stat_method", required=True)
    parser.add_argument("--stat_groups", type=int, default=4)
    parser.add_argument("--stat_use_orthogonal_basis", action="store_true")
    parser.add_argument("--stat_whitening", choices=("none", "full"), default="none")
    parser.add_argument("--stat_inv_only", action="store_true", default=True)
    parser.add_argument("--train_backbone", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--lr_head", type=float, default=5e-5)
    parser.add_argument("--lr_backbone", type=float, default=2e-5)
    parser.add_argument("--lr_stat", type=float, default=1e-4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=0, help="smoke-test cap; 0 is full data")
    return parser.parse_args()


def build_adapter(cli, model, loader, device):
    if cli.stat_method == "a0":
        return None
    if cli.stat_method == "gate":
        return ComplementaryGateDecomposer(model.d_model, 64).to(device)
    x, _, _, cycle = next(iter(loader))
    with torch.no_grad():
        hidden, _, _ = model.encode(x.float().to(device), cycle.to(device))
    if cli.stat_method in ("t2_orthogonal",) and not cli.stat_use_orthogonal_basis:
        raise ValueError("t2_orthogonal requires --stat_use_orthogonal_basis")
    if cli.stat_method in ("c1_covariance", "c3_dual") and cli.stat_whitening != "full":
        raise ValueError("covariance method requires --stat_whitening full")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cli.seed + 811)
        adapter = make_statistical_decomposer(cli.stat_method, hidden.shape[-1], cli.stat_groups,
                                               hidden.shape[1], hidden.shape[2])
    return adapter.to(device)


def decompose(adapter, hidden):
    if adapter is None:
        return hidden, None, {}
    output = adapter(hidden)
    if isinstance(output, tuple):
        inv, var, gate = output
        return inv, {"legacy_gate": gate}, {}
    return output["z_inv"], output["z_var_stats"], output["diagnostics"]


def descriptor_variance(stats):
    if stats is None:
        return 0.0
    if isinstance(stats, dict):
        return sum(descriptor_variance(item) for item in stats.values()) / max(len(stats), 1)
    pooled = stats.float().reshape(stats.shape[0], -1).mean(-1)
    return float(pooled.var(unbiased=False).detach())


def evaluate(model, adapter, loader, args, device, max_batches=0, blocks=False):
    model.eval()
    if adapter is not None:
        adapter.eval()
    sums = dict(mse=0.0, mae=0.0, descriptor_sample_variance=0.0,
                zinv_rms=0.0, hidden_rms=0.0)
    diag_sums = {}
    count = hidden_count = batches = 0
    block_sum = np.zeros(4)
    block_count = np.zeros(4)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        for batch_number, (x, y, sample_id, cycle) in enumerate(loader):
            if max_batches and batch_number >= max_batches:
                break
            x, y, cycle = x.float().to(device), y[:, -args.pred_len:].float().to(device), cycle.to(device)
            hidden, mean, std = model.encode(x, cycle)
            inv, stats, diagnostics = decompose(adapter, hidden)
            pred = prediction(model, inv, mean, std, cycle, head="active")
            error = pred - y
            n = error.numel()
            sums["mse"] += error.square().sum().item()
            sums["mae"] += error.abs().sum().item()
            sums["zinv_rms"] += inv.square().sum().item()
            sums["hidden_rms"] += hidden.square().sum().item()
            sums["descriptor_sample_variance"] += descriptor_variance(stats)
            for key, value in diagnostics.items():
                scalar = value.detach().float().mean().item() if isinstance(value, torch.Tensor) else float(value)
                diag_sums[key] = diag_sums.get(key, 0.0) + scalar
            if blocks:
                bid = (sample_id.long() * 4 // len(loader.dataset)).clamp(0, 3).cpu().numpy()
                sample_error = error.square().mean((1, 2)).cpu().numpy()
                for index in range(4):
                    chosen = bid == index
                    block_sum[index] += sample_error[chosen].sum()
                    block_count[index] += chosen.sum()
            count += n
            hidden_count += hidden.numel()
            batches += 1
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    if count == 0:
        raise RuntimeError("empty evaluation loader")
    result = {"mse": sums["mse"] / count, "mae": sums["mae"] / count,
              "zinv_rms": math.sqrt(sums["zinv_rms"] / hidden_count),
              "hidden_rms": math.sqrt(sums["hidden_rms"] / hidden_count),
              "descriptor_sample_variance": sums["descriptor_sample_variance"] / batches,
              "seconds": elapsed, "milliseconds_per_batch": 1000 * elapsed / batches,
              "diagnostics": {key: value / batches for key, value in diag_sums.items()}}
    if blocks:
        block = block_sum / np.maximum(block_count, 1)
        result["block_mse"] = block.tolist()
        result["worst_block_mse"] = float(block.max())
        result["block_variance_mse"] = float(block.var())
    return result


def configure_optimizer(cli, model, adapter):
    model.head_linear.requires_grad_(True)
    groups = [{"params": list(model.head_linear.parameters()), "lr": cli.lr_head}]
    if adapter is not None:
        params = list(adapter.parameters())
        if params:
            groups.append({"params": params, "lr": cli.lr_stat})
    if cli.train_backbone or cli.stat_method == "a1":
        names = ("patch_embedding", "inverted_embedding", "encoder", "modern_tcn_backbone")
        backbone = [p for n, p in model.named_parameters() if n.startswith(names)]
        for parameter in backbone:
            parameter.requires_grad_(True)
        groups.append({"params": backbone, "lr": cli.lr_backbone})
    return torch.optim.Adam(groups)


@torch.no_grad()
def decomposition_latency_ms(model, adapter, loader, device):
    if adapter is None:
        return 0.0
    model.eval()
    adapter.eval()
    x, _, _, cycle = next(iter(loader))
    hidden, _, _ = model.encode(x.float().to(device), cycle.to(device))
    for _ in range(3):
        decompose(adapter, hidden)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(20):
        decompose(adapter, hidden)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return 1000 * (time.perf_counter() - start) / 20


def main():
    cli = parse_args()
    output = Path(cli.output)
    output.mkdir(parents=True, exist_ok=True)
    args, model, loaders, device, digest = load_experiment(cli)
    if cli.stat_method == "a1":
        adapter = None
    else:
        adapter = build_adapter(cli, model, loaders["train"], device)
    if cli.stat_method == "a0":
        trainable = 0
        write_json(output / "run_config.json", {"cli": vars(cli), "reference_sha256": digest,
                                                 "trainable_parameters": 0, "test_selection": False})
        result = {"method": "a0", "groups": 0, "train_backbone": False, "best_epoch": 0,
                  "reference_sha256": digest, "trainable_parameters": 0,
                  "val": evaluate(model, None, loaders["val"], args, device, cli.max_batches, blocks=True),
                  "test": evaluate(model, None, loaders["test"], args, device, cli.max_batches)}
        write_json(output / "result.json", result)
        (output / "run_complete").touch()
        return
    initial = evaluate(model, adapter, loaders["val"], args, device,
                       max_batches=3, blocks=False)
    optimizer = configure_optimizer(cli, model, adapter)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable += sum(p.numel() for p in adapter.parameters()) if adapter is not None else 0
    write_json(output / "run_config.json", {
        "cli": vars(cli), "reference_sha256": digest, "backbone": args.backbone,
        "data": args.dataset_name, "pred_len": args.pred_len, "trainable_parameters": trainable,
        "environment_supervision": False, "test_selection": False,
        "method_design": "H-space adaptation; not a literal paper reproduction",
    })
    torch.manual_seed(cli.seed + 961)
    loader = train_loader(loaders["train"].dataset.dataset, args, cli.seed)
    best = math.inf
    history = []
    start = time.perf_counter()
    for epoch in range(1, cli.epochs + 1):
        model.train() if (cli.train_backbone or cli.stat_method == "a1") else model.eval()
        model.head_dropout.train()
        if adapter is not None:
            adapter.train()
        mse_sum = count = 0
        for batch_number, (x, y, _, cycle) in enumerate(loader):
            if cli.max_batches and batch_number >= cli.max_batches:
                break
            x, y, cycle = x.float().to(device), y[:, -args.pred_len:].float().to(device), cycle.to(device)
            hidden, mean, std = model.encode(x, cycle)
            inv, _, _ = decompose(adapter, hidden)
            pred = prediction(model, inv, mean, std, cycle, head="active")
            loss = (pred - y).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at epoch {epoch} batch {batch_number}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            mse_sum += loss.item() * y.numel()
            count += y.numel()
        validation = evaluate(model, adapter, loaders["val"], args, device, cli.max_batches, blocks=True)
        row = {"epoch": epoch, "train_mse": mse_sum / count, "val_mse": validation["mse"],
               "val_mae": validation["mae"]}
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation["mse"] < best:
            best = validation["mse"]
            torch.save({"model": model.state_dict(), "adapter": adapter.state_dict() if adapter else None,
                        "best_epoch": epoch, "val": validation}, output / "best.pt")
    saved = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(saved["model"])
    if adapter is not None:
        adapter.load_state_dict(saved["adapter"])
    test = evaluate(model, adapter, loaders["test"], args, device, cli.max_batches)
    latency = decomposition_latency_ms(model, adapter, loaders["test"], device)
    result = {"method": cli.stat_method, "groups": cli.stat_groups,
              "train_backbone": cli.train_backbone or cli.stat_method == "a1",
              "reference_sha256": digest, "trainable_parameters": trainable,
              "initial_val": initial, "best_epoch": saved["best_epoch"],
              "train_history": history, "val": saved["val"], "test": test,
              "training_seconds": time.perf_counter() - start,
              "decomposition_milliseconds_per_batch": latency,
              "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0}
    write_json(output / "result.json", result)
    (output / "run_complete").touch()


if __name__ == "__main__":
    main()
