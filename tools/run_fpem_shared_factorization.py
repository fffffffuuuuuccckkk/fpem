"""TRAIN-only dynamic-environment, VAL-selected shared-factorization pilot.

Reuses the existing A0 checkpoint, backbone adapter, predictive-conflict EIIL,
and forecasting head. TEST is evaluated exactly once after VAL selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.predictive_env.eiil_environment import PredictiveConflictEnvironment
from models.PatchTST_PredictiveEnvIV import Model
from models.predictive_env.shared_factorization import (
    EnvironmentPartitionAdapter, StreamingHiddenStatistics,
    make_shared_factor_decomposer,
)
from tools.run_predictive_env_iv_patchtst import (
    IndexedDataset, build_data, load_shared_reference_state,
    model_config, parse_args as legacy_parse_args, reference_signature,
    Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom,
)


METHODS = (
    "common_subspace", "shared_spectral", "robust_lowrank_sparse",
    "supervised_joint_private", "shared_private_lite", "pairwise_partial_shared",
    "sample_prevalence", "subspace_recurrence", "subspace_recurrence_poly",
)


def write_json(path, value):
    path = Path(path)
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
    defaults.update({"num_workers": 0, "select_invariant_on_val": True,
                     "variant_fusion_mode": "inv_only", "seed": cli.seed})
    args = SimpleNamespace(**defaults)
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)
    datasets, train, _, _ = build_data(args)
    if "val" not in datasets:
        # Older secondary-server copies of build_data omit VAL altogether.
        dataset_class = {"ett_hour": Dataset_ETT_hour, "ett_minute": Dataset_ETT_minute,
                         "custom": Dataset_Custom}[args.data_class]
        datasets["val"] = dataset_class(
            flag="val", args=SimpleNamespace(augmentation_ratio=0),
            root_path=args.root_path,
            size=[args.seq_len, min(48, args.seq_len // 2), args.pred_len],
            features="M", data_path=args.data_path, target=args.target,
            scale=True, timeenc=1, freq=args.freq)
    val = DataLoader(IndexedDataset(datasets["val"], cycle_len=args.cyclenet_cycle_len),
                     batch_size=args.batch_size, shuffle=False, num_workers=0)
    test = DataLoader(IndexedDataset(datasets["test"], cycle_len=args.cyclenet_cycle_len),
                      batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device(f"cuda:{cli.gpu}" if torch.cuda.is_available() else "cpu")
    cfg = model_config(args, "A0")
    modern_fields = (
        "modern_tcn_patch_size", "modern_tcn_patch_stride", "modern_tcn_num_stages",
        "modern_tcn_ffn_ratio", "modern_tcn_large_size", "modern_tcn_small_size",
        "modern_tcn_downsample_ratio", "modern_tcn_head_dropout",
    )
    if args.backbone == "moderntcn":
        for field in modern_fields:
            setattr(cfg, field, getattr(args, field))
    try:
        model = Model(cfg).to(device)
    except ValueError as error:
        if args.backbone != "moderntcn" or "predictive_env_backbone" not in str(error):
            raise
        # Server 3 retains an older main model. Use an isolated, identical
        # ModernTCN adapter rather than overwriting that user's model file.
        from models.PatchTST_PredictiveEnvIV_shared import Model as ModernCompatibleModel
        model = ModernCompatibleModel(cfg).to(device)
    reference = Path(args.reference_checkpoint)
    if not reference.is_file():
        raise FileNotFoundError(f"existing A0 required, never retrained: {reference}")
    payload = torch.load(reference, map_location="cpu", weights_only=False)
    expected_signature = reference_signature(args)
    if args.backbone == "moderntcn" and "modern_tcn_patch_size" not in expected_signature:
        expected_signature.update({field: getattr(args, field) for field in modern_fields})
    actual_signature = payload.get("signature")
    if actual_signature != expected_signature:
        if not getattr(cli, "allow_legacy_reference_signature", False):
            raise ValueError(f"A0 checkpoint signature mismatch: {reference}")
        # Earlier A0 checkpoints predate the archive/protocol signature fields.
        # Accept them only when every shared field agrees and the complete
        # backbone/data/shape identity is present. The checkpoint SHA remains
        # recorded in every trial for exact provenance.
        required = {"backbone", "dataset_name", "data_path", "seq_len",
                    "pred_len", "d_model", "d_ff", "n_heads", "e_layers",
                    "dropout", "seed", "warmup_epochs", "enc_in"}
        if (not isinstance(actual_signature, dict)
                or not required.issubset(actual_signature)
                or any(actual_signature[k] != expected_signature[k]
                       for k in actual_signature.keys() & expected_signature.keys())):
            raise ValueError(f"A0 legacy checkpoint identity mismatch: {reference}")
        print(f"LEGACY A0 signature accepted after shared-field verification: {reference}",
              flush=True)
    load_shared_reference_state(model, payload["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return args, model, {"train": train, "val": val, "test": test}, device, hashlib.sha256(reference.read_bytes()).hexdigest()


def prediction(model, hidden, mean, std, cycle_index, head="reference"):
    normalized = (model.frozen_h_head(hidden, cycle_index) if head == "reference"
                  else model.forecast_head(hidden, cycle_index))
    return normalized.permute(0, 2, 1) * std + mean


def train_loader(dataset, args, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(IndexedDataset(dataset, cycle_len=args.cyclenet_cycle_len),
                      batch_size=args.batch_size, shuffle=True, generator=generator,
                      num_workers=0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_run_config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--stage_epochs", type=int, default=0,
                   help="0 reuses the base run_config schedule")
    p.add_argument("--lr_head", type=float, default=5e-5)
    p.add_argument("--lr_decomposer", type=float, default=1e-4)
    p.add_argument("--lr_backbone", type=float, default=2e-5)
    p.add_argument("--train_backbone", action="store_true")
    p.add_argument("--environment_mode", choices=("predictive", "shuffled", "random",
                                                    "uniform", "no_env"), default="predictive")
    p.add_argument("--function_preserving_init", action="store_true")
    p.add_argument("--pair_affinity", choices=("env_affinity", "hybrid_affinity"),
                   default="env_affinity")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--max_batches", type=int, default=0, help="smoke only; full=0")
    p.add_argument("--eiil_steps", type=int, default=0,
                   help="0 reuses the base run_config")
    return p.parse_args()


def ordered_training_loader(dataset, args):
    return DataLoader(IndexedDataset(dataset, cycle_len=args.cyclenet_cycle_len),
                      batch_size=args.batch_size, shuffle=False, num_workers=0)


def factorize(adapter, hidden, ids=None, q=None):
    return adapter(hidden, sample_indices=ids, train_env_q=q, return_diagnostics=True)


@torch.no_grad()
def collect_train_predictions(model, adapter, loader, args, device, max_batches=0):
    model.eval(); adapter.eval()
    preds, targets, indices = [], [], []
    for n, (x, y, ids, cycle) in enumerate(loader):
        if max_batches and n >= max_batches:
            break
        x, cycle = x.float().to(device), cycle.to(device)
        hidden, mean, std = model.encode(x, cycle)
        inv = factorize(adapter, hidden)["z_inv_tokens"]
        preds.append(prediction(model, inv, mean, std, cycle, head="active").cpu())
        targets.append(y[:, -args.pred_len:].float().cpu())
        indices.append(ids.cpu())
    indices = torch.cat(indices)
    if not torch.equal(indices, torch.arange(len(indices))):
        raise RuntimeError("ordered TRAIN sample IDs required for EIIL")
    if not max_batches and len(indices) != len(loader.dataset):
        raise RuntimeError("TRAIN predictions are incomplete")
    return torch.cat(preds), torch.cat(targets), indices


@torch.no_grad()
def refit_statistics(model, adapter, loader, partition, device, max_batches=0):
    if not adapter.needs_fit:
        return {}
    model.eval(); adapter.eval()
    stats = StreamingHiddenStatistics(model.d_model,
                                      partition.get_train_assignments().shape[1],
                                      max_rows=4096 if adapter.method == "robust_lowrank_sparse" else 0)
    for n, (x, _, ids, cycle) in enumerate(loader):
        if max_batches and n >= max_batches:
            break
        hidden, _, _ = model.encode(x.float().to(device), cycle.to(device))
        stats.add(hidden, partition.get_assignments(ids))
    result = stats.finish()
    kwargs = {"model": model, "loader": loader, "device": device,
              "q": partition.get_train_assignments()}
    return adapter.fit(result, **kwargs)


@torch.no_grad()
def evaluate(model, adapter, loader, args, device, max_batches=0, a0=False):
    model.eval(); adapter.eval()
    totals = dict(mse=0., mae=0., zvar_mse=0., a0_prediction_difference=0.,
                  hidden_energy=0., inv_energy=0., var_energy=0., rare_energy=0.,
                  reconstruction_mae=0.)
    count = hidden_count = batches = 0
    diagnostics = {}
    started = time.perf_counter()
    for n, (x, y, ids, cycle) in enumerate(loader):
        if max_batches and n >= max_batches:
            break
        x, y, cycle = x.float().to(device), y[:, -args.pred_len:].float().to(device), cycle.to(device)
        hidden, mean, std = model.encode(x, cycle)
        out = factorize(adapter, hidden)
        inv, var = out["z_inv_tokens"], out["z_var_tokens"]
        pred = prediction(model, inv, mean, std, cycle, head="active")
        baseline = prediction(model, hidden, mean, std, cycle, head="reference")
        vpred = prediction(model, var, mean, std, cycle, head="active")
        rare = out["z_rare_tokens"]
        totals["mse"] += (pred - y).square().sum().item()
        totals["mae"] += (pred - y).abs().sum().item()
        totals["zvar_mse"] += (vpred - y).square().sum().item()
        totals["a0_prediction_difference"] += (pred - baseline).square().sum().item()
        totals["hidden_energy"] += hidden.square().sum().item()
        totals["inv_energy"] += inv.square().sum().item()
        totals["var_energy"] += var.square().sum().item()
        totals["rare_energy"] += rare.square().sum().item() if rare is not None else 0.
        totals["reconstruction_mae"] += (out["reconstruction"] - hidden).abs().sum().item()
        for key, value in out["diagnostics"].items():
            scalar = float(value.detach().float().mean() if torch.is_tensor(value) else value)
            diagnostics[key] = diagnostics.get(key, 0.) + scalar
        count += y.numel(); hidden_count += hidden.numel(); batches += 1
    if count == 0:
        raise RuntimeError("empty evaluation")
    return {**{k: v / (hidden_count if k.endswith("energy") or k == "reconstruction_mae" else count)
               for k, v in totals.items()},
            "diagnostics": {k: v / batches for k, v in diagnostics.items()},
            "seconds": time.perf_counter() - started,
            "milliseconds_per_batch": 1000 * (time.perf_counter() - started) / batches}


def build_optimizer(model, adapter, cli):
    model.head_linear.requires_grad_(True)
    groups = [{"name": "head", "params": list(model.head_linear.parameters()), "lr": cli.lr_head}]
    new = [p for p in adapter.parameters() if p.requires_grad]
    if new:
        groups.append({"name": "decomposer", "params": new, "lr": cli.lr_decomposer})
    if cli.train_backbone:
        prefixes = ("patch_embedding", "inverted_embedding", "encoder", "modern_tcn_backbone")
        backbone = [p for name, p in model.named_parameters() if name.startswith(prefixes)]
        for p in backbone:
            p.requires_grad_(True)
        groups.append({"name": "backbone", "params": backbone, "lr": cli.lr_backbone})
    return torch.optim.Adam(groups)


@torch.no_grad()
def decomposition_latency_ms(model, adapter, loader, device):
    model.eval(); adapter.eval()
    x, _, _, cycle = next(iter(loader))
    hidden, _, _ = model.encode(x.float().to(device), cycle.to(device))
    for _ in range(3):
        factorize(adapter, hidden)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(20):
        factorize(adapter, hidden)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return 1000 * (time.perf_counter() - started) / 20


def main():
    cli = parse_args()
    outdir = Path(cli.output)
    outdir.mkdir(parents=True, exist_ok=True)
    args, model, loaders, device, reference_hash = load_experiment(cli)
    ordered = ordered_training_loader(loaders["train"].dataset.dataset, args)
    x, _, _, cycle = next(iter(ordered))
    with torch.no_grad():
        hidden, _, _ = model.encode(x.float().to(device), cycle.to(device))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cli.seed + 811)
        adapter = make_shared_factor_decomposer(
            cli.method, hidden.shape[-1], seq_len=args.seq_len,
            affinity=cli.pair_affinity, function_preserving_init=cli.function_preserving_init,
        ).to(device)
    environment = PredictiveConflictEnvironment(
        len(ordered.dataset), env_num=args.env_num,
        steps=cli.eiil_steps or args.eiil_steps,
        lr=args.eiil_lr, balance_weight=args.balance_weight,
        entropy_weight=args.entropy_weight, seed=cli.seed,
        diagnostic_samples=min(512, len(ordered.dataset)),
        random_partition_repeats=10 if cli.max_batches else 100,
    )
    # For a capped smoke test, EIIL must receive exactly the capped sample count.
    if cli.max_batches:
        n = min(len(ordered.dataset), cli.max_batches * args.batch_size)
        environment = PredictiveConflictEnvironment(
            n, env_num=args.env_num, steps=cli.eiil_steps or args.eiil_steps,
            lr=args.eiil_lr, balance_weight=args.balance_weight,
            entropy_weight=args.entropy_weight, seed=cli.seed,
            diagnostic_samples=min(128, n), random_partition_repeats=10)
    partition = EnvironmentPartitionAdapter(environment.sample_count, args.env_num,
                                            cli.environment_mode, cli.seed)
    optimizer = build_optimizer(model, adapter, cli)
    if cli.max_batches:
        # Smoke uses only chronological prefix IDs, including its shuffled train loader.
        subset = Subset(IndexedDataset(loaders["train"].dataset.dataset,
                                        cycle_len=args.cyclenet_cycle_len),
                        range(environment.sample_count))
        training = DataLoader(subset, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(cli.seed), num_workers=0)
    else:
        training = train_loader(loaders["train"].dataset.dataset, args, cli.seed)
    stage_epochs = cli.stage_epochs or args.stage_epochs
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=False).stdout.strip()
    config = {"cli": vars(cli), "base_run_config": vars(args), "reference_sha256": reference_hash,
              "git_commit": commit, "selection_split": "val", "environment_source": "TRAIN only",
              "trainable_parameters": sum(p.numel() for p in list(model.parameters()) +
                                          list(adapter.parameters()) if p.requires_grad),
              "buffer_bytes": sum(b.numel() * b.element_size() for b in adapter.buffers())}
    write_json(outdir / "run_config.json", config)
    history, environments, fits = [], [], []
    best_val = math.inf
    begin = time.perf_counter()
    for stage_start in range(0, cli.epochs, stage_epochs):
        stage = stage_start // stage_epochs
        pred, true, ids = collect_train_predictions(model, adapter, ordered, args, device,
                                                     cli.max_batches)
        discovered, record = environment.update(pred, true, ids, source_split="train")
        partition.refresh_after_environment_update(discovered)
        record.update({"stage": stage, "supervision": cli.environment_mode,
                       "assignment_hash": hashlib.sha256(partition.q.numpy().tobytes()).hexdigest()})
        environments.append(record)
        fit_started = time.perf_counter()
        fit = refit_statistics(model, adapter, ordered, partition, device, cli.max_batches)
        fits.append({"stage": stage, "seconds": time.perf_counter() - fit_started, **fit})
        if stage == 0:
            initial = evaluate(model, adapter, loaders["val"], args, device,
                               max_batches=cli.max_batches)
        for epoch in range(stage_start, min(cli.epochs, stage_start + stage_epochs)):
            model.train() if cli.train_backbone else model.eval()
            model.head_dropout.train()
            adapter.train()
            total, count = 0., 0
            for batch_number, (x, y, sample_ids, cycle) in enumerate(training):
                if cli.max_batches and batch_number >= cli.max_batches:
                    break
                x, y, cycle = x.float().to(device), y[:, -args.pred_len:].float().to(device), cycle.to(device)
                hidden, mean, std = model.encode(x, cycle)
                q_batch = partition.get_assignments(sample_ids).to(device)
                factors = factorize(adapter, hidden, sample_ids, q_batch)
                pred = prediction(model, factors["z_inv_tokens"], mean, std, cycle,
                                  head="active")
                primary = (pred - y).square().mean()
                aux = sum(factors["aux_losses"].values(), primary.new_zeros(()))
                loss = primary + aux
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite loss stage={stage} epoch={epoch+1}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(primary.detach()) * y.numel(); count += y.numel()
            val = evaluate(model, adapter, loaders["val"], args, device,
                           max_batches=cli.max_batches)
            row = {"epoch": epoch + 1, "stage": stage, "train_mse": total / count,
                   "val_mse": val["mse"], "val_mae": val["mae"]}
            history.append(row)
            print(json.dumps(row), flush=True)
            if val["mse"] < best_val:
                best_val = val["mse"]
                torch.save({"model": model.state_dict(), "decomposer": adapter.state_dict(),
                            "best_epoch": epoch + 1, "val": val, "stage": stage,
                            "reference_sha256": reference_hash}, outdir / "best.pt")
        write_json(outdir / "environment_stages.json", environments)
        write_json(outdir / "fit_stages.json", fits)
    best = torch.load(outdir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    adapter.load_state_dict(best["decomposer"])
    test = evaluate(model, adapter, loaders["test"], args, device,
                    max_batches=cli.max_batches)
    latency = decomposition_latency_ms(model, adapter, loaders["val"], device)
    result = {"method": cli.method, "environment_mode": cli.environment_mode,
              "reference_sha256": reference_hash, "best_epoch": best["best_epoch"],
              "initial_val": initial, "val": best["val"], "test": test,
              "train_history": history, "environment_history": environments,
              "fit_history": fits, "training_seconds": time.perf_counter() - begin,
              "decomposition_milliseconds_per_batch": latency,
              "cpu_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "trainable_parameters": config["trainable_parameters"],
              "buffer_bytes": config["buffer_bytes"],
              "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device)
              if device.type == "cuda" else 0}
    write_json(outdir / "result.json", result)
    (outdir / "run_complete").touch()


if __name__ == "__main__":
    main()
