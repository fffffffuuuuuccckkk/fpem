#!/usr/bin/env python3
"""Paired, from-scratch A0 comparison for history-only calendar tokens.

This runner never writes to the source FPEM results tree.  It reuses the
existing A0 model configuration and the exact forecasting loss/Adam protocol,
adds a VAL-selected checkpoint for each fresh arm, and evaluates the old A0
checkpoint read-only for context.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class PeriodicDataset(Dataset):
    def __init__(self, base, calendar):
        self.base = base
        self.calendar = np.asarray(calendar, dtype=np.float32)
        if self.calendar.ndim != 2 or len(self.calendar) != len(base.data_x):
            raise ValueError("calendar grid must align one-to-one with data_x")
        if not np.isfinite(self.calendar).all():
            raise ValueError("calendar grid contains NaN or Inf")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        x, y, _, _ = self.base[index]
        marks = self.calendar[index:index + self.base.seq_len]
        if len(marks) != self.base.seq_len:
            raise IndexError("incomplete history calendar window")
        return x, y, marks


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_staged_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_project(repo_root, token_stage_root):
    """Use staged token code without overwriting the original repository."""
    sys.path.insert(0, str(repo_root))
    importlib.import_module("models.predictive_env")
    staged = token_stage_root / "models"
    for name in ("exogenous_token_fusion", "modern_tcn_backbone", "periodic_exog"):
        load_staged_module(
            f"models.predictive_env.{name}", staged / "predictive_env" / f"{name}.py"
        )
    model_module = load_staged_module(
        "models.PatchTST_PredictiveEnvIV", staged / "PatchTST_PredictiveEnvIV.py"
    )
    legacy = load_staged_module(
        "_fpem_original_a0_protocol", repo_root / "tools" / "run_predictive_env_iv_patchtst.py"
    )
    periodic = sys.modules["models.predictive_env.periodic_exog"]
    return model_module.Model, legacy, periodic.periodic_features_for_dataset


def resolved(repo_root, value):
    path = Path(value)
    return (path if path.is_absolute() else repo_root / path).resolve()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, args, shuffle=False):
    generator = torch.Generator().manual_seed(args.seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=args.num_workers,
        drop_last=False,
    )


def make_datasets(args, legacy, periodic_fn, periodic_mode):
    classes = {
        "ett_hour": legacy.Dataset_ETT_hour,
        "ett_minute": legacy.Dataset_ETT_minute,
        "custom": legacy.Dataset_Custom,
    }
    dataset_class = classes[args.data_class]
    common = dict(
        args=SimpleNamespace(augmentation_ratio=0),
        root_path=args.root_path,
        size=[args.seq_len, min(48, args.seq_len // 2), args.pred_len],
        features="M",
        data_path=args.data_path,
        target=args.target,
        scale=True,
        timeenc=1,
        freq=args.freq,
    )
    result = {}
    for split in ("train", "val", "test"):
        base = dataset_class(flag=split, **common)
        calendar = periodic_fn(base, args.dataset_name, split, mode=periodic_mode)
        result[split] = PeriodicDataset(base, calendar)
    actual_channels = int(result["train"].base.data_x.shape[-1])
    if actual_channels != args.enc_in:
        raise ValueError(f"enc_in mismatch: config={args.enc_in}, data={actual_channels}")
    return result


def prediction(model, x, marks, use_tokens):
    # This is the same A0 forward path and prediction head as original warmup.
    return model.forward_components(
        x, exog=marks if use_tokens else None
    )["prediction"]


def epoch_metrics(model, loader, args, device, use_tokens, optimizer=None,
                  max_batches=0):
    training = optimizer is not None
    model.train(training)
    square_sum = absolute_sum = 0.0
    count = batches = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for x, y, marks in loader:
            if max_batches and batches >= max_batches:
                break
            x = x.float().to(device)
            y = y[:, -args.pred_len:].float().to(device)
            marks = marks.float().to(device)
            forecast = prediction(model, x, marks, use_tokens)
            if forecast.shape != y.shape:
                raise RuntimeError(f"prediction shape {forecast.shape} != target {y.shape}")
            error = forecast - y
            loss = error.square().mean()
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            square_sum += float(error.detach().square().sum())
            absolute_sum += float(error.detach().abs().sum())
            count += error.numel()
            batches += 1
    if count == 0:
        raise RuntimeError("empty split or max_batches=0 produced no samples")
    return {"mse": square_sum / count, "mae": absolute_sum / count,
            "batches": batches, "values": count}


def state_on_cpu(model):
    return {key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()}


def save_checkpoint(path, model, args, arm, epoch, val):
    torch.save({
        "arm": arm,
        "selected_epoch": epoch,
        "validation": val,
        "model_config": vars(args),
        "state_dict": state_on_cpu(model),
    }, path)


def train_arm(arm, model, datasets, args, device, arm_dir, max_train_batches,
              max_eval_batches):
    arm_dir.mkdir(parents=True, exist_ok=False)
    use_tokens = arm == "token"
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = make_loader(datasets["train"], args, shuffle=True)
    val_loader = make_loader(datasets["val"], args)
    initial_val = epoch_metrics(model, val_loader, args, device, use_tokens,
                                max_batches=max_eval_batches)
    history = []
    best_mse = float("inf")
    best_epoch = None
    checkpoint = arm_dir / "best_val.pt"
    for epoch in range(1, args.warmup_epochs + 1):
        started = time.monotonic()
        train = epoch_metrics(model, train_loader, args, device, use_tokens,
                              optimizer, max_train_batches)
        val = epoch_metrics(model, val_loader, args, device, use_tokens,
                            max_batches=max_eval_batches)
        entry = {"epoch": epoch, "train": train, "val": val,
                 "seconds": time.monotonic() - started}
        history.append(entry)
        print(json.dumps({"arm": arm, **entry}), flush=True)
        if val["mse"] < best_mse:
            best_mse, best_epoch = val["mse"], epoch
            save_checkpoint(checkpoint, model, args, arm, epoch, val)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device)
    test = epoch_metrics(model, make_loader(datasets["test"], args), args,
                         device, use_tokens, max_batches=max_eval_batches)
    result = {"selected_epoch": best_epoch, "initial_val": initial_val,
              "val": payload["validation"],
              "test": test, "history": history, "checkpoint": str(checkpoint)}
    (arm_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def old_a0_metrics(Model, legacy, args, source_args, checkpoint, datasets, device,
                   max_eval_batches):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected = legacy.reference_signature(source_args)
    if payload.get("signature") != expected:
        raise ValueError("old A0 checkpoint signature differs from base run config")
    cfg = legacy.model_config(args, "A0")
    cfg.exog_token_enabled = False
    old = Model(cfg)
    legacy.load_shared_reference_state(old, payload["state_dict"])
    old.to(device)
    return {
        "checkpoint": str(checkpoint), "sha256": sha256(checkpoint),
        "val": epoch_metrics(old, make_loader(datasets["val"], args), args,
                             device, False, max_batches=max_eval_batches),
        "test": epoch_metrics(old, make_loader(datasets["test"], args), args,
                              device, False, max_batches=max_eval_batches),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--base-run-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--token-stage-root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--old-a0-checkpoint", type=Path)
    parser.add_argument("--skip-old-a0", action="store_true")
    parser.add_argument("--periodic-mode", choices=("day", "week", "both", "timeF"),
                        default="both")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index; -1 explicitly selects CPU")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Defaults to base warmup_epochs (normally 3)")
    return parser.parse_args()


def main():
    cli = parse_args()
    repo_root = cli.repo_root.resolve()
    stage_root = cli.token_stage_root.resolve()
    config_path = resolved(repo_root, cli.base_run_config)
    output_dir = cli.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory must be new: {output_dir}")
    original_results = repo_root / "results"
    if output_dir == repo_root or output_dir == original_results:
        raise ValueError("output-dir must be a new experiment directory")
    if original_results in output_dir.parents:
        relative = output_dir.relative_to(original_results)
        if not relative.parts[0].startswith("fpem_periodic_token_a0_"):
            raise ValueError("inside results, use a new fpem_periodic_token_a0_* root")
    Model, legacy, periodic_fn = load_project(repo_root, stage_root)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    # Saved A0 configurations can predate newly added unrelated FPEM flags.
    # Rehydrate current parser defaults first, as the original runner does.
    original_argv = sys.argv
    try:
        sys.argv = [str(repo_root / "tools" / "run_predictive_env_iv_patchtst.py")]
        defaults = legacy.parse_args()
    finally:
        sys.argv = original_argv
    for key, value in raw.items():
        setattr(defaults, key, value)
    source_args = SimpleNamespace(**vars(defaults))
    args = SimpleNamespace(**vars(defaults))
    if args.backbone not in ("patchtst", "moderntcn", "itransformer"):
        raise ValueError("token A0 supports patchtst, moderntcn, itransformer")
    if cli.epochs is not None:
        args.warmup_epochs = cli.epochs
    if cli.num_workers is not None:
        args.num_workers = cli.num_workers
    if args.warmup_epochs <= 0 or cli.max_train_batches < 0 or cli.max_eval_batches < 0:
        raise ValueError("epochs must be positive and batch limits nonnegative")
    args.root_path = str(resolved(repo_root, args.root_path))
    csv_path = Path(args.root_path) / args.data_path
    if not csv_path.is_file():
        raise FileNotFoundError(f"local dated CSV is required: {csv_path}")
    old_path = (resolved(repo_root, cli.old_a0_checkpoint) if cli.old_a0_checkpoint
                else resolved(repo_root, args.reference_checkpoint))
    if not cli.skip_old_a0 and not old_path.is_file():
        raise FileNotFoundError(f"old A0 checkpoint not found: {old_path}")
    gpu = args.gpu if cli.gpu is None else cli.gpu
    if gpu >= 0 and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; pass --gpu -1 for explicit CPU")
    device = torch.device(f"cuda:{gpu}" if gpu >= 0 else "cpu")
    datasets = make_datasets(args, legacy, periodic_fn, cli.periodic_mode)
    seed_all(args.seed)
    cfg_baseline = legacy.model_config(args, "A0")
    cfg_baseline.exog_token_enabled = False
    baseline = Model(cfg_baseline)
    seed_all(args.seed)
    cfg_token = legacy.model_config(args, "A0")
    cfg_token.exog_token_enabled = True
    token = Model(cfg_token)
    # Equalize every shared parameter and buffer, including the prediction head.
    common_state = baseline.state_dict()
    incompatible = token.load_state_dict(common_state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"unexpected baseline keys: {incompatible.unexpected_keys}")
    missing = incompatible.missing_keys
    allowed = ("exog_token_fusion.",
               "modern_tcn_backbone.exog_fusion.")
    if any(not key.startswith(allowed) for key in missing):
        raise RuntimeError(f"unexpected token-only keys: {missing}")
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "source_config": str(config_path), "source_config_sha256": sha256(config_path),
        "csv": str(csv_path), "csv_sha256": sha256(csv_path),
        "backbone": args.backbone, "dataset": args.dataset_name,
        "seq_len": args.seq_len, "pred_len": args.pred_len,
        "periodic_mode": cli.periodic_mode, "calendar_features":
            int(datasets["train"].calendar.shape[-1]),
        "seed": args.seed, "epochs": args.warmup_epochs, "lr": args.lr,
        "batch_size": args.batch_size, "num_workers": args.num_workers,
        "max_train_batches": cli.max_train_batches,
        "max_eval_batches": cli.max_eval_batches,
        "device": str(device), "old_a0_checkpoint": str(old_path) if not cli.skip_old_a0 else None,
    }
    (output_dir / "protocol.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    results = {"protocol": metadata}
    if not cli.skip_old_a0:
        results["old_a0"] = old_a0_metrics(
            Model, legacy, args, source_args, old_path, datasets, device,
            cli.max_eval_batches
        )
        print(json.dumps({"arm": "old_a0", **results["old_a0"]}), flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    arms = {"baseline": baseline, "token": token}
    del baseline, token
    for arm in ("baseline", "token"):
        model = arms.pop(arm)
        seed_all(args.seed)
        model.to(device)
        results[arm] = train_arm(
            arm, model, datasets, args, device, output_dir / arm,
            cli.max_train_batches, cli.max_eval_batches
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    results["delta_token_minus_baseline_test_mse"] = (
        results["token"]["test"]["mse"] - results["baseline"]["test"]["mse"]
    )
    (output_dir / "comparison.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    (output_dir / "complete").write_text("ok\n", encoding="utf-8")
    print(json.dumps({"comparison": str(output_dir / "comparison.json"),
                      "delta_test_mse": results["delta_token_minus_baseline_test_mse"]}),
          flush=True)


if __name__ == "__main__":
    main()
