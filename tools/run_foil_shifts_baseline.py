#!/usr/bin/env python3
"""Train portable FOIL/ShifTS controls on the same A0 data/backbone protocol.

No FPEM learning-rate search, probability head, or TEST-based selection is used.
The only test access is the final evaluation after training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_provider.data_loader import Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute
from models.PatchTST_PredictiveEnvIV import Model
from models.predictive_env.portable_foil_shifts import (
    FOILAdapter, ShifTSAdapter, backbone_forecast, train_only_aligned_windows,
)
from tools.run_predictive_env_iv_patchtst import IndexedDataset, shared_reference_state


DATA_CLASSES = {"ett_hour": Dataset_ETT_hour,
                "ett_minute": Dataset_ETT_minute, "custom": Dataset_Custom}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=("foil", "shifts"), required=True)
    p.add_argument("--backbone", choices=("patchtst", "itransformer", "moderntcn"), required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--pred-len", type=int, required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--data-class", choices=tuple(DATA_CLASSES), required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--freq", required=True)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--d-ff", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--modern-patch-size", type=int, default=8)
    p.add_argument("--modern-patch-stride", type=int, default=4)
    p.add_argument("--modern-stages", type=int, default=1)
    p.add_argument("--modern-ffn-ratio", type=int, default=1)
    p.add_argument("--modern-head-dropout", type=float, default=0.0)
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--env-epochs", type=int, default=2)
    p.add_argument("--env-num", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--max-eval-batches", type=int, default=0)
    p.add_argument("--reference-root", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def base_config(args, channels):
    return SimpleNamespace(
        seq_len=args.seq_len, pred_len=args.pred_len, d_model=args.d_model,
        d_ff=args.d_ff, n_heads=2, e_layers=1, dropout=args.dropout,
        factor=3, activation="gelu", enc_in=channels,
        predictive_env_backbone=args.backbone, predictive_env_ablation="A0",
        variant_fusion_mode="legacy", predictive_env_num=args.env_num,
        modern_tcn_patch_size=args.modern_patch_size,
        modern_tcn_patch_stride=args.modern_patch_stride,
        modern_tcn_num_stages=args.modern_stages,
        modern_tcn_ffn_ratio=args.modern_ffn_ratio,
        modern_tcn_head_dropout=args.modern_head_dropout,
    )


def make_data(args):
    klass = DATA_CLASSES[args.data_class]
    common = dict(
        args=SimpleNamespace(augmentation_ratio=0),
        root_path=args.data_root,
        size=[args.seq_len, min(48, args.seq_len // 2), args.pred_len],
        features="M", data_path=args.data_path, target="OT",
        scale=True, timeenc=1, freq=args.freq,
    )
    return {split: klass(flag=split, **common)
            for split in ("train", "val", "test")}


def loader(dataset, args, shuffle=False):
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    return DataLoader(
        IndexedDataset(dataset, cycle_len=24),
        batch_size=args.batch_size, shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=args.num_workers, drop_last=False,
    )


def reference_compatible(signature, args, channels):
    expected = {
        "backbone": args.backbone, "dataset_name": args.dataset,
        "data_path": args.data_path, "seq_len": args.seq_len,
        "pred_len": args.pred_len, "d_model": args.d_model,
        "d_ff": args.d_ff, "n_heads": 2, "e_layers": 1,
        "dropout": args.dropout, "seed": args.seed,
        "warmup_epochs": args.warmup_epochs, "enc_in": channels,
    }
    if args.backbone == "moderntcn":
        expected.update({
            "modern_tcn_patch_size": args.modern_patch_size,
            "modern_tcn_patch_stride": args.modern_patch_stride,
            "modern_tcn_num_stages": args.modern_stages,
            "modern_tcn_ffn_ratio": args.modern_ffn_ratio,
            "modern_tcn_head_dropout": args.modern_head_dropout,
        })
    return all(signature.get(key) == value for key, value in expected.items())


def find_reference(args, channels):
    root = Path(args.reference_root)
    patterns = [
        f"**/{args.backbone}/{args.dataset}/pred_{args.pred_len}/shared_reference.pt",
        f"**/{args.backbone}/{args.dataset}/pred_{args.pred_len}/anchor*/shared_reference.pt",
    ]
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if not path.is_file() or path.stat().st_size == 0:
                continue
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                if reference_compatible(payload["signature"], args, channels):
                    return path, payload["state_dict"]
            except (KeyError, OSError, RuntimeError, ValueError,
                    EOFError, pickle.UnpicklingError):
                continue
    return None, None


def fresh_reference(base, train_set, args, device, path):
    # Only used when there is no compatible existing checkpoint.  The same
    # warmup reference is then reused by both portable methods for this case.
    base.train()
    opt = torch.optim.Adam(base.parameters(), lr=args.lr)
    for epoch in range(args.warmup_epochs):
        total = count = 0
        for step, (x, y, _, _) in enumerate(loader(train_set, args, shuffle=True)):
            if args.max_train_batches and step >= args.max_train_batches:
                break
            x, y = x.float().to(device), y.float().to(device)
            opt.zero_grad(set_to_none=True)
            pred = backbone_forecast(base, x)[0]
            loss = F.mse_loss(pred, y[:, -args.pred_len:])
            loss.backward()
            opt.step()
            total += float(loss.detach())
            count += 1
        print(f"reference warmup {epoch + 1}: {total / max(count, 1):.6f}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"signature": {
        "backbone": args.backbone, "dataset_name": args.dataset,
        "data_path": args.data_path, "seq_len": args.seq_len,
        "pred_len": args.pred_len, "d_model": args.d_model,
        "d_ff": args.d_ff, "n_heads": 2, "e_layers": 1,
        "dropout": args.dropout, "seed": args.seed,
        "warmup_epochs": args.warmup_epochs,
        "enc_in": train_set.data_x.shape[-1],
        "modern_tcn_patch_size": args.modern_patch_size,
        "modern_tcn_patch_stride": args.modern_patch_stride,
        "modern_tcn_num_stages": args.modern_stages,
        "modern_tcn_ffn_ratio": args.modern_ffn_ratio,
        "modern_tcn_head_dropout": args.modern_head_dropout,
    }, "state_dict": shared_reference_state(base)}, path)
    return path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def evaluate(model, dataset, args, device):
    model.eval()
    total_sq = total_abs = elements = windows = 0
    for step, (x, y, _, _) in enumerate(loader(dataset, args)):
        if args.max_eval_batches and step >= args.max_eval_batches:
            break
        x, y = x.float().to(device), y.float().to(device)[:, -args.pred_len:]
        pred = model.predict(x)
        if pred.shape != y.shape:
            raise RuntimeError(f"prediction {pred.shape} != label {y.shape}")
        error = pred - y
        total_sq += float(error.square().sum())
        total_abs += float(error.abs().sum())
        elements += error.numel()
        windows += x.shape[0]
    if not elements:
        raise RuntimeError("evaluation had no batches")
    return {"MSE": total_sq / elements, "MAE": total_abs / elements,
            "windows": windows, "elements": elements}


@torch.no_grad()
def infer_foil_train_labels(adapter, dataset, args, device):
    adapter.eval()
    labels = np.zeros(len(dataset), dtype=np.int64)
    for x, y, ids, _ in loader(dataset, args):
        labels[ids.numpy()] = adapter.infer_environment(
            x.float().to(device), y.float().to(device)[:, -args.pred_len:]
        ).cpu().numpy()
    return labels


def run_foil(adapter, data, args, device):
    adapter.backbone.requires_grad_(False)
    head_opt = torch.optim.Adam([adapter.env_heads], lr=args.lr)
    history = []
    for epoch in range(args.env_epochs):
        adapter.train()
        adapter.backbone.eval()
        total = count = 0
        for step, (x, y, _, _) in enumerate(loader(data["train"], args, True)):
            if args.max_train_batches and step >= args.max_train_batches:
                break
            x, y = x.float().to(device), y.float().to(device)[:, -args.pred_len:]
            head_opt.zero_grad(set_to_none=True)
            loss = adapter.environment_head_loss(x, y)
            loss.backward()
            head_opt.step()
            total += float(loss.detach())
            count += 1
        history.append({"phase": "environment_head", "epoch": epoch + 1,
                        "loss": total / max(count, 1)})
        print(history[-1], flush=True)
    labels = infer_foil_train_labels(adapter, data["train"], args, device)
    counts = np.bincount(labels, minlength=args.env_num)
    print("FOIL TRAIN-only environment counts", counts.tolist(), flush=True)
    adapter.backbone.requires_grad_(True)
    adapter.env_heads.requires_grad_(False)
    opt = torch.optim.Adam(adapter.backbone.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        adapter.train()
        total = count = 0
        for step, (x, y, ids, _) in enumerate(loader(data["train"], args, True)):
            if args.max_train_batches and step >= args.max_train_batches:
                break
            x, y = x.float().to(device), y.float().to(device)[:, -args.pred_len:]
            q = torch.as_tensor(labels[ids.numpy()], device=device, dtype=torch.long)
            opt.zero_grad(set_to_none=True)
            loss, parts = adapter.final_loss(x, y, q)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite FOIL final loss")
            loss.backward()
            opt.step()
            total += float(loss.detach())
            count += 1
        history.append({"phase": "final", "epoch": epoch + 1,
                        "loss": total / max(count, 1)})
        print(history[-1], flush=True)
    return history, {"source_split": "TRAIN only", "counts": counts.tolist()}


def run_shifts(adapter, data, args, device):
    opt = torch.optim.Adam(adapter.parameters(), lr=args.lr)
    train_values = torch.as_tensor(data["train"].data_y, device=device, dtype=torch.float32)
    history = []
    for epoch in range(args.epochs):
        adapter.train()
        total = forecast_total = aux_total = count = 0
        for step, (x, y, ids, _) in enumerate(loader(data["train"], args, True)):
            if args.max_train_batches and step >= args.max_train_batches:
                break
            x, y = x.float().to(device), y.float().to(device)[:, -args.pred_len:]
            x_all = train_only_aligned_windows(
                train_values, ids.to(device), args.seq_len, args.pred_len
            )
            opt.zero_grad(set_to_none=True)
            pred, aux = adapter.predict_with_aux(x, x_all)
            forecast_loss = F.mse_loss(pred, y)
            loss = forecast_loss + aux
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite ShifTS loss")
            loss.backward()
            opt.step()
            total += float(loss.detach())
            forecast_total += float(forecast_loss.detach())
            aux_total += float(aux.detach())
            count += 1
        history.append({"phase": "train", "epoch": epoch + 1,
                        "loss": total / max(count, 1),
                        "forecast_loss": forecast_total / max(count, 1),
                        "aux_loss": aux_total / max(count, 1)})
        print(history[-1], flush=True)
    return history, {"auxiliary_source_split": "TRAIN only"}


def main():
    args = parse_args()
    if args.epochs <= 0 or args.warmup_epochs <= 0 or args.lr <= 0:
        raise ValueError("epochs, warmup_epochs and lr must be positive")
    if args.env_num < 2:
        raise ValueError("FOIL requires at least two environments")
    torch.set_num_threads(2)
    seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    data = make_data(args)
    channels = int(data["train"].data_x.shape[-1])
    base = Model(base_config(args, channels)).to(device)
    reference_path, reference_state = find_reference(args, channels)
    if reference_state is None:
        reference_path = output.parent / "shared_reference.pt"
        if reference_path.exists():
            payload = torch.load(reference_path, map_location="cpu", weights_only=False)
            if not reference_compatible(payload["signature"], args, channels):
                raise RuntimeError("local shared reference signature mismatch")
            reference_state = payload["state_dict"]
        else:
            seed_all(args.seed)
            fresh_reference(base, data["train"], args, device, reference_path)
            reference_state = torch.load(reference_path, map_location="cpu", weights_only=False)["state_dict"]
    from tools.run_predictive_env_iv_patchtst import load_shared_reference_state
    load_shared_reference_state(base, reference_state)
    # Source and all training settings are recorded before any optimizer update.
    config = vars(args).copy()
    config.update({"channels": channels, "reference_checkpoint": str(reference_path),
                   "reference_sha256": sha256(reference_path),
                   "train_windows": len(data["train"]), "val_windows": len(data["val"]),
                   "test_windows": len(data["test"]),
                   "protocol": "FPEM A0 split/normalization/seed; fixed backbone LR; no LR search"})
    (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    if args.method == "foil":
        adapter = FOILAdapter(base, args.d_model, args.pred_len, args.env_num).to(device)
        history, diagnostics = run_foil(adapter, data, args, device)
    else:
        adapter = ShifTSAdapter(base, args.seq_len, channels).to(device)
        history, diagnostics = run_shifts(adapter, data, args, device)
    val_metrics = evaluate(adapter, data["val"], args, device)
    test_metrics = evaluate(adapter, data["test"], args, device)
    metrics = {"method": args.method, "backbone": args.backbone,
               "dataset": args.dataset, "pred_len": args.pred_len,
               "val": val_metrics, "test": test_metrics,
               "diagnostics": diagnostics}
    (output / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    torch.save(adapter.state_dict(), output / "checkpoint.pt")
    (output / "run_complete").touch()
    print("FINAL", json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
