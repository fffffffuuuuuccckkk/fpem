"""Matched F03 continuation: frozen encoder versus low-LR encoder fine-tune.

Both arms start from the exact same VAL-selected F03 head and A0 backbone.
The extra epochs, batches, head LR and random order are identical. The only
experimental variable is whether encoder/embedding parameters receive Adam
updates at 2e-5. TEST is evaluated once after VAL checkpoint selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.predictive_env.direct_invariant_forecaster import DirectInvariantForecaster
from tools.run_fpem_shared_factorization_g1r3 import load_experiment
from tools.run_predictive_env_iv_patchtst import IndexedDataset


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--base_run_config", required=True)
    p.add_argument("--shared_selection", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=("head_only", "joint"), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--head_lr_multiplier", type=float, default=0.1)
    p.add_argument("--backbone_lr", type=float, default=2e-5)
    p.add_argument("--train_stride", type=int, default=16)
    p.add_argument("--val_stride", type=int, default=4)
    p.add_argument("--test_stride", type=int, default=1)
    p.add_argument("--allow_legacy_reference_signature", action="store_true")
    return p.parse_args()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_loader(dataset, args, stride, shuffle, seed):
    indexed = IndexedDataset(dataset, cycle_len=args.cyclenet_cycle_len)
    subset = Subset(indexed, range(0, len(dataset), stride))
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(subset, batch_size=args.batch_size, shuffle=shuffle,
                      generator=generator, num_workers=0)


def predict(model, head, loader, device):
    model.eval()
    head.eval()
    squares, absolutes, count = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y, _, cycle in loader:
            x, cycle = x.float().to(device), cycle.to(device)
            hidden, mean, std = model.encode(x, cycle)
            prediction = head(hidden).transpose(1, 2) * std + mean
            error = prediction - y[:, -model.pred_len:].float().to(device)
            squares += error.square().sum().item()
            absolutes += error.abs().sum().item()
            count += error.numel()
    return {"mse": squares / count, "mae": absolutes / count, "elements": count}


def main():
    cli = arguments()
    out = Path(cli.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete").is_file():
        print("SKIP complete", out, flush=True)
        return
    selected = json.loads(Path(cli.shared_selection).read_text(encoding="utf-8"))
    base_cfg = json.loads(Path(cli.base_run_config).read_text(encoding="utf-8"))
    if selected["seed"] != cli.seed:
        raise ValueError("seed mismatch between trial and selected F03 head")
    checkpoint = Path(selected["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    torch.set_num_threads(4)
    load_cli = argparse.Namespace(**vars(cli))
    load_cli.seed = base_cfg["seed"]
    args, model, loaders, device, sha = load_experiment(load_cli)
    if (selected["reference_sha256"] != sha or selected["backbone"] != args.backbone
            or selected["dataset"] != args.dataset_name
            or selected["pred_len"] != args.pred_len):
        raise ValueError("A0 or data identity mismatch")
    if args.backbone != "itransformer":
        raise ValueError("this controlled pilot supports iTransformer only")
    if any(p.requires_grad for p in model.parameters()):
        raise AssertionError("A0 loader must freeze all parameters")
    sample = next(iter(loaders["train"]))
    with torch.no_grad():
        hidden, _, _ = model.encode(sample[0].float().to(device), sample[3].to(device))
    head = DirectInvariantForecaster(hidden.shape[-2], hidden.shape[-1], args.pred_len,
                                    pretrained_head=model.h_forecast_head).to(device)
    head.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False))
    head_lr = float(selected["selected_lr"]) * cli.head_lr_multiplier
    if head_lr <= 0 or cli.backbone_lr <= 0:
        raise ValueError("LR must be positive")
    backbone_modules = (model.inverted_embedding, model.encoder)
    backbone_params = [p for module in backbone_modules for p in module.parameters()]
    if cli.mode == "joint":
        for p in backbone_params:
            p.requires_grad_(True)
    if cli.mode == "head_only" and any(p.requires_grad for p in backbone_params):
        raise AssertionError("head-only arm accidentally unfreezes backbone")
    optimizer_groups = [{"params": list(head.parameters()), "lr": head_lr}]
    if cli.mode == "joint":
        optimizer_groups.append({"params": backbone_params, "lr": cli.backbone_lr})
    optimizer = torch.optim.Adam(optimizer_groups)
    datasets = {name: loader.dataset.dataset for name, loader in loaders.items()}
    train = make_loader(datasets["train"], args, cli.train_stride, True, cli.seed)
    val = make_loader(datasets["val"], args, cli.val_stride, False, cli.seed)
    test = make_loader(datasets["test"], args, cli.test_stride, False, cli.seed)
    initial_val = predict(model, head, val, device)
    if abs(initial_val["mse"] - selected["val"]["mse"]) > 1e-5:
        raise AssertionError("initial VAL does not match the selected F03 head")
    best = {"epoch": 0, "val": initial_val,
            "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
            "backbone": ({k: v.detach().cpu().clone() for k, v in
                           model.state_dict().items() if k.startswith(("inverted_embedding.", "encoder."))}
                         if cli.mode == "joint" else None)}
    history = [{"epoch": 0, "val": initial_val, "train_mse": None}]
    started = time.monotonic()
    torch.manual_seed(cli.seed)
    for epoch in range(1, cli.epochs + 1):
        # Evaluation mode is intentional in both arms: no dropout changes the
        # original F03 feature distribution while the joint arm still has grads.
        model.eval()
        head.train()
        sum_loss, batches = 0.0, 0
        for x, y, _, cycle in train:
            x, y, cycle = x.float().to(device), y.float().to(device), cycle.to(device)
            hidden, mean, std = model.encode(x, cycle)
            prediction = head(hidden).transpose(1, 2) * std + mean
            loss = (prediction - y[:, -args.pred_len:]).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite continuation loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            if cli.mode == "joint":
                torch.nn.utils.clip_grad_norm_(backbone_params, 1.0)
            optimizer.step()
            sum_loss += loss.item()
            batches += 1
        score = predict(model, head, val, device)
        history.append({"epoch": epoch, "val": score, "train_mse": sum_loss / batches})
        print(cli.mode, args.dataset_name, args.pred_len, cli.seed, epoch,
              score["mse"], flush=True)
        if score["mse"] < best["val"]["mse"]:
            best = {"epoch": epoch, "val": score,
                    "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                    "backbone": ({k: v.detach().cpu().clone() for k, v in
                                   model.state_dict().items() if k.startswith(("inverted_embedding.", "encoder."))}
                                 if cli.mode == "joint" else None)}
    head.load_state_dict(best["head"])
    if best["backbone"] is not None:
        model.load_state_dict(best["backbone"], strict=False)
    final_test = predict(model, head, test, device)
    if best["epoch"] == 0 and abs(final_test["mse"] - selected["test"]["mse"]) > 1e-5:
        raise AssertionError("epoch-0 rollback failed to recover the selected F03 checkpoint")
    result = {"backbone": args.backbone, "dataset": args.dataset_name,
              "pred_len": args.pred_len, "seed": cli.seed, "mode": cli.mode,
              "reference_sha256": sha,
              "shared_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "source_selection": str(cli.shared_selection),
              "head_lr": head_lr, "backbone_lr": cli.backbone_lr if cli.mode == "joint" else 0.0,
              "extra_epochs": cli.epochs, "best_extra_epoch": best["epoch"],
              "initial_val": initial_val, "initial_test": selected["test"],
              "selected_val": best["val"], "selected_test": final_test,
              "history": history, "elapsed_seconds": time.monotonic() - started,
              "trainable_backbone_parameters": sum(p.numel() for p in backbone_params if p.requires_grad),
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "selection_protocol": "same F03 start; 5 extra epochs; VAL selects 0..5; TEST report-only",
              "caveat": "Historical A0 catalog was TEST-selected; resulting TEST is exploratory."}
    torch.save({"head": best["head"], "backbone": best["backbone"],
                "reference_sha256": sha}, out / "model.pt")
    write_json(out / "run_config.json", {"cli": vars(cli), "base_run_config": base_cfg,
                                        "selected_f03": selected})
    write_json(out / "metrics.json", result)
    (out / "complete").write_text("complete\n", encoding="utf-8")
    print("CONTINUATION COMPLETE", cli.mode, args.dataset_name, args.pred_len,
          cli.seed, final_test["mse"], flush=True)


if __name__ == "__main__":
    main()
