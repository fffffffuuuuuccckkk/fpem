#!/usr/bin/env python
"""Independent ProbTS CSDI baseline on the FPEM train/val/test CSV splits.

This is intentionally NOT a ninth shared FPEM head: the reference CSDI
conditions on observed history, not on Zvar or the PatchTST representation.
The copied diffusion network and DDPM sampling equations live in
``models/predictive_env/csdi_probts_layers.py`` and ``csdi_probts.py``.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from models.predictive_env.csdi_probts import ProbTSCSDI
from tools.evaluate_fpem_probabilistic import probabilistic_metrics


DATA = {
    "ETTh1": ("ett_hour", "ETT-small", "ETTh1.csv", 7),
    "ETTh2": ("ett_hour", "ETT-small", "ETTh2.csv", 7),
    "ETTm1": ("ett_minute", "ETT-small", "ETTm1.csv", 7),
    "ETTm2": ("ett_minute", "ETT-small", "ETTm2.csv", 7),
    "ExchangeRate": ("custom", "exchange_rate", "exchange_rate.csv", 8),
    "Weather": ("custom", "weather", "weather.csv", 21),
}
DATA_CLASSES = {"ett_hour": Dataset_ETT_hour,
                "ett_minute": Dataset_ETT_minute, "custom": Dataset_Custom}


def build_loaders(args):
    data_class, folder, file_name, channels = DATA[args.dataset]
    common = dict(
        args=SimpleNamespace(augmentation_ratio=0),
        root_path=str(args.data_root / folder),
        size=[args.seq_len, min(48, args.seq_len // 2), args.pred_len],
        features="M", data_path=file_name, target="OT", scale=True,
        timeenc=1, freq=("10min" if args.dataset == "Weather" else
                         {"ett_hour": "h", "ett_minute": "15min",
                          "custom": "d"}[data_class]),
    )
    datasets = {split: DATA_CLASSES[data_class](flag=split, **common)
                for split in ("train", "val", "test")}
    observed_channels = datasets["train"].data_x.shape[-1]
    if observed_channels != channels:
        raise RuntimeError(f"unexpected feature count: {observed_channels} != {channels}")
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {split: DataLoader(
        datasets[split], batch_size=args.batch_size,
        shuffle=(split == "train"),
        generator=generator if split == "train" else None,
        num_workers=0, drop_last=False,
    ) for split in datasets}
    return loaders, channels


@torch.no_grad()
def evaluate(model, loader, args, device, split, output):
    model.eval()
    totals = {}
    total_windows = 0
    max_windows = (args.max_val_windows if split == "val" else
                   args.max_test_windows)
    for batch_idx, (x, y, _, _) in enumerate(loader):
        if max_windows and total_windows >= max_windows:
            break
        x = x.float().to(device)
        y = y[:, -args.pred_len:].float().to(device)
        if max_windows:
            count = min(x.shape[0], max_windows - total_windows)
            x, y = x[:count], y[:count]
        sample_generator = torch.Generator(device=device)
        sample_generator.manual_seed(args.seed + (19001 if split == "val" else 25007)
                                     + batch_idx * 1009)
        samples = model.sample(x, args.num_samples, args.sample_chunk,
                               sample_generator)
        if not torch.isfinite(samples).all():
            raise FloatingPointError("CSDI generated non-finite samples")
        naive = x[:, -1:, :].expand_as(y)
        score = probabilistic_metrics(samples, y, naive)
        summed = probabilistic_metrics(
            samples.sum(-1, keepdim=True), y.sum(-1, keepdim=True),
            naive.sum(-1, keepdim=True),
        )
        scalars = {key: float(value) for key, value in score.items()
                   if torch.as_tensor(value).ndim == 0}
        scalars["ProbTS_CRPS_Sum"] = float(summed["ProbTS_CRPS"])
        scalars["CRPS_Sum"] = float(summed["CRPS"])
        for key, value in scalars.items():
            totals[key] = totals.get(key, 0.0) + x.shape[0] * value
        if batch_idx == 0:
            np.savez_compressed(
                output / f"{split}_trajectory_examples.npz",
                target=y[:2].cpu().numpy(),
                y_samples=samples[:2, :min(20, args.num_samples)].cpu().numpy(),
                y_mean=samples[:2].mean(1).cpu().numpy(),
            )
        total_windows += x.shape[0]
    if not total_windows:
        raise RuntimeError(f"empty {split} evaluation")
    result = {key: value / total_windows for key, value in totals.items()}
    result.update({
        "dataset": args.dataset, "pred_len": args.pred_len,
        "split": split, "evaluated_windows": total_windows,
        "total_split_windows": len(loader.dataset),
        "num_samples": args.num_samples,
        "model": "ProbTS CSDI independent raw-history baseline",
        "ProbTS_CRPS_definition":
            "quantiles_num=20; mean per-sequence normalized twice-pinball",
        "not_shared_FPEM_encoder": True,
        "naive_inv_MSE_not_FPEM_invariant": True,
    })
    (output / f"{split}_metrics.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATA, required=True)
    parser.add_argument("--pred_len", type=int, required=True)
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--data_root", type=Path,
                        default=ROOT / "dataset/all_datasets")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accumulate_grad_batches", type=int, default=8)
    parser.add_argument("--max_train_batches", type=int, default=800,
                        help="0 uses every TRAIN window each epoch")
    parser.add_argument("--max_val_windows", type=int, default=128)
    parser.add_argument("--max_test_windows", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--sample_chunk", type=int, default=2)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta_start", type=float, default=1e-3)
    parser.add_argument("--beta_end", type=float, default=0.5)
    parser.add_argument("--eval_every", type=int, default=2,
                        help="0 disables validation selection and uses the final epoch")
    parser.add_argument("--max_train_steps_smoke", type=int, default=0)
    args = parser.parse_args()
    if args.pred_len <= 0 or args.seq_len <= 0 or args.num_samples < 2:
        raise ValueError("invalid context/prediction/sample count")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "run_config.json").write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        indent=2,
    ))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    loaders, channels = build_loaders(args)
    model = ProbTSCSDI(
        channels, args.seq_len, args.pred_len,
        hidden_channels=args.hidden_channels, num_steps=args.num_steps,
        n_layers=args.num_layers, num_heads=args.num_heads,
        beta_start=args.beta_start, beta_end=args.beta_end,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    best_score = float("inf")
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss, total_batches = 0.0, 0
        for step, (x, y, _, _) in enumerate(loaders["train"]):
            if args.max_train_batches and step >= args.max_train_batches:
                break
            if args.max_train_steps_smoke and step >= args.max_train_steps_smoke:
                break
            x = x.float().to(device)
            y = y[:, -args.pred_len:].float().to(device)
            loss = model.training_loss(x, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite CSDI diffusion loss")
            (loss / args.accumulate_grad_batches).backward()
            batch_limit = min(
                len(loaders["train"]),
                args.max_train_batches or len(loaders["train"]),
                args.max_train_steps_smoke or len(loaders["train"]),
            )
            if ((step + 1) % args.accumulate_grad_batches == 0 or
                    step + 1 == batch_limit):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach())
            total_batches += 1
        record = {"epoch": epoch + 1,
                  "train_diffusion_loss": total_loss / max(total_batches, 1),
                  "train_batches": total_batches}
        if args.eval_every and ((epoch + 1) % args.eval_every == 0 or
                                epoch + 1 == args.epochs):
            validation = evaluate(model, loaders["val"], args, device,
                                  "val", args.output)
            record["val_ProbTS_CRPS"] = validation["ProbTS_CRPS"]
            if validation["ProbTS_CRPS"] < best_score:
                best_score = validation["ProbTS_CRPS"]
                torch.save({"state_dict": model.state_dict(),
                            "run_config": vars(args), "epoch": epoch + 1},
                           args.output / "best_val.pt")
        history.append(record)
        (args.output / "history.json").write_text(json.dumps(history, indent=2))
        print("EPOCH", record, flush=True)
    if args.eval_every == 0:
        torch.save({"state_dict": model.state_dict(),
                    "run_config": vars(args), "epoch": args.epochs},
                   args.output / "last_epoch.pt")
    checkpoint = (args.output / "best_val.pt" if args.eval_every else
                  args.output / "last_epoch.pt")
    best = torch.load(checkpoint, map_location=device,
                      weights_only=False)
    model.load_state_dict(best["state_dict"])
    result = evaluate(model, loaders["test"], args, device,
                      "test", args.output)
    (args.output / "complete").write_text("complete\n")
    print("COMPLETE", result, flush=True)


if __name__ == "__main__":
    main()
