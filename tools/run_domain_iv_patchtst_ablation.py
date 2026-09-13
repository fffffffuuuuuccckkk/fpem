#!/usr/bin/env python
"""Run leakage-safe D0--D6 DomainIV ablations on PatchTST/ETTh1."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from data_provider.data_loader import Dataset_ETT_hour
from models.PatchTST_DomainIV import Model, EXPERIMENTS
from models.domain_iv import DynamicEnvironmentProvider, domain_constraint_loss, representation_diagnostics


class IndexedDataset(Dataset):
    def __init__(self, dataset): self.dataset = dataset
    def __len__(self): return len(self.dataset)
    def __getitem__(self, index):
        x, y, _, _ = self.dataset[index]
        return x, y, index


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def config(args, experiment):
    return SimpleNamespace(
        pred_len=args.pred_len, seq_len=args.seq_len, enc_in=7,
        d_model=args.d_model, n_heads=args.n_heads, e_layers=args.e_layers,
        d_ff=args.d_ff, dropout=args.dropout, factor=3, activation="gelu",
        patch_len=args.patch_len, domain_iv_stride=args.stride,
        domain_iv_experiment=experiment, domain_iv_bottleneck=args.bottleneck,
        domain_iv_rank=args.rank, domain_iv_env_num=args.env_num,
        domain_constraint=args.domain_constraint,
    )


def make_dataset(args, split):
    dataset_args = SimpleNamespace(augmentation_ratio=0)
    return Dataset_ETT_hour(
        dataset_args, args.root_path, flag=split,
        size=[args.seq_len, args.label_len, args.pred_len], features="M",
        data_path="ETTh1.csv", target="OT", scale=True, timeenc=1, freq="h",
    )


def input_windows(dataset):
    return np.stack([dataset.data_x[i:i + dataset.seq_len] for i in range(len(dataset))]).astype("float32")


def prepare_data(args):
    datasets = {s: make_dataset(args, s) for s in ("train", "val", "test")}
    loaders = {
        s: DataLoader(IndexedDataset(ds), batch_size=args.batch_size, shuffle=(s == "train"),
                      num_workers=args.num_workers, drop_last=False)
        for s, ds in datasets.items()
    }
    provider = DynamicEnvironmentProvider(args.env_num, args.seed)
    train_x = input_windows(datasets["train"])
    environment = provider.load_or_create(args.foil_env_path, train_x, np.empty((len(train_x), 0, 0), dtype="float32"))
    soft = {"train": environment.soft}
    for split in ("val", "test"):
        soft[split] = provider.assign_inputs(args.foil_env_path, input_windows(datasets[split])).soft
    return datasets, loaders, soft


def train_epoch(model, loader, soft_q, optimizer, args, device):
    model.train(); totals = []
    for step, (x, y, index) in enumerate(loader):
        if args.max_train_batches and step >= args.max_train_batches: break
        x = x.float().to(device); y = y[:, -args.pred_len:].float().to(device)
        q = torch.as_tensor(soft_q[index.numpy()], device=device)
        optimizer.zero_grad(); output = model.forward_components(x)
        forecast = F.mse_loss(output["prediction"], y)
        if model.experiment == "D0":
            loss = forecast
        else:
            inv_forecast = F.mse_loss(output["invariant_prediction"], y)
            domain, _ = domain_constraint_loss(
                model.domain_constraint, output["z_inv"], output["z_var"], q,
                output["q_logits"], args.temperature, args.lambda_con_inv,
                (args.lambda_mi_inv, args.lambda_mi_var, args.lambda_sep),
            )
            loss = forecast + args.lambda_invpred * inv_forecast + args.lambda_domain * domain
            loss = loss + args.lambda_gate * output["gate"].abs().mean()
            loss = loss + args.lambda_delta * output["deltaW_to_WU"].square().mean()
        loss.backward(); optimizer.step(); totals.append(float(loss.detach()))
    return float(np.mean(totals))


def _quantiles(values):
    return [float(x) for x in np.quantile(values, [.1, .5, .9])]


@torch.no_grad()
def evaluate(model, loader, soft_q, args, device):
    model.eval(); full_error, inv_error = [], []
    gates, ratios, z_inv, z_var, all_q = [], [], [], [], []
    for step, (x, y, index) in enumerate(loader):
        if args.max_eval_batches and step >= args.max_eval_batches: break
        x = x.float().to(device); y = y[:, -args.pred_len:].float().to(device)
        q = torch.as_tensor(soft_q[index.numpy()], device=device)
        output = model.forward_components(x)
        full_error.append((output["prediction"] - y).cpu())
        inv_error.append((output["invariant_prediction"] - y).cpu())
        gates.append(output["gate"].cpu()); ratios.append(output["deltaW_to_WU"].cpu())
        z_inv.append(output["z_inv"].cpu()); z_var.append(output["z_var"].cpu()); all_q.append(q.cpu())
    full_error, inv_error = torch.cat(full_error), torch.cat(inv_error)
    gates, ratios = torch.cat(gates).numpy(), torch.cat(ratios).numpy()
    z_inv, z_var, all_q = torch.cat(z_inv), torch.cat(z_var), torch.cat(all_q)
    diagnostic = {k: float(v) for k, v in representation_diagnostics(z_inv, z_var, all_q).items()}
    env_similarity = (all_q @ all_q.t())[~torch.eye(len(all_q), dtype=torch.bool)].numpy()
    gp10, gp50, gp90 = _quantiles(gates)
    result = {
        "MSE": float(full_error.square().mean()), "MAE": float(full_error.abs().mean()),
        "full_MSE": float(full_error.square().mean()), "inv_only_MSE": float(inv_error.square().mean()),
        "inv_only_MAE": float(inv_error.abs().mean()),
        "gate_mean": float(gates.mean()), "gate_p10": gp10, "gate_p50": gp50, "gate_p90": gp90,
        "deltaW_to_WU_mean": float(ratios.mean()), "deltaW_to_WU_p90": float(np.quantile(ratios, .9)),
        "env_similarity_p10": float(np.quantile(env_similarity, .1)),
        "env_similarity_p50": float(np.quantile(env_similarity, .5)),
        "env_similarity_p90": float(np.quantile(env_similarity, .9)),
    }
    result.update(diagnostic)
    return result


def write_summary(path, rows):
    header = "experiment constraint mapping MSE MAE inv_only_MSE gate_mean deltaW/WU corr_inv_env corr_var_env HSIC_inv_env HSIC_var_env HSIC_inv_var"
    lines = [header]
    for r in rows:
        lines.append("{experiment} {domain_constraint} {mapping_mode} {MSE:.9f} {MAE:.9f} {inv_only_MSE:.9f} {gate_mean:.6f} {deltaW_to_WU_mean:.6f} {corr_inv_env:.6f} {corr_var_env:.6f} {hsic_inv_env:.9g} {hsic_var_env:.9g} {hsic_inv_var:.9g}".format(**r))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root_path", default="./dataset/ETT-small/")
    p.add_argument("--seq_len", type=int, default=96); p.add_argument("--label_len", type=int, default=48); p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--patch_len", type=int, default=16); p.add_argument("--stride", type=int, default=8)
    p.add_argument("--d_model", type=int, default=512); p.add_argument("--d_ff", type=int, default=2048); p.add_argument("--n_heads", type=int, default=2); p.add_argument("--e_layers", type=int, default=1); p.add_argument("--dropout", type=float, default=.1)
    p.add_argument("--bottleneck", type=int, default=64); p.add_argument("--rank", type=int, default=8); p.add_argument("--env_num", type=int, default=6)
    p.add_argument("--domain_constraint", choices=["none", "contrastive", "mutual_info"], default=None, help="optional single-mode override; D0-D6 otherwise select their declared constraint")
    p.add_argument("--experiments", default="D0,D1,D2,D3,D4,D5,D6")
    p.add_argument("--epochs", type=int, default=10); p.add_argument("--batch_size", type=int, default=32); p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--temperature", type=float, default=.2); p.add_argument("--lambda_invpred", type=float, default=.5); p.add_argument("--lambda_domain", type=float, default=.1); p.add_argument("--lambda_gate", type=float, default=1e-3); p.add_argument("--lambda_delta", type=float, default=1e-4)
    p.add_argument("--lambda_con_inv", type=float, default=1.0); p.add_argument("--lambda_mi_inv", type=float, default=1.0); p.add_argument("--lambda_mi_var", type=float, default=1.0); p.add_argument("--lambda_sep", type=float, default=.1)
    p.add_argument("--foil_env_path", default="foil_env/ETTh1_foil_env_k6.npz"); p.add_argument("--seed", type=int, default=2021); p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--max_train_batches", type=int, default=0); p.add_argument("--max_eval_batches", type=int, default=0)
    p.add_argument("--output", default="results/domain_iv_patchtst_ETTh1_96_96")
    args = p.parse_args(); seed_all(args.seed)
    unknown = set(args.experiments.split(",")) - set(EXPERIMENTS)
    if unknown: raise ValueError("unknown experiments: %s" % sorted(unknown))
    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")
    _, loaders, soft_q = prepare_data(args)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    (output / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    rows = []
    for experiment in args.experiments.split(","):
        seed_all(args.seed); model = Model(config(args, experiment)).to(device)
        initial_difference = model.initial_mapping_difference(next(iter(loaders["train"]))[0].float().to(device))
        print("%s initial prediction difference %.12g" % (experiment, initial_difference), flush=True)
        if initial_difference > 1e-7: raise RuntimeError("dynamic Mapping is not zero-initialized")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(args.epochs):
            loss = train_epoch(model, loaders["train"], soft_q["train"], optimizer, args, device)
            print("%s epoch %d/%d loss %.7f" % (experiment, epoch + 1, args.epochs, loss), flush=True)
        result = evaluate(model, loaders["test"], soft_q["test"], args, device)
        result.update({"experiment": experiment, "domain_constraint": model.domain_constraint,
                       "mapping_mode": model.mapping_mode, "initial_prediction_difference": initial_difference})
        exp_dir = output / experiment; exp_dir.mkdir(exist_ok=True)
        (exp_dir / "metrics_and_diagnostics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (exp_dir / "metrics_and_diagnostics.txt").write_text("\n".join("%s: %s" % item for item in result.items()) + "\n", encoding="utf-8")
        rows.append(result); write_summary(Path("results/domain_iv_patchtst_ablation_summary.txt"), rows)
        print("%s TEST MSE %.9f MAE %.9f inv-only %.9f gate %.6f" % (experiment, result["MSE"], result["MAE"], result["inv_only_MSE"], result["gate_mean"]), flush=True)


if __name__ == "__main__": main()
