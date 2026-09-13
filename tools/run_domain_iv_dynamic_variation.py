#!/usr/bin/env python
"""Run V0--V5 with epoch-wise TRAIN-only dynamic environments."""
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
from models.PatchTST_DomainIV_Dynamic import Model, VARIATIONS
from models.domain_iv.dynamic_environment import DynamicEnvironmentDiscovery
from models.domain_iv.losses import contrastive_constraint, representation_diagnostics


class IndexedDataset(Dataset):
    def __init__(self, dataset): self.dataset = dataset
    def __len__(self): return len(self.dataset)
    def __getitem__(self, index):
        x, y, _, _ = self.dataset[index]
        return x, y, index


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def make_dataset(args, split):
    return Dataset_ETT_hour(
        SimpleNamespace(augmentation_ratio=0), args.root_path, flag=split,
        size=[args.seq_len, args.label_len, args.pred_len], features="M",
        data_path="ETTh1.csv", target="OT", scale=True, timeenc=1, freq="h",
    )


def prepare_data(args):
    datasets = {s: make_dataset(args, s) for s in ("train", "val", "test")}
    train_update = DataLoader(IndexedDataset(datasets["train"]), batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, drop_last=False)
    loaders = {
        s: DataLoader(IndexedDataset(ds), batch_size=args.batch_size,
                      shuffle=(s == "train"), num_workers=args.num_workers, drop_last=False)
        for s, ds in datasets.items()
    }
    return datasets, loaders, train_update


def config(args, experiment):
    return SimpleNamespace(
        seq_len=args.seq_len, pred_len=args.pred_len, d_model=args.d_model,
        n_heads=args.n_heads, e_layers=args.e_layers, d_ff=args.d_ff,
        dropout=args.dropout, factor=3, activation="gelu", patch_len=args.patch_len,
        domain_iv_stride=args.stride, domain_iv_variation=experiment,
        domain_iv_bottleneck=args.bottleneck, domain_iv_rank=args.rank,
    )


@torch.no_grad()
def extract_environment_representations(model, loader, device, max_batches=0):
    """Chronological representation extraction; this accepts TRAIN loader only."""
    model.eval(); pieces, indices = [], []
    for step, (x, _, index) in enumerate(loader):
        if max_batches and step >= max_batches: break
        pieces.append(model.environment_representation(x.float().to(device)).cpu())
        indices.append(index)
    representation, index = torch.cat(pieces), torch.cat(indices)
    if not torch.equal(index, torch.arange(len(index))):
        raise RuntimeError("dynamic environment update must use chronological TRAIN indices")
    return representation.numpy()


def train_epoch(model, loader, q_train, optimizer, args, device):
    model.train(); values = []
    for step, (x, y, index) in enumerate(loader):
        if args.max_train_batches and step >= args.max_train_batches: break
        x, y = x.float().to(device), y[:, -args.pred_len:].float().to(device)
        q = torch.as_tensor(q_train[index.numpy()], device=device)
        optimizer.zero_grad(); output = model.forward_components(x)
        forecast = F.mse_loss(output["prediction"], y)
        invariant_forecast = F.mse_loss(output["invariant_prediction"], y)
        domain, _ = contrastive_constraint(
            output["z_inv"], output["z_var"], q,
            temperature=args.temperature, invariant_weight=args.lambda_con_inv,
        )
        loss = forecast + args.lambda_invpred * invariant_forecast + args.lambda_domain * domain
        loss = loss + args.lambda_feature_gate * output["feature_gate"].abs().mean()
        loss = loss + args.lambda_mapping_gate * output["mapping_gate"].abs().mean()
        loss = loss + args.lambda_delta * output["deltaW_to_Winv"].square().mean()
        loss.backward(); optimizer.step(); values.append(float(loss.detach()))
    return float(np.mean(values))


def _q(values): return [float(x) for x in np.quantile(values, [.1, .5, .9])]


@torch.no_grad()
def evaluate(model, loader, environment, args, device):
    model.eval(); records, representations = [], []
    for step, (x, y, _) in enumerate(loader):
        if args.max_eval_batches and step >= args.max_eval_batches: break
        x, y = x.float().to(device), y[:, -args.pred_len:].float().to(device)
        output = model.forward_components(x)
        records.append({
            "full_error": (output["prediction"] - y).cpu(),
            "inv_error": (output["invariant_prediction"] - y).cpu(),
            "gz": output["feature_gate"].reshape(-1).cpu(),
            "gm": output["mapping_gate"].cpu(),
            "feature_ratio": (output["feature_variation"].flatten(1).norm(dim=1) /
                              output["z_inv_tokens"].flatten(1).norm(dim=1).clamp_min(1e-8)).cpu(),
            "weight_ratio": output["deltaW_to_Winv"].cpu(),
            "z_inv": output["z_inv"].cpu(), "z_var": output["z_var"].cpu(),
        })
        representations.append(model.environment_representation(x).cpu())
    full = torch.cat([r["full_error"] for r in records]); inv = torch.cat([r["inv_error"] for r in records])
    gz = torch.cat([r["gz"] for r in records]).numpy(); gm = torch.cat([r["gm"] for r in records]).numpy()
    feature_ratio = torch.cat([r["feature_ratio"] for r in records]).numpy()
    weight_ratio = torch.cat([r["weight_ratio"] for r in records]).numpy()
    z_inv = torch.cat([r["z_inv"] for r in records]); z_var = torch.cat([r["z_var"] for r in records])
    q_test = torch.as_tensor(environment.assignment(torch.cat(representations).numpy()))
    relation = {k: float(v) for k, v in representation_diagnostics(z_inv, z_var, q_test).items()}
    gz10, gz50, gz90 = _q(gz); gm10, gm50, gm90 = _q(gm)
    result = {
        "MSE": float(full.square().mean()), "MAE": float(full.abs().mean()),
        "inv_only_MSE": float(inv.square().mean()),
        "g_z_mean": float(gz.mean()), "g_z_p10": gz10, "g_z_p50": gz50, "g_z_p90": gz90,
        "g_m_mean": float(gm.mean()), "g_m_p10": gm10, "g_m_p50": gm50, "g_m_p90": gm90,
        "feature_variation_to_Zinv": float(feature_ratio.mean()),
        "deltaW_to_Winv": float(weight_ratio.mean()),
    }
    result.update(relation); return result


def write_summary(path, rows):
    lines = ["experiment MSE MAE inv_MSE g_z g_m feature/Zinv DeltaW/Winv corr_inv_env corr_var_env final_q_change final_envsim_corr"]
    for r in rows:
        lines.append("{experiment} {MSE:.9f} {MAE:.9f} {inv_only_MSE:.9f} {g_z_mean:.6f} {g_m_mean:.6f} {feature_variation_to_Zinv:.6f} {deltaW_to_Winv:.6f} {corr_inv_env:.6f} {corr_var_env:.6f} {final_q_change:.6f} {final_env_similarity_correlation:.6f}".format(**r))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root_path", default="./dataset/ETT-small/"); p.add_argument("--seq_len", type=int, default=96); p.add_argument("--label_len", type=int, default=48); p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--patch_len", type=int, default=16); p.add_argument("--stride", type=int, default=8)
    p.add_argument("--d_model", type=int, default=512); p.add_argument("--d_ff", type=int, default=2048); p.add_argument("--n_heads", type=int, default=2); p.add_argument("--e_layers", type=int, default=1); p.add_argument("--dropout", type=float, default=.1)
    p.add_argument("--bottleneck", type=int, default=64); p.add_argument("--rank", type=int, default=8); p.add_argument("--env_num", type=int, default=6); p.add_argument("--env_temperature", type=float, default=.2); p.add_argument("--env_ema_beta", type=float, default=.9)
    p.add_argument("--experiments", default="V0,V1,V2,V3,V4,V5"); p.add_argument("--epochs", type=int, default=10); p.add_argument("--batch_size", type=int, default=32); p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--temperature", type=float, default=.2); p.add_argument("--lambda_invpred", type=float, default=.5); p.add_argument("--lambda_domain", type=float, default=.1); p.add_argument("--lambda_con_inv", type=float, default=1.0); p.add_argument("--lambda_feature_gate", type=float, default=1e-3); p.add_argument("--lambda_mapping_gate", type=float, default=1e-3); p.add_argument("--lambda_delta", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=2021); p.add_argument("--gpu", type=int, default=0); p.add_argument("--max_train_batches", type=int, default=0); p.add_argument("--max_eval_batches", type=int, default=0); p.add_argument("--max_environment_batches", type=int, default=0)
    p.add_argument("--output", default="results/domain_iv_dynamic_ETTh1_96_96")
    args = p.parse_args(); seed_all(args.seed)
    experiments = [x.strip().upper() for x in args.experiments.split(",") if x.strip()]
    if set(experiments) - set(VARIATIONS): raise ValueError("experiments must be V0...V5")
    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")
    _, loaders, update_loader = prepare_data(args)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    (output / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    rows = []
    for experiment in experiments:
        seed_all(args.seed); model = Model(config(args, experiment)).to(device)
        identity = model.identity_diagnostics(next(iter(loaders["train"]))[0].float().to(device))
        print(experiment, "initial identity", identity, flush=True)
        if max(identity.values()) > 1e-7: raise RuntimeError("variation branches are not zero-initialized")
        environment = DynamicEnvironmentDiscovery(args.env_num, args.env_temperature, args.env_ema_beta, args.seed)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr); history = []
        for epoch in range(args.epochs):
            train_rep = extract_environment_representations(model, update_loader, device, args.max_environment_batches)
            q_train, update = environment.update(train_rep, source_split="train")
            loss = train_epoch(model, loaders["train"], q_train, optimizer, args, device)
            history.append(update)
            print("%s epoch %d/%d loss %.7f q_change=%s envsim_corr=%s TRAIN-only" % (
                experiment, epoch + 1, args.epochs, loss, update["mean_q_change"],
                update["environment_similarity_epoch_correlation"]), flush=True)
        result = evaluate(model, loaders["test"], environment, args, device)
        finite_changes = [x["mean_q_change"] for x in history if np.isfinite(x["mean_q_change"])]
        finite_corr = [x["environment_similarity_epoch_correlation"] for x in history if np.isfinite(x["environment_similarity_epoch_correlation"])]
        result.update({
            "experiment": experiment, "feature_mode": model.feature_mode, "mapping_mode": model.mapping_mode,
            "initial_identity": identity, "dynamic_environment_source": "TRAIN only",
            "environment_updates": history,
            "final_q_change": finite_changes[-1] if finite_changes else float("nan"),
            "final_env_similarity_correlation": finite_corr[-1] if finite_corr else float("nan"),
        })
        exp_dir = output / experiment; exp_dir.mkdir(exist_ok=True)
        (exp_dir / "metrics_and_diagnostics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (exp_dir / "metrics_and_diagnostics.txt").write_text("\n".join("%s: %s" % item for item in result.items()) + "\n", encoding="utf-8")
        (exp_dir / "environment_updates.txt").write_text("\n".join(json.dumps(x) for x in history) + "\n", encoding="utf-8")
        np.save(exp_dir / "final_environment_centers.npy", environment.centers)
        rows.append(result); write_summary(Path("results/domain_iv_dynamic_ablation_summary.txt"), rows)
        print("%s TEST MSE %.9f MAE %.9f gz %.6f gm %.6f" % (experiment, result["MSE"], result["MAE"], result["g_z_mean"], result["g_m_mean"]), flush=True)


if __name__ == "__main__": main()
