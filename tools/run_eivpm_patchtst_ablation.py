#!/usr/bin/env python
"""Leakage-safe E0--E9 PatchTST EIVPM experiment driver."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_provider.data_loader import Dataset_ETT_hour
from models.PatchTST_EIVPM import Model, ABLATIONS
from models.eivpm.environment_decomposer import FoilEnvironmentProvider
from models.eivpm.pattern_bank import _two_component_soft


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        x, y, xm, ym = self.dataset[index]
        return x, y, index


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def model_config(args, ablation):
    return SimpleNamespace(
        task_name="long_term_forecast", seq_len=args.seq_len, pred_len=args.pred_len,
        enc_in=7, d_model=args.d_model, n_heads=args.n_heads, e_layers=args.e_layers,
        d_ff=args.d_ff, dropout=args.dropout, factor=3, activation="gelu",
        patch_len=args.patch_len, eivpm_stride=args.stride, eivpm_ablation=ablation,
        eivpm_pattern_count=args.pattern_count, eivpm_adapter_dim=args.adapter_dim,
        eivpm_mapping_rank=args.mapping_rank,
    )


def dataset_args():
    return SimpleNamespace(augmentation_ratio=0)


def make_dataset(args, split):
    return Dataset_ETT_hour(
        dataset_args(), args.root_path, flag=split,
        size=[args.seq_len, args.label_len, args.pred_len], features="M",
        data_path="ETTh1.csv", target="OT", scale=True, timeenc=1, freq="h",
    )


def window_inputs(dataset):
    x = np.stack([dataset.data_x[i:i + dataset.seq_len] for i in range(len(dataset))]).astype("float32")
    y = np.stack([dataset.data_y[i + dataset.seq_len:i + dataset.seq_len + dataset.pred_len] for i in range(len(dataset))]).astype("float32")
    return x, y


def loaders(args):
    data = {split: make_dataset(args, split) for split in ("train", "val", "test")}
    result = {}
    for split, dataset in data.items():
        result[split] = DataLoader(
            IndexedDataset(dataset), batch_size=args.batch_size,
            shuffle=(split == "train"), num_workers=args.num_workers, drop_last=False,
        )
    return data, result


def build_banks(model, loader, train_env, device, max_batches=0):
    model.eval()
    hidden_batches, env_batches = [], []
    with torch.no_grad():
        for step, (x, _, indices) in enumerate(loader):
            if max_batches and step >= max_batches: break
            hidden, _, _ = model.encode(x.float().to(device))
            hidden_batches.append(hidden.cpu())
            env_batches.append(torch.as_tensor(train_env[indices.numpy()], dtype=torch.long))
    pattern_stats = model.pattern_bank.fit(hidden_batches, env_batches)
    resp_batches = []
    with torch.no_grad():
        for hidden in hidden_batches:
            _, _, resp = model.pattern_bank.query(hidden.to(device))
            resp_batches.append(resp.cpu())
    mapping_stats = model.mapping_bank.fit(resp_batches, env_batches)
    return pattern_stats, mapping_stats


def copy_banks(source, target):
    target.pattern_bank.salience.load_state_dict(source.pattern_bank.salience.state_dict())
    target.pattern_bank.prototypes.copy_(source.pattern_bank.prototypes)
    target.pattern_bank.p_inv.copy_(source.pattern_bank.p_inv)
    target.pattern_bank.env_occurrence = source.pattern_bank.env_occurrence.clone()
    target.pattern_bank.fitted.copy_(source.pattern_bank.fitted)
    target.mapping_bank.p_inv.copy_(source.mapping_bank.p_inv)
    target.mapping_bank.env_transition = source.mapping_bank.env_transition.clone()
    target.mapping_bank.fitted.copy_(source.mapping_bank.fitted)


def train_epoch(model, loader, optimizer, environment, stage, args, device):
    model.train(); losses = []
    _, mapping_mode = ABLATIONS[model.ablation]
    for step, (x, y, indices) in enumerate(loader):
        if args.max_train_batches and step >= args.max_train_batches: break
        x, y = x.float().to(device), y[:, -args.pred_len:].float().to(device)
        env = torch.as_tensor(environment[indices.numpy()], device=device)
        optimizer.zero_grad()
        prediction = model(x)
        sample_loss = (prediction - y).square().mean((1, 2))
        inv_strength = model.last_diagnostics["invariant_mapping_strength_values"].detach()
        if stage == "A" and mapping_mode != "off":
            weight = inv_strength / inv_strength.mean().clamp_min(1e-6)
            forecast = (weight * sample_loss).mean()
        else:
            forecast = sample_loss.mean()
        env_risks = [sample_loss[env == e].mean() for e in env.unique() if int((env == e).sum()) > 0]
        env_cons = torch.stack(env_risks).var(unbiased=False) if len(env_risks) > 1 else forecast.new_zeros(())
        loss = forecast + (args.lambda_env * env_cons if stage == "A" and mapping_mode != "off" else 0.0)
        if stage == "B":
            diag = model.last_diagnostics
            loss = loss + args.lambda_pattern * diag["gP"] + args.lambda_mapping * diag["gM"]
            loss = loss + args.lambda_delta * diag["mapping_update_ratio"].square()
        loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model, loader, args, device, environments=None):
    model.eval(); squared, absolute = [], []
    gp, gm, pcr, mur, dwr = [], [], [], [], []
    env_sq, env_dw = {}, {}
    for step, (x, y, indices) in enumerate(loader):
        if args.max_eval_batches and step >= args.max_eval_batches: break
        x, y = x.float().to(device), y[:, -args.pred_len:].float().to(device)
        pred = model(x); error = pred - y
        per_sample = error.square().mean((1, 2))
        squared.append(error.square().reshape(-1).cpu()); absolute.append(error.abs().reshape(-1).cpu())
        d = model.last_diagnostics
        gp.append(d["pattern_gate_values"].cpu()); gm.append(d["mapping_gate_values"].cpu())
        pcr.append(float(d["pattern_correction_ratio"])); mur.append(float(d["mapping_update_ratio"]))
        dwr.append(d["deltaW_to_WU_values"].cpu())
        if environments is not None:
            batch_env = environments[indices.numpy()]
            for e in np.unique(batch_env):
                mask = torch.as_tensor(batch_env == e, device=device)
                env_sq.setdefault(int(e), []).append(per_sample[mask].cpu())
                env_dw.setdefault(int(e), []).append(d["deltaW_to_WU_values"][mask].cpu())
    gates_p = torch.cat(gp).numpy(); gates_m = torch.cat(gm).numpy()
    result = {
        "MSE": float(torch.cat(squared).mean()), "MAE": float(torch.cat(absolute).mean()),
        "gP_mean": float(gates_p.mean()), "gM_mean": float(gates_m.mean()),
        "gP_p10": float(np.quantile(gates_p, .1)), "gP_p50": float(np.quantile(gates_p, .5)), "gP_p90": float(np.quantile(gates_p, .9)),
        "gM_p10": float(np.quantile(gates_m, .1)), "gM_p50": float(np.quantile(gates_m, .5)), "gM_p90": float(np.quantile(gates_m, .9)),
        "pattern_correction_to_H": float(np.mean(pcr)), "deltaW_effect_ratio": float(np.mean(mur)),
        "deltaW_to_WU": float(torch.cat(dwr).mean()),
    }
    result["environment_MSE"] = {str(e): float(torch.cat(v).mean()) for e, v in env_sq.items()}
    result["environment_deltaW_to_WU"] = {str(e): float(torch.cat(v).mean()) for e, v in env_dw.items()}
    return result


def pattern_score_diagnostics(occurrence, p_inv=None):
    occurrence = np.asarray(occurrence, dtype=np.float64)
    normalized = occurrence / np.clip(occurrence.sum(1, keepdims=True), 1e-8, None)
    env_num = occurrence.shape[1]
    entropy = -(normalized * np.log(np.clip(normalized, 1e-8, None))).sum(1) / np.log(max(env_num, 2))
    coverage = (occurrence / np.clip(occurrence.max(1, keepdims=True), 1e-8, None)).mean(1)
    raw_score = 0.5 * (entropy + coverage)
    raw_std = raw_score.std(ddof=1) if len(raw_score) > 1 else 0.0
    standardized = np.zeros_like(raw_score) if raw_std < 1e-7 else (raw_score - raw_score.mean()) / raw_std
    if p_inv is None:
        mixture_input = torch.as_tensor(raw_score if raw_std < 1e-7 else standardized, dtype=torch.float32)
        p_inv = _two_component_soft(mixture_input).numpy()
    else:
        p_inv = np.asarray(p_inv, dtype=np.float64)
    gap = occurrence.max(1) - occurrence.min(1)
    cv = occurrence.std(1) / np.clip(occurrence.mean(1), 1e-8, None)
    correlation = float("nan") if raw_score.std() == 0 or p_inv.std() == 0 else float(np.corrcoef(raw_score, p_inv)[0, 1])
    return {
        "pattern_p_inv": p_inv.tolist(),
        "pattern_entropy": entropy.tolist(),
        "pattern_coverage": coverage.tolist(),
        "pattern_raw_invariant_score": raw_score.tolist(),
        "pattern_standardized_score": standardized.tolist(),
        "pattern_max_min_gap": gap.tolist(),
        "pattern_cv": cv.tolist(),
        "raw_score_quantiles": np.quantile(raw_score, [0, .1, .5, .9, 1]).tolist(),
        "p_inv_quantiles": np.quantile(p_inv, [0, .1, .5, .9, 1]).tolist(),
        "cv_quantiles": np.quantile(cv, [0, .1, .5, .9, 1]).tolist(),
        "correlation_raw_score_p_inv": correlation,
    }


def bank_diagnostics(model, pattern_stats, mapping_stats):
    pp = model.pattern_bank.p_inv.detach().cpu().numpy(); mp = model.mapping_bank.p_inv.detach().cpu().numpy()
    occurrence = model.pattern_bank.env_occurrence.detach().cpu().numpy()
    transitions = model.mapping_bank.env_transition.detach().cpu().numpy()
    support = mapping_stats["support"].detach().cpu().numpy()
    observed = support > 1e-8
    observed_mp = mp[observed]
    support_mass = support / max(float(support.sum()), 1e-8)
    pattern_quantiles = np.quantile(pp, [0, .1, .5, .9, 1])
    pattern_support = occurrence.mean(1)
    pattern_support_weight = pattern_support / max(float(pattern_support.mean()), 1e-8)
    pattern_mass = float((pp * pattern_support_weight).sum())
    result = {
        "pattern_count": int(len(pp)), "pattern_env_occurrence": occurrence.tolist(),
        "pattern_p_inv_quantiles": pattern_quantiles.tolist(),
        "pattern_p_inv_min": float(pattern_quantiles[0]),
        "pattern_p_inv_p10": float(pattern_quantiles[1]),
        "pattern_p_inv_p50": float(pattern_quantiles[2]),
        "pattern_p_inv_p90": float(pattern_quantiles[3]),
        "pattern_p_inv_max": float(pattern_quantiles[4]),
        "pattern_p_var_quantiles": np.quantile(1 - pp, [0, .1, .5, .9, 1]).tolist(),
        "invariant_pattern_probability_sum": float(pp.sum()),
        "invariant_pattern_effective_mass": pattern_mass,
        "pattern_mass_inv": pattern_mass,
        "variant_pattern_environment_concentration": float((1 - pattern_stats["entropy"]).mean()),
        "mapping_count": int((mapping_stats["support"] > 1e-8).sum()),
        "mapping_env_transitions": transitions.tolist(),
        "mapping_p_inv_quantiles": np.quantile(observed_mp, [0, .1, .5, .9, 1]).tolist(),
        "mapping_p_var_quantiles": np.quantile(1 - observed_mp, [0, .1, .5, .9, 1]).tolist(),
        "invariant_mapping_effective_mass": float((support_mass * mp).sum()),
        "variant_mapping_effective_mass": float((support_mass * (1 - mp)).sum()),
    }
    result.update(pattern_score_diagnostics(occurrence, pp))
    return result


def write_summary(path, rows):
    lines = ["experiment\tMSE\tMAE\tgP\tgM\tpattern_mass_inv\tmapping_mass_inv\tcorrection/H\tdeltaW_effect"]
    for row in rows:
        lines.append("{experiment}\t{MSE:.9f}\t{MAE:.9f}\t{gP_mean:.6f}\t{gM_mean:.6f}\t{invariant_pattern_effective_mass:.4f}\t{invariant_mapping_effective_mass:.4f}\t{pattern_correction_to_H:.6f}\t{deltaW_effect_ratio:.6f}".format(**row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_bank_report(path, diagnostics):
    keys = [
        "pattern_count", "pattern_p_inv_min", "pattern_p_inv_p10",
        "pattern_p_inv_p50", "pattern_p_inv_p90", "pattern_p_inv_max",
        "pattern_mass_inv", "mapping_count", "invariant_mapping_effective_mass",
        "variant_mapping_effective_mass",
    ]
    lines = ["%s: %s" % (key, diagnostics[key]) for key in keys]
    lines.append("pattern-level raw invariant score -> mixture posterior audit:")
    for pattern_id, values in enumerate(diagnostics["pattern_env_occurrence"]):
        lines.append(
            "Pattern %02d | env_occurrence=[%s] | entropy=%.12f | coverage=%.12f | "
            "raw_invariant_score=%.12f | standardized_score=%.9f | p_inv=%.9f | "
            "max_min_gap=%.12g | cv=%.12g" % (
                pattern_id, ", ".join("%.10f" % value for value in values),
                diagnostics["pattern_entropy"][pattern_id],
                diagnostics["pattern_coverage"][pattern_id],
                diagnostics["pattern_raw_invariant_score"][pattern_id],
                diagnostics["pattern_standardized_score"][pattern_id],
                diagnostics["pattern_p_inv"][pattern_id],
                diagnostics["pattern_max_min_gap"][pattern_id],
                diagnostics["pattern_cv"][pattern_id],
            )
        )
    lines.append("raw_score quantiles [min,p10,p50,p90,max]: %s" % diagnostics["raw_score_quantiles"])
    lines.append("p_inv quantiles [min,p10,p50,p90,max]: %s" % diagnostics["p_inv_quantiles"])
    lines.append("cv quantiles [min,p10,p50,p90,max]: %s" % diagnostics["cv_quantiles"])
    lines.append("correlation(raw_score, p_inv): %s" % diagnostics["correlation_raw_score_p_inv"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root_path", default="./dataset/ETT-small/")
    p.add_argument("--seq_len", type=int, default=96); p.add_argument("--label_len", type=int, default=48); p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--patch_len", type=int, default=16); p.add_argument("--stride", type=int, default=8)
    # Match the repository's official PatchTST_ETTh1.sh baseline defaults.
    p.add_argument("--d_model", type=int, default=512); p.add_argument("--d_ff", type=int, default=2048); p.add_argument("--n_heads", type=int, default=2); p.add_argument("--e_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=.1); p.add_argument("--pattern_count", type=int, default=32); p.add_argument("--adapter_dim", type=int, default=32); p.add_argument("--mapping_rank", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=32); p.add_argument("--epochs", type=int, default=10); p.add_argument("--warmup_epochs", type=int, default=3); p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--env_num", type=int, default=6); p.add_argument("--foil_env_path", default="foil_env/ETTh1_foil_env_k6.npz")
    p.add_argument("--experiments", default="E0,E1,E2,E3,E4,E5,E6,E7,E8,E9")
    p.add_argument("--lambda_env", type=float, default=.1); p.add_argument("--lambda_pattern", type=float, default=1e-3); p.add_argument("--lambda_mapping", type=float, default=1e-3); p.add_argument("--lambda_delta", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=2021); p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--max_train_batches", type=int, default=0); p.add_argument("--max_eval_batches", type=int, default=0); p.add_argument("--max_bank_batches", type=int, default=0)
    p.add_argument("--output", default="results/eivpm_patchtst")
    p.add_argument("--diagnostics_json", default="", help="augment an existing bank_diagnostics.json and exit")
    args = p.parse_args(); seed_all(args.seed)
    if args.diagnostics_json:
        json_path = Path(args.diagnostics_json)
        diagnostics = json.loads(json_path.read_text(encoding="utf-8"))
        diagnostics.update(pattern_score_diagnostics(diagnostics["pattern_env_occurrence"]))
        pp = np.asarray(diagnostics["pattern_p_inv"])
        occurrence = np.asarray(diagnostics["pattern_env_occurrence"])
        support_weight = occurrence.mean(1) / max(float(occurrence.mean(1).mean()), 1e-8)
        diagnostics["pattern_p_inv_min"] = float(pp.min())
        diagnostics["pattern_p_inv_p10"] = float(np.quantile(pp, .1))
        diagnostics["pattern_p_inv_p50"] = float(np.quantile(pp, .5))
        diagnostics["pattern_p_inv_p90"] = float(np.quantile(pp, .9))
        diagnostics["pattern_p_inv_max"] = float(pp.max())
        diagnostics["pattern_mass_inv"] = float((pp * support_weight).sum())
        diagnostics["invariant_pattern_effective_mass"] = diagnostics["pattern_mass_inv"]
        json_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        report_path = json_path.with_suffix(".txt")
        write_bank_report(report_path, diagnostics)
        print(report_path.read_text(encoding="utf-8"), flush=True)
        return
    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")
    data, dl = loaders(args)
    train_x, train_y = window_inputs(data["train"])
    provider = FoilEnvironmentProvider(args.env_num, args.seed)
    env = provider.load_or_create(args.foil_env_path, train_x, train_y)
    split_env = {"train": env.labels}
    for split in ("val", "test"):
        split_x, _ = window_inputs(data[split]); split_env[split] = provider.assign_inputs(args.foil_env_path, split_x).labels
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    experiments = [x.strip().upper() for x in args.experiments.split(",") if x.strip()]
    unknown = set(experiments) - set(ABLATIONS)
    if unknown: raise ValueError("Unknown experiments: %s" % sorted(unknown))

    # E0 supplies the identical initialization checkpoint to every ablation.
    seed_all(args.seed); baseline = Model(model_config(args, "E0")).to(device)
    baseline.configure_stage("A")
    identity = baseline.initial_identity_differences(next(iter(dl["train"]))[0].float().to(device))
    print("EIVPM identity sanity:", identity, flush=True)
    optimizer = torch.optim.Adam([p for p in baseline.parameters() if p.requires_grad], lr=args.lr)
    for epoch in range(args.warmup_epochs):
        loss = train_epoch(baseline, dl["train"], optimizer, split_env["train"], "A", args, device)
        print("shared warmup epoch %d/%d loss %.7f" % (epoch + 1, args.warmup_epochs, loss), flush=True)
    baseline_state = {k: v.detach().cpu() for k, v in baseline.state_dict().items()}

    bank_model = Model(model_config(args, "E9")).to(device)
    bank_model.load_state_dict(baseline_state)
    pattern_stats, mapping_stats = build_banks(bank_model, dl["train"], split_env["train"], device, args.max_bank_batches)
    identity = bank_model.initial_identity_differences(next(iter(dl["train"]))[0].float().to(device))
    print("EIVPM post-bank identity sanity:", identity, flush=True)
    common = bank_diagnostics(bank_model, pattern_stats, mapping_stats)
    pattern_prob = bank_model.pattern_bank.p_inv.detach().cpu()
    if torch.allclose(pattern_prob, torch.full_like(pattern_prob, 0.5), atol=1e-6, rtol=0.0):
        raise RuntimeError("degenerate Pattern split: every p_inv equals 0.5")
    if abs(common["pattern_mass_inv"] - args.pattern_count * 0.5) < 1e-6:
        raise RuntimeError("degenerate Pattern mass: K * 0.5")
    (output / "bank_diagnostics.json").write_text(json.dumps(common, indent=2), encoding="utf-8")
    write_bank_report(output / "bank_diagnostics.txt", common)
    print((output / "bank_diagnostics.txt").read_text(encoding="utf-8"), flush=True)
    (output / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    rows = []
    for experiment in experiments:
        seed_all(args.seed); model = Model(model_config(args, experiment)).to(device)
        model.load_state_dict(baseline_state); copy_banks(bank_model, model)
        variant = ABLATIONS[experiment][0] in ("forced", "gated") or ABLATIONS[experiment][1] in ("forced", "gated")
        stage_a_epochs = args.epochs // 2 if variant else args.epochs
        stage_b_epochs = args.epochs - stage_a_epochs if variant else 0
        model.configure_stage("A")
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        for epoch in range(stage_a_epochs):
            loss = train_epoch(model, dl["train"], optimizer, split_env["train"], "A", args, device)
            print("%s Stage-A %d/%d loss %.7f" % (experiment, epoch + 1, stage_a_epochs, loss), flush=True)
        if stage_b_epochs:
            model.configure_stage("B")
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
            for epoch in range(stage_b_epochs):
                loss = train_epoch(model, dl["train"], optimizer, split_env["train"], "B", args, device)
                print("%s Stage-B %d/%d loss %.7f" % (experiment, epoch + 1, stage_b_epochs, loss), flush=True)
        result = evaluate(model, dl["test"], args, device, split_env["test"])
        result.update(common); result.update({"experiment": experiment, "identity": identity, "shared_warmup_epochs": args.warmup_epochs, "stage_A_epochs": stage_a_epochs, "stage_B_epochs": stage_b_epochs})
        exp_dir = output / experiment; exp_dir.mkdir(exist_ok=True)
        (exp_dir / "metrics_and_diagnostics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        readable = "\n".join("%s: %s" % (k, v) for k, v in result.items() if k not in ("pattern_env_occurrence", "mapping_env_transitions")) + "\n"
        (exp_dir / "metrics_and_diagnostics.txt").write_text(readable, encoding="utf-8")
        rows.append(result); write_summary(Path("results/eivpm_patchtst_ablation_summary.txt"), rows)
        print("%s TEST MSE %.9f MAE %.9f" % (experiment, result["MSE"], result["MAE"]), flush=True)
    print("completed", ",".join(experiments), flush=True)


if __name__ == "__main__":
    main()
