#!/usr/bin/env python3
"""Per-case frozen-A0 EIIL3 search with VAL-selected A0-safe direct head.

With --full-train, every sampled TRAIN window fits the router, F03 and detached
R3. --selection-split controls LR/epoch/safe-alpha selection; selecting on
TEST is an explicitly exploratory, non-independent evaluation protocol.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.predictive_env.eiil3_modules import soft_risks_and_full_head_penalty
from models.predictive_env.eiil3_linear_correction import EnvironmentLinearCorrection, correction_design
from tools.run_fpem_eiil3 import SPACES, dump, event_shuffle, file_hash, fit_router, get_data
from tools.run_fpem_eiil3_cosa import fit as fit_variation, predict as predict_variation
from tools.run_fpem_f03_safe import forecast, interpolation, mse_mae, soft_env_risk_variance


def parse_floats(value):
    result = [float(x) for x in value.split(",")]
    if not result or any(not np.isfinite(x) or x <= 0 for x in result):
        raise ValueError("expected positive finite comma-separated values")
    return result


def supervision_for_ablation(teacher, starts, mode, seed):
    """Change only TRAIN teacher q; the same router predicts q at inference."""
    if mode == "predictive":
        return teacher, hashlib.sha256(np.ascontiguousarray(teacher["A"]).tobytes()).hexdigest()
    original = np.asarray(teacher["A"], dtype=np.float32)
    if mode == "shuffled":
        changed = event_shuffle(original, starts, seed + 91).astype(np.float32)
    elif mode == "random":
        unique, inverse = np.unique(starts, return_inverse=True)
        rng = np.random.default_rng(seed + 193)
        changed = rng.dirichlet(np.ones(original.shape[1]), size=len(unique)).astype(np.float32)[inverse]
    elif mode == "uniform":
        changed = np.full_like(original, 1.0 / original.shape[1])
    else:
        raise ValueError(mode)
    if changed.shape != original.shape or not np.isfinite(changed).all():
        raise AssertionError("invalid ablation q")
    if not np.allclose(changed.sum(axis=1), 1., atol=1e-5):
        raise AssertionError("ablation q is not normalized")
    updated = dict(teacher)
    for space in SPACES[:4]:
        updated[space] = changed
    return updated, hashlib.sha256(np.ascontiguousarray(changed).tobytes()).hexdigest()


def score_head_batches(head, record, device, batch=512):
    total, count = 0., 0
    with torch.no_grad():
        for start in range(0, len(record["design"]), batch):
            end = start + batch
            x = torch.from_numpy(record["design"][start:end]).to(device)
            ratio = torch.from_numpy(record["ratio"][start:end]).to(device)
            y = torch.from_numpy(record["y"][start:end]).to(device)
            diff = head(x) * ratio - y
            total += float(diff.square().sum().item())
            count += diff.numel()
    return total / count


def fit_head(cli, data, q, original, mode, lam, lr, device):
    torch.manual_seed(cli.seed)
    head = torch.nn.Linear(original["weight"].shape[1], original["weight"].shape[0]).to(device)
    head.load_state_dict({k: v.to(device) for k, v in original.items()})
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    best, chosen, best_epoch, trace = float("inf"), None, 0, []
    train = data["A"]
    select_name = cli.selection_split if cli.full_train else "B1"
    select = data[select_name]
    for epoch in range(cli.epochs + 1):
        head.eval()
        valid = score_head_batches(head, select, device)
        trace.append({"epoch": epoch, "selection_mse": valid})
        if valid < best:
            best, best_epoch = valid, epoch
            chosen = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        if epoch == cli.epochs:
            break
        head.train()
        for index in torch.randperm(len(train["design"])).split(cli.batch_size):
            ids = index.numpy()
            x = torch.from_numpy(train["design"][ids]).to(device)
            y = torch.from_numpy(train["y"][ids]).to(device)
            ratio = torch.from_numpy(train["ratio"][ids]).to(device)
            prediction = head(x) * ratio
            mse = (prediction - y).square().mean()
            if mode == "mse_only":
                loss = mse
            else:
                weights = torch.from_numpy(q["A"][ids]).to(device)
                _, penalty, _, _ = soft_risks_and_full_head_penalty(head, x, y, ratio, weights)
                loss = mse + lam * penalty
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite F03 loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return chosen, {"mode": mode, "lambda_inv": lam, "lr": lr,
                    "selection_mse": best, "selection_split": select_name,
                    "epoch": best_epoch, "trace": trace}


def val_alpha_and_mse(record, original_forecast, state):
    # Exact streaming sufficient statistics for a frozen linear head.
    numerator, denominator, e2, n = 0., 0., 0., 0
    weight = state["weight"].numpy()
    bias = state["bias"].numpy()
    for start in range(0, len(record["design"]), 256):
        end = start + 256
        y1 = (record["design"][start:end] @ weight.T + bias) * record["ratio"][start:end]
        y0 = original_forecast[start:end]
        target = record["y"][start:end]
        d = y1.astype(np.float64) - y0.astype(np.float64)
        e = y0.astype(np.float64) - target.astype(np.float64)
        numerator += float(np.sum(e * d))
        denominator += float(np.sum(d * d))
        e2 += float(np.sum(e * e))
        n += e.size
    alpha = float(np.clip(-numerator / denominator, 0, 1)) if denominator > 1e-20 else 0.
    mse = (e2 + 2 * alpha * numerator + alpha * alpha * denominator) / n
    return alpha, float(mse)


def fit_r3(cli, data, q, stable, label, device):
    if cli.full_train:
        features = {space: correction_design(stable[space], data[space]["key"])
                    for space in ("A", "val", "test")}
        features.update({space: features["A"] for space in ("B1", "B2", "C")})
    else:
        features = {space: correction_design(stable[space], data[space]["key"]) for space in SPACES}
    best, trials = None, []
    fit_split = "A" if cli.full_train else "B1"
    select_split = cli.selection_split if cli.full_train else "B2"
    anchor = mse_mae(stable[select_split], data[select_split]["y"])["mse"]
    for lr in parse_floats(cli.variation_lrs):
        model, info = fit_variation(features[fit_split], stable[fit_split], data[fit_split]["y"], q[fit_split],
                                    features[select_split], stable[select_split], data[select_split]["y"], q[select_split],
                                    lr, cli.epochs, cli.batch_size, cli.seed, device)
        trials.append(info)
        if best is None or info["B2_mse"] < best[0]:
            best = (info["B2_mse"], copy.deepcopy(model.cpu().state_dict()), info)
        if cli.stop_on_gain and best[0] < anchor - 1e-6:
            break
    path = cli.output / f"{label}_r3.pt"
    torch.save(best[1], path)
    model = EnvironmentLinearCorrection(features["B1"].shape[1], data["B1"]["y"].shape[1]).to(device)
    model.load_state_dict(best[1])
    scores = {}
    for space in ("val", "test") if cli.full_train else ("B2", "val", "test"):
        correction, _ = predict_variation(model, features[space], q[space], device)
        scores[space] = mse_mae(stable[space] + correction, data[space]["y"])
    return {"selected": best[2], "selection_split": select_split,
            "fit_split": fit_split, "trials": trials, "scores": scores,
            "checkpoint_sha256": file_hash(path)}


def run(cli, *, data_loader=get_data, asset_override=None):
    if asset_override is None:
        manifest = {r["case_key"]: r for r in (json.loads(line) for line in cli.manifest.read_text().splitlines())}
        if cli.case_key not in manifest:
            raise KeyError(cli.case_key)
        cli.asset = manifest[cli.case_key]
    else:
        cli.asset = asset_override
    if cli.asset["status"] == "MISSING_A0":
        raise FileNotFoundError(f"exact A0 missing: {cli.case_key}")
    cli.output = cli.output_root / cli.case_key
    cli.output.mkdir(parents=True, exist_ok=True)
    if cli.resume and (cli.output / "complete").exists():
        prior = json.loads((cli.output / "metrics.json").read_text())
        if (prior["A0_sha256"] == cli.asset["A0_sha256"]
                and prior.get("environment_supervision", "predictive") == cli.environment_supervision
                and prior.get("selection_split", "val") == cli.selection_split
                and prior.get("train_protocol") == ("full_train_no_A_B1_B2_C_split" if cli.full_train else "purged_four_blocks")):
            print(json.dumps({"case": cli.case_key, "status": "REUSED"}), flush=True)
            return
    # get_data's ephemeral tensors are deliberately not saved as a large NPZ.
    cli.pilot_search = False
    cli.persist_features = False
    cli.generate_missing_q = True
    cli.eiil_steps = 50
    cli.ridge_alpha = 100.
    cli.lr_i3 = parse_floats(cli.head_lrs)[0]
    data, meta, raw_head, teacher = data_loader(cli)
    teacher, supervision_sha = supervision_for_ablation(
        teacher, data["A"]["start"], cli.environment_supervision, cli.seed)
    q, router = fit_router(cli, data, teacher, meta)
    original = {"weight": torch.from_numpy(raw_head[0]), "bias": torch.from_numpy(raw_head[1])}
    device = torch.device(f"cuda:{cli.gpu}" if torch.cuda.is_available() else "cpu")
    candidate_grid = [("mse_only", 0.)] + [("mse_first_irm", lam) for lam in parse_floats(cli.lambda_values)]
    best = None
    trials = []
    selection_split = cli.selection_split if cli.full_train else "val"
    a0_selected = mse_mae(data[selection_split]["a0"], data[selection_split]["y"])["mse"]
    for mode, lam in candidate_grid:
        for lr in parse_floats(cli.head_lrs):
            state, info = fit_head(cli, data, q, original, mode, lam, lr, device)
            alpha, selected_mse = val_alpha_and_mse(data[selection_split],
                                                     data[selection_split]["a0"], state)
            entry = {**info, "alpha": alpha, "selection_mse": selected_mse,
                     "selection_split": selection_split, "A0_selection_mse": a0_selected,
                     "beats_A0_selection": selected_mse < a0_selected - 1e-6}
            trials.append(entry)
            dump(cli.output / "head_search.json", {"trials": trials,
                  "selection": f"{selection_split} epoch, alpha, LR and invariant coefficient; "
                               + ("TEST-selected exploratory protocol" if selection_split == "test" else "TEST report-only")
                               if cli.full_train else
                               "TRAIN B1 epoch; VAL alpha, LR and invariant coefficient; TEST report-only"})
            if best is None or selected_mse < best[0] - 1e-8:
                best = (selected_mse, alpha, state, entry)
    _, alpha, trained, chosen = best
    safe_state = interpolation(original, trained, alpha)
    torch.save(safe_state, cli.output / "selected_safe_head.pt")
    if cli.full_train:
        stable = {space: forecast(data[space], safe_state) for space in ("A", "val", "test")}
        stable.update({space: stable["A"] for space in ("B1", "B2", "C")})
    else:
        stable = {space: forecast(data[space], safe_state) for space in SPACES}
    a0 = {space: data[space]["a0"] for space in SPACES}
    selected_scores = {space: mse_mae(stable[space], data[space]["y"]) for space in SPACES}
    a0_scores = {space: mse_mae(a0[space], data[space]["y"]) for space in SPACES}
    if selected_scores[selection_split]["mse"] > a0_selected + 1e-5:
        raise AssertionError(f"safe head violates {selection_split} A0 protection")
    result = {"case": cli.case_key, "status": "head_complete", "A0_sha256": meta["A0_sha256"],
              "q_sha256": meta["q_sha256"], "router_sha256": router["router_sha256"],
              "environment_supervision": cli.environment_supervision,
              "supervision_sha256": supervision_sha,
              "head_sha256": file_hash(cli.output / "selected_safe_head.pt"),
              "head_selected": chosen, "head_trial_count": len(trials),
              "selection": f"full TRAIN fit; {selection_split} selects F03/R3 LR, epochs and safe alpha; "
                           + ("TEST is exploratory, not independent" if selection_split == "test" else "TEST report-only")
                           if cli.full_train else
                           "TRAIN B1 epoch; VAL alpha/head config; TRAIN B2 R3; TEST report-only",
              "train_protocol": "full_train_no_A_B1_B2_C_split" if cli.full_train else "purged_four_blocks",
              "selection_split": selection_split,
              "backbone_frozen": True, "variation_detached": True,
              "A0": a0_scores, "safe_head": selected_scores, "r3": {}}
    dump(cli.output / "metrics.json", result)
    for label, base in (("A0_no_F03", a0), ("Safe_F03", stable)):
        result["r3"][label] = fit_r3(cli, data, q, base, label, device)
        dump(cli.output / "metrics.json", result)
    result["status"] = "complete"
    result["versions"] = {
        "A0": a0_scores,
        "F03": selected_scores,
        "R3": result["r3"]["A0_no_F03"]["scores"],
        "F03+R3": result["r3"]["Safe_F03"]["scores"],
    }
    dump(cli.output / "metrics.json", result)
    dump(cli.output / "run_config.json", {k: str(v) if isinstance(v, Path) else v
                                          for k, v in vars(cli).items() if k != "asset"})
    (cli.output / "complete").write_text("complete\n")
    print(json.dumps({"case": cli.case_key, "status": "complete",
                      "A0": a0_scores["test"], "safe": selected_scores["test"],
                      "safe_r3": result["r3"]["Safe_F03"]["scores"]["test"]}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--case-key", required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--train-stride", type=int, default=8)
    p.add_argument("--val-stride", type=int, default=8)
    p.add_argument("--test-stride", type=int, default=1)
    p.add_argument("--router-epochs", type=int, default=30)
    p.add_argument("--router-lr", type=float, default=.01)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--head-lrs", default="0.00002,0.0001,0.0003,0.001")
    p.add_argument("--lambda-values", default="0.0001,0.01")
    p.add_argument("--variation-lrs", default="0.0001,0.0003,0.001,0.003")
    p.add_argument("--stop-on-gain", action="store_true")
    p.add_argument("--full-train", action="store_true")
    p.add_argument("--selection-split", choices=("val", "test"), default="val")
    p.add_argument("--environment-supervision",
                   choices=("predictive", "shuffled", "random", "uniform"),
                   default="predictive")
    p.add_argument("--resume", action="store_true")
    cli = p.parse_args()
    torch.set_num_threads(4)
    run(cli)


if __name__ == "__main__":
    main()
