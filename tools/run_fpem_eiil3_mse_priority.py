#!/usr/bin/env python3
"""One-case EIIL3 F03 V0-V5 comparison with a frozen A0 and detached R3.

TRAIN A fits the direct head, purged TRAIN B1 selects its LR/epoch; TRAIN B1
fits R3 and purged TRAIN B2 selects R3 LR/epoch. VAL/TEST are report-only.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.predictive_env.eiil3_linear_correction import EnvironmentLinearCorrection, correction_design
from models.predictive_env.eiil3_mse_priority import priority_step
from tools.run_fpem_eiil3 import SPACES, dump, file_hash, fit_router, get_data
from tools.run_fpem_eiil3_cosa import fit as fit_variation, predict as predict_variation
from tools.run_fpem_mechanism_fusion import score


def predict_head(head, data, device):
    head.eval()
    result = {}
    with torch.no_grad():
        for space in SPACES:
            record = data[space]
            chunks = []
            for start in range(0, len(record["design"]), 512):
                x = torch.from_numpy(record["design"][start:start + 512]).to(device)
                scale = torch.from_numpy(record["ratio"][start:start + 512]).to(device)
                chunks.append((head(x) * scale).cpu().numpy())
            result[space] = np.concatenate(chunks).astype(np.float32)
    return result


def fit_head(cli, data, q, initial, mode, rho, lr, device):
    torch.manual_seed(cli.seed)
    weight, bias = initial
    head = torch.nn.Linear(weight.shape[1], weight.shape[0]).to(device)
    with torch.no_grad():
        head.weight.copy_(torch.from_numpy(weight).to(device))
        head.bias.copy_(torch.from_numpy(bias).to(device))
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    a = {key: torch.from_numpy(data["A"][key]).to(device)
         for key in ("design", "y", "ratio")}
    b = {key: torch.from_numpy(data["B1"][key]).to(device)
         for key in ("design", "y", "ratio")}
    q_a = torch.from_numpy(q["A"]).to(device)
    best, chosen_epoch, chosen = float("inf"), -1, None
    trace, diagnostics = [], []
    for epoch in range(cli.epochs + 1):
        head.eval()
        with torch.no_grad():
            valid = float(((head(b["design"]) * b["ratio"] - b["y"]) ** 2).mean())
        trace.append({"epoch": epoch, "B1_mse": valid})
        if valid < best:
            best, chosen_epoch = valid, epoch
            chosen = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
        if epoch == cli.epochs:
            break
        head.train()
        epoch_diag = []
        for index in torch.randperm(len(a["design"]), device=device).split(cli.batch_size):
            epoch_diag.append(priority_step(
                head, optimizer, a["design"][index], a["y"][index],
                a["ratio"][index], q_a[index], mode, cli.lambda_inv, rho))
        diagnostics.append({"epoch": epoch + 1, "batches": len(epoch_diag),
                            "conflict_rate": float(np.mean([x["gradient_conflict"] for x in epoch_diag])),
                            "gradient_cosine": float(np.mean([x["gradient_cosine"] for x in epoch_diag])),
                            "main_norm": float(np.mean([x["main_norm"] for x in epoch_diag])),
                            "aux_norm": float(np.mean([x["aux_norm"] for x in epoch_diag])),
                            "effective_aux_ratio": float(np.mean([x["effective_aux_ratio"] for x in epoch_diag]))})
    head.load_state_dict(chosen)
    head.requires_grad_(False)
    forecasts = predict_head(head, data, device)
    info = {"mode": mode, "rho": rho, "lr": lr, "lambda_inv": cli.lambda_inv,
            "B1_mse": best, "best_epoch": chosen_epoch, "trace": trace,
            "diagnostics": diagnostics,
            "weight_relative_drift": float(np.linalg.norm(chosen["weight"].numpy() - weight)
                                           / max(np.linalg.norm(weight), 1e-8))}
    return forecasts, chosen, info


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
    p.add_argument("--lambda-inv", type=float, default=1e-4)
    p.add_argument("--eiil-steps", type=int, default=50)
    p.add_argument("--head-lrs", default="0.0003,0.0001,0.001,0.00002")
    p.add_argument("--variation-lrs", default="0.0003,0.001,0.0001,0.003")
    p.add_argument("--stop-on-gain", action="store_true")
    p.add_argument("--resume", action="store_true")
    cli = p.parse_args()
    torch.set_num_threads(4)
    cli.pilot_search = False
    cli.persist_features = True
    cli.generate_missing_q = True
    cli.lr_i3 = float(cli.head_lrs.split(",")[0])
    cli.ridge_alpha = 100.
    manifest = {row["case_key"]: row for row in
                (json.loads(line) for line in cli.manifest.read_text().splitlines())}
    cli.asset = manifest[cli.case_key]
    if cli.asset["status"] == "MISSING_A0":
        raise FileNotFoundError(f"exact A0 unavailable: {cli.case_key}")
    cli.output = cli.output_root / cli.case_key
    cli.output.mkdir(parents=True, exist_ok=True)
    if cli.resume and (cli.output / "point_complete").exists():
        prior = json.loads((cli.output / "point_metrics.json").read_text())
        if prior["A0_sha256"] == cli.asset["A0_sha256"]:
            print(json.dumps({"case": cli.case_key, "status": "REUSED"}), flush=True)
            return
    data, meta, initial, teacher = get_data(cli)
    q, router = fit_router(cli, data, teacher, meta)
    device = torch.device(f"cuda:{cli.gpu}" if torch.cuda.is_available() else "cpu")
    variants = [("V0", "V0", None), ("V1", "V1", None), ("V2", "V2", None),
                ("V3", "V3", None)] + [(f"V4_rho_{rho:g}", "V4", rho)
                                           for rho in (.05, .1, .2)] + [("V5", "V5", None)]
    head_lrs = [float(part) for part in cli.head_lrs.split(",")]
    variation_lrs = [float(part) for part in cli.variation_lrs.split(",")]
    if any(lr <= 0 for lr in head_lrs + variation_lrs):
        raise ValueError("learning rates must be positive")
    a0_b1 = score(data["B1"]["a0"], data["B1"]["y"])["mse"]
    result = {"case": cli.case_key, "A0_sha256": cli.asset["A0_sha256"],
              "q_sha256": meta["q_sha256"], "router_sha256": router["router_sha256"],
              "selection": "purged TRAIN B1 head; purged TRAIN B2 variation; TEST report-only",
              "backbone_frozen": True, "variation_inputs_detached": True,
              "A0_init_error_max": meta["init_error_max"],
              "A0": {space: score(data[space]["a0"], data[space]["y"])
                     for space in SPACES}, "variants": {}}
    for name, mode, rho in variants:
        selected_head, head_trials = None, []
        for lr in head_lrs:
            forecast, state, info = fit_head(cli, data, q, initial, mode, rho, lr, device)
            head_trials.append(info)
            if selected_head is None or info["B1_mse"] < selected_head[0]:
                selected_head = (info["B1_mse"], forecast, state, info)
            dump(cli.output / "progress.json", {"completed": list(result["variants"]),
                                                  "running": name, "head_trials": head_trials})
            if cli.stop_on_gain and info["B1_mse"] < a0_b1 - 1e-6:
                break
        _, stable, head_state, head_info = selected_head
        torch.save(head_state, cli.output / f"{name}_head.pt")
        if name == "V0":
            if abs(score(stable["B1"], data["B1"]["y"])["mse"] - head_info["B1_mse"]) > 1e-4:
                raise AssertionError("V0 head score mismatch")
        features = {space: correction_design(stable[space], data[space]["key"])
                    for space in SPACES}
        b1, b2 = data["B1"], data["B2"]
        anchor_b2 = score(stable["B2"], b2["y"])["mse"]
        best_var, var_trials = None, []
        for lr in variation_lrs:
            model, info = fit_variation(
                features["B1"], stable["B1"], b1["y"], q["B1"],
                features["B2"], stable["B2"], b2["y"], q["B2"],
                lr, cli.epochs, cli.batch_size, cli.seed, device)
            var_trials.append(info)
            if best_var is None or info["B2_mse"] < best_var[0]:
                best_var = (info["B2_mse"], copy.deepcopy(model.cpu().state_dict()), info)
            if cli.stop_on_gain and info["B2_mse"] < anchor_b2 - 1e-6:
                break
        _, var_state, var_info = best_var
        torch.save(var_state, cli.output / f"{name}_variation.pt")
        model = EnvironmentLinearCorrection(features["B1"].shape[1],
                                            b1["y"].shape[1]).to(device)
        model.load_state_dict(var_state)
        scores = {}
        for space in SPACES:
            correction, _ = predict_variation(model, features[space], q[space], device)
            scores[space] = {"head": score(stable[space], data[space]["y"]),
                             "final": score(stable[space] + correction, data[space]["y"])}
        result["variants"][name] = {"head_selected": head_info,
                                    "variation_selected": var_info,
                                    "head_candidates": len(head_trials),
                                    "variation_candidates": len(var_trials),
                                    "head_trials": head_trials, "variation_trials": var_trials,
                                    "head_sha256": file_hash(cli.output / f"{name}_head.pt"),
                                    "variation_sha256": file_hash(cli.output / f"{name}_variation.pt"),
                                    "scores": scores}
        dump(cli.output / "point_metrics.json", result)
        print(json.dumps({"case": cli.case_key, "variant": name,
                          "test_a0": result["A0"]["test"],
                          "test_head": scores["test"]["head"],
                          "test_final": scores["test"]["final"]}), flush=True)
    result["status"] = "complete"
    dump(cli.output / "point_metrics.json", result)
    dump(cli.output / "run_config.json", {key: str(value) if isinstance(value, Path) else value
                                          for key, value in vars(cli).items() if key != "asset"})
    (cli.output / "point_complete").write_text("complete\n")


if __name__ == "__main__":
    main()
