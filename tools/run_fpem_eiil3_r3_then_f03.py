#!/usr/bin/env python3
"""Reverse-stage EIIL3 ablation: freeze pure R3, then adapt only the A0 head.

The original R3 checkpoint always consumes [A0 forecast, history] and is never
retrained. The added head starts from the exact A0 head and predicts the full
future from the unchanged frozen Backbone H. Its training target is Y minus the
fixed R3 correction. Alpha=0 recovers pure R3 exactly. This script matches the
source full-TRAIN, TEST-selected exploratory search; TEST is not independent.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.predictive_env.eiil3_linear_correction import (  # noqa: E402
    EnvironmentLinearCorrection, correction_design,
)
from models.predictive_env.eiil3_modules import soft_risks_and_full_head_penalty  # noqa: E402
from models.predictive_env.relation_prototypes import route  # noqa: E402
from tools.run_fpem_eiil3 import dump, file_hash, get_data  # noqa: E402
from tools.run_fpem_eiil3_cosa import (  # noqa: E402
    fit as fit_variation, predict as predict_variation, router_from_checkpoint,
)
from tools.run_fpem_eiil3_r3_input_ablation import correction_from_checkpoint  # noqa: E402
from tools.run_fpem_eiil3_safe_full import (  # noqa: E402
    parse_floats, score_head_batches, val_alpha_and_mse,
)
from tools.run_fpem_f03_safe import forecast, interpolation, mse_mae  # noqa: E402


def corrected_record(record: dict, fixed_correction: np.ndarray) -> dict:
    if fixed_correction.shape != record["y"].shape:
        raise ValueError("fixed R3 correction and target shapes differ")
    return {**record, "y": np.asarray(record["y"] - fixed_correction,
                                       dtype=np.float32)}


def fit_head_after_r3(cli, data, q, original, mode, lam, lr, device):
    """Identical head/Adam/epochs to F03, but Y is the remaining R3 residual."""
    torch.manual_seed(cli.seed)
    head = torch.nn.Linear(original["weight"].shape[1],
                           original["weight"].shape[0]).to(device)
    head.load_state_dict({key: value.to(device) for key, value in original.items()})
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    train, select = data["A"], data[cli.selection_split]
    best = float("inf")
    chosen, best_epoch, trace = None, 0, []
    for epoch in range(cli.epochs + 1):
        head.eval()
        mse = score_head_batches(head, select, device)
        trace.append({"epoch": epoch, "selection_mse": mse})
        if mse < best:
            best, best_epoch = mse, epoch
            chosen = {key: value.detach().cpu().clone()
                      for key, value in head.state_dict().items()}
        if epoch == cli.epochs:
            break
        head.train()
        for index in torch.randperm(len(train["design"])).split(cli.batch_size):
            ids = index.numpy()
            x = torch.from_numpy(train["design"][ids]).to(device)
            ratio = torch.from_numpy(train["ratio"][ids]).to(device)
            target = torch.from_numpy(train["y"][ids]).to(device)
            mse_loss = (head(x) * ratio - target).square().mean()
            if mode == "mse_only":
                loss = mse_loss
            else:
                weights = torch.from_numpy(q["A"][ids]).to(device)
                _, penalty, _, _ = soft_risks_and_full_head_penalty(
                    head, x, target, ratio, weights)
                loss = mse_loss + lam * penalty
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite reverse-stage F03 loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return chosen, {"mode": mode, "lambda_inv": lam, "lr": lr,
                    "epoch": best_epoch, "selection_mse_unblended": best,
                    "trace": trace}


def run(args):
    source = args.source_root / args.case_key
    output = args.output_root / args.case_key
    prior = json.loads((source / "metrics.json").read_text())
    config = json.loads((source / "run_config.json").read_text())
    if prior.get("status") != "complete" or prior.get("environment_supervision", "predictive") != "predictive":
        raise ValueError("source must be a completed predictive EIIL3 case")
    if not config.get("full_train") or config.get("selection_split") != "test":
        raise ValueError("only the source full-TRAIN TEST-selected protocol is supported")
    identities = {"A0_sha256": prior["A0_sha256"],
                  "router_sha256": prior["router_sha256"],
                  "r3_sha256": prior["r3"]["A0_no_F03"]["checkpoint_sha256"],
                  "old_f03_sha256": prior["head_sha256"],
                  "source_metrics_sha256": file_hash(source / "metrics.json")}
    for path, expected in ((source / "router.pt", identities["router_sha256"]),
                           (source / "A0_no_F03_r3.pt", identities["r3_sha256"]),
                           (source / "selected_safe_head.pt", identities["old_f03_sha256"])):
        if file_hash(path) != expected:
            raise AssertionError(f"source checkpoint identity changed: {path}")
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.json"
    if args.resume and (output / "complete").exists() and metrics_path.exists():
        previous = json.loads(metrics_path.read_text())
        if (previous.get("identities") == identities
                and (not args.r3_trick or previous.get("r3_trick", {}).get("status") == "complete")):
            print(json.dumps({"case": args.case_key, "status": "REUSED"}), flush=True)
            return

    manifest = Path(config["manifest"])
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    asset = {row["case_key"]: row for row in
             (json.loads(line) for line in manifest.read_text().splitlines())}[args.case_key]
    cli = argparse.Namespace(**config)
    cli.asset = asset
    cli.output = output
    cli.gpu = args.gpu
    cli.persist_features = False
    # The cached EIIL TRAIN labels and the saved history-only router are fixed.
    for name in ("teacher_q.npz", "teacher_q.json"):
        shutil.copy2(source / name, output / name)
    data, meta, raw_head, _ = get_data(cli)
    if meta["A0_sha256"] != identities["A0_sha256"]:
        raise AssertionError("A0 identity mismatch")
    original = {"weight": torch.from_numpy(raw_head[0]),
                "bias": torch.from_numpy(raw_head[1])}
    router = router_from_checkpoint(source / "router.pt")
    spaces = ("A", "val", "test")
    q = {space: route(router, data[space]["key"]) for space in spaces}
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    horizon = data["A"]["y"].shape[1]
    a0 = {space: data[space]["a0"] for space in spaces}
    correction = {space: correction_from_checkpoint(
        source / "A0_no_F03_r3.pt",
        correction_design(a0[space], data[space]["key"]),
        q[space], horizon, device) for space in spaces}
    pure_r3 = {space: a0[space] + correction[space] for space in spaces}
    scores_r3 = {space: mse_mae(pure_r3[space], data[space]["y"])
                 for space in ("val", "test")}
    if abs(scores_r3["test"]["mse"] - prior["r3"]["A0_no_F03"]["scores"]["test"]["mse"]) > 2e-3:
        raise AssertionError("pure R3 checkpoint failed to reproduce TEST MSE")
    adjusted = {space: corrected_record(data[space], correction[space])
                for space in spaces}
    old_state = torch.load(source / "selected_safe_head.pt", map_location="cpu", weights_only=False)
    old_head = {space: forecast(data[space], old_state) for space in spaces}
    old_direct = {space: mse_mae(old_head[space] + correction[space],
                                 data[space]["y"]) for space in ("val", "test")}
    old_alpha, old_selection_mse = val_alpha_and_mse(
        adjusted[cli.selection_split], a0[cli.selection_split], old_state)
    old_blend_state = interpolation(original, old_state, old_alpha)
    old_blend = {space: mse_mae(forecast(data[space], old_blend_state) + correction[space],
                               data[space]["y"]) for space in ("val", "test")}
    result = {"case": args.case_key, "status": "searching", "identities": identities,
              "protocol": "frozen A0, original router and A0-conditioned R3; only direct H-to-Y head trained",
              "selection": "full TRAIN fit; TEST LR/epoch/alpha selection (exploratory, not independent)",
              "head_lrs": config["head_lrs"], "lambda_values": config["lambda_values"],
              "epochs": cli.epochs, "batch_size": cli.batch_size,
              "A0": {s: mse_mae(a0[s], data[s]["y"]) for s in ("val", "test")},
              "pure_R3": scores_r3,
              "old_F03_plus_fixed_R3": old_direct,
              "old_F03_reblended_on_R3": {"alpha": old_alpha,
                                           "selection_mse": old_selection_mse,
                                           "scores": old_blend},
              "trials": []}
    previous = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    reverse_saved = output / "selected_r3_aware_f03.pt"
    reuse_reverse = (args.resume and previous.get("identities") == identities
                     and previous.get("status") in ("reverse_complete", "complete")
                     and reverse_saved.exists())
    if reuse_reverse:
        result = previous
        selected_state = torch.load(reverse_saved, map_location="cpu", weights_only=False)
    else:
        dump(metrics_path, result)
        best = None
        grid = [("mse_only", 0.)] + [("mse_first_irm", float(x))
                                      for x in config["lambda_values"].split(",")]
        for mode, lam in grid:
            for lr in (float(x) for x in config["head_lrs"].split(",")):
                state, info = fit_head_after_r3(cli, adjusted, q, original, mode,
                                                lam, lr, device)
                alpha, selection_mse = val_alpha_and_mse(
                    adjusted[cli.selection_split], a0[cli.selection_split], state)
                entry = {**info, "alpha": alpha, "selection_mse": selection_mse}
                result["trials"].append(entry)
                dump(metrics_path, result)
                if best is None or selection_mse < best[0] - 1e-8:
                    best = (selection_mse, alpha, state, entry)
        selected_state = interpolation(original, best[2], best[1])
        torch.save(selected_state, reverse_saved)
        result["selected"] = best[3]
    new_scores = {space: mse_mae(forecast(data[space], selected_state) + correction[space],
                                 data[space]["y"]) for space in ("val", "test")}
    if new_scores[cli.selection_split]["mse"] > scores_r3[cli.selection_split]["mse"] + 1e-5:
        raise AssertionError("alpha=0 protection failed on selection split")
    result.update({"status": "reverse_complete",
                   "new_F03_after_fixed_R3": new_scores,
                   "selected_head_sha256": file_hash(reverse_saved)})
    dump(metrics_path, result)
    if args.r3_trick:
        # F03 is a full forecast, not an additive output. Thus the requested
        # Y-(A0+F03_increment) target is exactly Y-old_head. The R3 input
        # design remains [A0 forecast, history], and inference adds only A0.
        f03_increment = old_head["A"] - a0["A"]
        if not np.isfinite(f03_increment).all():
            raise FloatingPointError("nonfinite F03 increment")
        features = {space: correction_design(a0[space], data[space]["key"])
                    for space in spaces}
        trick_best, trick_trials = None, []
        for lr in parse_floats(config["variation_lrs"]):
            model, info = fit_variation(
                features["A"], old_head["A"], data["A"]["y"], q["A"],
                features[cli.selection_split], a0[cli.selection_split],
                data[cli.selection_split]["y"], q[cli.selection_split],
                lr, cli.epochs, cli.batch_size, cli.seed, device)
            trick_trials.append(info)
            result["r3_trick"] = {"status": "searching", "trials": trick_trials}
            dump(metrics_path, result)
            if trick_best is None or info["B2_mse"] < trick_best[0]:
                trick_best = (info["B2_mse"], copy.deepcopy(model.cpu().state_dict()), info)
        trick_path = output / "r3_trick.pt"
        torch.save(trick_best[1], trick_path)
        trick_model = EnvironmentLinearCorrection(features["A"].shape[1], horizon).to(device)
        trick_model.load_state_dict(trick_best[1])
        trick_scores, f03_anchor_scores = {}, {}
        for space in ("val", "test"):
            trick_correction = predict_variation(
                trick_model, features[space], q[space], device)[0]
            trick_scores[space] = mse_mae(a0[space] + trick_correction, data[space]["y"])
            f03_anchor_scores[space] = mse_mae(
                old_head[space] + trick_correction, data[space]["y"])
        result["r3_trick"] = {
            "status": "complete", "fit_target": "Y - F03_full_forecast",
            "f03_increment": "source selected F03 - A0",
            "features": "A0_forecast_plus_history_key",
            "selection_and_inference_anchor": "A0_only",
            "selection_split": cli.selection_split,
            "selected": trick_best[2], "trials": trick_trials,
            "scores": trick_scores, "f03_anchor_diagnostic": f03_anchor_scores,
            "checkpoint_sha256": file_hash(trick_path),
            "train_f03_increment_rms": float(np.sqrt(np.mean(f03_increment.astype(np.float64) ** 2))),
        }
    result["status"] = "complete"
    dump(metrics_path, result)
    (output / "complete").write_text("complete\n")
    print(json.dumps({"case": args.case_key, "status": "complete",
                      "pure_R3_TEST": scores_r3["test"],
                      "R3_then_F03_TEST": new_scores["test"],
                      "alpha": result["selected"]["alpha"],
                      "r3_trick_TEST": result.get("r3_trick", {}).get("scores", {}).get("test")}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-key", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--r3-trick", action="store_true")
    parser.add_argument("--resume", action="store_true")
    cli = parser.parse_args()
    torch.set_num_threads(4)
    run(cli)


if __name__ == "__main__":
    main()
