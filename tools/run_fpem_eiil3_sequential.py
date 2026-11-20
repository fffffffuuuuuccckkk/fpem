#!/usr/bin/env python3
"""Per-case EIIL3 search: invariant head, then detached linear variation.

The frozen A0/teacher assets are reused. B1 selects the invariant head; B2
selects the variation adapter. TEST is report-only. No Flow is run here.
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
from models.predictive_env.eiil3_linear_correction import correction_design
from tools.run_fpem_eiil3 import SPACES, dump, file_hash, fit_head, fit_router, get_data
from tools.run_fpem_eiil3_cosa import fit as fit_variation, predict as predict_variation
from tools.run_fpem_mechanism_fusion import score


def forecast(state, record):
    weight = state["weight"].numpy()
    bias = state["bias"].numpy()
    return ((record["design"] @ weight.T + bias) * record["ratio"]).astype(np.float32)


def parse_lrs(value):
    result = [float(part) for part in value.split(",")]
    if not result or any(not np.isfinite(v) or v <= 0 for v in result):
        raise ValueError("learning rates must be finite and positive")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case-key", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--old-root", type=Path, default=Path("results/fpem_eiil3_20261110"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--train-stride", type=int, default=8)
    parser.add_argument("--val-stride", type=int, default=8)
    parser.add_argument("--test-stride", type=int, default=1)
    parser.add_argument("--router-epochs", type=int, default=30)
    parser.add_argument("--router-lr", type=float, default=.01)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lambda-inv", type=float, default=1e-4)
    parser.add_argument("--eiil-steps", type=int, default=50)
    parser.add_argument("--head-lrs", default="0.0003,0.0001,0.001,0.00002")
    parser.add_argument("--variation-lrs", default="0.0003,0.001,0.0001,0.003")
    parser.add_argument("--generate-missing-q", action="store_true")
    parser.add_argument("--reuse-features", action="store_true")
    parser.add_argument("--stop-on-gain", action="store_true")
    parser.add_argument("--resume", action="store_true")
    cli = parser.parse_args()
    torch.set_num_threads(4)
    cli.pilot_search = False
    cli.persist_features = False
    cli.lr_i3 = parse_lrs(cli.head_lrs)[0]
    cli.ridge_alpha = 100.
    rows = (json.loads(line) for line in cli.manifest.read_text().splitlines())
    asset = next((row for row in rows if row["case_key"] == cli.case_key), None)
    if asset is None:
        raise KeyError(cli.case_key)
    if asset["status"] == "MISSING_A0":
        print(json.dumps({"case": cli.case_key, "status": "BLOCKED_MISSING_A0"}), flush=True)
        return
    cli.asset = asset
    cli.output = cli.output_root / cli.case_key
    cli.output.mkdir(parents=True, exist_ok=True)
    if cli.resume and (cli.output / "point_complete").exists():
        prior = json.loads((cli.output / "point_metrics.json").read_text())
        if prior["A0_sha256"] == asset["A0_sha256"]:
            print(json.dumps({"case": cli.case_key, "status": "REUSED"}), flush=True)
            return
    old = cli.old_root / cli.case_key
    cache_used = bool(cli.reuse_features and (old / "features.npz").exists()
                      and (old / "features.json").exists())
    if cache_used:
        old_meta = json.loads((old / "features.json").read_text())
        if old_meta["A0_sha256"] != asset["A0_sha256"]:
            raise ValueError("old feature cache A0 SHA mismatch")
        cli.output = old
    data, meta, initial, teacher = get_data(cli)
    cli.output = cli.output_root / cli.case_key
    cli.output.mkdir(parents=True, exist_ok=True)
    if cache_used:
        # The old features are read-only. Preserve small teacher identity in new run.
        dump(cli.output / "input_cache.json", {"path": str(old), "A0_sha256": asset["A0_sha256"],
                                               "q_sha256": meta["q_sha256"]})
        for name in ("teacher_q.npz", "teacher_q.json"):
            if (old / name).exists():
                shutil.copy2(old / name, cli.output / name)
    q, router = fit_router(cli, data, teacher, meta)
    head_lrs = parse_lrs(cli.head_lrs)
    a0_b1 = score(data["B1"]["a0"], data["B1"]["y"])["mse"]
    head_trials = []
    best_head = None
    for lr in head_lrs:
        pred, state, info = fit_head(cli, data, q, initial, "I3_SoftIRM_EIIL3", lr, cli.lambda_inv)
        entry = {**info, "beats_A0_B1": info["B1_mse"] < a0_b1}
        head_trials.append(entry)
        if best_head is None or info["B1_mse"] < best_head[0]:
            best_head = (info["B1_mse"], state, pred, info)
        dump(cli.output / "head_search.json", {"A0_B1_mse": a0_b1, "trials": head_trials,
                                                "stop_rule": "first B1 MSE below A0"})
        if cli.stop_on_gain and entry["beats_A0_B1"]:
            break
    _, head_state, stable, head_info = best_head
    torch.save(head_state, cli.output / "invariant_head.pt")
    # The stable forecasts below are detached NumPy arrays. Variation fit never
    # sees the backbone or invariant head optimizer, so it cannot update them.
    features = {space: correction_design(stable[space], data[space]["key"])
                for space in SPACES}
    stable_hash = file_hash(cli.output / "invariant_head.pt")
    device = torch.device(f"cuda:{cli.gpu}" if torch.cuda.is_available() else "cpu")
    a, b = data["B1"], data["B2"]
    anchor_b2 = score(stable["B2"], b["y"])["mse"]
    variation_trials = []
    best_variation = None
    for lr in parse_lrs(cli.variation_lrs):
        model, info = fit_variation(features["B1"], stable["B1"], a["y"], q["B1"],
                                    features["B2"], stable["B2"], b["y"], q["B2"],
                                    lr, cli.epochs, cli.batch_size, cli.seed, device)
        entry = {**info, "beats_invariant_B2": info["B2_mse"] < anchor_b2}
        variation_trials.append(entry)
        if best_variation is None or info["B2_mse"] < best_variation[0]:
            best_variation = (info["B2_mse"], copy.deepcopy(model.cpu().state_dict()), info)
        dump(cli.output / "variation_search.json", {"invariant_B2_mse": anchor_b2,
             "trials": variation_trials, "stop_rule": "first B2 MSE below frozen invariant"})
        if cli.stop_on_gain and entry["beats_invariant_B2"]:
            break
    _, variation_state, variation_info = best_variation
    torch.save(variation_state, cli.output / "linear_variation.pt")
    from models.predictive_env.eiil3_linear_correction import EnvironmentLinearCorrection
    model = EnvironmentLinearCorrection(features["B1"].shape[1], a["y"].shape[1]).to(device)
    model.load_state_dict(variation_state)
    model.eval()
    scores = {}
    flow_data = {}
    for space in SPACES:
        correction, uncertainty = predict_variation(model, features[space], q[space], device)
        final = stable[space] + correction
        scores[space] = {"A0": score(data[space]["a0"], data[space]["y"]),
                         "I3": score(stable[space], data[space]["y"]),
                         "I3_R3_linear": score(final, data[space]["y"])}
        if space in ("val", "test") and meta["horizon"] in (24, 48):
            for field, value in (("key", data[space]["key"]), ("y", data[space]["y"]),
                                 ("start", data[space]["start"]), ("point", final),
                                 ("u", uncertainty)):
                flow_data[f"{space}_{field}"] = value
    if flow_data:
        np.savez_compressed(cli.output / "flow_input.npz", **flow_data)
    dump(cli.output / "features.json", meta)
    result = {"case": cli.case_key, "status": "complete", "A0_sha256": asset["A0_sha256"],
              "q_sha256": meta["q_sha256"], "router_sha256": router["router_sha256"],
              "I3_sha256": stable_hash, "R3_sha256": file_hash(cli.output / "linear_variation.pt"),
              "head_lr_selected": head_info["lr"], "head_epoch_selected": head_info["best_epoch"],
              "variation_lr_selected": variation_info["lr"],
              "variation_epoch_selected": variation_info["best_epoch"],
              "head_candidates_evaluated": len(head_trials),
              "variation_candidates_evaluated": len(variation_trials),
              "selection": "TRAIN B1/B2; TEST report only", "cache_used": cache_used,
              "detached_variation_input": True, "scores": scores}
    dump(cli.output / "point_metrics.json", result)
    dump(cli.output / "run_config.json", {k: str(v) if isinstance(v, Path) else v
                                        for k, v in vars(cli).items() if k != "asset"})
    (cli.output / "point_complete").write_text("complete\n")
    print(json.dumps({"case": cli.case_key, "status": "complete", "test": scores["test"]}), flush=True)


if __name__ == "__main__":
    main()
