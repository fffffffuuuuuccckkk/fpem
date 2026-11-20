#!/usr/bin/env python3
"""A0-safe interpolation of existing EIIL3 F03 heads; no Backbone retraining.

The source cache contains frozen-A0 hidden designs, original predictions and
TRAIN/VAL/TEST targets. VAL selects alpha; TEST is report-only. Optionally fit
fresh detached R3 corrections to A0 and to the selected safe head.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.predictive_env.eiil3_linear_correction import EnvironmentLinearCorrection, correction_design
from models.predictive_env.relation_prototypes import route
from tools.run_fpem_eiil3 import SPACES, dump, file_hash
from tools.run_fpem_eiil3_cosa import fit as fit_variation, predict as predict_variation, router_from_checkpoint


def mse_mae(a: np.ndarray, y: np.ndarray) -> dict:
    d = a.astype(np.float64) - y.astype(np.float64)
    return {"mse": float(np.mean(d * d)), "mae": float(np.mean(np.abs(d)))}


def forecast(record: dict, state: dict, batch: int = 256) -> np.ndarray:
    weight = state["weight"].detach().cpu().numpy()
    bias = state["bias"].detach().cpu().numpy()
    chunks = []
    for start in range(0, len(record["design"]), batch):
        end = start + batch
        chunks.append(((record["design"][start:end] @ weight.T + bias)
                       * record["ratio"][start:end]).astype(np.float32))
    return np.concatenate(chunks)


def interpolation(a0: dict, trained: dict, alpha: float) -> dict:
    alpha = float(alpha)
    return {key: a0[key] + alpha * (trained[key] - a0[key]) for key in ("weight", "bias")}


def analytic_alpha(y0: np.ndarray, y1: np.ndarray, target: np.ndarray) -> float:
    e = y0.astype(np.float64) - target.astype(np.float64)
    d = y1.astype(np.float64) - y0.astype(np.float64)
    denominator = float(np.sum(d * d))
    return float(np.clip(-np.sum(e * d) / denominator, 0, 1)) if denominator > 1e-20 else 0.0


def largest_safe_alpha(y0, y1, target, requested: float, delta: float) -> float:
    """Largest alpha <= requested whose VAL MSE meets the A0-relative cap."""
    baseline = mse_mae(y0, target)["mse"]
    limit = (1 + delta) * baseline + 1e-12
    if mse_mae(y0 + requested * (y1 - y0), target)["mse"] <= limit:
        return float(requested)
    lo, hi = 0.0, float(requested)
    for _ in range(48):
        mid = (lo + hi) / 2
        if mse_mae(y0 + mid * (y1 - y0), target)["mse"] <= limit:
            lo = mid
        else:
            hi = mid
    return lo


def soft_env_risk_variance(pred: np.ndarray, truth: np.ndarray, q: np.ndarray) -> float:
    sample = ((pred.astype(np.float64) - truth.astype(np.float64)) ** 2).mean(axis=1)
    weights = q.astype(np.float64)
    risks = (weights * sample[:, None]).sum(0) / np.maximum(weights.sum(0), 1e-12)
    return float(np.var(risks))


def load_source(source: Path):
    if not (source / "point_complete").exists():
        raise FileNotFoundError(f"incomplete source: {source}")
    original = json.loads((source / "point_metrics.json").read_text())
    meta = json.loads((source / "features.json").read_text())
    if meta["A0_sha256"] != original["A0_sha256"]:
        raise ValueError("A0 SHA mismatch between source metrics and feature cache")
    with np.load(source / "features.npz") as cache:
        data = {space: {field: cache[f"{space}_{field}"] for field in
                        ("design", "key", "y", "a0", "ratio", "start")}
                for space in SPACES}
        teacher = {space: cache[f"q_teacher_{space}"] for space in SPACES[:4]}
    a0 = torch.load(source / "a0_head.pt", map_location="cpu", weights_only=False)
    a0_state = {"weight": torch.from_numpy(a0[0]), "bias": torch.from_numpy(a0[1])}
    for space in SPACES:
        predicted = forecast(data[space], a0_state)
        error = float(np.max(np.abs(predicted - data[space]["a0"])))
        if error > 2e-3:
            raise AssertionError(f"A0 function mismatch {space}: {error}")
    return original, meta, data, teacher, a0_state


def rows_for_variant(case, variant, a0_state, trained_state, data, teacher, eps_relative):
    trained_forecasts = {space: forecast(data[space], trained_state) for space in SPACES}
    y0_val = data["val"]["a0"]
    y1_val = trained_forecasts["val"]
    truth_val = data["val"]["y"]
    optimum = analytic_alpha(y0_val, y1_val, truth_val)
    candidates = [("A0", 0.0), ("Original_F03", 1.0),
                  ("Safe_0.1", .1), ("Safe_0.25", .25), ("Safe_0.5", .5),
                  ("Safe_VAL", optimum)]
    for delta in (0.0, .001, .003):
        candidates.append((f"Protected_{delta:g}",
                           largest_safe_alpha(y0_val, y1_val, truth_val, 1.0, delta)))
    scored = []
    for label, alpha in candidates:
        pred_val = y0_val + alpha * (y1_val - y0_val)
        val = mse_mae(pred_val, truth_val)
        b1 = data["B1"]["a0"] + alpha * (trained_forecasts["B1"] - data["B1"]["a0"])
        inv = soft_env_risk_variance(b1, data["B1"]["y"], teacher["B1"])
        scored.append((label, alpha, val["mse"], inv))
    best_val = min(row[2] for row in scored)
    tolerance = eps_relative * mse_mae(y0_val, truth_val)["mse"]
    lex = min((row for row in scored if row[2] <= best_val + tolerance),
              key=lambda row: (row[3], row[2], row[1]))
    candidates.append(("Safe_Lex", lex[1]))
    rows = []
    selected = {}
    for label, alpha in candidates:
        state = interpolation(a0_state, trained_state, alpha)
        values = {space: mse_mae(data[space]["a0"] + alpha *
                                  (trained_forecasts[space] - data[space]["a0"]), data[space]["y"])
                  for space in SPACES}
        inv = soft_env_risk_variance(data["B1"]["a0"] + alpha *
                                     (trained_forecasts["B1"] - data["B1"]["a0"]),
                                     data["B1"]["y"], teacher["B1"])
        row = {"case": case, "source_variant": variant, "method": label,
               "alpha": alpha, "B1_env_risk_variance": inv,
               "lex_tolerance_relative_to_A0_val": eps_relative,
               **{f"{space}_{metric}": values[space][metric]
                  for space in SPACES for metric in ("mse", "mae")}}
        row["test_mse_change_vs_A0_pct"] = 100 * (row["test_mse"] / mse_mae(
            data["test"]["a0"], data["test"]["y"])["mse"] - 1)
        rows.append(row)
        if label in ("Safe_VAL", "Safe_Lex"):
            selected[label] = state
    # An interpolation of a linear head must match interpolation in prediction space.
    check = forecast(data["val"], selected["Safe_VAL"])
    expected = y0_val + optimum * (y1_val - y0_val)
    if np.max(np.abs(check - expected)) > 3e-3:
        raise AssertionError("head interpolation differs from prediction interpolation")
    if next(row for row in rows if row["method"] == "Safe_VAL")["val_mse"] > rows[0]["val_mse"] + 1e-7:
        raise AssertionError("VAL-optimal alpha did worse than A0")
    return rows, selected


def fresh_r3(source, output, data, stable, label, device, seed, epochs, lrs):
    router = router_from_checkpoint(source / "router.pt")
    q = {space: route(router, data[space]["key"]) for space in SPACES}
    features = {space: correction_design(stable[space], data[space]["key"]) for space in SPACES}
    b1, b2 = data["B1"], data["B2"]
    best, trials = None, []
    for lr in lrs:
        model, info = fit_variation(features["B1"], stable["B1"], b1["y"], q["B1"],
                                    features["B2"], stable["B2"], b2["y"], q["B2"],
                                    lr, epochs, 128, seed, device)
        trials.append(info)
        if best is None or info["B2_mse"] < best[0]:
            best = (info["B2_mse"], {k: v.detach().cpu().clone()
                                     for k, v in model.state_dict().items()}, info)
        if best[0] < mse_mae(stable["B2"], b2["y"])["mse"] - 1e-6:
            break
    torch.save(best[1], output / f"{label}_r3.pt")
    model = EnvironmentLinearCorrection(features["B1"].shape[1], b1["y"].shape[1]).to(device)
    model.load_state_dict(best[1])
    scores = {}
    for space in ("B2", "val", "test"):
        correction, _ = predict_variation(model, features[space], q[space], device)
        scores[space] = mse_mae(stable[space] + correction, data[space]["y"])
    return {"label": label, "selected": best[2], "trials": trials, "scores": scores,
            "r3_sha256": file_hash(output / f"{label}_r3.pt")}


def run(cli):
    source, output = cli.source_case, cli.output_case
    output.mkdir(parents=True, exist_ok=True)
    complete = output / "complete"
    source_sha = file_hash(source / "point_metrics.json")
    if cli.resume and complete.exists():
        prior = json.loads((output / "summary.json").read_text())
        if prior["source_metrics_sha256"] == source_sha:
            print(json.dumps({"case": prior["case"], "status": "REUSED"}), flush=True)
            return
    original, meta, data, teacher, a0_state = load_source(source)
    variants = [name for name in cli.variants.split(",") if name]
    all_rows, states, original_states = [], {}, {}
    for variant in variants:
        path = source / f"{variant}_head.pt"
        if not path.exists():
            continue
        trained = torch.load(path, map_location="cpu", weights_only=False)
        rows, picked = rows_for_variant(original["case"], variant, a0_state, trained,
                                        data, teacher, cli.lex_eps_relative)
        all_rows.extend(rows)
        states[variant] = picked
        original_states[variant] = trained
    if not all_rows:
        raise ValueError("no source F03 head checkpoint exists")
    with (output / "all_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, all_rows[0].keys())
        writer.writeheader(); writer.writerows(all_rows)
    ranked = sorted((row for row in all_rows if row["method"] == "Safe_VAL"),
                    key=lambda row: row["val_mse"])
    chosen = ranked[0]
    chosen_variant = chosen["source_variant"]
    torch.save(states[chosen_variant]["Safe_VAL"], output / "selected_safe_head.pt")
    torch.save(a0_state, output / "a0_head.pt")
    shutil.copyfile(source / "point_metrics.json", output / "source_point_metrics.json")
    shutil.copyfile(source / "features.json", output / "source_features.json")
    shutil.copyfile(source / "router.pt", output / "router.pt")
    manifest_row = None
    if cli.manifest:
        manifest_row = next((json.loads(line) for line in cli.manifest.read_text().splitlines()
                             if json.loads(line)["case_key"] == original["case"]), None)
        if manifest_row is None or manifest_row["A0_sha256"] != original["A0_sha256"]:
            raise ValueError("manifest A0 identity mismatch")
        dump(output / "source_manifest_row.json", manifest_row)
    summary = {"case": original["case"], "status": "point_complete", "A0_sha256": original["A0_sha256"],
               "source_metrics_sha256": source_sha, "source_features_sha256": file_hash(source / "features.json"),
               "source": str(source), "selection": "VAL MSE, then TRAIN B1 environment-risk variance for Lex; TEST report-only",
               "lex_eps_relative": cli.lex_eps_relative, "chosen_variant": chosen_variant,
               "chosen_alpha": chosen["alpha"], "chosen_val_mse": chosen["val_mse"],
               "chosen_test_mse": chosen["test_mse"],
               "zero_alpha_variants": sum(row["alpha"] < 1e-8 for row in ranked),
               "total_variants": len(ranked), "r3": {}}
    dump(output / "summary.json", summary)
    if cli.fit_r3:
        device = torch.device(f"cuda:{cli.gpu}" if torch.cuda.is_available() else "cpu")
        for label, state in (("A0_no_F03", a0_state),
                             ("Original_F03", original_states[chosen_variant]),
                             ("Safe_VAL", states[chosen_variant]["Safe_VAL"])):
            stable = {space: forecast(data[space], state) for space in SPACES}
            summary["r3"][label] = fresh_r3(source, output, data, stable, label,
                                               device, cli.seed, cli.epochs,
                                               [float(v) for v in cli.r3_lrs.split(",")])
            dump(output / "summary.json", summary)
    summary["status"] = "complete"
    dump(output / "summary.json", summary)
    complete.write_text("complete\n")
    if cli.cleanup_source:
        if manifest_row is None or not Path(manifest_row["A0_ckpt"]).exists():
            raise ValueError("cannot clean source without intact A0 checkpoint")
        if file_hash(manifest_row["A0_ckpt"]) != original["A0_sha256"]:
            raise ValueError("cannot clean source: A0 checkpoint SHA mismatch")
        source_rel = source.resolve().relative_to(
            (ROOT / "results" / "fpem_eiil3_mse_priority_20261112").resolve())
        if len(source_rel.parts) != 4 or source.resolve() == output.resolve():
            raise ValueError("source cleanup target is not one exact pilot case")
        if not all((output / filename).exists() for filename in
                   ("complete", "summary.json", "all_results.csv", "selected_safe_head.pt",
                    "source_point_metrics.json", "router.pt", "source_manifest_row.json")):
            raise ValueError("cannot clean source before outputs are verified")
        shutil.rmtree(source)
        summary["source_cleanup"] = "removed prior V0-V5 case files after preserving metrics, A0 SHA and selected new head"
        dump(output / "summary.json", summary)
    print(json.dumps({"case": original["case"], "status": "complete",
                      "chosen_variant": chosen_variant, "alpha": chosen["alpha"],
                      "val_mse": chosen["val_mse"], "test_mse": chosen["test_mse"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-case", required=True, type=Path)
    parser.add_argument("--output-case", required=True, type=Path)
    parser.add_argument("--variants", default="V0,V1,V2,V3,V4_rho_0.05,V4_rho_0.1,V4_rho_0.2,V5")
    parser.add_argument("--lex-eps-relative", type=float, default=.001)
    parser.add_argument("--fit-r3", action="store_true")
    parser.add_argument("--r3-lrs", default="0.0003,0.001,0.0001,0.003")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cleanup-source", action="store_true")
    cli = parser.parse_args()
    torch.set_num_threads(4)
    run(cli)


if __name__ == "__main__":
    main()
