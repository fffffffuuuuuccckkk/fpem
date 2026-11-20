"""Small G1/G2/G3 diagnostic under a purged, chronological A0-OOS protocol.

Early VAL is historical support, later disjoint VAL selects parameters, and
TEST is report-only. TRAIN histories define prototype frequency/centers. This
does not modify the production DPPM forecast fusion or any checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from sklearn.cluster import MiniBatchKMeans

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.diagnose_predictive_rescue import (
    history_density, per_sample_mae, per_sample_mse, residual_ridge,
    ridge_query, sparse_record,
)
from tools.run_fpem_dppm import BASE, CASES, normalized_record
from tools.run_fpem_dppm_next import block_shuffle
from tools.run_fpem_shared_factorization import load_experiment, write_json

GLOBAL_RIDGES = (1.0, 10.0, 100.0, 1000.0)
LOCAL_RIDGES = (1.0, 10.0, 100.0)
GLOBAL_SCALES = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0)
LOCAL_SCALES = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("case", choices=CASES)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--eval_stride", type=int, default=4)
    p.add_argument("--n_shuffles", type=int, default=20)
    p.add_argument("--output", default="results/fpem_dual_pattern_memory/global_local_diagnostic")
    return p.parse_args()


def subset(record, mask):
    return {name: value[mask] for name, value in record.items()}


def fit_g1(source, selection):
    target = source["truth"] - source["base"]
    best = None
    for ridge in GLOBAL_RIDGES:
        weight = residual_ridge(source["key"], target, ridge)
        correction = ridge_query(selection["key"], weight)
        for scale in GLOBAL_SCALES:
            mse = float(per_sample_mse(selection, selection["base"] + scale * correction).mean())
            if best is None or mse < best["selection_mse"]:
                best = {"ridge": ridge, "scale": scale,
                        "selection_mse": mse, "weight": weight}
    return best


def prototype_centers(train_key, starts, span, seed, count=8):
    key = train_key.reshape(-1, train_key.shape[-1]).float().numpy()
    km = MiniBatchKMeans(n_clusters=count, random_state=seed, n_init=3,
                         batch_size=min(256, len(key)), reassignment_ratio=.01)
    km.fit(key)
    centers = torch.from_numpy(km.cluster_centers_.copy()).float()
    labels = torch.cdist(train_key.reshape(-1, train_key.shape[-1]).float(),
                         centers).argmin(-1)
    events = (starts.long() // span).repeat_interleave(train_key.shape[1])
    # Each overlapping time event casts one unit of total mass, distributed
    # across its channels/windows; mere presence would mark all centers common.
    support = []
    for k in range(count):
        mass = 0.0
        for event in events.unique():
            owned = events == event
            mass += float((labels[owned] == k).float().mean())
        support.append(mass)
    raw = [int((labels == k).sum()) for k in range(count)]
    rare = sorted(range(count), key=lambda k: (support[k], raw[k], k))[:count//2]
    return centers, rare, support, raw


def soft_match(key, centers, temperature=0.5):
    flat = key.reshape(-1, key.shape[-1]).float()
    distance = torch.cdist(flat, centers).square() / flat.shape[-1]
    return (-distance / temperature).softmax(-1)


def fit_local(source_key, target, centers, rare, ridge):
    x = source_key.reshape(-1, source_key.shape[-1]).double()
    y = target.reshape(-1, target.shape[-1]).double()
    design = torch.cat((x, x.new_ones(len(x), 1)), -1)
    match = soft_match(source_key, centers).double()
    penalty = torch.eye(design.shape[1], dtype=design.dtype) * ridge
    penalty[-1, -1] = ridge * 0.01
    weights = {}
    for k in rare:
        w = match[:, k]
        if float(w.sum()) < 1e-8:
            weights[k] = torch.zeros(design.shape[1], y.shape[1])
            continue
        w = w / w.sum()
        gram = design.T @ (design * w[:, None]) + penalty
        rhs = design.T @ (y * w[:, None])
        weights[k] = torch.linalg.solve(gram, rhs).float()
    return weights


def local_query(key, centers, rare, weights):
    x = key.reshape(-1, key.shape[-1]).float()
    design = torch.cat((x, x.new_ones(len(x), 1)), -1)
    match = soft_match(key, centers)
    output = torch.zeros(len(x), next(iter(weights.values())).shape[1])
    for k in rare:
        output += match[:, k, None] * (design @ weights[k])
    return output.reshape(key.shape[0], key.shape[1], -1)


def fit_g2(source, selection, global_model, centers, rare, target=None):
    shared_source = global_model["scale"] * ridge_query(source["key"], global_model["weight"])
    residual = source["truth"] - source["base"] - shared_source
    if target is not None:
        residual = target
    shared_selection = global_model["scale"] * ridge_query(selection["key"], global_model["weight"])
    best = None
    for ridge in LOCAL_RIDGES:
        weights = fit_local(source["key"], residual, centers, rare, ridge)
        correction = local_query(selection["key"], centers, rare, weights)
        for scale in LOCAL_SCALES:
            prediction = selection["base"] + shared_selection + scale * correction
            mse = float(per_sample_mse(selection, prediction).mean())
            if best is None or mse < best["selection_mse"]:
                best = {"ridge": ridge, "scale": scale,
                        "selection_mse": mse, "weights": weights,
                        "raw_selection_mse": float(per_sample_mse(
                            selection, selection["base"] + shared_selection + correction).mean())}
    return best, residual


def report_prediction(record, prediction, masks):
    mse = per_sample_mse(record, prediction)
    mae = per_sample_mae(record, prediction)
    result = {"mse": float(mse.mean()), "mae": float(mae.mean())}
    for name, mask in masks.items():
        result[f"{name}_mse"] = float(mse[mask].mean()) if bool(mask.any()) else None
        result[f"{name}_n"] = int(mask.sum())
    return result


def main():
    cli = parse_args()
    if cli.eval_stride < 1 or cli.n_shuffles < 1:
        raise ValueError("stride and shuffle count must be positive")
    torch.manual_seed(cli.seed)
    torch.set_num_threads(4)
    out = ROOT / cli.output / cli.case
    out.mkdir(parents=True, exist_ok=True)
    cli.base_run_config = str(BASE / (f"{cli.case}_A0_complementary_gate_predictive_"
                              "sub0_orth0_refnone_fp0_rank8_bias-4/run_config.json"))
    cli.allow_legacy_reference_signature = cli.case.startswith("moderntcn")
    args, model, loaders, device, sha = load_experiment(cli)
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise AssertionError("A0 must stay frozen")
    old = ROOT / "results/fpem_dual_pattern_memory/point_pilot" / cli.case
    bundle = torch.load(old / "dual_pattern_memory.pt", map_location="cpu", weights_only=False)
    cache = torch.load(ROOT / "results/fpem_predictive_memory/pilot_20261024" /
                       cli.case / "phase0_cache.pt", map_location="cpu", weights_only=False)
    if bundle["reference_sha256"] != sha or cache["reference_sha256"] != sha:
        raise ValueError("reference checkpoint mismatch")
    train_dataset = loaders["train"].dataset.dataset
    val_dataset = loaders["val"].dataset.dataset
    test_dataset = loaders["test"].dataset.dataset
    raw_train = cache["records"]
    first_val_history = len(train_dataset.data_x) - args.seq_len
    train_ends = raw_train["starts"].long() + args.seq_len + args.pred_len
    train_mask = train_ends <= first_val_history
    train = normalized_record({"x": raw_train["x"][train_mask],
                               "target": raw_train["target"][train_mask],
                               "base": raw_train["a0_pred"][train_mask]}, bundle["floor"])
    val, val_ids = sparse_record(model, val_dataset, args, device, cli.eval_stride, bundle["floor"])
    test, _ = sparse_record(model, test_dataset, args, device, cli.eval_stride, bundle["floor"])
    cut = len(val_dataset.data_x) // 2
    first_test_history_in_val = len(val_dataset.data_x) - args.seq_len
    source_mask = val_ids + args.seq_len + args.pred_len <= cut
    selection_mask = ((val_ids >= cut) &
                      (val_ids + args.seq_len + args.pred_len <= first_test_history_in_val))
    source, selection = subset(val, source_mask), subset(val, selection_mask)
    if len(source["key"]) < 8 or len(selection["key"]) < 8:
        raise ValueError("insufficient non-overlapping VAL support/selection")
    if int((val_ids[source_mask] + args.seq_len + args.pred_len).max()) > int(val_ids[selection_mask].min()):
        raise AssertionError("historical support overlaps selection history")
    if int((val_ids[selection_mask] + args.seq_len + args.pred_len).max()) > first_test_history_in_val:
        raise AssertionError("selection targets overlap first TEST history")

    centers, rare, support, raw_count = prototype_centers(
        train["key"], raw_train["starts"][train_mask], args.seq_len+args.pred_len, cli.seed)
    train_flat = train["key"].reshape(-1, train["key"].shape[-1]).float()
    pair_distance = torch.cdist(train_flat, train_flat).square() / train_flat.shape[-1]
    pair_distance.fill_diagonal_(float("inf"))
    bandwidth = float(pair_distance.kthvalue(min(10, len(train_flat)-1), dim=-1).values.median().clamp_min(1e-4))
    train_density = history_density(train["key"], train["key"], bandwidth, True)
    test_density = history_density(train["key"], test["key"], bandwidth)
    absolute_cut = torch.quantile(train_density, torch.tensor([1/3, 2/3]))
    relative_cut = torch.quantile(test_density, torch.tensor([1/3, 2/3]))
    masks = {"rare": test_density <= relative_cut[0],
             "common": test_density >= relative_cut[1]}

    g1 = fit_g1(source, selection)
    test_shared = g1["scale"] * ridge_query(test["key"], g1["weight"])
    g1_prediction = test["base"] + test_shared
    g2, local_target = fit_g2(source, selection, g1, centers, rare)
    local_true = local_query(test["key"], centers, rare, g2["weights"])
    g2_prediction = g1_prediction + g2["scale"] * local_true
    g2_raw = g1_prediction + local_true
    rows = [dict(method="A0", **report_prediction(test, test["base"], masks)),
            dict(method="G1_global", **report_prediction(test, g1_prediction, masks)),
            dict(method="G2_shared_rare_selected", **report_prediction(test, g2_prediction, masks)),
            dict(method="G2_shared_rare_raw", **report_prediction(test, g2_raw, masks))]

    early_blocks = (torch.arange(len(source["key"])) * 4 // len(source["key"])).clamp_max(3)
    shuffled_rows = []
    for rep in range(cli.n_shuffles):
        shuffled_target = block_shuffle(local_target, early_blocks, cli.seed + 211 + rep)
        null_model, _ = fit_g2(source, selection, g1, centers, rare, shuffled_target)
        null_correction = local_query(test["key"], centers, rare, null_model["weights"])
        selected = report_prediction(test, g1_prediction + null_model["scale"] * null_correction, masks)
        raw_result = report_prediction(test, g1_prediction + null_correction, masks)
        shuffled_rows.append({"rep": rep, "ridge": null_model["ridge"],
                              "scale": null_model["scale"],
                              "selection_mse": null_model["selection_mse"],
                              "test_mse": selected["mse"],
                              "rare_test_mse": selected["rare_mse"],
                              "common_test_mse": selected["common_mse"],
                              "raw_test_mse": raw_result["mse"],
                              "raw_rare_test_mse": raw_result["rare_mse"]})

    for name, records in (("results.csv", rows), ("shuffle_null.csv", shuffled_rows)):
        with (out / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
    g2_mse = rows[2]["mse"]
    g2_rare = rows[2]["rare_mse"]
    g2_raw_mse = rows[3]["mse"]
    g2_raw_rare = rows[3]["rare_mse"]
    null_better = sum(row["test_mse"] <= g2_mse for row in shuffled_rows)
    null_rare_better = sum(row["rare_test_mse"] <= g2_rare for row in shuffled_rows)
    null_raw_better = sum(row["raw_test_mse"] <= g2_raw_mse for row in shuffled_rows)
    null_raw_rare_better = sum(row["raw_rare_test_mse"] <= g2_raw_rare for row in shuffled_rows)
    summary = {"case": cli.case, "reference_sha256": sha,
               "a0_test_mse": rows[0]["mse"], "g1_test_mse": rows[1]["mse"],
               "g2_test_mse": g2_mse, "g2_raw_test_mse": g2_raw_mse,
               "g1_rare_test_mse": rows[1]["rare_mse"],
               "g2_rare_test_mse": g2_rare,
               "g2_raw_rare_test_mse": g2_raw_rare,
               "g1_selection": {k: g1[k] for k in ("ridge", "scale", "selection_mse")},
               "g2_selection": {k: g2[k] for k in ("ridge", "scale", "selection_mse", "raw_selection_mse")},
               "rare_prototype_ids": rare,
               "prototype_independent_train_event_support": support,
               "prototype_raw_train_pair_support": raw_count,
               "rare_match_mass_source": float(soft_match(source["key"], centers)[:, rare].sum(-1).mean()),
               "rare_match_mass_test": float(soft_match(test["key"], centers)[:, rare].sum(-1).mean()),
               "rare_match_mass_test_rare_group": float(soft_match(test["key"][masks["rare"]], centers)[:, rare].sum(-1).mean()),
               "shuffle_null_better_or_equal": null_better,
               "shuffle_null_rare_better_or_equal": null_rare_better,
               "shuffle_null_raw_better_or_equal": null_raw_better,
               "shuffle_null_raw_rare_better_or_equal": null_raw_rare_better,
               "shuffle_null_n": cli.n_shuffles,
               "shuffle_mean_test_mse": sum(r["test_mse"] for r in shuffled_rows)/len(shuffled_rows),
               "shuffle_mean_rare_test_mse": sum(r["rare_test_mse"] for r in shuffled_rows)/len(shuffled_rows),
               "source_val_rows": len(source["key"]),
               "selection_val_rows": len(selection["key"]),
               "test_rows": len(test["key"]),
               "absolute_train_rare_test_count": int((test_density <= absolute_cut[0]).sum())}
    write_json(out / "summary.json", summary)
    write_json(out / "run_config.json", {
        "case": cli.case, "reference_sha256": sha, "seed": cli.seed,
        "eval_stride": cli.eval_stride, "n_shuffles": cli.n_shuffles,
        "training_data_use": "TRAIN X only for 8 prototype centers and history frequency; no TRAIN Y in G1/G2",
        "support": "early VAL with A0-out-of-sample predictions; historical Y observed before selection histories",
        "selection": "later non-overlapping VAL; TEST report only",
        "max_source_future_end_local": int((val_ids[source_mask] + args.seq_len + args.pred_len).max()),
        "min_selection_history_start_local": int(val_ids[selection_mask].min()),
        "max_selection_future_end_local": int((val_ids[selection_mask] + args.seq_len + args.pred_len).max()),
        "first_test_history_start_local_in_val": first_test_history_in_val,
        "rare_prototype_rule": "bottom half of 8 TRAIN-X KMeans centers by independent time-event support",
        "history_frequency_rule": "TRAIN-only kernel density; TEST input-only relative tertiles for balanced diagnostic groups",
        "absolute_train_tertiles": [float(x) for x in absolute_cut],
        "relative_test_input_tertiles": [float(x) for x in relative_cut],
        "global_ridges": GLOBAL_RIDGES, "local_ridges": LOCAL_RIDGES,
        "global_scales": GLOBAL_SCALES, "local_scales": LOCAL_SCALES,
        "shuffling": "only residual after fixed true G1; permute complete sample futures within early VAL time blocks",
        "scope": "offline diagnostic; no production DPPM forward/fusion/checkpoint changed",
    })
    (out / "diagnostic_complete").write_text("complete\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
