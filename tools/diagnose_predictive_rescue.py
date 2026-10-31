"""A/B/C diagnostic: can historical X->Y pairs rescue frozen A0 errors?

All fitted targets come from purged TRAIN. VAL chooses regularization and blend
strength; TEST is read once for reporting. Historical frequency uses TRAIN X
only. This is a diagnostic, not a new forecast fusion method.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.run_fpem_dppm import BASE, CASES, normalized_record
from tools.run_fpem_dppm_next import ALPHAS, block_shuffle
from tools.run_fpem_dppm_validation import candidate, fit_control
from tools.run_fpem_shared_factorization import load_experiment, write_json
from tools.run_fpem_smrva import collect_split
from tools.run_predictive_env_iv_patchtst import IndexedDataset

RIDGES = (1.0, 10.0, 100.0, 1000.0)


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("case", choices=CASES)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--eval_stride", type=int, default=24)
    p.add_argument("--n_shuffles", type=int, default=20)
    p.add_argument("--n_adaptation_shuffles", type=int, default=20)
    p.add_argument("--output", default="results/fpem_dual_pattern_memory/predictive_rescue_diagnostic")
    return p.parse_args()


def sparse_record(model, dataset, args, device, stride, floor):
    wrapped = IndexedDataset(dataset, cycle_len=args.cyclenet_cycle_len)
    loader = DataLoader(Subset(wrapped, range(0, len(dataset), stride)),
                        batch_size=args.batch_size, shuffle=False, num_workers=0)
    raw = collect_split(model, loader, device)
    return normalized_record(raw, floor), raw["ids"].long()


def residual_ridge(key, target, ridge):
    x = key.reshape(-1, key.shape[-1]).double()
    y = target.reshape(-1, target.shape[-1]).double()
    design = torch.cat((x, torch.ones(len(x), 1, dtype=x.dtype)), -1)
    gram = design.T @ design / len(design)
    penalty = torch.eye(len(gram), dtype=x.dtype) * ridge
    penalty[-1, -1] = ridge * 0.01
    rhs = design.T @ y / len(design)
    return torch.linalg.solve(gram + penalty, rhs).float()


def ridge_query(key, weights):
    x = key.reshape(-1, key.shape[-1])
    design = torch.cat((x, torch.ones(len(x), 1)), -1)
    return (design @ weights).reshape(key.shape[0], key.shape[1], -1)


def per_sample_mse(record, prediction):
    error = (prediction - record["truth"]) * record["scale"].transpose(1, 2)
    return error.square().mean((1, 2))


def per_sample_mae(record, prediction):
    error = (prediction - record["truth"]) * record["scale"].transpose(1, 2)
    return error.abs().mean((1, 2))


def select(val, query_fn):
    best = None
    for ridge, val_candidate, test_candidate in query_fn():
        for alpha in ALPHAS:
            val_pred = val["base"] + alpha * (val_candidate - val["base"])
            score = float(per_sample_mse(val, val_pred).mean())
            if best is None or score < best["val_mse"]:
                best = {"ridge": ridge, "alpha": alpha, "val_mse": score,
                        "val_candidate": val_candidate, "test_candidate": test_candidate}
    return best


def history_density(source_key, query_key, bandwidth, exclude_self=False):
    source = source_key.reshape(-1, source_key.shape[-1]).float()
    query = query_key.reshape(-1, query_key.shape[-1]).float()
    distance = torch.cdist(query, source).square() / source.shape[-1]
    kernel = (-distance / bandwidth).exp()
    if exclude_self:
        kernel.fill_diagonal_(0)
        density = kernel.sum(-1) / max(1, len(source) - 1)
    else:
        density = kernel.mean(-1)
    return density.reshape(query_key.shape[0], query_key.shape[1]).mean(-1)


def grouped_rows(case, test, score, thresholds, predictions, grouping_scheme):
    bounds = {"rare": score <= thresholds[0],
              "middle": (score > thresholds[0]) & (score < thresholds[1]),
              "common": score >= thresholds[1]}
    base_mse = per_sample_mse(test, test["base"])
    base_mae = per_sample_mae(test, test["base"])
    rows = []
    for group, chosen in bounds.items():
        if not bool(chosen.any()):
            continue
        for method, prediction in predictions.items():
            mse = per_sample_mse(test, prediction)
            mae = per_sample_mae(test, prediction)
            rows.append({"case": case, "grouping_scheme": grouping_scheme,
                         "group": group, "method": method,
                         "n_anchors": int(chosen.sum()),
                         "a0_mse": float(base_mse[chosen].mean()),
                         "a0_mae": float(base_mae[chosen].mean()),
                         "mse": float(mse[chosen].mean()),
                         "mae": float(mae[chosen].mean()),
                         "mse_gain_vs_a0": float((base_mse[chosen] - mse[chosen]).mean()),
                         "positive_gain_ratio": float(((base_mse-mse)[chosen] > 0).float().mean()),
                         "history_density_mean": float(score[chosen].mean())})
    return rows


def main():
    cli = args_parser()
    if cli.eval_stride < 1 or cli.n_shuffles < 1 or cli.n_adaptation_shuffles < 1:
        raise ValueError("eval_stride and shuffle counts must be positive")
    torch.manual_seed(cli.seed)
    torch.set_num_threads(4)
    out = ROOT / cli.output / cli.case
    out.mkdir(parents=True, exist_ok=True)
    source_dir = ROOT / "results/fpem_dual_pattern_memory/next_pilot" / cli.case
    chosen = json.loads((source_dir / "stage_c_selected.json").read_text())
    settings = json.loads((source_dir / f"{chosen['choice']}.json").read_text())["settings"]
    source_config = json.loads((source_dir / "run_config.json").read_text())
    cli.base_run_config = str(BASE / (f"{cli.case}_A0_complementary_gate_predictive_"
                              "sub0_orth0_refnone_fp0_rank8_bias-4/run_config.json"))
    cli.allow_legacy_reference_signature = cli.case.startswith("moderntcn")
    args, model, loaders, device, sha = load_experiment(cli)
    if sha != source_config["reference_sha256"]:
        raise ValueError("A0 checkpoint hash mismatch")
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise AssertionError("A0 must be frozen")
    old = ROOT / "results/fpem_dual_pattern_memory/point_pilot" / cli.case
    bundle = torch.load(old / "dual_pattern_memory.pt", map_location="cpu", weights_only=False)
    if bundle["reference_sha256"] != sha:
        raise ValueError("normalization floor checkpoint mismatch")
    cache = torch.load(ROOT / "results/fpem_predictive_memory/pilot_20261024" /
                       cli.case / "phase0_cache.pt", map_location="cpu", weights_only=False)
    if cache["reference_sha256"] != sha:
        raise ValueError("TRAIN cache checkpoint mismatch")
    raw = cache["records"]
    train_set = loaders["train"].dataset.dataset
    val_set = loaders["val"].dataset.dataset
    test_set = loaders["test"].dataset.dataset
    first_val_history_start = len(train_set.data_x) - args.seq_len
    future_end = raw["starts"].long() + args.seq_len + args.pred_len
    usable = future_end <= first_val_history_start
    if not bool(usable.any()):
        raise ValueError("no purged TRAIN support before VAL history")
    train = normalized_record({"x": raw["x"][usable], "target": raw["target"][usable],
                               "base": raw["a0_pred"][usable]}, bundle["floor"])
    blocks = (torch.arange(len(train["key"])) * 6 // len(train["key"])).clamp_max(5)
    val, val_ids = sparse_record(model, val_set, args, device, cli.eval_stride, bundle["floor"])
    test, _ = sparse_record(model, test_set, args, device, cli.eval_stride, bundle["floor"])

    source = train["key"].reshape(-1, train["key"].shape[-1])
    d = torch.cdist(source.float(), source.float()).square() / source.shape[-1]
    d.fill_diagonal_(float("inf"))
    kth = min(10, len(source)-1)
    bandwidth = float(d.kthvalue(kth, dim=-1).values.median().clamp_min(1e-4))
    train_density = history_density(train["key"], train["key"], bandwidth, True)
    thresholds = torch.quantile(train_density, torch.tensor([1/3, 2/3]))
    test_density = history_density(train["key"], test["key"], bandwidth)

    true_residual = train["truth"] - train["base"]
    shuffled_truth = block_shuffle(train["truth"], blocks, cli.seed + 211)
    shuffled_residual = shuffled_truth - train["base"]
    selections = {}
    predictions = {"A0": test["base"]}
    for label, target in (("global_true", true_residual),
                          ("global_shuffled", shuffled_residual)):
        def candidates():
            for ridge in RIDGES:
                weights = residual_ridge(train["key"], target, ridge)
                yield ridge, val["base"] + ridge_query(val["key"], weights), \
                      test["base"] + ridge_query(test["key"], weights)
        best = select(val, candidates)
        predictions[label] = test["base"] + best["alpha"] * (best["test_candidate"] - test["base"])
        selections[label] = {k: best[k] for k in ("ridge", "alpha", "val_mse")}
    for label, control in (("prototype_true", "E1-1_full"),
                           ("prototype_shuffled", "E1-2_shuffled_xy")):
        fitted = fit_control(control, train, blocks, settings, cli.seed)
        val_candidate = candidate(control, val, fitted, settings)
        test_candidate = candidate(control, test, fitted, settings)
        best = select(val, lambda: [(float(settings["ridge"]), val_candidate, test_candidate)])
        predictions[label] = test["base"] + best["alpha"] * (test_candidate - test["base"])
        selections[label] = {k: best[k] for k in ("ridge", "alpha", "val_mse")}

    null_rows = []
    # The relative groups below are fixed from X-only scores before inspecting Y.
    relative_thresholds = torch.quantile(test_density, torch.tensor([1/3, 2/3]))
    group_masks = {"rare": test_density <= relative_thresholds[0],
                   "common": test_density >= relative_thresholds[1]}
    for rep in range(cli.n_shuffles):
        if rep == 0:
            null_predictions = {name: predictions[name] for name in
                                ("global_shuffled", "prototype_shuffled")}
            null_choices = {name: selections[name] for name in null_predictions}
        else:
            permuted = block_shuffle(train["truth"], blocks, cli.seed + 211 + rep)
            target = permuted - train["base"]
            def null_global():
                for ridge in RIDGES:
                    weights = residual_ridge(train["key"], target, ridge)
                    yield ridge, val["base"] + ridge_query(val["key"], weights), \
                          test["base"] + ridge_query(test["key"], weights)
            global_best = select(val, null_global)
            global_prediction = test["base"] + global_best["alpha"] * (
                global_best["test_candidate"] - test["base"])
            shuffled_record = dict(train)
            shuffled_record["truth"] = permuted
            fitted = fit_control("E1-1_full", shuffled_record, blocks, settings, cli.seed)
            val_candidate = candidate("E1-1_full", val, fitted, settings)
            test_candidate = candidate("E1-1_full", test, fitted, settings)
            proto_best = select(val, lambda: [(float(settings["ridge"]),
                                                val_candidate, test_candidate)])
            proto_prediction = test["base"] + proto_best["alpha"] * (
                test_candidate - test["base"])
            null_predictions = {"global_shuffled": global_prediction,
                                "prototype_shuffled": proto_prediction}
            null_choices = {"global_shuffled": global_best,
                            "prototype_shuffled": proto_best}
        for method, prediction in null_predictions.items():
            mse = per_sample_mse(test, prediction)
            row = {"case": cli.case, "rep": rep, "method": method,
                   "test_mse": float(mse.mean()),
                   "val_mse": null_choices[method]["val_mse"],
                   "ridge": null_choices[method]["ridge"],
                   "alpha": null_choices[method]["alpha"]}
            for group, mask in group_masks.items():
                row[f"{group}_test_mse"] = float(mse[mask].mean())
            null_rows.append(row)

    rows = grouped_rows(cli.case, test, test_density, thresholds, predictions,
                        "absolute_train_tertiles")
    # Input-only relative ranking is useful when all TEST histories are novel
    # under the absolute TRAIN calibration. Neither score uses TEST labels.
    rows += grouped_rows(cli.case, test, test_density, relative_thresholds,
                         predictions, "relative_test_input_rank")
    all_rows = []
    for label, prediction in predictions.items():
        all_rows.append({"case": cli.case, "method": label, "n_anchors": len(test["key"]),
                         "test_mse": float(per_sample_mse(test, prediction).mean()),
                         "test_mae": float(per_sample_mae(test, prediction).mean()),
                         "val_selected_ridge": selections.get(label, {}).get("ridge"),
                         "val_selected_alpha": selections.get(label, {}).get("alpha"),
                         "val_mse": selections.get(label, {}).get("val_mse")})
    for filename, entries in (("all_results.csv", all_rows),
                              ("frequency_groups.csv", rows),
                              ("shuffle_null.csv", null_rows)):
        with (out / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=entries[0].keys())
            writer.writeheader()
            writer.writerows(entries)
    audit = {"case": cli.case, "a0_sha256": sha, "eval_stride": cli.eval_stride,
             "source_train_rows": int(usable.sum()), "purged_train_rows": int((~usable).sum()),
             "max_source_future_end": int(future_end[usable].max()),
             "first_val_history_start": first_val_history_start,
             "source_before_val": bool(int(future_end[usable].max()) <= first_val_history_start),
             "history_frequency": "TRAIN-only Gaussian kernel density on normalized 64-point history keys; tertile thresholds from TRAIN only",
             "density_bandwidth": bandwidth,
             "density_tertile_thresholds": [float(x) for x in thresholds],
             "relative_test_input_rank_thresholds": [float(x) for x in relative_thresholds],
             "prototype_settings": settings,
             "selection": "regularization and shrinkage selected on VAL; TEST report only",
             "source_caveat": "A0 TRAIN predictions were generated by an A0 checkpoint trained on TRAIN; mapping targets are not strict A0 OOF, though VAL/TEST are chronologically held out.",
             "shuffling": "within-TRAIN chronological block; same architecture, source count, and VAL selection budget",
             "n_shuffles": cli.n_shuffles,
             "selections": selections}
    write_json(out / "run_config.json", audit)
    # Secondary chronology check: fit on an early, genuinely A0-out-of-sample
    # part of VAL, choose on a later disjoint VAL part, then report TEST. This
    # is a historical-adaptation diagnostic, not the deployable TRAIN protocol.
    val_cut = len(val_set.data_x) // 2
    first_test_history_in_val = len(val_set.data_x) - args.seq_len
    val_source_mask = val_ids + args.seq_len + args.pred_len <= val_cut
    val_select_mask = ((val_ids >= val_cut) &
                       (val_ids + args.seq_len + args.pred_len <= first_test_history_in_val))
    if int(val_source_mask.sum()) >= 8 and int(val_select_mask.sum()) >= 8:
        early = {k: v[val_source_mask] for k, v in val.items()}
        late = {k: v[val_select_mask] for k, v in val.items()}
        early_blocks = (torch.arange(len(early["key"])) * 4 // len(early["key"])).clamp_max(3)
        adaptation_rows = []
        for rep in range(-1, cli.n_adaptation_shuffles):
            label = "true" if rep == -1 else "shuffled"
            target = (early["truth"] if rep == -1 else
                      block_shuffle(early["truth"], early_blocks,
                                    cli.seed + 211 + rep)) - early["base"]
            def adaptation_candidates():
                for ridge in RIDGES:
                    weights = residual_ridge(early["key"], target, ridge)
                    yield ridge, late["base"] + ridge_query(late["key"], weights), \
                          test["base"] + ridge_query(test["key"], weights)
            chosen_adaptation = select(late, adaptation_candidates)
            predicted = test["base"] + chosen_adaptation["alpha"] * (
                chosen_adaptation["test_candidate"] - test["base"])
            predicted_mse = per_sample_mse(test, predicted)
            adaptation_rows.append({"case": cli.case, "pairing": label, "rep": rep,
                                    "source_n": len(early["key"]),
                                    "selection_n": len(late["key"]),
                                    "max_source_future_end_local": int((val_ids[val_source_mask]
                                        + args.seq_len + args.pred_len).max()),
                                    "min_selection_history_start_local": int(val_ids[val_select_mask].min()),
                                    "ridge": chosen_adaptation["ridge"],
                                    "alpha": chosen_adaptation["alpha"],
                                    "selection_val_mse": chosen_adaptation["val_mse"],
                                    "test_mse": float(predicted_mse.mean()),
                                    "rare_test_mse": float(predicted_mse[group_masks["rare"]].mean()),
                                    "common_test_mse": float(predicted_mse[group_masks["common"]].mean())})
        with (out / "val_history_oos_global.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=adaptation_rows[0].keys())
            writer.writeheader()
            writer.writerows(adaptation_rows)
        audit["val_history_oos_global"] = {
            "scope": "separate exploratory adaptation check; early VAL labels form historical support; no TEST labels used",
            "source_before_selection": int((val_ids[val_source_mask]
                + args.seq_len + args.pred_len).max()) <= int(val_ids[val_select_mask].min()),
            "selection_end_before_first_test_history": int((val_ids[val_select_mask]
                + args.seq_len + args.pred_len).max()) <= first_test_history_in_val,
            "results": adaptation_rows}
        write_json(out / "run_config.json", audit)
    (out / "diagnostic_complete").write_text("complete\n", encoding="utf-8")
    print(json.dumps({"case": cli.case, "results": all_rows,
                      "group_counts": {g: next((r["n_anchors"] for r in rows if r["group"] == g), 0)
                                       for g in ("common", "middle", "rare")}}), flush=True)


if __name__ == "__main__":
    main()
