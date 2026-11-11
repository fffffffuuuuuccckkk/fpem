"""Exploratory, TEST-ranked G1-only pilot with purged forward OOF diagnostics.

TEST labels are used only to *rank* already fitted candidates and alpha values.
They are never passed to an estimator, proxy-group builder, OOF gate or shrinker.
Consequently every reported TEST winner is a development result, not an
independent generalization estimate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.fpem_g1_stability_methods import G1Correction, fit, registry_rows
from tools.run_fpem_dppm_g1_r3_full import collect, metric, record, subset
from tools.run_fpem_shared_factorization_g1r3 import load_experiment

ALPHAS = (0., .05, .1, .2, .5, 1.)
RESULT_ROOT = Path("results/fpem_g1_stability_search_20261102")
OLD_ROOT = Path("results/fpem_dual_pattern_memory/g1_r3_full_testlr_search")
BASE_ROOT = Path("results/fpem_dual_pattern_memory/g1_r3_full_point")
EXTENDED_ROOT = Path("results/fpem_dual_pattern_memory/g1_r3_extended_test_search")


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True, choices=("patchtst", "itransformer", "moderntcn"))
    p.add_argument("--dataset", required=True)
    p.add_argument("--pred_len", type=int, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--result_root", type=Path, default=RESULT_ROOT)
    p.add_argument("--max_methods", type=int, default=0, help="test-only smoke limiter; 0 runs all registered pilot candidates")
    p.add_argument("--methods", default="", help="comma-separated Stage-4 shortlist; blank runs full pilot")
    return p.parse_args()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str)+"\n", encoding="utf-8")


def append_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerow(row)


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _cache_source(cli, output):
    source_path = OLD_ROOT/cli.backbone/cli.dataset/f"pred_{cli.pred_len}"/"run_config.json"
    base_path = BASE_ROOT/cli.backbone/cli.dataset/f"pred_{cli.pred_len}"/"base_run_config.json"
    if not source_path.is_file():
        case_dir = EXTENDED_ROOT/cli.backbone/cli.dataset/f"pred_{cli.pred_len}"
        source_path = case_dir/"search"/"run_config.json"
        base_path = case_dir/"base_run_config.json"
    if not (source_path.is_file() and base_path.is_file()):
        raise FileNotFoundError(f"existing A0/G1 source required: {source_path}, {base_path}")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    old_metrics = json.loads((source_path.parent/"metrics.json").read_text(encoding="utf-8"))
    cache_path = output/"frozen_a0_cache.pt"
    base_config = json.loads(base_path.read_text(encoding="utf-8"))
    checkpoint = Path(base_config["reference_checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    expected_sha = source["reference_sha256"]
    actual_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if actual_sha != expected_sha:
        raise ValueError("A0 checkpoint SHA differs from the old G1 trial")
    if cache_path.is_file():
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cache["a0_sha256"] != actual_sha or cache["base_run_config"] != str(base_path):
            raise ValueError("cached A0 data identity differs; do not silently reuse")
        return cache, source, old_metrics
    old_cli = source["cli"]
    loader_cli = SimpleNamespace(**{**old_cli, "gpu": cli.gpu, "seed": cli.seed,
                                    "base_run_config": str(base_path),
                                    "allow_legacy_reference_signature": True})
    args, model, loaders, device, sha = load_experiment(loader_cli)
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise AssertionError("A0 must remain frozen")
    torch.set_num_threads(4)
    raw = collect(model, loaders["train"].dataset.dataset, args, device, old_cli["train_stride"])
    starts = raw["ids"].long()
    train_history_end = len(loaders["train"].dataset.dataset.data_x)-args.seq_len
    legal = starts+args.seq_len+args.pred_len <= train_history_end
    raw = subset(raw, legal)
    starts = starts[legal]
    std = raw["x"].float().var(1, unbiased=False).add(1e-5).sqrt()
    floor = (std.median(0).values*.25).clamp_min(.01)
    train = record(raw, floor)
    del raw
    val = record(collect(model, loaders["val"].dataset.dataset, args, device, old_cli["val_stride"]), floor)
    test = record(collect(model, loaders["test"].dataset.dataset, args, device, old_cli["test_stride"]), floor)
    a0 = metric(test, test["base"])
    if abs(a0["mse"]-old_metrics["a0"]["mse"]) > 2e-5:
        raise AssertionError(f"A0 regression failed: {a0} vs {old_metrics['a0']}")
    cache = {"train": train, "val": val, "test": test, "starts": starts,
             "span": args.seq_len+args.pred_len, "a0_sha256": sha,
             "base_run_config": str(base_path), "a0": a0,
             "old_g1": old_metrics["g1"], "train_stride": old_cli["train_stride"]}
    temporary_cache = cache_path.with_suffix(".building")
    torch.save(cache, temporary_cache)
    temporary_cache.replace(cache_path)
    return cache, source, old_metrics


def candidate_specs(old_ridge, seed):
    """Finite one-factor Stage-1 budget; mathematical duplicates are omitted."""
    base = {"ridge": old_ridge, "seed": seed}
    items = [("RIDGE_original", "RIDGE", base), ("ZERO", "ZERO", base)]
    for value in (1e-3, .01, 1., 100.):
        if value != old_ridge:
            items.append((f"RIDGE_{value:g}", "RIDGE", {**base, "ridge": value}))
    for method in ("U01", "U02", "U03", "U04", "U05", "U06", "U07", "U10", "U11", "U12",
                   "U16", "U17", "U18", "U19", "U20", "U21", "U23", "U24", "U25", "U26",
                   "U27", "U28", "U29", "U30", "U31", "U35", "U37", "U38"):
        cfg = dict(base)
        if method in ("U01", "U02"):
            cfg["delta"] = .5
        if method in ("U05", "U06", "U07", "U11", "U12"):
            cfg["gamma"] = .2
        if method in ("U17", "U18", "U19", "U20"):
            cfg["repeats"] = 8
        if method in ("U20", "U26"):
            cfg["rank"] = 8
        items.append((method, method, cfg))
    for proxy in ("time", "key", "dynamics", "residual"):
        for objective in ("G07", "G08", "G09"):
            items.append((f"{objective}_{proxy}", objective,
                          {**base, "proxy": proxy, "num_groups": 4, "gamma": .2}))
        items.append((f"G12_{proxy}", "G12", {**base, "proxy": proxy, "num_groups": 4}))
    return items


def forward_folds(starts, span):
    starts = starts.long()
    cuts = torch.quantile(starts.double(), torch.tensor([.45, .65, .82], dtype=torch.double)).long().tolist()
    upper = cuts[1:] + [int(starts.max())+1]
    folds = []
    for cut, stop in zip(cuts, upper):
        fit_rows = (starts+span <= cut).nonzero().flatten()
        eval_rows = ((starts >= cut)&(starts < stop)).nonzero().flatten()
        if min(len(fit_rows), len(eval_rows)) < 12:
            continue
        if int((starts[fit_rows]+span).max()) > int(starts[eval_rows].min()):
            raise AssertionError("OOF embargo violated")
        folds.append((fit_rows, eval_rows, cut))
    return folds


def _sample_gain(data, delta):
    residual = data["truth"]-data["base"]
    scale = data["scale"].transpose(1, 2)
    return ((residual.square()-(residual-delta).square())*scale.square()).mean((1, 2))


def _diagnostics(data, delta):
    gains = _sample_gain(data, delta).double()
    negative = gains[gains < 0]
    return {"ntr": float((gains < 0).double().mean()),
            "mean_gain": float(gains.mean()),
            "worst_5pct_gain": float(gains.quantile(.05)),
            "worst_10pct_gain": float(gains.quantile(.1)),
            "negative_gain_mean": float(negative.mean()) if len(negative) else 0.,
            "zero_correction_ratio": float((delta.abs().mean((1, 2)) < 1e-7).float().mean())}


def _oof_fit(method, config, cache):
    train, starts, span = cache["train"], cache["starts"], cache["span"]
    residual = train["truth"]-train["base"]
    predictions = []
    fold_rows = []
    for fit_rows, eval_rows, cut in forward_folds(starts, span):
        fitted = fit(method, train["key"][fit_rows], residual[fit_rows],
                     {"starts": starts[fit_rows]}, config)
        part = subset(train, eval_rows)
        part["starts"] = starts[eval_rows]
        part["embargo"] = span
        delta = fitted.predict(part["key"])
        predictions.append((part, delta))
        for alpha in (0., 1.):
            value = metric(part, part["base"]+alpha*delta)
            fold_rows.append({"fold_cut": cut, "fit_n": len(fit_rows), "eval_n": len(eval_rows),
                              "embargo": span, "alpha": alpha, **value})
    return predictions, fold_rows


def _oof_alpha(predictions):
    if not predictions:
        return 0., 0., 0.
    scored = []
    for alpha in ALPHAS:
        squared = count = 0.
        for part, delta in predictions:
            err = (part["base"]+alpha*delta-part["truth"])*part["scale"].transpose(1, 2)
            squared += float(err.square().sum())
            count += err.numel()
        scored.append((squared/count, alpha))
    best_mse, best_alpha = min(scored)
    # Coarse non-overlapping *event* blocks, not independent sliding windows.
    block_values = []
    for part, delta in predictions:
        gains = _sample_gain(part, best_alpha*delta).double()
        group = torch.div(part["starts"], max(1, int(part.get("embargo", 0) or 1)), rounding_mode="floor")
        # starts are spaced by train_stride; a block of up to 64 sampled rows
        # is more conservative than using overlapping windows as iid evidence.
        if len(group.unique()) > 1:
            for g in group.unique():
                block_values.append(gains[group == g].mean())
        else:
            for chunk in gains.split(64):
                block_values.append(chunk.mean())
    block_gains = torch.stack(block_values) if block_values else torch.zeros(1, dtype=torch.double)
    lower = float(block_gains.mean()-1.64*block_gains.std(unbiased=False)/math.sqrt(len(block_gains)))
    safe_alpha = best_alpha if lower > 0 and len(block_gains) >= 5 else 0.
    return best_alpha, safe_alpha, lower


def _test_alpha(test, raw_delta):
    scored = []
    for alpha in ALPHAS:
        value = metric(test, test["base"]+alpha*raw_delta)
        scored.append((value["mse"], value["mae"], alpha))
    return min(scored)


def _run_trial(output, name, method, config, cache, old_metrics):
    train, test = cache["train"], cache["test"]
    residual = train["truth"]-train["base"]
    started = time.monotonic()
    oof_pred, folds = _oof_fit(method, config, cache)
    oof_alpha, safe_alpha, lower = _oof_alpha(oof_pred)
    model = fit(method, train["key"], residual, {"starts": cache["starts"]}, config)
    raw = model.predict(test["key"])
    test_mse, test_mae, alpha = _test_alpha(test, raw)
    delta = alpha*raw
    diag = _diagnostics(test, delta)
    row = {"trial": name, "method_id": method,
           "family": "LATENT-GROUP" if method.startswith("G") else "STRICT-GROUP-FREE",
           "config": json.dumps(config, sort_keys=True), "alpha_test_selected": alpha,
           "alpha_oof": oof_alpha, "alpha_safe_ci": safe_alpha, "oof_lower_gain": lower,
           "test_mse": test_mse, "test_mae": test_mae,
           "a0_mse": cache["a0"]["mse"], "old_g1_mse": old_metrics["g1"]["mse"],
           "mse_gain_vs_old_g1": old_metrics["g1"]["mse"]-test_mse,
           "oof_folds": len(folds), "elapsed_seconds": time.monotonic()-started,
           "selection_status": "TEST-selected exploratory development result", **diag}
    model_path = output/"best_models"/f"{name}.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.export(), "config": config,
                "alpha_test_selected": alpha, "alpha_oof": oof_alpha,
                "alpha_safe_ci": safe_alpha,
                "a0_sha256": cache["a0_sha256"], "base_run_config": cache["base_run_config"],
                "seed": config.get("seed", 2021), "selection_metric": "TEST_MSE_exploratory"}, model_path)
    for f in folds:
        append_csv(output/"time_forward_results.csv", {"trial": name, **f})
    append_csv(output/"single_method_trials.csv", row)
    append_csv(output/"negative_transfer_results.csv",
               {"trial": name, "test_mse": test_mse, "alpha": alpha, **diag})
    # Separately record OOF-only safety selector.  It never sees TEST labels.
    if alpha != safe_alpha and method != "ZERO":
        safe_value = metric(test, test["base"]+safe_alpha*raw)
        append_csv(output/"combination_trials.csv", {"trial": f"{name}+U41", "components": name,
                   "alpha": safe_alpha, "test_mse": safe_value["mse"],
                   "test_mae": safe_value["mae"], "selection_status": "TEST-ranked exploratory; alpha from TRAIN forward OOF"})
    return row


def _combination_stage(output, cache):
    rows = read_csv(output/"single_method_trials.csv")
    valid = [r for r in rows if r["method_id"] not in ("ZERO", "RIDGE")]
    if not valid:
        return
    best = min(valid, key=lambda r: float(r["test_mse"]))
    old = float(best["old_g1_mse"])
    if float(best["test_mse"]) >= old-1e-5:
        append_csv(output/"failure_log.csv", {"trial": "combinations", "status": "pruned",
                   "reason": "no single method exceeded original G1 by 1e-5"})
        return
    # Only complementary pairs are tested; no Cartesian product.
    candidate_names = ["U01", "U17", "U20", "U23", "U26", "U29", "G07_time", "G08_key"]
    by_name = {r["trial"]: r for r in rows}
    names = [n for n in candidate_names if n in by_name]
    if best["trial"] not in names:
        names = [best["trial"]]+names
    names = names[:5]
    test = cache["test"]
    existing = {r["trial"] for r in read_csv(output/"combination_trials.csv")}
    for i, left in enumerate(names):
        for right in names[i+1:]:
            if left == right:
                continue
            trial = f"C_AUTO_{left}_{right}"
            if any(saved.startswith(trial+"_") for saved in existing):
                continue
            first = torch.load(output/"best_models"/f"{left}.pt", map_location="cpu", weights_only=False)
            second = torch.load(output/"best_models"/f"{right}.pt", map_location="cpu", weights_only=False)
            a = G1Correction.load(first["model"]).predict(test["key"])
            b = G1Correction.load(second["model"]).predict(test["key"])
            # Convex weights fixed from candidate list; TEST selects among them.
            for mix in (.25, .5, .75):
                delta = mix*a+(1-mix)*b
                mse, mae, alpha = _test_alpha(test, delta)
                append_csv(output/"combination_trials.csv", {"trial": trial+f"_{mix:g}",
                           "components": f"{left}+{right}", "alpha": alpha,
                           "test_mse": mse, "test_mae": mae,
                           "selection_status": "TEST-selected exploratory development result"})


def _finalize(output, cache, old_metrics):
    rows = read_csv(output/"single_method_trials.csv")
    ranking = sorted(rows, key=lambda r: (float(r["test_mse"]), float(r["test_mae"])))
    with (output/"pilot_ranking.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ranking[0]))
        writer.writeheader()
        writer.writerows(ranking)
    winner = ranking[0]
    write_json(output/"result.json", {"a0": cache["a0"], "old_g1": old_metrics["g1"],
               "winner": winner, "trials": len(rows), "source_a0_sha256": cache["a0_sha256"],
               "warning": "TEST-selected exploratory development result; not independent generalization evidence"})
    (output/"report.md").write_text(
        f"# G1 stability pilot\n\nA0 MSE: {cache['a0']['mse']:.6f}; original G1 MSE: {old_metrics['g1']['mse']:.6f}.\n\n"
        f"Best single candidate: {winner['trial']}, TEST MSE {float(winner['test_mse']):.6f}, "
        f"MAE {float(winner['test_mae']):.6f}, negative-transfer rate {float(winner['ntr']):.4f}.\n\n"
        "**TEST-selected exploratory development result.** TEST is used to rank methods and alphas; "
        "TRAIN-only labels fit every map, and purged forward OOF alone estimates safety. "
        "No environment invariance is claimed.\n", encoding="utf-8")
    (output/"complete").write_text("complete\n", encoding="utf-8")


def main():
    cli = arguments()
    torch.manual_seed(cli.seed)
    torch.set_num_threads(4)
    output = cli.result_root/cli.backbone/cli.dataset/f"pred_{cli.pred_len}"
    output.mkdir(parents=True, exist_ok=True)
    if (output/"complete").exists():
        print("SKIP complete", output, flush=True)
        return
    cache, source, old_metrics = _cache_source(cli, output)
    write_json(output/"run_config.json", {"cli": vars(cli), "source_run": source,
               "a0_sha256": cache["a0_sha256"], "protocol": "TRAIN fit; purged forward TRAIN OOF; TEST-ranked exploratory selection"})
    registry_path = output/"method_registry.csv"
    if not registry_path.exists():
        with registry_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(registry_rows()[0]))
            writer.writeheader()
            writer.writerows(registry_rows())
    specs = candidate_specs(float(old_metrics["g1"]["ridge"]), cli.seed)
    if cli.methods:
        selected_methods = set(cli.methods.split(",")) | {"RIDGE", "ZERO"}
        specs = [spec for spec in specs if spec[1] in selected_methods or spec[0] in selected_methods]
    if cli.max_methods:
        specs = specs[:cli.max_methods]
    completed = {r["trial"] for r in read_csv(output/"single_method_trials.csv")}
    failures = {r["trial"] for r in read_csv(output/"failure_log.csv")}
    for name, method, config in specs:
        if name in completed or name in failures:
            continue
        try:
            row = _run_trial(output, name, method, config, cache, old_metrics)
            print("TRIAL", cli.backbone, cli.dataset, cli.pred_len, name,
                  f"{row['test_mse']:.6f}", flush=True)
        except (ArithmeticError, RuntimeError, ValueError) as error:
            append_csv(output/"failure_log.csv", {"trial": name, "status": "failed",
                       "reason": f"{type(error).__name__}: {str(error)[:300]}"})
            print("TRIAL FAILED", name, str(error)[:160], flush=True)
    if cli.max_methods:
        print("SMOKE ONLY, no complete marker", flush=True)
        return
    if not cli.methods:
        _combination_stage(output, cache)
    _finalize(output, cache, old_metrics)
    print("COMPLETE", cli.backbone, cli.dataset, cli.pred_len, flush=True)


if __name__ == "__main__":
    main()
