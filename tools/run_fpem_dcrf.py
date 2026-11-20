"""DCRF frozen-backbone B0-B11 pilot with causal TRAIN memory.

Selection uses the first half of VAL. The second half remains untouched for a
later, independent probability-calibration experiment. No TEST target enters
retrieval, fitting, evidence scores, or hyperparameter selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.predictive_env.dual_consensus_retrieval import (
    selection_from_memory, two_stage_weights, weighted_ridge_evidence,
    reliable_correction)
from models.predictive_env.mapping_stability import history_scale
from models.predictive_env.paired_memory import PairedMemoryBank
from models.predictive_env.paired_memory.memory_bank import history_key
from models.predictive_env.horizon_future_variant import HorizonFutureVariant
from models.predictive_env.probability_heads import ProbabilityHeadBank
from models.predictive_env.residual_difference.condition import ResidualDifferenceCondition
from tools.run_fpem_shared_factorization import load_experiment, write_json
from tools.run_predictive_env_iv_patchtst import IndexedDataset
from tools.run_fpem_smrva import collect_split


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/fpem_latent_decomposition_search/pilot_20261015"
CASES = ("patchtst_ETTh1_720", "itransformer_ETTm1_720", "moderntcn_ETTm2_720")
MODES = tuple(f"B{i}" for i in range(12))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=CASES)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--output", default="results/fpem_dual_consensus_retrieval/point_pilot")
    parser.add_argument("--max_batches", type=int, default=0, help="smoke only")
    return parser.parse_args()


def _loader(dataset, args, start, stop):
    return DataLoader(Subset(IndexedDataset(dataset, cycle_len=args.cyclenet_cycle_len),
                             list(range(start, stop))), batch_size=args.batch_size,
                      shuffle=False, num_workers=0)


def eligible_rows(ends, split, sample_id, val_start):
    if split == "train":
        return ends <= int(sample_id)
    if split == "val":
        return ends <= int(val_start)
    if split == "test":
        return torch.ones_like(ends, dtype=torch.bool)
    raise ValueError(split)


def _mismatch(indices, eligible):
    pool = eligible.nonzero(as_tuple=True)[0]
    if len(pool) < 2:
        return indices
    rank = torch.searchsorted(pool.contiguous(), indices.contiguous())
    return pool[(rank + max(1, len(pool) // 2)) % len(pool)]


def _source_quality(quality, eligible):
    return quality.masked_fill(~eligible, 0.)


def query_one(key, row_features, memory_key, memory_features,
              memory_common, memory_variation, residual, variation_target,
              quality_stable, quality_variation, eligible, config, mode,
              sample_seed, want_trace=False):
    """One variable of one query, with all N eligible TRAIN candidates."""
    horizon = residual.shape[-1]
    zero = residual.new_zeros(horizon)
    if mode == "B0" or not bool(eligible.any()):
        return zero, zero, {"initial_count": 0, "stable_count": 0,
                            "variation_count": 0, "null_weight": 1.,
                            "stable_gain": 0., "variation_gain": 0.,
                            "stable_consensus": 0., "variation_consensus": 0.,
                            "stable_effect": 0., "variation_effect": 0.}
    raw, _ = selection_from_memory(key, memory_key, eligible,
                    config["tau0"], random=mode == "B9",
                    fixed_topk=5 if mode == "B8" else 0,
                    random_seed=sample_seed)
    initial = int((raw > 0).sum())
    if mode == "B2":
        head = weighted_ridge_evidence(row_features, memory_features, residual,
                                       raw, config["lambda_stable"])
        candidate = head.candidate if head.supported else zero
        return candidate, zero, {"initial_count": initial,
                "stable_count": initial, "variation_count": 0,
                "null_weight": 1., "stable_gain": float(head.gain.mean()),
                "variation_gain": 0., "stable_consensus": 1.,
                "variation_consensus": 0., "stable_effect": float(candidate.abs().mean()),
                "variation_effect": 0.}
    common, variation, stable_weight, variant_weight, null_weight = two_stage_weights(
        raw, key, memory_key, memory_common, memory_variation,
        _source_quality(quality_stable, eligible),
        _source_quality(quality_variation, eligible), config["tau_stable"],
        config["tau_variation"], config["beta_stable"], config["beta_variation"],
        config["null_score"], use_null=mode not in ("B3", "B4"),
        use_quality=mode != "B7")
    if mode == "B3":
        variant_weight = variant_weight.new_zeros(variant_weight.shape)
        null_weight = null_weight.new_tensor(1.)
    source_residual = residual
    source_variation = variation_target
    if mode == "B10":
        mismatch = _mismatch(torch.arange(len(eligible), device=eligible.device), eligible)
        source_residual = residual[mismatch]
        source_variation = variation_target[mismatch]
    use_consensus = mode not in ("B3", "B4", "B5")
    use_reliability = mode not in ("B3", "B4", "B5", "B7")
    stable = weighted_ridge_evidence(row_features, memory_features, source_residual,
                                     stable_weight, config["lambda_stable"])
    stable_center, stable_cons, stable_rel = reliable_correction(
        stable, config["gain_scale"], use_consensus, use_reliability)
    if mode == "B11":
        stable_constant = weighted_ridge_evidence(row_features, memory_features,
                         source_residual, stable_weight, config["lambda_stable"],
                         constant=True)
        stable_center = stable_constant.common if use_consensus else stable_constant.candidate
    stable_delta = stable_center * stable_cons * stable_rel
    var_delta = zero
    var_gain = zero
    var_cons_mean = 0.
    if mode != "B3" and bool((variant_weight > 0).any()):
        var = weighted_ridge_evidence(row_features, memory_features,
                    source_variation, variant_weight, config["lambda_variation"])
        var_center, var_cons, var_rel = reliable_correction(
            var, config["gain_scale"], use_consensus, use_reliability)
        if mode == "B11":
            constant = weighted_ridge_evidence(row_features, memory_features,
                        source_variation, variant_weight, config["lambda_variation"],
                        constant=True)
            var_center = constant.common if use_consensus else constant.candidate
        var_delta = (1 - null_weight) * var_center * var_cons * var_rel
        var_gain = var.gain
        var_cons_mean = float(var_cons.mean())
    info = {"initial_count": initial,
            "stable_count": int((stable_weight > 0).sum()),
            "variation_count": int((variant_weight > 0).sum()),
            "null_weight": float(null_weight),
            "stable_gain": float(stable.gain.mean()),
            "variation_gain": float(var_gain.mean()),
            "stable_consensus": float(stable_cons.mean()),
            "variation_consensus": var_cons_mean,
            "stable_effect": float(stable_delta.abs().mean()),
            "variation_effect": float(var_delta.abs().mean())}
    if want_trace:
        info["initial_indices"] = (raw > 0).nonzero(as_tuple=True)[0].tolist()
        info["stable_indices"] = (stable_weight > 0).nonzero(as_tuple=True)[0].tolist()
        info["variation_indices"] = (variant_weight > 0).nonzero(as_tuple=True)[0].tolist()
    return stable_delta, var_delta, info


@torch.no_grad()
def build_common_variation(memory_key, ends, starts, tau0):
    """Each historical row's descriptors use only futures already finished then."""
    common = torch.zeros_like(memory_key)
    variation = torch.zeros_like(memory_key)
    for i in range(len(memory_key)):
        eligible = ends <= starts[i]
        for c in range(memory_key.shape[1]):
            if not bool(eligible.any()):
                common[i, c] = memory_key[i, c]
                continue
            weight, _ = selection_from_memory(memory_key[i, c], memory_key[:, c],
                                               eligible, tau0)
            common[i, c] = weight @ memory_key[:, c]
            variation[i, c] = memory_key[i, c] - common[i, c]
    return common, variation


@torch.no_grad()
def causal_evidence(bank, features, common, variation, config):
    """Strictly chronological OOF stable/variation targets and quality scores."""
    n, channels, horizon = bank.residual.shape
    stable_remaining = bank.residual.new_zeros(n, channels, horizon)
    full_remaining = stable_remaining.clone()
    stable_quality = bank.residual.new_zeros(n, channels)
    variation_quality = stable_quality.clone()
    records = []
    for i in range(n):
        eligible = bank.ends <= bank.starts[i]
        if bool((eligible & (torch.arange(n, device=eligible.device) >= i)).any()):
            raise AssertionError("OOF memory contains unavailable futures")
        for c in range(channels):
            stable_delta, variant_delta, info = query_one(
                bank.keys[i,c], features[i,c], bank.keys[:,c], features[:,c],
                common[:,c], variation[:,c], bank.residual[:,c],
                stable_remaining[:,c], stable_quality[:,c], variation_quality[:,c],
                eligible, config, "B6", int(bank.starts[i]) * 17 + c)
            truth = bank.residual[i,c]
            stable_remaining[i,c] = truth - stable_delta
            full_remaining[i,c] = truth - stable_delta - variant_delta
            base_error = truth.square().mean()
            stable_error = stable_remaining[i,c].square().mean()
            full_error = full_remaining[i,c].square().mean()
            stable_quality[i,c] = ((base_error - stable_error) /
                                  base_error.clamp_min(1e-5)).clamp(-2,2)
            variation_quality[i,c] = ((stable_error - full_error) /
                                     stable_error.clamp_min(1e-5)).clamp(-2,2)
            records.append({"source": i, "channel": c,
                            "eligible_count": int(eligible.sum()),
                            "stable_gain_relative": float(stable_quality[i,c]),
                            "variation_gain_relative": float(variation_quality[i,c]),
                            "full_oof_normalized_mse": float(full_error), **info})
        if i % 64 == 0:
            print(f"OOF {i + 1}/{n}", flush=True)
    return {"stable_remaining": stable_remaining,
            "full_remaining": full_remaining, "stable_quality": stable_quality,
            "variation_quality": variation_quality, "rows": records}


@torch.no_grad()
def evaluate(split_record, split, mode, bank, features, common, variation,
             evidence, config, val_start, max_rows=0, diagnostics=False):
    n = len(split_record["x"])
    horizon = bank.pred_len
    channels = bank.keys.shape[1]
    total = {"mse": 0., "mae": 0., "a0_mse": 0., "a0_mae": 0.,
             "stable_effect": 0., "variation_effect": 0.,
             "initial_count": 0., "stable_count": 0., "variation_count": 0.,
             "null_weight": 0., "stable_gain": 0., "variation_gain": 0.,
             "stable_consensus": 0., "variation_consensus": 0.}
    trace = []
    max_rows = min(n, max_rows) if max_rows else n
    started = time.monotonic()
    for i in range(max_rows):
        x = split_record["x"][i:i+1].to(bank.keys.device)
        truth = split_record["target"][i].to(bank.keys.device)
        base = split_record["base"][i].to(bank.keys.device)
        sample_id = int(split_record["ids"][i])
        key = history_key(x, bank.history_size)[0]
        scale = history_scale(x, bank.historical_std_floor)[1][0, 0]
        eligible = eligible_rows(bank.ends, split, sample_id, val_start)
        prediction = base.clone()
        for c in range(channels):
            stable_delta, variant_delta, info = query_one(
                key[c], torch.cat((key[c], key.new_ones(1))),
                bank.keys[:,c], features[:,c], common[:,c], variation[:,c],
                bank.residual[:,c], evidence["stable_remaining"][:,c],
                evidence["stable_quality"][:,c], evidence["variation_quality"][:,c],
                eligible, config, mode, sample_id * 17 + c,
                want_trace=diagnostics and i < 32)
            prediction[:,c] += (stable_delta + variant_delta) * scale[c]
            for field in ("initial_count", "stable_count", "variation_count",
                          "null_weight", "stable_gain", "variation_gain",
                          "stable_consensus", "variation_consensus",
                          "stable_effect", "variation_effect"):
                total[field] += info[field]
            if diagnostics and i < 32:
                trace.append({"sample_id": sample_id, "channel": c, **info,
                              "a0_mse": float((base[:,c]-truth[:,c]).square().mean()),
                              "final_mse": float((prediction[:,c]-truth[:,c]).square().mean()),
                              "base_prediction": base[:,c].cpu().tolist(),
                              "final_prediction": prediction[:,c].cpu().tolist(),
                              "observed_future_for_offline_diagnostics_only": truth[:,c].cpu().tolist()})
        total["mse"] += float((prediction-truth).square().sum())
        total["mae"] += float((prediction-truth).abs().sum())
        total["a0_mse"] += float((base-truth).square().sum())
        total["a0_mae"] += float((base-truth).abs().sum())
    scalar = max_rows * channels * horizon
    for field in ("mse", "mae", "a0_mse", "a0_mae"):
        total[field] /= scalar
    for field in set(total) - {"mse", "mae", "a0_mse", "a0_mae"}:
        total[field] /= max_rows * channels
    total["sample_count"] = max_rows
    total["seconds"] = time.monotonic() - started
    if not all(torch.isfinite(torch.tensor(x)) for x in total.values()):
        raise FloatingPointError("nonfinite DCRF result")
    return total, trace


def main():
    cli = parse_args()
    torch.set_num_threads(4)
    out = ROOT / cli.output / cli.case
    out.mkdir(parents=True, exist_ok=True)
    if (out / "point_pilot_complete").exists():
        print("already complete", flush=True)
        return
    cli.base_run_config = str(BASE / (f"{cli.case}_A0_complementary_gate_predictive_"
                                     "sub0_orth0_refnone_fp0_rank8_bias-4/run_config.json"))
    cli.allow_legacy_reference_signature = cli.case.startswith("moderntcn")
    args, model, loaders, device, sha = load_experiment(cli)
    cached = torch.load(ROOT / "results/fpem_predictive_memory/pilot_20261024" /
                        cli.case / "phase0_cache.pt", map_location="cpu", weights_only=False)
    if cached.get("reference_sha256") != sha:
        raise ValueError("TRAIN anchor cache/A0 checkpoint SHA mismatch")
    records = cached["records"]
    bank = PairedMemoryBank(args.seq_len, args.pred_len, history_size=64)
    bank_audit = bank.fit(records)
    bank.to(device)
    flow_audit = ProbabilityHeadBank(args.d_model, args.pred_len,
                                    patch_len=16, heads=("residual_flow",))
    adapter_audit = ResidualDifferenceCondition(7, args.d_model, 16, "film")
    future_audit = HorizonFutureVariant(args.d_model, args.pred_len,
                                      future_patch_len=16, environment_count=0)
    count_parameters = lambda module: sum(p.numel() for p in module.parameters())
    write_json(out / "head_audit.json", {
        "head_linear": {"class": type(model.head_linear).__name__,
                        "parameters": count_parameters(model.head_linear),
                        "frozen_in_DCRF": True, "training": "A0 checkpoint only",
                        "inference": "A0 reference forecast (via h_forecast_head)"},
        "h_forecast_head": {"class": type(model.h_forecast_head).__name__,
                            "parameters": count_parameters(model.h_forecast_head),
                            "frozen_in_DCRF": True, "training": "A0 reference copy",
                            "inference": "yes, frozen A0"},
        "stable_local_ridge": {"parameters": 0, "optimizer": "none",
                               "learning_rate": None, "hyperparameter": "lambda_stable",
                               "target": "A0 normalized residual", "inference": "yes"},
        "variation_local_ridge": {"parameters": 0, "optimizer": "none",
                                  "learning_rate": None, "hyperparameter": "lambda_variation",
                                  "target": "causal stable-OOF residual", "inference": "yes"},
        "existing_probability_only_not_used_in_point_pilot": {
            "ResidualDifferenceCondition": {"parameters": count_parameters(adapter_audit),
                "historical_optimizer": "Adam", "historical_lr": 1e-3,
                "loss": "Flow Matching through zero-initialized FiLM"},
            "ProbabilityHeadBank_residual_flow": {"parameters": count_parameters(flow_audit),
                "historical_optimizer": "Adam", "historical_lr": 1e-4,
                "loss": "residual Flow Matching"},
            "HorizonFutureVariant": {"parameters": count_parameters(future_audit),
                "historical_optimizer": "Adam", "historical_lr": 1e-4}},
        "backbone": {"frozen": True, "learning_rate": 0.,
                     "parameter_count": count_parameters(model)},
        "note": "Counts of probability modules are a structural audit; those modules are not instantiated for training in this point pilot."})
    del flow_audit, adapter_audit, future_audit
    features = torch.cat((bank.keys, torch.ones_like(bank.keys[...,:1])), -1)
    train_data = loaders["train"].dataset.dataset
    val_data = loaders["val"].dataset.dataset
    val_start = len(train_data.data_x) - args.seq_len
    validation = collect_split(model,
        _loader(val_data, args, 0, max(1, len(val_data)//2)), device, cli.max_batches)
    test = collect_split(model, loaders["test"], device, cli.max_batches)
    write_json(out / "run_config.json", {"case": cli.case, "seed": cli.seed,
       "reference_sha256": sha, "frozen_backbone": True, "train_memory": bank_audit,
       "selection": "first-half VAL only; TEST report only after configuration fixed",
       "initial_retrieval": "full causal candidate set, entmax1.5",
       "stable_head": "weighted float64 Cholesky ridge with analytic LOO",
       "variation_head": "weighted float64 Cholesky ridge with NULL candidate",
       "probability_status": "not started until new OOF point anchor is validated"})
    default = {"tau0": 1., "tau_stable": 1., "tau_variation": 1.,
               "lambda_stable": 100., "lambda_variation": 100.,
               "beta_stable": 1., "beta_variation": 1.,
               "null_score": 0., "gain_scale": 1.}
    # O4: a small VAL-only coarse search of raw retrieval temperature.
    raw_temperature = []
    blank_common = bank.keys.clone()
    blank_variation = bank.keys.new_zeros(bank.keys.shape)
    blank_evidence = {"stable_remaining": bank.residual,
                      "stable_quality": bank.keys.new_zeros(bank.keys.shape[:2]),
                      "variation_quality": bank.keys.new_zeros(bank.keys.shape[:2])}
    for temperature in (.25, 1., 4.):
        setting = {**default, "tau0": temperature}
        result, _ = evaluate(validation, "val", "B2", bank, features,
            blank_common, blank_variation, blank_evidence, setting, val_start,
            max_rows=cli.max_batches * args.batch_size if cli.max_batches else 0)
        raw_temperature.append({"tau0": temperature, "val_mse": result["mse"]})
    default["tau0"] = min(raw_temperature, key=lambda row: row["val_mse"])["tau0"]
    common, variation = build_common_variation(bank.keys, bank.ends, bank.starts,
                                               default["tau0"])
    print("Causal memory descriptions complete", flush=True)
    evidence = causal_evidence(bank, features, common, variation, default)
    # O1: separate VAL-selected regularization of the two different targets.
    stable_trials = []
    for value in (1., 10., 100., 1000.):
        setting = {**default, "lambda_stable": value}
        result, _ = evaluate(validation, "val", "B3", bank, features,
            common, variation, evidence, setting, val_start,
            max_rows=cli.max_batches * args.batch_size if cli.max_batches else 0)
        stable_trials.append({"lambda_stable": value, "val_mse": result["mse"]})
    default["lambda_stable"] = min(stable_trials, key=lambda row: row["val_mse"])["lambda_stable"]
    evidence = causal_evidence(bank, features, common, variation, default)
    variation_trials = []
    for value in (1., 10., 100., 1000.):
        setting = {**default, "lambda_variation": value}
        result, _ = evaluate(validation, "val", "B6", bank, features,
            common, variation, evidence, setting, val_start,
            max_rows=cli.max_batches * args.batch_size if cli.max_batches else 0)
        variation_trials.append({"lambda_variation": value, "val_mse": result["mse"]})
    default["lambda_variation"] = min(variation_trials,
                                      key=lambda row: row["val_mse"])["lambda_variation"]
    null_trials = []
    for value in (-1., 0., 1.):
        setting = {**default, "null_score": value}
        result, _ = evaluate(validation, "val", "B6", bank, features,
            common, variation, evidence, setting, val_start,
            max_rows=cli.max_batches * args.batch_size if cli.max_batches else 0)
        null_trials.append({"null_score": value, "val_mse": result["mse"]})
    default["null_score"] = min(null_trials,
                                key=lambda row: row["val_mse"])["null_score"]
    evidence = causal_evidence(bank, features, common, variation, default)
    write_json(out / "optimization_only.json", {
        "raw_temperature": raw_temperature, "stable_ridge": stable_trials,
        "variation_ridge": variation_trials, "null_score": null_trials,
        "selected": default, "selection_split": "first-half VAL"})
    torch.save({"reference_sha256": sha, "starts": bank.starts.cpu(),
                "ends": bank.ends.cpu(), "common": common.cpu(),
                "variation": variation.cpu(), "stable_quality": evidence["stable_quality"].cpu(),
                "variation_quality": evidence["variation_quality"].cpu(),
                "stable_oof_residual": evidence["stable_remaining"].cpu(),
                "full_oof_residual": evidence["full_remaining"].cpu()},
               out / "causal_memory.pt")
    with (out / "causal_evidence.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=evidence["rows"][0].keys())
        writer.writeheader(); writer.writerows(evidence["rows"])
    rows = []
    sm_root = ROOT / "results/fpem_mapping_stability/pilot_20261027" / cli.case
    selected_sm = json.loads((sm_root / "point_selection.json").read_text(encoding="utf-8"))["selected_mode"]
    sm = json.loads((sm_root / f"{selected_sm}_result.json").read_text(encoding="utf-8"))
    if sm.get("reference_sha256") != sha:
        raise ValueError("SMRVA comparator uses a different A0 checkpoint")
    for mode in MODES:
        path = out / f"{mode}_result.json"
        if path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
            if result.get("reference_sha256") != sha:
                raise ValueError("cached result A0 mismatch")
            rows.append(result)
            continue
        if mode == "B1":
            result = {"case": cli.case, "mode": mode, "source": "SMRVA existing VAL-selected",
                      "smrva_mode": selected_sm, "val": sm["val"], "test": sm["test"],
                      "reference_sha256": sha, "config": sm["config"]}
        else:
            val, _ = evaluate(validation, "val", mode, bank, features,
                 common, variation, evidence, default, val_start,
                 max_rows=cli.max_batches * args.batch_size if cli.max_batches else 0)
            test_metrics, trace = evaluate(test, "test", mode, bank, features,
                 common, variation, evidence, default, val_start,
                 max_rows=cli.max_batches * args.batch_size if cli.max_batches else 0,
                 diagnostics=mode == "B6")
            result = {"case": cli.case, "mode": mode, "source": "DCRF",
                      "val": val, "test": test_metrics, "reference_sha256": sha,
                      "config": default}
            if trace:
                write_json(out / "B6_prediction_traces_32.json", trace)
        write_json(path, result)
        rows.append(result)
        print(json.dumps({"mode": mode, "val_mse": result["val"]["mse"],
                          "test_mse": result["test"]["mse"]}), flush=True)
    fields = ("case", "mode", "val_mse", "val_mae", "test_mse", "test_mae",
              "a0_test_mse", "initial_count", "stable_count", "variation_count",
              "null_weight", "stable_effect", "variation_effect")
    with (out / "all_point_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            metric = row["test"]
            writer.writerow({"case": cli.case, "mode": row["mode"],
                "val_mse": row["val"]["mse"], "val_mae": row["val"]["mae"],
                "test_mse": metric["mse"], "test_mae": metric["mae"],
                "a0_test_mse": metric.get("a0_mse", rows[0]["test"]["mse"]),
                **{key: metric.get(key, "") for key in fields[7:]}})
    selected = min((r for r in rows if r["mode"] not in {"B8", "B9", "B10", "B11"}),
                   key=lambda row: row["val"]["mse"])
    write_json(out / "point_selection.json", {"selected_mode": selected["mode"],
        "selected_test_mse": selected["test"]["mse"],
        "selected_val_mse": selected["val"]["mse"],
        "B6_beats_A0_B2_random_shuffled_constant_on_VAL":
        all(next(r for r in rows if r["mode"] == "B6")["val"]["mse"] <
            next(r for r in rows if r["mode"] == other)["val"]["mse"]
            for other in ("B0", "B2", "B9", "B10", "B11")),
        "reference_sha256": sha})
    text = [f"# DCRF point pilot: {cli.case}", "",
            "All parameters selected from first-half VAL; TEST is report-only.",
            f"VAL-selected mode: {selected['mode']}",
            "| Mode | VAL MSE | TEST MSE | TEST MAE |",
            "|---|---:|---:|---:|"]
    text.extend(f"| {r['mode']} | {r['val']['mse']:.6f} | {r['test']['mse']:.6f} | {r['test']['mae']:.6f} |"
                for r in rows)
    text.extend(("", "This pilot does not train or reuse a Flow under the new point anchor.",
                 "OOF memory is TRAIN-causal. TEST future is metrics-only."))
    (out / "summary.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    (out / "point_pilot_complete").touch()


if __name__ == "__main__":
    main()
