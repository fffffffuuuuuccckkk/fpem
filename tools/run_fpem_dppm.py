"""Frozen-backbone dual-pattern memory point pilot.

All fitting/OOF reliability uses TRAIN. Hyperparameters use the first half of
VAL; TEST is evaluated after selection. A0's TRAIN predictions are in-sample,
so saved OOF labels describe *memory-module* OOF, not full-system OOF.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.predictive_env.dual_pattern_memory.prototype_bank import PatternBank, normalized_pairs
from models.predictive_env.dual_pattern_memory.forecast_fusion import (
    stable_direction_basis, stable_forecast, variation_forecast,
    prototype_segment_reliability)
from models.predictive_env.dual_pattern_memory.uncertainty_memory import fit_uncertainty_memory
from models.predictive_env.paired_memory.memory_bank import history_key
from models.predictive_env.paired_memory import PairedMemoryBank
from tools.run_fpem_shared_factorization import load_experiment, write_json
from tools.run_predictive_env_iv_patchtst import IndexedDataset
from tools.run_fpem_smrva import collect_split

CASES = ("patchtst_ETTh1_720", "itransformer_ETTm1_720", "moderntcn_ETTm2_720")
BASE = ROOT / "results/fpem_latent_decomposition_search/pilot_20261015"


def args_parse():
    p = argparse.ArgumentParser()
    p.add_argument("case", choices=CASES)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--output", default="results/fpem_dual_pattern_memory/point_pilot")
    p.add_argument("--max_batches", type=int, default=0, help="smoke only")
    p.add_argument("--ms", type=int, default=16)
    p.add_argument("--mv", type=int, default=8)
    p.add_argument("--ridge", type=float, default=.05)
    return p.parse_args()


def loader_slice(dataset, config, start, stop):
    return DataLoader(Subset(IndexedDataset(dataset, cycle_len=config.cyclenet_cycle_len),
                             list(range(start, stop))), batch_size=config.batch_size,
                      shuffle=False, num_workers=0)


def blocks_for(n):
    return (torch.arange(n) * 5 // n).clamp_max(4)


def normalized_record(record, floor):
    key = history_key(record["x"], 64).cpu().float()
    truth, mean, scale = normalized_pairs(record["x"], record["target"], floor)
    base = ((record["base"].float() - mean) / scale).transpose(1, 2)
    return {"key": key, "truth": truth.cpu(), "base": base.cpu(),
            "mean": mean.cpu(), "scale": scale.cpu()}


def fit_bank(key, truth, blocks, count, ridge, seed, mapping=None):
    bank = PatternBank(count, truth.shape[-1], key.shape[-1], ridge=ridge, seed=seed)
    audit = bank.fit(key, truth, blocks, mapping)
    return bank, audit


def prefix_fit(key, truth, groups, stop, count, ridge, seed, mapping=None):
    chosen = groups < stop
    if not bool(chosen.any()):
        raise ValueError("empty chronological prefix")
    mapped = mapping[chosen] if mapping is not None else None
    return fit_bank(key[chosen], truth[chosen], groups[chosen], count, ridge, seed, mapped)[0]


def clamp_variation(v):
    # Fixed robust protection, not a learned split or sample-specific label.
    return v.clamp(-5., 5.)


def variation_key(stable_bank, key):
    return clamp_variation(key - stable_bank.query(key)["matched_history"])


def stable_oof(training, groups, ms, ridge, seed):
    key, truth, base = training["key"], training["truth"], training["base"]
    memory = torch.zeros_like(truth)
    held = torch.zeros(len(key), dtype=torch.bool)
    for fold in (2, 3, 4):
        bank = prefix_fit(key, truth, groups, fold, ms, ridge, seed)
        rows = groups == fold
        memory[rows] = bank.query(key[rows])["prediction"]
        held |= rows
    # Folds 2/3 select directions. Fold 4 remains unseen for variation evidence.
    basis_rows = held & (groups < 4)
    basis, audit = stable_direction_basis(memory[basis_rows], base[basis_rows],
                                          truth[basis_rows], groups[basis_rows], 8)
    return memory, basis, audit


def train_evidence(training, groups, ms, mv, ridge, seed, basis, eta, segments=8):
    """Chronological fold-4 evidence; no fold-4 target in either map."""
    key, truth, base = training["key"], training["truth"], training["base"]
    stable = prefix_fit(key, truth, groups, 4, ms, ridge, seed)
    v_prefix = variation_key(stable, key[groups < 4])
    v_eval = variation_key(stable, key[groups == 4])
    x_prefix = torch.cat((key[groups < 4], v_prefix), -1)
    x_eval = torch.cat((key[groups == 4], v_eval), -1)
    variation = prefix_fit(v_prefix, truth[groups < 4], groups[groups < 4],
                           4, mv, ridge, seed + 11, x_prefix)
    stable_query = stable.query(key[groups == 4])
    stable_prediction = stable_forecast(base[groups == 4],
                      stable_query["prediction"], basis, eta)
    var_query = variation.query(v_eval, x_eval)
    candidate = var_query["candidate"]
    if basis.shape[-1]:
        offset = candidate - stable_prediction[:,:,None]
        candidate = stable_prediction[:,:,None] + offset - (offset @ basis) @ basis.T
    reliability, audit = prototype_segment_reliability(
        stable_prediction, candidate, truth[groups == 4],
        var_query["weight"], segments)
    return reliability, audit, {"stable": stable_prediction,
        "variation_query": var_query, "truth": truth[groups == 4],
        "base": base[groups == 4], "key": key[groups == 4],
        "fold": 4, "provenance": "TRAIN chronological memory-OOF; A0 may be in-sample"}, stable, variation


def predict(mode, row, stable_bank, variation_bank, basis, eta,
            reliability, null_score, segments, shuffled_bank=None):
    key = row["key"]
    base = row["base"]
    if mode == "B0":
        return base, {"null": 1., "stable_effect": 0., "variation_effect": 0.}
    sq = stable_bank.query(key)
    if mode == "B1":
        prediction = (1-eta)*base + eta*sq["prediction"]
        return prediction, {"null": 1., "stable_effect": float((prediction-base).abs().mean()), "variation_effect": 0.}
    stable = stable_forecast(base, sq["prediction"], basis, eta)
    if mode == "B2":
        return stable, {"null": 1., "stable_effect": float((stable-base).abs().mean()), "variation_effect": 0.}
    variation_key_query = clamp_variation(key - sq["matched_history"])
    bank = shuffled_bank if mode == "B7" and shuffled_bank is not None else variation_bank
    vq = bank.query(variation_key_query, torch.cat((key, variation_key_query), -1),
                    null_score if mode in ("B5", "B6", "B7") else None)
    if mode == "B3":
        rel = reliability.new_ones(reliability.shape)
        use_segments = False
    elif mode == "B4":
        rel = reliability
        use_segments = False
        rel = rel.mean(-1, keepdim=True).expand_as(rel)
    else:
        rel = reliability
        use_segments = True
    full, _ = variation_forecast(stable, vq["candidate"], vq["weight"],
                                  rel, basis, use_segments, segments)
    return full, {"null": float(vq["null_weight"].mean()),
                  "stable_effect": float((stable-base).abs().mean()),
                  "variation_effect": float((full-stable).abs().mean())}


def evaluate(record, modes, stable_bank, variation_bank, basis, eta,
             reliability, null_score, segments=8, shuffled_bank=None):
    out = {}
    for mode in modes:
        pred, diag = predict(mode, record, stable_bank, variation_bank, basis,
                             eta, reliability, null_score, segments, shuffled_bank)
        restored = pred.transpose(1, 2) * record["scale"] + record["mean"]
        truth = record["truth"].transpose(1, 2) * record["scale"] + record["mean"]
        out[mode] = {"mse": float((restored-truth).square().mean()),
                     "mae": float((restored-truth).abs().mean()), **diag}
    return out


def main():
    cli = args_parse()
    torch.manual_seed(cli.seed)
    torch.set_num_threads(4)
    out = ROOT / cli.output / cli.case
    out.mkdir(parents=True, exist_ok=True)
    if (out / "point_pilot_complete").exists():
        print("completed; preserving existing results", flush=True)
        return
    cli.base_run_config = str(BASE / (f"{cli.case}_A0_complementary_gate_predictive_"
                                   "sub0_orth0_refnone_fp0_rank8_bias-4/run_config.json"))
    cli.allow_legacy_reference_signature = cli.case.startswith("moderntcn")
    args, model, loaders, device, sha = load_experiment(cli)
    cache = torch.load(ROOT / "results/fpem_predictive_memory/pilot_20261024" /
                       cli.case / "phase0_cache.pt", map_location="cpu", weights_only=False)
    if cache.get("reference_sha256") != sha:
        raise ValueError("A0 checkpoint and TRAIN cache SHA differ")
    records = cache["records"]
    floor_bank = PairedMemoryBank(args.seq_len, args.pred_len, history_size=64)
    floor_bank.fit(records)
    floor = floor_bank.historical_std_floor.cpu()
    training = normalized_record({"x": records["x"], "target": records["target"],
                                  "base": records["a0_pred"]}, floor)
    group = blocks_for(len(training["key"]))
    for parameter in model.parameters():
        if parameter.requires_grad:
            raise AssertionError("A0 must be frozen")
    val_data = loaders["val"].dataset.dataset
    val_half = max(1, len(val_data)//2)
    val = collect_split(model, loader_slice(val_data, args, 0, val_half), device, cli.max_batches)
    val = normalized_record(val, floor)
    write_json(out / "run_config.json", {"case": cli.case, "seed": cli.seed,
        "reference_sha256": sha, "frozen_backbone": True, "ms": cli.ms,
        "mv": cli.mv, "ridge": cli.ridge,
        "selection": "first half VAL; TEST report only",
        "oof_scope": "memory module only; A0 TRAIN predictions may be in-sample",
        "future_target": "complete normalized future, never A0 residual"})
    print("stable TRAIN/OOF bank", flush=True)
    memory_oof, basis, direction_audit = stable_oof(training, group, cli.ms, cli.ridge, cli.seed)
    write_json(out / "stable_direction_audit.json", direction_audit)
    key, truth = training["key"], training["truth"]
    stable_bank, stable_audit = fit_bank(key, truth, group, cli.ms, cli.ridge, cli.seed)
    write_json(out / "stable_bank_audit.json", stable_audit)
    selected_eta, best_stable = 0., float("inf")
    for eta in (0., .25, .5, 1.):
        score = evaluate(val, ("B2",), stable_bank, None, basis, eta,
                         torch.zeros(cli.mv, 8), -1.)["B2"]["mse"]
        if score < best_stable:
            selected_eta, best_stable = eta, score
    print(f"selected stable eta={selected_eta} rank={basis.shape[-1]}", flush=True)
    reliability, reliability_audit, held, prefix_stable, prefix_variation = train_evidence(training, group,
                    cli.ms, cli.mv, cli.ridge, cli.seed, basis, selected_eta)
    write_json(out / "variation_reliability_audit.json", reliability_audit)
    v = variation_key(stable_bank, key)
    mapping_input = torch.cat((key, v), -1)
    variation_bank, variation_audit = fit_bank(v, truth, group, cli.mv,
                                 cli.ridge, cli.seed+11, mapping_input)
    write_json(out / "variation_bank_audit.json", variation_audit)
    selected_null, best_var = 0., float("inf")
    for null in (-4., -2., -1., 0., 1.):
        score = evaluate(val, ("B6",), stable_bank, variation_bank, basis,
                         selected_eta, reliability, null)["B6"]["mse"]
        if score < best_var:
            selected_null, best_var = null, score
    selected = {"eta": selected_eta, "null_score": selected_null,
                "stable_rank": basis.shape[-1], "stable_val_mse": best_stable,
                "full_val_mse": best_var}
    write_json(out / "val_selection.json", selected)
    modes = tuple(f"B{i}" for i in range(7))
    val_metrics = evaluate(val, modes, stable_bank, variation_bank, basis,
                           selected_eta, reliability, selected_null)
    write_json(out / "val_results.json", val_metrics)
    print("validation modes complete", flush=True)
    test = normalized_record(collect_split(model, loaders["test"], device,
                                           cli.max_batches), floor)
    test_metrics = evaluate(test, modes, stable_bank, variation_bank, basis,
                            selected_eta, reliability, selected_null)
    write_json(out / "test_results.json", test_metrics)
    rows = [{"case":cli.case,"mode":m,"val_mse":val_metrics[m]["mse"],
             "val_mae":val_metrics[m]["mae"],"test_mse":test_metrics[m]["mse"],
             "test_mae":test_metrics[m]["mae"],"a0_test_mse":test_metrics["B0"]["mse"],
             "null":test_metrics[m]["null"],"stable_effect":test_metrics[m]["stable_effect"],
             "variation_effect":test_metrics[m]["variation_effect"]} for m in modes]
    with (out / "all_point_results.csv").open("w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    # Fold-4 full point prediction uses earlier TRAIN labels only in banks.
    # It is an exploratory subset; never call it full-system OOF.
    oof_mode = "B6"
    oof_pred, _ = predict(oof_mode, held, prefix_stable, prefix_variation,
        basis, selected_eta, reliability, selected_null, 8)
    residual = held["truth"] - oof_pred
    stable_match = prefix_stable.query(held["key"])["weight"]
    var_key = variation_key(prefix_stable, held["key"])
    var_match = prefix_variation.query(var_key,
        torch.cat((held["key"],var_key),-1), selected_null)["weight"]
    stable_uncertainty = fit_uncertainty_memory(residual, stable_match)
    variation_uncertainty = fit_uncertainty_memory(residual, var_match)
    torch.save({"reference_sha256":sha,"stable":stable_bank.state(),
                "variation":variation_bank.state(),"basis":basis,
                "reliability":reliability,"selection":selected,"floor":floor,
                "stable_uncertainty_memory":stable_uncertainty,
                "variation_uncertainty_memory":variation_uncertainty,
                "memory_oof_fold4_residual":residual,"oof_scope":held["provenance"]},
               out / "dual_pattern_memory.pt")
    write_json(out / "point_summary.json", {"case":cli.case,"reference_sha256":sha,
        "selected":selected,"val":val_metrics,"test":test_metrics,
        "probability_status":"pending point-pilot evidence; old Flow checkpoint not reused",
        "oof_scope":held["provenance"],"memory_size_bytes":(out/"dual_pattern_memory.pt").stat().st_size})
    (out / "point_pilot_complete").write_text("complete\n",encoding="utf-8")
    print(json.dumps({"case":cli.case,"selected":selected,
                      "test":test_metrics},ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
