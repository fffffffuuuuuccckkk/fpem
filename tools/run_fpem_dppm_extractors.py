"""Compare TRAIN-only DPPM prototype extraction with unchanged downstream use.

P0 is routed through the original implementation.  The frozen A0 TRAIN cache
is in-sample, so reported block-held-out results are *memory-module* OOF, not
full-system OOF.  P5 is refused unless a genuinely aligned A0 OOF cache exists.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.predictive_env.dual_pattern_memory.extractors import PatternExtractor
from models.predictive_env.dual_pattern_memory.prototype_bank import PatternBank
from models.predictive_env.dual_pattern_memory.forecast_fusion import (
    stable_direction_basis, stable_forecast, variation_forecast,
    prototype_segment_reliability)
from models.predictive_env.paired_memory import PairedMemoryBank
from tools.run_fpem_dppm import (
    BASE, CASES, blocks_for, clamp_variation, fit_bank as original_fit_bank,
    loader_slice, normalized_record, predict as original_predict,
    stable_oof as original_stable_oof,
    train_evidence as original_evidence)
from tools.run_fpem_shared_factorization import load_experiment, write_json
from tools.run_fpem_smrva import collect_split


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("case", choices=CASES)
    p.add_argument("--mode", choices=[f"P{i}" for i in range(9)], required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--ms", type=int, default=16)
    p.add_argument("--mv", type=int, default=8)
    p.add_argument("--ridge", type=float, default=.05)
    p.add_argument("--max_batches", type=int, default=0)
    p.add_argument("--controls", action="store_true")
    p.add_argument("--a0_oof_cache", default="", help="P5 only: strict chronological, row-aligned A0 OOF cache")
    p.add_argument("--output", default="results/fpem_dual_pattern_memory/extractor_pilot")
    return p.parse_args()


def make_extractor(mode, cli, train, groups):
    instance = PatternExtractor(mode, cli.ms, cli.mv, cli.seed,
                                span=cli.seq_len+cli.pred_len)
    instance.fit(train["key"], train["truth"], train["base"], groups,
                 train["starts"], train.get("a0_oof"))
    return instance


def fit_banks(train, groups, cli, stable_only=False):
    mode = cli.mode
    ex = make_extractor(mode, cli, train, groups)
    key, truth = train["key"], train["truth"]
    keep = ex.train_mask if ex.train_mask is not None else torch.ones(len(key), dtype=torch.bool)
    fit_key, fit_truth, fit_groups = key[keep], truth[keep], groups[keep]
    sk = ex.train_stable_key(key)[keep]
    if mode == "P0":
        stable, stable_audit = original_fit_bank(sk, fit_truth, fit_groups, cli.ms,
                                                 cli.ridge, cli.seed)
    else:
        stable = PatternBank(cli.ms, fit_truth.shape[-1], sk.shape[-1],
                             ridge=cli.ridge, seed=cli.seed)
        stable_audit = stable.fit(sk, fit_truth, fit_groups,
                                  preset_centers=ex.stable_centers)
    if stable_only:
        return ex, stable, None, stable_audit, None
    sq = stable.query(sk)
    vk = ex.rare_key(fit_key, sq) if mode != "P4" else sk
    mapping = torch.cat((fit_key, vk), -1)
    if mode == "P0":
        rare, rare_audit = original_fit_bank(vk, fit_truth, fit_groups, cli.mv,
                                            cli.ridge, cli.seed+11, mapping)
    else:
        rare = PatternBank(cli.mv, fit_truth.shape[-1], vk.shape[-1],
                           ridge=cli.ridge, seed=cli.seed+11)
        rare_audit = rare.fit(vk, fit_truth, fit_groups, mapping,
                              preset_centers=ex.rare_centers)
    return ex, stable, rare, stable_audit, rare_audit


def subset(train, rows):
    return {k: v[rows] if isinstance(v, torch.Tensor) and len(v) == len(rows)
            else v for k,v in train.items()}


def stable_oof(train, groups, cli):
    if cli.mode == "P0":
        return original_stable_oof(train, groups, cli.ms, cli.ridge, cli.seed)
    memory = torch.zeros_like(train["truth"])
    held = torch.zeros(len(groups), dtype=torch.bool)
    for fold in (2, 3, 4):
        rows = groups < fold
        part = subset(train, rows)
        ex, stable, _, _, _ = fit_banks(part, groups[rows], cli, stable_only=True)
        target_rows = groups == fold
        memory[target_rows] = stable.query(ex.stable_key(train["key"][target_rows]))["prediction"]
        held |= target_rows
    basis_rows = held & (groups < 4)
    basis, audit = stable_direction_basis(memory[basis_rows],
        train["base"][basis_rows], train["truth"][basis_rows], groups[basis_rows], 8)
    return memory, basis, audit


def predict(row, ex, stable, rare, basis, eta, reliability, null,
            mode="full", segments=8):
    key, base = row["key"], row["base"]
    if ex.mode == "P0" and mode == "full":
        return original_predict("B6", row, stable, rare, basis, eta,
                                reliability, null, segments)
    if mode == "a0":
        return base, {"null": 1., "stable_effect": 0., "variation_effect": 0.}
    sk = ex.stable_key(key)
    sq = stable.query(sk)
    stable_pred = stable_forecast(base, sq["prediction"], basis, eta)
    if mode == "stable":
        return stable_pred, {"null": 1., "stable_effect": float((stable_pred-base).abs().mean()),
                             "variation_effect": 0.}
    vk = ex.rare_key(key, sq)
    vq = rare.query(vk, torch.cat((key, vk), -1), null)
    anchor = base if mode == "rare" else stable_pred
    if basis.shape[-1] and mode != "rare":
        offset = vq["candidate"] - anchor[:,:,None]
        candidates = anchor[:,:,None] + offset - (offset @ basis) @ basis.T
    else:
        candidates = vq["candidate"]
    rel = torch.ones_like(reliability) if mode == "no_reliability" else reliability
    use_basis = basis if mode != "rare" else basis[:, :0]
    out, _ = variation_forecast(anchor, candidates, vq["weight"],
                                 rel, use_basis, True, segments)
    return out, {"null": float(vq["null_weight"].mean()),
                 "stable_effect": float((stable_pred-base).abs().mean()) if mode != "rare" else 0.,
                 "variation_effect": float((out-anchor).abs().mean())}


def metrics(row, prediction, diag):
    restored = prediction.transpose(1,2)*row["scale"]+row["mean"]
    truth = row["truth"].transpose(1,2)*row["scale"]+row["mean"]
    return {"mse": float((restored-truth).square().mean()),
            "mae": float((restored-truth).abs().mean()), **diag}


def score(row, ex, stable, rare, basis, eta, reliability, null, mode="full"):
    pred, diag = predict(row, ex, stable, rare, basis, eta, reliability, null, mode)
    return metrics(row, pred, diag)


def select_eta(val, ex, stable, basis, mv):
    reliability = torch.zeros(mv, 8)
    return min((0., .25, .5, 1.), key=lambda e: score(val, ex, stable, None,
                     basis, e, reliability, -1., "stable")["mse"])


def select_null(val, ex, stable, rare, basis, eta, reliability, mode="full"):
    return min((-4., -2., -1., 0., 1.), key=lambda n: score(val, ex, stable,
                  rare, basis, eta, reliability, n, mode)["mse"])


def reliability_oof(train, groups, cli, basis, eta):
    rows, held_rows = groups < 4, groups == 4
    prefix = subset(train, rows)
    ex, stable, rare, _, _ = fit_banks(prefix, groups[rows], cli)
    held = subset(train, held_rows)
    sk = ex.stable_key(held["key"])
    sq = stable.query(sk)
    stable_pred = stable_forecast(held["base"], sq["prediction"], basis, eta)
    vk = ex.rare_key(held["key"], sq)
    vq = rare.query(vk, torch.cat((held["key"], vk), -1))
    candidate = vq["candidate"]
    if basis.shape[-1]:
        offset = candidate - stable_pred[:,:,None]
        candidate = stable_pred[:,:,None] + offset - (offset @ basis) @ basis.T
    rel, audit = prototype_segment_reliability(stable_pred, candidate,
                                held["truth"], vq["weight"], 8)
    audit = {"segment_reliability": audit}
    audit["stable_oof_mse"] = float((stable_pred-held["truth"]).square().mean())
    audit["a0_oof_mse"] = float((held["base"]-held["truth"]).square().mean())
    audit["rare_candidate_oof_mse"] = float((vq["prediction"]-held["truth"]).square().mean())
    audit["stable_oof_gain"] = audit["a0_oof_mse"]-audit["stable_oof_mse"]
    audit["rare_oof_gain"] = audit["stable_oof_mse"]-audit["rare_candidate_oof_mse"]
    return rel, audit, held


def group_metrics(test, ex, stable, rare, basis, eta, rel, null):
    key = test["key"]
    sq = stable.query(ex.stable_key(key))
    distance = (ex.stable_key(key).reshape(-1,64)[:,None]-stable.centers[None]).square().mean(-1).min(-1).values
    rarity = distance.reshape(len(key), -1).mean(-1)
    median = rarity.median()
    result = []
    for name, take in (("common", rarity <= median), ("rare", rarity > median)):
        part = subset(test, take)
        if len(part["key"]) == 0:
            continue
        a0 = score(part, ex, stable, rare, basis, eta, rel, null, "a0")
        stable_score = score(part, ex, stable, rare, basis, eta, rel, null, "stable")
        rare_score = score(part, ex, stable, rare, basis, eta, rel, null, "rare")
        full = score(part, ex, stable, rare, basis, eta, rel, null, "full")
        result.append({"group": name, "samples": int(take.sum()),
                       "a0_mse": a0["mse"], "stable_mse": stable_score["mse"],
                       "rare_only_mse": rare_score["mse"], "full_mse": full["mse"],
                       "full_gain": a0["mse"]-full["mse"]})
    return result


def row_write(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def prototype_diagnostics(train, groups, cli, ex, stable, rare):
    key = train["key"]
    sk = ex.stable_key(key)
    sq = stable.query(sk)
    vk = ex.rare_key(key, sq)
    vq = rare.query(vk, torch.cat((key,vk),-1))
    event = (train["starts"].long()//(cli.seq_len+cli.pred_len))[:,None].expand(
        -1,key.shape[1]).reshape(-1)
    result = {}
    for name, query, bank in (("stable",sq,stable),("rare",vq,rare)):
        weight = query["weight"].reshape(-1, bank.n_patterns)
        hard = weight.argmax(-1)
        result[name+"_independent_events"] = [int(torch.unique(event[hard==j]).numel())
                    for j in range(bank.n_patterns)]
        result[name+"_match_entropy"] = float((-(weight*weight.clamp_min(1e-9).log()).sum(-1)).mean())
        result[name+"_train_mapping_mse"] = float((query["prediction"]-train["truth"]).square().mean())
    return result


def main():
    cli = parse_args()
    torch.manual_seed(cli.seed)
    torch.set_num_threads(4)
    out = ROOT / cli.output / cli.case / cli.mode
    out.mkdir(parents=True, exist_ok=True)
    if (out/"extractor_complete").exists():
        print("complete; preserving existing result", flush=True); return
    cli.base_run_config = str(BASE / (f"{cli.case}_A0_complementary_gate_predictive_"
                                     "sub0_orth0_refnone_fp0_rank8_bias-4/run_config.json"))
    cli.allow_legacy_reference_signature = cli.case.startswith("moderntcn")
    args, model, loaders, device, sha = load_experiment(cli)
    cli.seq_len, cli.pred_len = args.seq_len, args.pred_len
    cache = torch.load(ROOT / "results/fpem_predictive_memory/pilot_20261024" /
                       cli.case / "phase0_cache.pt", map_location="cpu", weights_only=False)
    if cache.get("reference_sha256") != sha:
        raise ValueError("frozen A0 checkpoint/cache SHA mismatch")
    records = cache["records"]
    floor_bank = PairedMemoryBank(args.seq_len, args.pred_len, history_size=64)
    floor_bank.fit(records)
    floor = floor_bank.historical_std_floor.cpu()
    train = normalized_record({"x": records["x"], "target": records["target"],
                               "base": records["a0_pred"]}, floor)
    train["starts"] = records["starts"].cpu()
    groups = blocks_for(len(train["key"]))
    if cli.mode == "P5" and cli.a0_oof_cache:
        source = torch.load(cli.a0_oof_cache, map_location="cpu", weights_only=False)
        if source.get("protocol") != "strict_chronological_a0_oof":
            raise ValueError("P5 cache must certify strict chronological A0 OOF")
        if not torch.equal(source["starts"].long(), train["starts"].long()):
            raise ValueError("P5 cache rows do not match TRAIN starts")
        cutoff = source["source_max_target_end"].long()
        if cutoff.shape != train["starts"].shape or not bool((cutoff[groups>=2] < train["starts"][groups>=2]).all()):
            raise ValueError("P5 A0 OOF source labels overlap held-out query")
        raw_oof = source["predictions"].float()
        if raw_oof.shape != records["a0_pred"].shape:
            raise ValueError("P5 A0 OOF prediction shape mismatch")
        train["a0_oof"] = ((raw_oof-train["mean"])/train["scale"]).transpose(1,2)
    if any(p.requires_grad for p in model.parameters()):
        raise AssertionError("A0 must be frozen")
    val_data = loaders["val"].dataset.dataset
    val = normalized_record(collect_split(model,
        loader_slice(val_data, args, 0, max(1, len(val_data)//2)),
        device, cli.max_batches), floor)
    config = {"case": cli.case, "mode": cli.mode, "seed": cli.seed,
        "reference_sha256": sha, "ms": cli.ms, "mv": cli.mv, "ridge": cli.ridge,
        "selection": "first half VAL; TEST report only", "controls": cli.controls,
        "oof_scope": "memory-module chronological OOF; A0 TRAIN cache in-sample",
        "probability": "existing Flow unchanged and not retrained in this pilot"}
    write_json(out/"run_config.json", config)
    if cli.mode == "P5" and not cli.a0_oof_cache:
        # In-sample A0 errors cannot stand in for a Backbone weakness signal.
        write_json(out/"blocked.json", {"reason": "aligned chronological A0 OOF cache missing",
            "required": "core-only A0 checkpoint per chronological fold and row-aligned predictions"})
        print("P5 safely deferred: aligned strict A0 OOF unavailable", flush=True)
        return
    started = time.monotonic()
    memory, basis, basis_audit = stable_oof(train, groups, cli)
    ex, stable, rare, stable_audit, rare_audit = fit_banks(train, groups, cli)
    ex.stats.update(prototype_diagnostics(train, groups, cli, ex, stable, rare))
    eta = select_eta(val, ex, stable, basis, cli.mv)
    if cli.mode == "P0":
        rel, rel_audit, held, _, _ = original_evidence(train, groups, cli.ms, cli.mv,
                                    cli.ridge, cli.seed, basis, eta)
        rel_audit = {"segment_reliability": rel_audit}
        rel_audit["a0_oof_mse"] = float((held["base"]-held["truth"]).square().mean())
        rel_audit["stable_oof_mse"] = float((held["stable"]-held["truth"]).square().mean())
        rel_audit["stable_oof_gain"] = rel_audit["a0_oof_mse"]-rel_audit["stable_oof_mse"]
        rel_audit["rare_candidate_oof_mse"] = float((held["variation_query"]["prediction"]-held["truth"]).square().mean())
        rel_audit["rare_oof_gain"] = rel_audit["stable_oof_mse"]-rel_audit["rare_candidate_oof_mse"]
    else:
        rel, rel_audit, held = reliability_oof(train, groups, cli, basis, eta)
    null = select_null(val, ex, stable, rare, basis, eta, rel)
    run_modes = ("a0", "stable", "rare", "full", "no_reliability")
    vals = {m: score(val, ex, stable, rare, basis, eta, rel, null, m) for m in run_modes}
    test = normalized_record(collect_split(model, loaders["test"], device,
                                            cli.max_batches), floor)
    query_started = time.monotonic()
    tests = {m: score(test, ex, stable, rare, basis, eta, rel, null, m) for m in run_modes}
    test_query_seconds = time.monotonic()-query_started
    controls = {}
    if cli.controls:
        # Matched capacities and validation rules for every extraction method.
        shuffled = dict(train)
        permutation = torch.arange(len(train["key"]))
        generator = torch.Generator().manual_seed(cli.seed+881)
        for block in range(5):
            idx = (groups == block).nonzero().flatten()
            permutation[idx] = idx[torch.randperm(len(idx), generator=generator)]
        shuffled["truth"] = train["truth"][permutation]
        shuffled_ex, shuffled_stable, shuffled_rare, _, _ = fit_banks(shuffled, groups, cli)
        shuffled_eta = select_eta(val, shuffled_ex, shuffled_stable, basis, cli.mv)
        shuffled_rel, _, _ = reliability_oof(shuffled, groups, cli, basis, shuffled_eta)
        shuffled_null = select_null(val, shuffled_ex, shuffled_stable, shuffled_rare,
                                    basis, shuffled_eta, shuffled_rel)
        controls["shuffled_xy"] = score(test, shuffled_ex, shuffled_stable,
                                  shuffled_rare, basis, shuffled_eta, shuffled_rel,
                                  shuffled_null)
        # Same total prototype capacity, one undivided future map.
        single, _ = original_fit_bank(train["key"], train["truth"], groups,
                                     cli.ms+cli.mv, cli.ridge, cli.seed)
        single_val = single.query(val["key"])["prediction"]
        single_test = single.query(test["key"])["prediction"]
        single_eta = min((0., .25, .5, 1.), key=lambda e: metrics(val,
                     stable_forecast(val["base"], single_val, basis, e), {})["mse"])
        controls["single_bank"] = metrics(test, stable_forecast(test["base"],
                                                   single_test, basis, single_eta), {})
        # Capacity-matched global linear map, same input and ridge penalty.
        design = torch.cat((train["key"].reshape(-1,64).double(),
                            torch.ones(train["key"].numel()//64,1, dtype=torch.float64)), -1)
        target = train["truth"].reshape(len(design), -1).double()
        eye = torch.eye(65, dtype=torch.float64); eye[-1,-1] = 1e-4
        weights = torch.linalg.solve(design.T@design/len(design)+cli.ridge*eye,
                                     design.T@target/len(design)).float()
        def global_predict(record):
            x = record["key"].reshape(-1,64)
            return (torch.cat((x,torch.ones(len(x),1)), -1)@weights).reshape(*record["base"].shape)
        global_val, global_test = global_predict(val), global_predict(test)
        global_eta = min((0., .25, .5, 1.), key=lambda e: metrics(val,
                       stable_forecast(val["base"],global_val,basis,e),{})["mse"])
        controls["global_linear"] = metrics(test,
             stable_forecast(test["base"], global_test, basis, global_eta), {})
    write_json(out/"extractor_statistics.json", ex.stats)
    write_json(out/"bank_statistics.json", {"stable":stable_audit,"rare":rare_audit})
    write_json(out/"mapping_oof.json", rel_audit)
    write_json(out/"point_results.json", {"val":vals,"test":tests,"controls":controls,
                                          "eta":eta,"null":null,"basis_rank":basis.shape[-1]})
    group_rows = group_metrics(test, ex, stable, rare, basis, eta, rel, null)
    row_write(out/"common_rare_group_results.csv", group_rows)
    rows = [{"case":cli.case,"method":cli.mode,"variant":variant,
             "val_mse": vals.get(variant,{}).get("mse", ""),
             "val_mae": vals.get(variant,{}).get("mae", ""),
             "test_mse":item["mse"],"test_mae":item["mae"],
             "a0_test_mse":tests["a0"]["mse"],"eta":eta,"null":null}
            for variant,item in {**tests, **controls}.items()]
    row_write(out/"all_results.csv", rows)
    write_json(out/"efficiency.json", {"total_wall_seconds": time.monotonic()-started,
        "test_query_seconds_five_modes":test_query_seconds,
        "test_samples":len(test["key"]),
        "prototype_count":cli.ms+cli.mv,"prototype_bytes":
            stable.centers.numel()*4+rare.centers.numel()*4,
        "mapping_parameters":stable.mapping.numel()+rare.mapping.numel()})
    torch.save({"stable":stable.state(),"rare":rare.state(),"basis":basis,
                "reliability":rel,"eta":eta,"null":null,"extractor":ex,
                "reference_sha256":sha},out/"extractor_memory.pt")
    (out/"extractor_complete").write_text("complete\n",encoding="utf-8")
    print(json.dumps({"case":cli.case,"method":cli.mode,"test":tests,
                      "controls":controls},ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
