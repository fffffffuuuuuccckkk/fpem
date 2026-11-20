"""VAL-selected SMRVA point pilot; no reconstruction GMM or H decomposition.

Frozen A0 supplies only its forecast. All new operators use fixed raw-history
keys, TRAIN historical futures and causal non-overlapping memory anchors.
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
from models.predictive_env.mapping_stability import (
    history_features, history_scale, chronological_blocks, fit_global_ridge,
    fit_shared_mapping, predict_mapping, cross_temporal_audit,
    variation_description, variation_activity, local_variation_mapping,
    coverage_from_ecdf)
from models.predictive_env.mapping_stability.variation_gate import nearest_variation_distance
from models.predictive_env.paired_memory import PairedMemoryBank
from tools.run_fpem_shared_factorization import load_experiment, write_json
from tools.run_predictive_env_iv_patchtst import IndexedDataset


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/fpem_latent_decomposition_search/pilot_20261015"
CASES = ("patchtst_ETTh1_720", "itransformer_ETTm1_720", "moderntcn_ETTm2_720")
MODES = tuple(f"P{i}" for i in range(11))


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("case", choices=CASES)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--output", default="results/fpem_mapping_stability/pilot_20261027")
    p.add_argument("--max_batches", type=int, default=0, help="smoke only")
    p.add_argument("--allow_legacy_reference_signature", action="store_true")
    return p.parse_args()


@torch.no_grad()
def collect_split(model, loader, device, max_batches=0):
    parts = {name: [] for name in ("x", "target", "base", "ids")}
    for batch_index, (x, y, ids, cycle) in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        x = x.float().to(device)
        hidden, mean, std = model.encode(x, cycle.to(device))
        base = model.frozen_h_head(hidden, cycle.to(device)).permute(0, 2, 1) * std + mean
        for key, value in (("x", x), ("target", y[:, -model.pred_len:]),
                           ("base", base), ("ids", ids)):
            parts[key].append(value.detach().cpu())
    return {key: torch.cat(value) for key, value in parts.items()}


def _loader(dataset, args, start, stop):
    wrapped = IndexedDataset(dataset, cycle_len=args.cyclenet_cycle_len)
    return DataLoader(Subset(wrapped, list(range(start, stop))),
                      batch_size=args.batch_size, shuffle=False, num_workers=0)


def _mismatched_indices(bank, retrieved, ids, split, val_start):
    """Deterministic label mismatch inside each query's causally eligible pool."""
    index = retrieved["indices"].cpu()
    mapped = torch.zeros_like(index)
    ends = bank.ends.cpu()
    for bi in range(len(index)):
        if split == "train":
            eligible = (ends <= int(ids[bi])).nonzero(as_tuple=True)[0]
        elif split == "val":
            eligible = (ends <= val_start).nonzero(as_tuple=True)[0]
        else:
            eligible = torch.arange(len(ends))
        if len(eligible) < 2:
            continue
        pos = torch.searchsorted(eligible, index[bi]).clamp_max(len(eligible)-1)
        mapped[bi] = eligible[(pos + max(1, len(eligible)//2)) % len(eligible)]
    return mapped.to(retrieved["indices"].device)


@torch.no_grad()
def train_coverage(bank, records, psi_bank, center, spread, k, device, batch_size=64):
    distances = []
    channel = torch.arange(bank.keys.shape[1], device=device)[None, :, None]
    for ids in torch.arange(len(records["x"])).split(batch_size):
        x = records["x"][ids].to(device)
        starts = records["starts"][ids]
        retrieved = bank.retrieve(x, view="all", common_k=k, rare_k=0,
                                  sample_ids=starts, split="train")
        query = variation_description(x, bank.historical_std_floor,
                                      center, spread)
        source = psi_bank[retrieved["indices"], channel]
        distance = nearest_variation_distance(query, source, retrieved["valid"])
        distances.append(distance.cpu())
    full = torch.cat(distances)
    eligible = torch.isfinite(full).all(-1)
    if not bool(eligible.any()):
        raise RuntimeError("no causal TRAIN neighbours for coverage ECDF")
    return full[eligible].sort(dim=0).values.transpose(0, 1).contiguous().to(device)


@torch.no_grad()
def train_gain_scale(bank, records, u_bank, stable, k, ridge, device):
    channel = torch.arange(bank.keys.shape[1], device=device)[None, :, None]
    gains = []
    # TRAIN-only sub-sample bounds the cost of tuning the smooth sigmoid scale.
    ids = torch.linspace(0, len(records["x"]) - 1,
                         steps=min(128, len(records["x"]))).long().unique()
    for part in ids.split(32):
        x = records["x"][part].to(device)
        retrieved = bank.retrieve(x, view="all", common_k=k, rare_k=0,
                                  sample_ids=records["starts"][part], split="train")
        index = retrieved["indices"]
        source = u_bank[index, channel]
        residual = bank.residual[index, channel]
        residual = residual - torch.einsum("bckf,cft->bckt", source, stable)
        local = local_variation_mapping(history_features(x), source,
                  residual, retrieved["valid"], ridge)
        valid_query = retrieved["valid"].sum(-1) >= 2
        if bool(valid_query.any()):
            gains.append(local["gain"][valid_query].abs().flatten().cpu())
    return max(float(torch.cat(gains).median()) if gains else 0., 1e-4)


@torch.no_grad()
def evaluate(split_record, split, mode, bank, u_bank, psi_bank,
             psi_center, psi_spread, ecdf, stable, global_weight,
             ridge, tau, k, val_start, batch_size=32,
             collect_diagnostics=False):
    device = bank.keys.device
    channel = torch.arange(bank.keys.shape[1], device=device)[None, :, None]
    sums = {"mse": 0., "mae": 0., "a0_mse": 0., "stable_mse": 0.,
            "correction_abs": 0., "coverage": 0., "activity": 0.,
            "mapping_reliability": 0., "gate": 0., "retrieval_distance": 0.,
            "effective_count": 0., "map_loo_error": 0.,
            "constant_loo_error": 0., "future_disagreement": 0.}
    count = 0
    detailed = []
    started = time.monotonic()
    for positions in torch.arange(len(split_record["x"])).split(batch_size):
        x = split_record["x"][positions].to(device)
        truth = split_record["target"][positions].to(device)
        base = split_record["base"][positions].to(device)
        ids = split_record["ids"][positions]
        _, scale = history_scale(x, bank.historical_std_floor)
        query = history_features(x)
        stable_delta = predict_mapping(query, stable)
        if mode == "P0":
            stable_delta = torch.zeros_like(stable_delta)
        if mode == "P1":
            stable_delta = predict_mapping(query, global_weight)
        if mode == "P9":
            stable_delta = torch.zeros_like(stable_delta)
        stable_prediction = base + stable_delta.permute(0, 2, 1) * scale
        output = stable_prediction
        correction = torch.zeros_like(stable_delta)
        coverage = torch.zeros_like(stable_delta[..., 0])
        activity = coverage.clone()
        reliability = torch.zeros_like(stable_delta)
        gate = torch.zeros_like(stable_delta)
        distance = torch.zeros_like(coverage)
        effective = coverage.clone()
        map_error = stable_delta.clone()
        const_error = stable_delta.clone()
        disagreement = stable_delta.clone()
        retrieved = None
        if mode not in ("P0", "P1", "P2"):
            retrieved = bank.retrieve(x, view="random" if mode == "P8" else "all",
                 common_k=k, rare_k=0, sample_ids=ids, split=split,
                 val_first_start=val_start)
            index, valid = retrieved["indices"], retrieved["valid"]
            source = u_bank[index, channel]
            labels = index if mode != "P7" else _mismatched_indices(
                bank, retrieved, ids, split, val_start)
            target = bank.residual[labels, channel] * valid[..., None]
            target = target - torch.einsum("bckf,cft->bckt", source, stable)
            if mode == "P9":
                target = bank.residual[labels, channel] * valid[..., None]
            local = local_variation_mapping(query, source, target, valid, ridge, tau)
            correction = local["constant"] if mode == "P10" else local["candidate"]
            activity = variation_activity(x, bank.historical_std_floor)
            psi_query = variation_description(x, bank.historical_std_floor,
                                               psi_center, psi_spread)
            source_psi = psi_bank[index, channel]
            distance = nearest_variation_distance(psi_query, source_psi, valid)
            coverage = coverage_from_ecdf(distance, ecdf, valid.any(-1))
            reliability = local["map_reliability"]
            effective = valid.float().sum(-1)
            map_error = local["map_loo_error"]
            const_error = local["constant_loo_error"]
            loo = local["loo_residual"].masked_fill(~valid[..., None], float("nan"))
            midpoint = torch.nan_to_num(torch.nanmedian(loo, dim=2).values, nan=0.)
            disagreement = torch.nan_to_num(
                torch.nanmedian((loo - midpoint[:, :, None]).abs(), dim=2).values,
                nan=0.)
            if mode == "P3":
                gate = torch.ones_like(correction)
            elif mode == "P4":
                gate = coverage[..., None].expand_as(correction)
            elif mode == "P5":
                gate = coverage[..., None] * reliability
            else:
                gate = activity[..., None] * coverage[..., None] * reliability
            gate = gate * valid.any(-1)[..., None]
            output = stable_prediction + (gate * correction).permute(0, 2, 1) * scale
        error = (output - truth)
        sums["mse"] += float(error.square().sum())
        sums["mae"] += float(error.abs().sum())
        sums["a0_mse"] += float((base-truth).square().sum())
        sums["stable_mse"] += float((stable_prediction-truth).square().sum())
        sums["correction_abs"] += float((gate*correction).abs().sum())
        sums["coverage"] += float(coverage.sum()) * truth.shape[-2]
        sums["activity"] += float(activity.sum()) * truth.shape[-2]
        sums["mapping_reliability"] += float(reliability.sum())
        sums["gate"] += float(gate.sum())
        sums["retrieval_distance"] += float(retrieved["distance"].masked_fill(
            ~retrieved["valid"], 0).sum()) if retrieved is not None else 0.
        sums["effective_count"] += float(effective.sum()) * truth.shape[-2]
        sums["map_loo_error"] += float(map_error.sum())
        sums["constant_loo_error"] += float(const_error.sum())
        sums["future_disagreement"] += float(disagreement.sum())
        count += truth.numel()
        if collect_diagnostics and retrieved is not None:
            for bi in range(len(x)):
                for ci in range(x.shape[-1]):
                    detailed.append({
                        "sample_id": int(ids[bi]), "channel": ci,
                        "retrieval_indices": json.dumps(retrieved["indices"][bi,ci].tolist()),
                        "retrieval_distance": float(retrieved["distance"][bi,ci].mean()),
                        "effective_count": int(effective[bi,ci]),
                        "variation_distance": float(distance[bi,ci]),
                        "variation_activity": float(activity[bi,ci]),
                        "variation_coverage": float(coverage[bi,ci]),
                        "mapping_reliability": float(reliability[bi,ci].mean()),
                        "variation_gate": float(gate[bi,ci].mean()),
                        "stable_correction_abs": float(stable_delta[bi,ci].abs().mean()),
                        "variation_candidate_abs": float(correction[bi,ci].abs().mean()),
                        "future_disagreement": float(disagreement[bi,ci].mean()),
                        "map_loo_error": float(map_error[bi,ci].mean()),
                        "constant_loo_error": float(const_error[bi,ci].mean()),
                        "base_mse": float((base[bi,:,ci]-truth[bi,:,ci]).square().mean()),
                        "stable_mse": float((stable_prediction[bi,:,ci]-truth[bi,:,ci]).square().mean()),
                        "final_mse": float(error[bi,:,ci].square().mean()),
                    })
    result = {key: value / count for key, value in sums.items()}
    result["seconds"] = time.monotonic() - started
    result["sample_count"] = len(split_record["x"])
    if not all(torch.isfinite(torch.tensor(value)) for value in result.values()):
        raise FloatingPointError("SMRVA metrics non-finite")
    return result, detailed


def main():
    cli = arguments()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    out = ROOT / cli.output / cli.case
    out.mkdir(parents=True, exist_ok=True)
    if (out / "point_pilot_complete").exists():
        print("point pilot already complete", flush=True)
        return
    cli.base_run_config = str(BASE / (f"{cli.case}_A0_complementary_gate_predictive_"
                                      "sub0_orth0_refnone_fp0_rank8_bias-4/run_config.json"))
    if cli.case.startswith("moderntcn"):
        cli.allow_legacy_reference_signature = True
    args, model, loaders, device, ref_hash = load_experiment(cli)
    source_cache = ROOT / "results/fpem_predictive_memory/pilot_20261024" / cli.case / "phase0_cache.pt"
    stored = torch.load(source_cache, map_location="cpu", weights_only=False)
    if stored.get("reference_sha256") != ref_hash:
        raise ValueError("TRAIN anchor/A0 checkpoint SHA mismatch")
    records = stored["records"]
    bank = PairedMemoryBank(args.seq_len, args.pred_len, history_size=64)
    bank_audit = bank.fit(records)
    bank.to(device)
    u_bank = torch.cat((bank.keys, torch.ones_like(bank.keys[..., :1])), -1)
    psi_raw = variation_description(records["x"].to(device),
                                     bank.historical_std_floor)
    psi_center = psi_raw.mean(0)
    psi_spread = psi_raw.std(0, unbiased=False).clamp_min(1e-3)
    psi_bank = ((psi_raw - psi_center[None]) / psi_spread[None]).clamp(-20, 20)
    block_ids, block_valid = chronological_blocks(records["starts"], args.seq_len,
                                                   args.pred_len, 4)
    u_cpu = u_bank.cpu()
    residual_cpu = bank.residual.cpu()
    train_data = loaders["train"].dataset.dataset
    val_data = loaders["val"].dataset.dataset
    val_start = len(train_data.data_x) - args.seq_len
    half = max(1, len(val_data)//2)
    validation = collect_split(model, _loader(val_data, args, 0, half), device,
                                cli.max_batches)
    test = collect_split(model, loaders["test"], device, cli.max_batches)
    zero = torch.zeros(bank.keys.shape[1], u_bank.shape[-1], args.pred_len,
                       device=device)
    ecdfs = {k: train_coverage(bank, records, psi_bank, psi_center, psi_spread,
                               k, device) for k in (5,8,16)}
    def score(split, mode, shared=zero, ordinary=zero, k=5, ridge=100., tau=1.,
              detail=False):
        return evaluate(split, "val" if split is validation else "test", mode,
            bank, u_bank, psi_bank, psi_center, psi_spread, ecdfs[k], shared,
            ordinary, ridge, tau, k, val_start, args.batch_size, detail)
    a0, _ = score(validation, "P0")
    global_candidates = []
    for lam in (10., 100., 1000.):
        weight = fit_global_ridge(u_cpu, residual_cpu, lam).to(device)
        metric, _ = score(validation, "P1", ordinary=weight)
        global_candidates.append((metric["mse"], lam, weight))
    _, global_lam, global_weight = min(global_candidates, key=lambda item: item[0])
    stable_candidates = []
    for lam_s in (10., 100., 1000.):
        for lam_v in (100., 1000.):
            weight, offsets = fit_shared_mapping(u_cpu, residual_cpu, block_ids,
                                       block_valid, lam_s, lam_v)
            metric, _ = score(validation, "P2", shared=weight.to(device))
            stable_candidates.append((metric["mse"], lam_s, lam_v,
                                      weight.to(device), offsets))
    _, lam_s, lam_v, stable_fit, offsets = min(stable_candidates, key=lambda item: item[0])
    transfer = cross_temporal_audit(u_cpu, residual_cpu, block_ids, block_valid,
                                    lam_s, lam_v, global_lam)
    stable_enabled = min(stable_candidates, key=lambda item: item[0])[0] < a0["mse"]
    stable = stable_fit if stable_enabled else zero
    write_json(out / "stable_mapping_audit.json", {
        "reference_sha256": ref_hash, "train_only": True,
        "block_ids": block_ids.tolist(), "purged_anchor_count": int((~block_valid).sum()),
        "lambda_shared": lam_s, "lambda_private": lam_v,
        "lambda_global": global_lam, "stable_enabled_by_val": stable_enabled,
        "shared_norm": float(stable_fit.norm()), "private_norm": float(offsets.norm()),
        "private_zero_sum_error": float(offsets.sum(0).abs().max()),
        "cross_temporal_transfer": transfer, "memory": bank_audit,
    })
    write_json(out / "run_config.json", {"case": cli.case, "seed": cli.seed,
        "reference_sha256": ref_hash, "backbone_frozen": True,
        "feature": "per-channel normalized raw history interpolated to 64 + bias",
        "chronological_blocks": 4, "train_anchor_stride": args.seq_len,
        "topk_candidates": [5,8,16], "ridge_candidates": [10,100,1000],
        "selection": "first half VAL MSE; TEST only evaluated after mode config fixed",
        "probability_phase": "not launched by this point pilot"})
    torch.save({"reference_sha256": ref_hash, "shared": stable.cpu(),
                "shared_unfiltered": stable_fit.cpu(), "private_audit_only": offsets.cpu(),
                "global": global_weight.cpu(), "psi_center": psi_center.cpu(),
                "psi_spread": psi_spread.cpu(),
                "coverage_ecdf": {k: value.cpu() for k, value in ecdfs.items()},
                "bank_state": bank.state_dict()}, out / "mapping_cache.pt")
    rows = []
    configs = {}
    for mode in MODES:
        result_path = out / f"{mode}_result.json"
        if result_path.exists():
            saved = json.loads(result_path.read_text(encoding="utf-8"))
            rows.append(saved)
            configs[mode] = saved["config"]
            print(json.dumps({"mode": mode, "reused": True}), flush=True)
            continue
        if mode in ("P0", "P1", "P2"):
            configs[mode] = {"k": 5, "ridge": 100., "tau": 1.}
        elif mode in ("P7", "P8", "P10"):
            configs[mode] = dict(configs["P6"])
        elif mode == "P9":
            configs[mode] = dict(configs["P6"])
        else:
            candidates = []
            for k in (5,8,16):
                for ridge in (10., 100., 1000.):
                    tau = train_gain_scale(bank, records, u_bank, stable, k, ridge, device)
                    metric, _ = score(validation, mode, stable, global_weight,
                                      k, ridge, tau)
                    candidates.append((metric["mse"], k, ridge, tau))
            _, k, ridge, tau = min(candidates, key=lambda item: item[0])
            configs[mode] = {"k": k, "ridge": ridge, "tau": tau}
        config = configs[mode]
        val, _ = score(validation, mode, stable, global_weight, **config)
        test_metrics, details = score(test, mode, stable, global_weight,
                                      **config, detail=mode == "P6")
        result = {"case": cli.case, "mode": mode, "config": config,
                  "val": val, "test": test_metrics, "reference_sha256": ref_hash,
                  "stable_mapping_enabled": stable_enabled}
        write_json(result_path, result)
        rows.append(result)
        if details:
            with (out / "P6_sample_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=details[0].keys())
                writer.writeheader(); writer.writerows(details)
        print(json.dumps({"mode": mode, "val_mse": val["mse"],
                          "test_mse": test_metrics["mse"]}), flush=True)
    with (out / "all_point_results.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ("case", "mode", "k", "ridge", "tau", "val_mse", "val_mae",
                  "test_mse", "test_mae", "a0_test_mse", "stable_enabled")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({"case": cli.case, "mode": row["mode"],
                **row["config"], "val_mse": row["val"]["mse"],
                "val_mae": row["val"]["mae"], "test_mse": row["test"]["mse"],
                "test_mae": row["test"]["mae"],
                "a0_test_mse": row["test"]["a0_mse"],
                "stable_enabled": stable_enabled})
    by_mode = {row["mode"]: row for row in rows}
    a, b, c, d = (by_mode[name]["val"]["mse"] for name in ("P6","P7","P8","P10"))
    ready = a < min(b,c,d,by_mode["P0"]["val"]["mse"])
    write_json(out / "point_selection.json", {
        "selected_mode": min((row for row in rows if row["mode"] in
                              {"P0","P1","P2","P3","P4","P5","P6","P9"}),
                              key=lambda row: row["val"]["mse"])["mode"],
        "P6_passes_key_val_controls": ready, "P6_val_mse": a,
        "P7_shuffled_val_mse": b, "P8_random_val_mse": c,
        "P10_constant_val_mse": d, "A0_val_mse": by_mode["P0"]["val"]["mse"],
        "reference_sha256": ref_hash})
    lines = [f"# SMRVA point pilot: {cli.case}", "",
             "TEST was not used for configuration/checkpoint selection.",
             f"Cross-temporal shared mapping enabled by VAL: {stable_enabled}.",
             f"P6 passes A0/shuffled/random/constant VAL controls: {ready}.", "",
             "| Mode | VAL MSE | TEST MSE | TEST MAE |",
             "|---|---:|---:|---:|"]
    lines += [f"| {row['mode']} | {row['val']['mse']:.6f} | "
              f"{row['test']['mse']:.6f} | {row['test']['mae']:.6f} |" for row in rows]
    lines += ["", "Cross-temporal TRAIN-only transfer:", ""]
    lines += [f"- block {r['holdout_block']}: A0={r['a0_normalized_mse']:.6f}, "
              f"ordinary={r['ordinary_normalized_mse']:.6f}, "
              f"shared={r['shared_normalized_mse']:.6f}" for r in transfer]
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "point_pilot_complete").touch()


if __name__ == "__main__":
    main()
