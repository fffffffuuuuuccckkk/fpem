"""VAL-selected local-operator pilot on three existing A0 checkpoints.

P0-P8 share one frozen A0 and real TRAIN historical/future memory. TEST is
evaluated only after each row's VAL configuration is locked. A0's TRAIN error
is not misrepresented as a strict A0-OOF estimate.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.predictive_env.consensus_adaptation import local_operator_adapt
from models.predictive_env.consensus_adaptation.local_head_adapter import unified_operator_adapt
from models.predictive_env.future_consensus import future_consensus
from models.predictive_env.learned_memory import LearnabilityMemory, fit_learnability
from tools.run_fpem_shared_factorization import load_experiment, write_json
from tools.run_predictive_env_iv_patchtst import IndexedDataset


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/fpem_latent_decomposition_search/pilot_20261015"
CASES = ("patchtst_ETTh1_720", "itransformer_ETTm1_720", "moderntcn_ETTm2_720")
POLICIES = {
    "P0": ("none", "none", False, False, False),
    "P1": ("none", "all", False, True, False),
    "P2": ("joint", "rare", False, False, False),
    "P3": ("joint", "mixed", True, False, False),
    "P4": ("random", "mixed", True, False, False),
    "P5": ("reconstruction", "mixed", True, False, False),
    "P6": ("joint", "mixed", True, False, False),
    "P7": ("joint", "mixed", True, False, True),
    "P8": ("joint", "random_group", True, False, False),
    "P9_density": ("density", "mixed", True, False, False),
}
NEW_MODES = ("A0", "A1", "A2", "A3", "A4", "B0", "B1", "B2", "B3",
             "B4", "B5", "B6", "B7", "B8")


def new_mode_retrieval(bank, x, ids, mode, split, val_start, topk):
    """Retrieve first, then read reconstruction posterior. No class quota."""
    view = "random" if mode == "B7" else "all"
    result = bank.retrieve(x, view=view, common_k=topk, rare_k=0,
                           sample_ids=ids, split=split,
                           val_first_start=val_start)
    if mode == "B6":
        # The entire retrieved-future multiset must change: merely rolling
        # within Top-K leaves its median/MAD consensus exactly unchanged.
        index = result["indices"].cpu()
        valid = result["valid"].cpu()
        ends = bank.ends.cpu()
        mapped = torch.zeros_like(index)
        for bi in range(len(index)):
            if split == "train":
                eligible = (ends <= int(ids[bi])).nonzero(as_tuple=True)[0]
            elif split == "val":
                eligible = (ends <= int(val_start)).nonzero(as_tuple=True)[0]
            else:
                eligible = torch.arange(len(ends))
            if len(eligible) < 2:
                continue
            shift = max(1, len(eligible) // 2)
            for ci in range(index.shape[1]):
                chosen = index[bi, ci]
                positions = torch.searchsorted(eligible, chosen).clamp_max(len(eligible) - 1)
                mapped[bi, ci] = eligible[(positions + shift) % len(eligible)]
        device = result["indices"].device
        mapped = mapped.to(device)
        channel = torch.arange(index.shape[1], device=device)[None, :, None]
        result["future"] = bank.future.to(device)[mapped, channel] * result["valid"][..., None]
        result["residual_indices"] = mapped
        result["label_source_ends"] = bank.ends.to(device)[mapped]
    return result


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("case", choices=CASES)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output", default="results/fpem_learnability/pilot_20261026")
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--stable_k", type=int, default=3)
    p.add_argument("--under_k", type=int, default=2)
    p.add_argument("--reconstruction_epochs", type=int, default=20)
    p.add_argument("--max_batches", type=int, default=0, help="smoke only; 0 means full")
    p.add_argument("--allow_legacy_reference_signature", action="store_true")
    p.add_argument("--protocol", choices=("legacy", "unified"), default="legacy")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--huber_delta", type=float, default=1.)
    p.add_argument("--lambda_huber", type=float, default=1.)
    return p.parse_args()


def _source_residual(bank: LearnabilityMemory, retrieved: dict, records: dict):
    # Exact A0 source residual in the reference head's normalized coordinates.
    source = bank.source_residual_modelnorm
    channel = torch.arange(source.shape[1], device=retrieved["indices"].device)[None, :, None]
    indices = retrieved.get("residual_indices", retrieved["indices"])
    return source.to(channel.device)[indices, channel] * retrieved["valid"][..., None]


@torch.no_grad()
def run_batch(model, bank, records, mode, config, x, ids, cycle, split, val_start,
              stable_k=3, under_k=2):
    hidden, mean, std = model.encode(x, cycle)
    base = model.frozen_h_head(hidden, cycle)
    original = base.permute(0, 2, 1) * std + mean
    if mode in ("P0", "A0"):
        return original, original, {"update_norm": 0., "valid_fraction": 0.}
    if mode == "A1":
        return run_batch(model, bank, records, "P5", config, x, ids, cycle,
                         split, val_start, stable_k, under_k)
    if mode in NEW_MODES:
        group = "random_reconstruction" if mode == "A4" else (
            "none" if mode == "A3" else "reconstruction")
        bank.set_group(config["learnability"], group, config["seed"])
        retrieved = new_mode_retrieval(
            bank, x, ids, mode, split, val_start, config.get("topk", 5))
        source_residual = _source_residual(bank, retrieved, records)
        consensus = future_consensus(retrieved["future"], retrieved["valid"],
                   bank.typical_future_disagreement)
        adapter_mode = ("legacy_protect" if mode in ("A2", "A4", "B0") else
                        "all_huber" if mode == "B8" else
                        "both_square" if mode in ("A3", "B1") else
                        "well_square_poor_huber")
        delta, diagnostics = unified_operator_adapt(
            hidden, retrieved["hidden"], source_residual, retrieved["valid"],
            retrieved["p_well"], consensus["weight"],
            lambda_ridge=config["lambda_ridge"],
            lambda_huber=config.get("lambda_huber", 1.),
            huber_delta=config.get("huber_delta", 1.),
            mode=adapter_mode, use_loss_weight=mode in ("B3", "B5", "B6", "B7"),
            use_output_weight=mode in ("B4", "B5", "B6", "B7"),
            shuffle_future=False)
        # The retrieved future and residual use this TRAIN-history-derived
        # floor, so the query correction must use the same coordinate system.
        correction_scale = std.clamp_min(config["learnability"]["history_std_floor"])
        output = original + delta.permute(0, 2, 1) * correction_scale
        diagnostics["retrieved_count"] = float(consensus["count"].float().mean())
        diagnostics["future_disagreement"] = float(consensus["disagreement"].mean())
        return output, original, diagnostics
    effective = config.get("control_base", mode) if mode in ("P7", "P8") else mode
    group, view, protect, all_supervised, _ = POLICIES[effective]
    if mode == "P8":
        view = "random_group"
    shuffle = mode == "P7"
    bank.set_group(config["learnability"], group, config["seed"])
    sk = 0 if view in ("rare", "all") else stable_k
    uk = (stable_k + under_k) if view == "all" else under_k
    retrieved = bank.retrieve(x, view=view, common_k=sk, rare_k=uk,
                              sample_ids=ids, split=split,
                              val_first_start=val_start)
    source_residual = _source_residual(bank, retrieved, records)
    delta, diagnostics = local_operator_adapt(
        hidden, retrieved["hidden"], source_residual, retrieved["valid"],
        retrieved["stable_probability"], retrieved["underlearned_probability"],
        stable_k=sk, lambda_stable=config["lambda_stable"],
        lambda_ridge=config["lambda_ridge"], protect=protect,
        only_under=mode == "P2", all_supervised=all_supervised,
        shuffle_future=shuffle)
    if not torch.isfinite(delta).all():
        raise FloatingPointError("local ridge produced a non-finite correction")
    output = (base + delta).permute(0, 2, 1) * std + mean
    return output, original, diagnostics


@torch.no_grad()
def evaluate(model, bank, records, mode, config, loader, split, val_start,
             max_batches=0, stable_k=3, under_k=2):
    device = next(model.parameters()).device
    total = abs_total = original_total = original_abs = 0.
    count = 0
    summary = {}
    started = time.monotonic()
    for batch_number, (x, y, ids, cycle) in enumerate(loader):
        if max_batches and batch_number >= max_batches:
            break
        x = x.float().to(device)
        cycle = cycle.to(device)
        target = y[:, -model.pred_len:].float().to(device)
        output, original, diagnostics = run_batch(
            model, bank, records, mode, config, x, ids, cycle, split, val_start,
            stable_k, under_k)
        total += float((output - target).square().sum())
        abs_total += float((output - target).abs().sum())
        original_total += float((original - target).square().sum())
        original_abs += float((original - target).abs().sum())
        count += target.numel()
        for key, value in diagnostics.items():
            summary[key] = summary.get(key, 0.) + float(value) * len(x)
        summary["sample_count"] = summary.get("sample_count", 0) + len(x)
    if count == 0:
        raise RuntimeError("empty evaluation split")
    samples = summary.pop("sample_count")
    return {"mse": total / count, "mae": abs_total / count,
            "a0_mse": original_total / count, "a0_mae": original_abs / count,
            "time_seconds": time.monotonic() - started,
            **{key: value / samples for key, value in summary.items()}}


def _loader(dataset, args, start=0, stop=None):
    if stop is None:
        stop = len(dataset)
    indices = range(start, stop)
    return DataLoader(Subset(IndexedDataset(dataset, cycle_len=args.cyclenet_cycle_len),
                             list(indices)), batch_size=args.batch_size,
                      shuffle=False, num_workers=0)


def _audit_group(bank, fitted, method, seed):
    bank.set_group(fitted, method, seed)
    stable = bank.common.float().mean(1).cpu()
    rec = fitted["reconstruction_error"]
    pred = fitted["train_fitted_prediction_error"]
    mask = stable >= .5
    def stat(values, selection):
        return float(values[selection].mean()) if bool(selection.any()) else None
    return {"group": method, "stable_count": int(mask.sum()),
            "underlearned_count": int((~mask).sum()),
            "stable_rec_error": stat(rec, mask), "under_rec_error": stat(rec, ~mask),
            "stable_pred_error": stat(pred, mask), "under_pred_error": stat(pred, ~mask),
            "stable_first_half_fraction": float(mask[:len(mask)//2].float().mean()),
            "stable_second_half_fraction": float(mask[len(mask)//2:].float().mean())}


def main():
    cli = arguments()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    out = ROOT / cli.output / cli.case
    out.mkdir(parents=True, exist_ok=True)
    if (out / "point_pilot_complete").exists():
        print("point pilot already complete", flush=True)
        return
    base = BASE / (f"{cli.case}_A0_complementary_gate_predictive_"
                   "sub0_orth0_refnone_fp0_rank8_bias-4/run_config.json")
    cli.base_run_config = str(base)
    if cli.case.startswith("moderntcn"):
        cli.allow_legacy_reference_signature = True
    args, model, loaders, device, ref_hash = load_experiment(cli)
    anchor = ROOT / "results/fpem_predictive_memory/pilot_20261024" / cli.case / "phase0_cache.pt"
    stored = torch.load(anchor, map_location="cpu", weights_only=False)
    if stored.get("reference_sha256") != ref_hash:
        raise ValueError("TRAIN anchor cache/A0 mismatch")
    records = stored["records"]
    cache = out / "learnability.pt"
    if cache.exists():
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        if payload["reference_sha256"] != ref_hash:
            raise ValueError("cached learnability/A0 mismatch")
        fitted = payload["learnability"]
        bank_state = payload["bank"]
        bank = LearnabilityMemory(args.seq_len, args.pred_len)
        for name, value in bank_state.items():
            setattr(bank, name, value.clone())
        memory_audit = payload["memory_audit"]
    else:
        bank = LearnabilityMemory(args.seq_len, args.pred_len)
        memory_audit = bank.fit(records)
        fitted = fit_learnability(records, cli.seed, cli.reconstruction_epochs, device=device)
        torch.save({"reference_sha256": ref_hash, "learnability": fitted,
                    "bank": bank.state_dict(), "memory_audit": memory_audit}, cache)
    bank.to(device)
    source_x = records["x"].float().cpu()
    source_std = source_x.var(1, keepdim=True, unbiased=False).add(1e-5).sqrt()
    source_std = source_std.clamp_min(fitted["history_std_floor"])
    bank.source_residual_modelnorm = (
        (records["target"].float().cpu() - records["a0_pred"].float().cpu()) /
        source_std).transpose(1, 2).contiguous().to(device)
    bank.future = ((records["target"].float().cpu() - source_x.mean(1, keepdim=True)) /
                   source_std).transpose(1, 2).contiguous().to(device)
    bank.typical_future_disagreement = (
        bank.future - bank.future.median(dim=0).values[None]).abs().median().clamp_min(1e-4)
    audit = {"reference_sha256": ref_hash, "memory": memory_audit,
             "prediction_error_provenance": fitted["prediction_error_provenance"],
             "gmm_rec": fitted["gmm_rec"], "gmm_joint": fitted["gmm_joint"],
             "groups": [_audit_group(bank, fitted, method, cli.seed) for method in
                        ("density", "reconstruction", "joint", "random", "none")]}
    write_json(out / "learnability_audit.json", audit)
    write_json(out / "run_config.json", {
        "protocol": cli.protocol, "case": cli.case, "seed": cli.seed,
        "reference_sha256": ref_hash, "topk": cli.topk,
        "reconstruction_epochs": cli.reconstruction_epochs,
        "lambda_huber": cli.lambda_huber, "huber_delta": cli.huber_delta,
        "backbone_frozen": True, "group_source": "TRAIN reconstruction only",
        "gmm_two_components_bic_supported": fitted["gmm_rec"]["two_components_bic_supported"],
        "selection": "first half VAL MSE, one checkpoint/config for MSE and MAE",
        "test_selection": False,
    })
    train_data = loaders["train"].dataset.dataset
    val_data = loaders["val"].dataset.dataset
    val_start = len(train_data.data_x) - args.seq_len
    val_select = _loader(val_data, args, 0, max(1, len(val_data) // 2))
    test = loaders["test"]
    rows = []
    selected_main_config = None
    main_modes = (("P0", "P1", "P2", "P3", "P4", "P5", "P6", "P9_density")
                  if cli.protocol == "legacy" else NEW_MODES)
    for mode in ((*main_modes, "P7", "P8") if cli.protocol == "legacy" else main_modes):
        result_path = out / f"{mode}_result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            rows.append(result)
            print(json.dumps({"mode": mode, "reused": True}), flush=True)
            continue
        if mode == "P6":
            # P3 and P6 are mathematically identical in the supplied task:
            # both are joint-GMM Under-Learned adaptation + Stable protection.
            # Reuse rather than rerun different RNG and imply a false ablation.
            reused = next(row for row in rows if row["mode"] == "P3")
            result = {**reused, "mode": mode,
                      "identical_to_P3": True}
            write_json(result_path, result)
            rows.append(result)
            continue
        if mode in ("P7", "P8", "B6", "B7"):
            main = min((row for row in rows if row["mode"] in
                        ({"P1", "P2", "P3", "P5", "P6", "P9_density"}
                         if cli.protocol == "legacy" else {"B5"})),
                       key=lambda row: row["val"]["mse"])
            candidates = [(main["ridge"], main["stable_weight"])]
        elif mode in ("P0", "A0"):
            candidates = [(1000., 1.)]
        else:
            candidates = [
                (ridge, stable) for ridge in (100., 1000., 10000., 100000.)
                for stable in ((.1, 1., 10.) if cli.protocol == "legacy" and
                                POLICIES[mode][2] else (1.,))]
        comparisons = []
        for ridge, stable in candidates:
            config = {"learnability": fitted, "seed": cli.seed,
                      "lambda_ridge": ridge, "lambda_stable": stable,
                      "topk": cli.topk, "lambda_huber": cli.lambda_huber,
                      "huber_delta": cli.huber_delta}
            if mode in ("P7", "P8"):
                config["control_base"] = main["mode"]
            metrics = evaluate(model, bank, records, mode, config, val_select,
                               "val", val_start, cli.max_batches,
                               cli.stable_k, cli.under_k)
            comparisons.append((metrics["mse"], ridge, stable, metrics))
        _, ridge, stable, val = min(comparisons, key=lambda row: row[0])
        config = {"learnability": fitted, "seed": cli.seed,
                  "lambda_ridge": ridge, "lambda_stable": stable,
                  "topk": cli.topk, "lambda_huber": cli.lambda_huber,
                  "huber_delta": cli.huber_delta}
        if mode in ("P7", "P8"):
            config["control_base"] = main["mode"]
        metrics = evaluate(model, bank, records, mode, config, test,
                           "test", val_start, cli.max_batches,
                           cli.stable_k, cli.under_k)
        source_mode = main["mode"] if mode in ("P7", "P8", "B6", "B7") else mode
        group = (POLICIES[source_mode][0] if cli.protocol == "legacy" else
                 "random_reconstruction" if mode == "A4" else
                 "none" if mode == "A3" else "reconstruction")
        view = (POLICIES[source_mode][1] if cli.protocol == "legacy" else
                "random" if mode == "B7" else "all")
        result = {"case": cli.case, "mode": mode, "group": group,
                  "view": view,
                  "control_base": source_mode if mode in ("P7", "P8", "B6", "B7") else None,
                  "ridge": ridge,
                  "stable_weight": stable, "val": val, "test": metrics,
                  "selection": "first half of VAL", "reference_sha256": ref_hash,
                  "protocol": cli.protocol, "topk": cli.topk,
                  "lambda_huber": cli.lambda_huber, "huber_delta": cli.huber_delta}
        write_json(result_path, result)
        rows.append(result)
        print(json.dumps({"mode": mode, "val_mse": val["mse"],
                          "test_mse": metrics["mse"]}), flush=True)
    flat = [{"case": row["case"], "mode": row["mode"], "group": row["group"],
             "ridge": row["ridge"], "stable_weight": row["stable_weight"],
             "val_mse": row["val"]["mse"], "val_mae": row["val"]["mae"],
             "test_mse": row["test"]["mse"], "test_mae": row["test"]["mae"],
             "a0_test_mse": row["test"]["a0_mse"],
             "a0_test_mae": row["test"]["a0_mae"],
             "update_norm": row["test"].get("update_norm", 0.),
             "under_before_mse": row["test"].get("under_before_mse", 0.),
             "under_after_mse": row["test"].get("under_after_mse", 0.),
             "stable_shift_mse": row["test"].get("stable_prediction_shift_mse", 0.)}
            for row in rows]
    with (out / "all_point_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=flat[0].keys())
        writer.writeheader()
        writer.writerows(flat)
    candidate_modes = ({"P0", "P1", "P2", "P3", "P5", "P6", "P9_density"}
                       if cli.protocol == "legacy" else
                       {"A0", "A1", "A2", "A3", "B0", "B1", "B2", "B3", "B4", "B5", "B8"})
    winner = min((row for row in rows if row["mode"] in candidate_modes),
                 key=lambda row: row["val"]["mse"])
    write_json(out / "point_selection.json", {"selected_mode": winner["mode"],
               "selected_val_mse": winner["val"]["mse"],
               "selected_test_mse": winner["test"]["mse"],
               "a0_test_mse": winner["test"]["a0_mse"],
               "point_improved_on_val": winner["val"]["mse"] < rows[0]["val"]["mse"],
               "reference_sha256": ref_hash, "protocol": cli.protocol})
    (out / "point_pilot_complete").touch()


if __name__ == "__main__":
    main()
