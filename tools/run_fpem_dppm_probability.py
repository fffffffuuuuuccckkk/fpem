"""E3: A0-anchored, strictly chronological residual-Flow memory ablation.

The A0 used here is trained only on the early TRAIN core. Later TRAIN windows
provide out-of-fit residual statistics and Flow targets. The same model, Flow
architecture, budget and calibration protocol are used for U0-U4.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.predictive_env.conditional_innovation_prior import strict_chronological_indices
from models.predictive_env.residual_difference.condition import ResidualDifferenceCondition
from models.predictive_env.probability_heads import ProbabilityHeadBank
from models.predictive_env.paired_memory.memory_bank import history_key
from tools.evaluate_fpem_probabilistic import probabilistic_metrics
from tools.run_fpem_csif_probabilistic import base_data, fresh_core_model, subset_loader, core_forecast
from tools.run_fpem_dppm import BASE, CASES, normalized_record
from tools.run_fpem_dppm_next import fit_memory, history_description, mapping_feature
from tools.run_fpem_shared_factorization import write_json

SOURCE = ROOT / "results/fpem_dual_pattern_memory/next_pilot"
CORE = ROOT / "results/csif_factor_innovation/probabilistic_20261022"
OUTPUT = "results/fpem_dual_pattern_memory/validation_pilot"
MODES = ("U0", "U1", "U2", "U3", "U4")
METRIC_NAMES = ("inv_MSE", "inv_MAE", "mean_MSE", "mean_MAE", "CRPS",
                "WIS", "PICP_95", "MPIW_95", "sample_std_mean")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("case", choices=CASES)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--output", default=OUTPUT)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--flow_steps", type=int, default=12)
    p.add_argument("--max_batches", type=int, default=0, help="smoke only")
    return p.parse_args()


def core_setup(cli, out):
    cli.base_run_config = str(BASE / (f"{cli.case}_A0_complementary_gate_predictive_"
                             "sub0_orth0_refnone_fp0_rank8_bias-4/run_config.json"))
    args, datasets, device = base_data(cli)
    original = CORE / cli.case / "core_only_a0.pt"
    if not original.exists():
        raise FileNotFoundError(f"strict core-only A0 required: {original}")
    state = torch.load(original, map_location="cpu", weights_only=False)
    metadata = state["metadata"]
    core, cal, cut = strict_chronological_indices(len(datasets["train"]),
                                                   args.seq_len, args.pred_len)
    if metadata["boundary"] != cut or metadata["core_count"] != len(core):
        raise ValueError("existing core-only A0 has different chronological split")
    if metadata["core_target_last"] >= metadata["calibration_target_first"]:
        raise ValueError("core-only A0 targets overlap held-out TRAIN windows")
    center = np.asarray(metadata["core_only_scaler_mean"], dtype=np.float32)
    std = np.asarray(metadata["core_only_scaler_std"], dtype=np.float32)
    for dataset in datasets.values():
        raw = dataset.inverse_transform(dataset.data_x).astype(np.float32)
        scaled = (raw - center) / std
        dataset.data_x = scaled
        dataset.data_y = scaled
    model = fresh_core_model(args, cli.seed, device)
    model.load_state_dict(state["model"])
    model.eval().requires_grad_(False)
    write_json(out / "core_a0_provenance.json", {
        "source_checkpoint": str(original), "metadata": metadata,
        "note": "core-only A0 is common to U0-U4; not identical to full-TRAIN A0"})
    return args, datasets, device, model, core, cal


@torch.no_grad()
def collect(model, dataset, indices, args, device, seed, max_batches=0):
    values = {name: [] for name in ("key", "truth", "base", "scale", "ids")}
    loader = subset_loader(dataset, list(map(int, indices)), args, seed, False)
    for step, (x, y, ids, cycle) in enumerate(loader):
        if max_batches and step >= max_batches:
            break
        x = x.float().to(device)
        y = y[:, -args.pred_len:].float().to(device)
        base = core_forecast(model, x, cycle.to(device))
        values["key"].append(history_key(x.detach().cpu(), 64))
        values["truth"].append(y.cpu())
        values["base"].append(base.cpu())
        values["scale"].append(x.std(1, keepdim=True, unbiased=False)
                                .clamp_min(1e-4).cpu())
        values["ids"].append(ids.cpu())
    if not values["key"]:
        raise ValueError("empty chronological data split")
    return {name: torch.cat(parts) for name, parts in values.items()}


def split_heldout(cal, seq_len, pred_len):
    # A complete gap ensures a residual-statistics future cannot overlap a
    # Flow-training history or its future label.
    # Long horizons leave little held-out TRAIN after a full no-overlap gap.
    # Keep a modest chronological prefix for statistics and the remainder for
    # Flow; record the effective independent support rather than overclaiming.
    pivot = int(cal[min(len(cal) - 16, seq_len + pred_len + 64)])
    sources = [int(i) for i in cal if int(i) + seq_len + pred_len <= pivot]
    flow = [int(i) for i in cal if int(i) >= pivot]
    if len(sources) < 8 or len(flow) < 8:
        raise ValueError(f"too few causally separated TRAIN rows for E3: "
                         f"cal={len(cal)} source={len(sources)} flow={len(flow)} "
                         f"seq={seq_len} pred={pred_len}")
    return sources, flow


def split_validation(n, seq_len, pred_len):
    pivot = n // 2
    selection = list(range(0, pivot - seq_len - pred_len))
    calibration = list(range(pivot, n))
    if len(selection) < 8 or len(calibration) < 8:
        raise ValueError("validation too short for independent selection/calibration")
    return selection, calibration


def build_banks(core_data, settings, seed):
    # Bank centers are fit only on the A0 core, before OOF residual-stat rows.
    # Targets are core TRAIN future labels; no held-out or TEST targets fit banks.
    dummy_x = core_data["key"].transpose(1, 2)
    dummy_y = core_data["truth"]
    floor = torch.ones(dummy_x.shape[-1]) * 1e-4
    record = normalized_record({"x": dummy_x, "target": dummy_y,
                                "base": core_data["base"]}, floor)
    # normalized_record expects an actual history window. Use the already
    # normalized key directly for bank centers; the future target only fits
    # unused mapping matrices and cannot affect retrieval weights.
    record["key"] = core_data["key"]
    record["truth"] = core_data["truth"].transpose(1, 2)
    blocks = (torch.arange(len(record["key"])) * 6 // len(record["key"])).clamp_max(5)
    return fit_memory(torch.ones(len(blocks), dtype=torch.bool), record,
                      blocks, settings, seed)


def soft_assign(key, stable, variation, settings):
    stable_weight = stable.query(key)["weight"]
    description = history_description(key, stable,
                                      settings.get("representation", "V0"))
    var_feature = mapping_feature(key, description,
                                  settings.get("input", "concat"))
    var_weight = variation.query(description, var_feature)["weight"]
    return stable_weight, var_weight


def prototype_statistics(residual, weights):
    # Residual rows are genuinely out-of-fit for the core-only A0. The
    # prototype bank was built strictly earlier. Use sample-level matching;
    # both banks can describe one sample, so their variances are never added.
    n, c, horizon = residual.shape
    k = weights.shape[-1]
    flat = residual.reshape(n * c, horizon)
    assignment = weights.reshape(n * c, k).argmax(-1)
    output = torch.zeros(k, horizon, 7)
    supports = []
    for idx in range(k):
        selected = flat[assignment == idx]
        supports.append(int(len(selected)))
        if not len(selected):
            continue
        center = selected.median(0).values
        mad = (selected - center).abs().median(0).values
        lower = torch.quantile(selected, .1, 0)
        upper = torch.quantile(selected, .9, 0)
        output[idx] = torch.stack((center, mad,
            selected.clamp_min(0).mean(0),
            selected.clamp_max(0).mean(0), lower, upper,
            torch.full_like(center, len(selected) / max(1, len(flat)))), -1)
    return output, supports


def features(record, banks, settings, stable_stats, variation_stats, mode, seed):
    sw, vw = soft_assign(record["key"], *banks, settings)
    if mode == "U4":
        generator = torch.Generator().manual_seed(seed + 4029)
        stable_stats = stable_stats[torch.randperm(len(stable_stats), generator=generator)]
        variation_stats = variation_stats[torch.randperm(len(variation_stats), generator=generator)]
    s = torch.einsum("bck,khf->bchf", sw, stable_stats)
    v = torch.einsum("bck,khf->bchf", vw, variation_stats)
    if mode == "U0":
        s = torch.zeros_like(s); v = torch.zeros_like(v)
    elif mode == "U1":
        v = torch.zeros_like(v)
    elif mode == "U2":
        s = torch.zeros_like(s)
    return torch.cat((s, v), -1)


def batches(record, args, batch_size=None):
    size = batch_size or args.batch_size
    for start in range(0, len(record["key"]), size):
        stop = min(len(record["key"]), start + size)
        yield {key: value[start:stop] for key, value in record.items()}


def condition(bank, adapter, data, mem, device):
    y_inv = data["base"].to(device)
    scale = data["scale"].to(device)
    count = math.ceil(y_inv.shape[1] / 16)
    future = data["key"].to(device).unsqueeze(2).expand(-1, -1, count, -1)
    encoded = bank.condition(future, y_inv, scale)
    return adapter(encoded, mem.to(device)), future, y_inv, scale


@torch.no_grad()
def evaluate(bank, adapter, record, mem, args, device, samples, seed,
             multiplier=1.0, return_ratio=False):
    bank.eval(); adapter.eval()
    total = {name: 0.0 for name in METRIC_NAMES}
    count = 0
    ratio_parts = []
    generator = torch.Generator(device=device).manual_seed(seed)
    for start, data in enumerate(batches(record, args)):
        a = start * args.batch_size; b = a + len(data["key"])
        cond, future, y_inv, scale = condition(bank, adapter, data, mem[a:b], device)
        sampled = bank.sample_head("residual_flow", cond, future, y_inv,
                                   scale, samples, generator, chunk_size=8,
                                   steps=args.flow_steps)["y_samples"]
        sampled = y_inv[:, None] + multiplier * (sampled - y_inv[:, None])
        truth = data["truth"].to(device)
        stats = probabilistic_metrics(sampled, truth, y_inv)
        for name in METRIC_NAMES:
            total[name] += float(stats[name]) * len(truth)
        count += len(truth)
        if return_ratio:
            lower = torch.quantile(sampled, .025, 1)
            upper = torch.quantile(sampled, .975, 1)
            midpoint = .5 * (lower + upper)
            halfwidth = .5 * (upper - lower).clamp_min(1e-5)
            ratio_parts.append(((truth - midpoint).abs() / halfwidth)
                                .reshape(-1).cpu())
    result = {name: value / count for name, value in total.items()}
    if return_ratio:
        result["coverage_ratios"] = torch.cat(ratio_parts)
    return result


def main():
    cli = parse_args()
    torch.manual_seed(cli.seed)
    torch.set_num_threads(4)
    out = ROOT / cli.output / cli.case / "probability"
    out.mkdir(parents=True, exist_ok=True)
    args, datasets, device, model, core, cal = core_setup(cli, out)
    cli.batch_size = args.batch_size
    source, flow_rows = split_heldout(cal, args.seq_len, args.pred_len)
    val_select, val_cal = split_validation(len(datasets["val"]), args.seq_len,
                                            args.pred_len)
    core_record = collect(model, datasets["train"], core, args, device,
                          cli.seed, cli.max_batches)
    stat_record = collect(model, datasets["train"], source, args, device,
                          cli.seed, cli.max_batches)
    train_record = collect(model, datasets["train"], flow_rows, args, device,
                           cli.seed, cli.max_batches)
    val_record = collect(model, datasets["val"], val_select, args, device,
                         cli.seed, cli.max_batches)
    cal_record = collect(model, datasets["val"], val_cal, args, device,
                         cli.seed, cli.max_batches)
    test_record = collect(model, datasets["test"], range(len(datasets["test"])),
                          args, device, cli.seed, cli.max_batches)
    source_setting = json.loads((SOURCE / cli.case / "stage_c_selected.json").read_text())
    selected = source_setting["choice"]
    settings = json.loads((SOURCE / cli.case / f"{selected}.json").read_text())["settings"]
    banks = build_banks(core_record, settings, cli.seed)
    normalized_residual = ((stat_record["truth"] - stat_record["base"])
                           / stat_record["scale"]).transpose(1, 2)
    sw, vw = soft_assign(stat_record["key"], *banks, settings)
    stable_stats, stable_support = prototype_statistics(normalized_residual, sw)
    var_stats, var_support = prototype_statistics(normalized_residual, vw)
    write_json(out / "run_config.json", {
        "case": cli.case, "seed": cli.seed, "core_rows": len(core),
        "stats_rows": len(source), "flow_rows": len(flow_rows),
        "stats_future_end_max": max(source) + args.seq_len + args.pred_len - 1,
        "flow_history_start_min": min(flow_rows),
        "val_selection_rows": len(val_select), "val_calibration_rows": len(val_cal),
        "capacity_source": selected, "settings": settings,
        "epochs": cli.epochs, "lr": cli.lr,
        "samples_test": cli.samples, "flow_steps": cli.flow_steps,
        "stable_support": stable_support, "variation_support": var_support,
        "note": "Core-only A0 checkpoint precedes OOF TRAIN residuals. All modes use identical Flow architecture; only memory statistics differ. Both bank conditions may overlap; their variances are not summed."})
    if max(source) + args.seq_len + args.pred_len - 1 >= min(flow_rows):
        raise AssertionError("OOF statistics future overlaps Flow TRAIN input")
    summary = {}
    for mode_index, mode in enumerate(MODES):
        result_path = out / f"{mode}.json"
        if result_path.exists():
            summary[mode] = json.loads(result_path.read_text())
            print(f"resumed {mode}", flush=True)
            continue
        cached_mem = [features(record, banks, settings, stable_stats,
                               var_stats, mode, cli.seed)
                      for record in (train_record, val_record, cal_record,
                                     test_record)]
        train_mem, val_mem, cal_mem, test_mem = cached_mem
        torch.manual_seed(cli.seed + 100 * mode_index)
        bank = ProbabilityHeadBank(64, args.pred_len, patch_len=16,
                                   heads=("residual_flow",)).to(device)
        adapter = ResidualDifferenceCondition(14, 64, 16).to(device)
        optimizer = torch.optim.Adam(list(bank.parameters()) +
                                     list(adapter.parameters()), lr=cli.lr)
        trace = []
        best = float("inf")
        checkpoint = out / f"{mode}_best.pt"
        for epoch in range(cli.epochs):
            bank.train(); adapter.train()
            total = 0.0; seen = 0
            for batch_number, data in enumerate(batches(train_record, args)):
                a = batch_number * args.batch_size
                b = a + len(data["key"])
                cond, future, y_inv, scale = condition(bank, adapter, data,
                                                        train_mem[a:b], device)
                y = data["truth"].to(device)
                loss, _ = bank.training_losses(cond, future, y_inv, y, scale)["residual_flow"]
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite Flow loss in {mode}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(y)
                seen += len(y)
            val = evaluate(bank, adapter, val_record, val_mem, cli, device,
                           min(32, cli.samples), cli.seed + 902)
            trace.append({"epoch": epoch + 1, "train_flow_loss": total / seen,
                          "val": val})
            print(json.dumps({"mode": mode, "epoch": epoch + 1,
                              "train": total / seen, "val_crps": val["CRPS"]}),
                  flush=True)
            if val["CRPS"] < best:
                best = val["CRPS"]
                torch.save({"bank": bank.state_dict(), "adapter": adapter.state_dict(),
                            "epoch": epoch + 1, "val_crps": best}, checkpoint)
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        bank.load_state_dict(state["bank"])
        adapter.load_state_dict(state["adapter"])
        calibration = evaluate(bank, adapter, cal_record, cal_mem, cli, device,
                               min(32, cli.samples), cli.seed + 903,
                               return_ratio=True)
        ratio = calibration.pop("coverage_ratios")
        factor = float(torch.quantile(ratio, .95).clamp(.1, 10))
        test = evaluate(bank, adapter, test_record, test_mem, cli, device,
                        cli.samples, cli.seed + 904, multiplier=factor)
        result = {"mode": mode, "best_epoch_by_val_crps": state["epoch"],
                  "val_crps": best, "calibration_factor": factor,
                  "calibration": calibration, "test": test,
                  "trace": trace, "same_flow_capacity": True}
        write_json(result_path, result)
        summary[mode] = result
    write_json(out / "summary.json", {"case": cli.case, "modes": summary,
        "selection": "VAL first part CRPS; interval scaling VAL second part; TEST report only"})
    (out / "E3_complete").write_text("complete\n", encoding="utf-8")


if __name__ == "__main__":
    main()
