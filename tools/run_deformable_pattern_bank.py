#!/usr/bin/env python3
"""Build PB0..PB3 from ETTh1 TRAIN and render isolated bank diagnostics."""

import argparse
import math
import os
import sys
from types import SimpleNamespace

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_provider.data_loader import Dataset_ETT_hour
from models.PatchTST_FPEM import extract_raw_patches
from models.fpem import DeformablePatternBank, DeformablePatternBankBuilder
from models.fpem.pattern_graph_builder import PatternMappingGraphBuilder


def dataset(root, flag, seq_len, pred_len):
    return Dataset_ETT_hour(
        SimpleNamespace(augmentation_ratio=0), root, flag=flag,
        size=[seq_len, seq_len // 2, pred_len], features="M", data_path="ETTh1.csv",
        target="OT", scale=True, timeenc=1, freq="h",
    )


def collect(data, patch_len, stride, maximum_windows=0):
    normalized, raw, future = [], [], []
    for batch_x, batch_y, _, _ in DataLoader(data, batch_size=256, shuffle=False, num_workers=2):
        patch = extract_raw_patches(batch_x.float(), patch_len, stride, stride)
        normalized.append(patch["normalized"])
        raw.append(patch["patches"])
        future.append(batch_y[:, -data.pred_len:, :].float().permute(0, 2, 1))
    normalized, raw, future = torch.cat(normalized), torch.cat(raw), torch.cat(future)
    if maximum_windows:
        normalized = normalized[:maximum_windows]
        raw = raw[:maximum_windows]
        future = future[:maximum_windows]
    return normalized, raw, future


def future_patches(values, patch_len, stride):
    patch = extract_raw_patches(values.permute(0, 2, 1), patch_len, stride, stride)
    return patch["normalized"], patch["patches"]


def legacy_bank(normalized, future, seq_len):
    windows, channels, patches, length = normalized.shape
    points = normalized.reshape(-1, length)
    window_anchor = torch.div(torch.arange(windows), seq_len, rounding_mode="floor")
    anchors = window_anchor[:, None, None].expand(windows, channels, patches).reshape(-1)
    target = future[:, :, None, :].expand(windows, channels, patches, future.shape[-1]).reshape(-1, future.shape[-1])
    builder = PatternMappingGraphBuilder(
        length, seq_len=seq_len, representation_space="raw", patch_len=length,
    )
    pattern, assignment, radius = builder._pattern_statistics(points, anchors, target, windows)
    anchor_count = int(window_anchor.max()) + 1
    pattern["coverage"] = pattern["window_support"] / float(anchor_count)
    pattern["assignment"] = assignment
    pattern["coverage_radius"] = torch.tensor(radius)
    return pattern


def canonical_query(points, patterns):
    distance = torch.cdist(points.float(), patterns.float()).square() / points.shape[-1]
    error, index = distance.min(-1)
    return error, index, patterns[index]


def spearman(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(np.argsort(np.argsort(left)), np.argsort(np.argsort(right)))[0, 1])


def save_figure(fig, directory, name):
    fig.tight_layout()
    fig.savefig(os.path.join(directory, name + ".png"), dpi=180, bbox_inches="tight")
    fig.savefig(os.path.join(directory, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def pattern_pages(patterns, coverage, labels, directory, prefix, shared_ylim):
    order = np.argsort(-coverage)
    for page, start in enumerate(range(0, len(order), 32), 1):
        chosen = order[start:start + 32]
        rows, columns = int(math.ceil(len(chosen) / 4)), 4
        fig, axes = plt.subplots(rows, columns, figsize=(13, max(2.5, rows * 2.2)), squeeze=False)
        for axis in axes.ravel():
            axis.axis("off")
        for axis, index in zip(axes.ravel(), chosen):
            axis.axis("on")
            axis.plot(patterns[index], linewidth=1.5)
            axis.set_ylim(*shared_ylim)
            axis.set_title(labels(index), fontsize=8)
            axis.grid(alpha=.2)
        save_figure(fig, directory, "{}_page_{:02d}".format(prefix, page))
        if page == 1:
            fig, axes = plt.subplots(rows, columns, figsize=(13, max(2.5, rows * 2.2)), squeeze=False)
            for axis in axes.ravel(): axis.axis("off")
            for axis, index in zip(axes.ravel(), chosen):
                axis.axis("on"); axis.plot(patterns[index], linewidth=1.5); axis.set_ylim(*shared_ylim)
                axis.set_title(labels(index), fontsize=8); axis.grid(alpha=.2)
            save_figure(fig, directory, prefix + "_grid")


def write_method_result(root, name, count, oracle, oracle_signal,
                        train_error, coverage, relation, arrays):
    directory = os.path.join(root, "pattern_bank_" + name)
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "pattern_bank_diagnostics.txt"), "w", encoding="utf-8") as handle:
        handle.write("method: {}\nsource_split: train\noracle_diagnostic_only: true\n".format(name))
        handle.write("pattern_count: {}\n".format(count))
        handle.write("oracle_normalized_shape_mse: {:.10g}\n".format(oracle))
        handle.write("oracle_signal_mse_gt_geometry_diagnostic_only: {:.10g}\n".format(oracle_signal))
        handle.write("train_reconstruction_mse: {:.10g}\n".format(train_error))
        handle.write("coverage_mean: {:.10g}\n".format(coverage))
        handle.write("p_inv_relation: {}\n".format(relation))
    np.savez_compressed(os.path.join(directory, "pattern_bank_diagnostics.npz"), **arrays)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", default="./dataset/ETT-small/")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--setting", default="deformable_pattern_bank_ETTh1_96_96")
    parser.add_argument("--max_windows", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(2026)
    np.random.seed(2026)
    train_data = dataset(args.root_path, "train", args.seq_len, args.pred_len)
    test_data = dataset(args.root_path, "test", args.seq_len, args.pred_len)
    train_normalized, train_raw, train_future = collect(
        train_data, args.patch_len, args.stride, args.max_windows
    )
    _, _, test_future = collect(test_data, args.patch_len, args.stride, args.max_windows)
    oracle_normalized, oracle_raw = future_patches(test_future, args.patch_len, args.stride)
    train_points = train_normalized.reshape(-1, args.patch_len)
    oracle_points = oracle_normalized.reshape(-1, args.patch_len)
    oracle_raw_points = oracle_raw.reshape(-1, args.patch_len)
    oracle_mean = oracle_raw_points.mean(-1, keepdim=True)
    oracle_scale = oracle_raw_points.var(-1, keepdim=True, unbiased=False).sqrt().clamp_min(1e-5)

    legacy = legacy_bank(train_normalized, train_future, args.seq_len)
    legacy_train_error, _, _ = canonical_query(train_points, legacy["means"])
    legacy_oracle_error, legacy_oracle_id, legacy_oracle_reconstruction = canonical_query(
        oracle_points, legacy["means"]
    )
    legacy_oracle_signal = oracle_mean + oracle_scale * legacy_oracle_reconstruction
    legacy_oracle_signal_mse = (legacy_oracle_signal - oracle_raw_points).square().mean()
    banks = {}
    for name, rank, consolidate in (("PB1", 0, False), ("PB2", 2, False), ("PB3", 2, True)):
        print("Building {} rank={} consolidate={}".format(name, rank, consolidate), flush=True)
        stats = DeformablePatternBankBuilder(
            args.patch_len, args.seq_len, rank,
            invariant_consolidation=consolidate, local_merge=consolidate,
        ).fit(train_normalized, train_future, source_split="train")
        banks[name] = (stats, DeformablePatternBank(stats))

    oracle_results = {}
    for name, (stats, bank) in banks.items():
        oracle_results[name] = bank.query(oracle_normalized, include_responsibility=False)

    result_root = os.path.join("./results", args.setting)
    os.makedirs(result_root, exist_ok=True)
    write_method_result(
        result_root, "PB0_legacy", legacy["means"].shape[0], legacy_oracle_error.mean(),
        legacy_oracle_signal_mse,
        legacy_train_error.mean(), legacy["coverage"].mean(), "legacy_stability",
        {"patterns": legacy["means"].numpy(), "coverage": legacy["coverage"].numpy(),
         "train_reconstruction_errors": legacy_train_error.numpy(),
         "oracle_reconstruction_errors": legacy_oracle_error.numpy()},
    )
    rows = [("PB0 Legacy", legacy["means"].shape[0], legacy_oracle_error.mean().item(),
             legacy_train_error.mean().item(), legacy["coverage"].mean().item(), "legacy")]
    for name in ("PB1", "PB2", "PB3"):
        stats, _ = banks[name]
        query = oracle_results[name]
        oracle_signal = oracle_mean + oracle_scale * query["shape_reconstruction"].reshape(-1, args.patch_len)
        oracle_signal_mse = (oracle_signal - oracle_raw_points).square().mean().item()
        rho = spearman(stats["coverage"].numpy(), stats["pattern_p_inv_all"].numpy())
        row = (name, stats["canonical_patterns"].shape[0], query["reconstruction_error"].mean().item(),
               stats["final_train_reconstruction_error"].mean().item(),
               stats["coverage"][stats["final_source_indices"]].mean().item(), "rho={:.4f}".format(rho))
        rows.append(row)
        write_method_result(
            result_root, name.lower(), row[1], row[2], oracle_signal_mse,
            row[3], row[4], row[5],
            {key: value.numpy() for key, value in stats.items() if torch.is_tensor(value)},
        )

    stats2, bank2 = banks["PB2"]
    stats3, bank3 = banks["PB3"]
    local = stats3["local_to_canonical_index"] >= 0
    explained_ratio = float(stats3["local_explained"][local].float().mean()) if bool(local.any()) else 0.0
    rho_coverage = spearman(stats3["coverage"], stats3["pattern_p_inv_all"])
    rho_entropy = spearman(stats3["occurrence_entropy"], stats3["pattern_p_inv_all"])
    rho_concentration = spearman(stats3["concentration"], stats3["pattern_p_local_all"])

    visual_dir = os.path.join("./pattern_visualizations", args.setting)
    os.makedirs(visual_dir, exist_ok=True)
    all_values = torch.cat([legacy["means"].flatten(), stats3["canonical_patterns"].flatten()]).numpy()
    shared_ylim = (float(np.quantile(all_values, .005)), float(np.quantile(all_values, .995)))
    pattern_pages(
        legacy["means"].numpy(), legacy["coverage"].numpy(),
        lambda i: "P{} cov={:.2f} sup={:.0f} stable={:.2f}".format(
            i, legacy["coverage"][i], legacy["window_support"][i], legacy["stability"][i]
        ), visual_dir, "legacy_patterns", shared_ylim,
    )
    final_source = stats3["final_source_indices"]
    pattern_pages(
        stats3["canonical_patterns"].numpy(), stats3["coverage"][final_source].numpy(),
        lambda i: "C{} cov={:.2f} p_inv={:.2f} rank={}".format(
            i, stats3["coverage"][final_source[i]], stats3["pattern_p_inv"][i], 2
        ), visual_dir, "deformable_patterns", shared_ylim,
    )

    legacy_patterns = legacy["means"]
    matches = bank2.query(legacy_patterns)
    order = torch.argsort(legacy["coverage"], descending=True)[:16]
    fig, axes = plt.subplots(8, 2, figsize=(14, 18), squeeze=False)
    for axis, index in zip(axes.ravel(), order):
        canonical_id = int(matches["canonical_pattern_id"][index])
        axis.plot(legacy_patterns[index], label="Legacy P{}".format(index))
        axis.plot(matches["canonical_reconstruction"][index], "--", label="Canonical C{}".format(canonical_id))
        axis.plot(matches["shape_reconstruction"][index], ":", label="Canonical+Shape")
        raw_error = float((legacy_patterns[index] - matches["canonical_reconstruction"][index]).square().mean())
        axis.set_title("raw={:.3g}, deform={:.3g}, coverage={:.2f}".format(
            raw_error, matches["reconstruction_error"][index], legacy["coverage"][index]
        ), fontsize=9)
        axis.legend(fontsize=7); axis.grid(alpha=.2)
    save_figure(fig, visual_dir, "legacy_vs_deformable_matches")

    chosen = torch.argsort(stats3["coverage"][final_source], descending=True)[:8]
    fig, axes = plt.subplots(4, 2, figsize=(13, 11), squeeze=False)
    for axis, index in zip(axes.ravel(), chosen):
        pattern, basis = stats3["canonical_patterns"][index], stats3["shape_basis"][index]
        axis.plot(pattern, linewidth=2, label="C{}".format(index))
        for component in range(min(2, basis.shape[-1])):
            amplitude = .75
            axis.plot(pattern + amplitude * basis[:, component], "--", label="+u{}".format(component + 1))
            axis.plot(pattern - amplitude * basis[:, component], ":", label="-u{}".format(component + 1))
        axis.legend(fontsize=7); axis.grid(alpha=.2)
    save_figure(fig, visual_dir, "canonical_shape_basis")

    example_index = torch.linspace(0, train_points.shape[0] - 1, 16).round().long()
    examples = bank2.query(train_points[example_index])
    fig, axes = plt.subplots(8, 2, figsize=(14, 19), squeeze=False)
    for row, (axis, point) in enumerate(zip(axes.ravel(), train_points[example_index])):
        axis.plot(point, label="Observed")
        axis.plot(examples["canonical_reconstruction"][row], "--", label="Canonical")
        axis.plot(examples["shape_reconstruction"][row], ":", label="Canonical+Shape")
        axis.plot(point - examples["shape_reconstruction"][row], alpha=.5, label="Residual")
        before = float((point - examples["canonical_reconstruction"][row]).square().mean())
        axis.set_title("C{} p_inv={:.2f} before={:.3g} after={:.3g} c={}".format(
            int(examples["canonical_pattern_id"][row]), examples["pattern_p_inv"][row], before,
            examples["reconstruction_error"][row], np.round(examples["shape_coefficients"][row].numpy(), 2)
        ), fontsize=8); axis.legend(fontsize=6); axis.grid(alpha=.2)
    save_figure(fig, visual_dir, "patch_decomposition_examples")

    local_indices = torch.nonzero(local, as_tuple=False).flatten()[:12]
    fig, axes = plt.subplots(max(1, math.ceil(max(1, len(local_indices)) / 2)), 2,
                             figsize=(14, max(4, math.ceil(max(1, len(local_indices)) / 2) * 2.4)), squeeze=False)
    for axis in axes.ravel(): axis.axis("off")
    for axis, index in zip(axes.ravel(), local_indices):
        axis.axis("on")
        target_id = int(stats3["local_to_canonical_index"][index])
        canonical = stats3["canonical_patterns_all"][target_id]
        basis = stats3["shape_basis_all"][target_id]
        deformed = canonical + basis @ stats3["local_shape_coefficients"][index]
        local_pattern = stats3["canonical_patterns_all"][index]
        axis.plot(local_pattern, label="Local {}".format(index)); axis.plot(canonical, "--", label="Canonical {}".format(target_id))
        axis.plot(deformed, ":", label="Canonical+Shape")
        axis.set_title("cov={:.2f}, before={:.3g}, after={:.3g}".format(
            stats3["coverage"][index], (local_pattern - canonical).square().mean(), stats3["local_merge_error"][index]
        ), fontsize=8); axis.legend(fontsize=6); axis.grid(alpha=.2)
    save_figure(fig, visual_dir, "local_pattern_consolidation")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].scatter(stats3["coverage"], stats3["pattern_p_inv_all"], s=16)
    axes[0].set(xlabel="Pattern sample coverage", ylabel="p_inv", title="rho={:.3f}".format(rho_coverage))
    axes[1].scatter(stats3["occurrence_entropy"], stats3["pattern_p_inv_all"], s=16)
    axes[1].set(xlabel="Occurrence entropy", ylabel="p_inv", title="rho={:.3f}".format(rho_entropy))
    axes[2].scatter(stats3["concentration"], stats3["pattern_p_local_all"], s=16)
    axes[2].set(xlabel="Concentration", ylabel="p_local", title="rho={:.3f}".format(rho_concentration))
    for axis in axes: axis.grid(alpha=.2)
    save_figure(fig, visual_dir, "pattern_coverage_invariance")

    train_errors = [legacy_train_error.numpy()] + [banks[name][0]["final_train_reconstruction_error"].numpy() for name in ("PB1", "PB2", "PB3")]
    oracle_errors = [legacy_oracle_error.numpy()] + [oracle_results[name]["reconstruction_error"].reshape(-1).numpy() for name in ("PB1", "PB2", "PB3")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    labels = ["Legacy", "PB1", "PB2", "PB3"]
    axes[0].boxplot([x[np.linspace(0, len(x)-1, min(5000, len(x))).astype(int)] for x in train_errors], labels=labels, showfliers=False)
    axes[0].set_title("TRAIN normalized shape reconstruction MSE")
    axes[1].bar(labels, [float(np.mean(x)) for x in oracle_errors])
    axes[1].set_title("Oracle future normalized shape MSE")
    for axis in axes: axis.grid(axis="y", alpha=.2)
    save_figure(fig, visual_dir, "pattern_reconstruction_comparison")

    np.savez_compressed(
        os.path.join(visual_dir, "pattern_bank_comparison.npz"),
        legacy_patterns=legacy["means"].numpy(), legacy_coverage=legacy["coverage"].numpy(),
        canonical_patterns=stats3["canonical_patterns"].numpy(),
        canonical_coverage=stats3["coverage"][final_source].numpy(),
        canonical_p_inv=stats3["pattern_p_inv"].numpy(), shape_basis=stats3["shape_basis"].numpy(),
        legacy_to_canonical=matches["canonical_pattern_id"].numpy(),
        train_reconstruction_errors=np.array([float(np.mean(x)) for x in train_errors]),
        oracle_reconstruction_errors=np.array([float(np.mean(x)) for x in oracle_errors]),
        example_patch=train_points[example_index].numpy(),
        example_canonical=examples["canonical_reconstruction"].numpy(),
        example_shape_reconstruction=examples["shape_reconstruction"].numpy(),
    )
    with open(os.path.join(visual_dir, "pattern_bank_summary.txt"), "w", encoding="utf-8") as handle:
        handle.write("Method | Pattern count | Oracle shape MSE | Train recon MSE | Coverage mean | p_inv relation\n")
        for row in rows:
            handle.write("{} | {} | {:.10g} | {:.10g} | {:.10g} | {}\n".format(*row))
        handle.write("provisional_pattern_count: {}\n".format(stats3["provisional_patterns"].shape[0]))
        handle.write("final_canonical_invariant_count: {}\n".format(stats3["canonical_patterns"].shape[0]))
        handle.write("mean_error_before_shape: {:.10g}\n".format(rows[1][3]))
        handle.write("mean_error_after_shape: {:.10g}\n".format(rows[2][3]))
        handle.write("explained_local_ratio: {:.10g}\n".format(explained_ratio))
        handle.write("unexplained_local_ratio: {:.10g}\n".format(1.0 - explained_ratio))
        handle.write("rho_coverage_p_inv: {:.10g}\n".format(rho_coverage))
        handle.write("rho_entropy_p_inv: {:.10g}\n".format(rho_entropy))
        handle.write("rho_concentration_p_local: {:.10g}\n".format(rho_concentration))
        handle.write("source_split: TRAIN only\noracle: DIAGNOSTIC ONLY\n")
    print(open(os.path.join(visual_dir, "pattern_bank_summary.txt"), encoding="utf-8").read(), flush=True)


if __name__ == "__main__":
    main()
