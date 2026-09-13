#!/usr/bin/env python3
"""Render interpretable views from a TRAIN-only raw pattern graph artifact."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def _normalize_shape(values, eps=1e-5):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    return (values - values.mean()) / (values.std() + eps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", help="raw_pattern_graph.npz")
    parser.add_argument("--output-dir", default="raw_pattern_visualizations")
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--test-patch-npy", default="", help="optional single raw test patch")
    args = parser.parse_args()

    graph = np.load(args.artifact)
    os.makedirs(args.output_dir, exist_ok=True)
    means = graph["mean_shape"]
    medoids = graph["medoid_shape"]
    shape_std = graph["shape_std"]
    stability = graph["overall_stability"]
    order = np.argsort(-stability)[: min(args.top, len(stability))]

    fig, axes = plt.subplots(len(order), 1, figsize=(9, max(3, 2.4 * len(order))), squeeze=False)
    for axis, pattern_id in zip(axes[:, 0], order):
        x = np.arange(means.shape[1])
        axis.plot(x, means[pattern_id], label="mean shape", linewidth=2)
        axis.fill_between(x, means[pattern_id] - shape_std[pattern_id],
                          means[pattern_id] + shape_std[pattern_id], alpha=0.2)
        axis.plot(x, medoids[pattern_id], "--", label="medoid")
        axis.set_title(
            "pattern {} | anchor support {:.0f} | predictive {:.3f} | overall {:.3f}".format(
                pattern_id, graph["anchor_support"][pattern_id],
                graph["predictive_stability"][pattern_id], stability[pattern_id]
            )
        )
        axis.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "top_stable_raw_patterns.png"), dpi=180)
    plt.close(fig)

    edge_order = np.argsort(-graph["mapping_stability"])[: min(args.top, len(graph["mapping_stability"]))]
    fig, axes = plt.subplots(len(edge_order), 1, figsize=(9, max(3, 2.4 * len(edge_order))), squeeze=False)
    for axis, edge_index in zip(axes[:, 0], edge_order):
        src = int(graph["src_pattern"][edge_index])
        dst = int(graph["dst_pattern"][edge_index])
        axis.plot(means[src], label="pattern {}".format(src))
        axis.plot(means[dst], label="pattern {}".format(dst))
        axis.set_title(
            "{} -> {} | support {:.1f} | mapping stability {:.3f}".format(
                src, dst, graph["mapping_support"][edge_index], graph["mapping_stability"][edge_index]
            )
        )
        axis.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "top_stable_raw_mappings.png"), dpi=180)
    plt.close(fig)

    if args.test_patch_npy:
        query = _normalize_shape(np.load(args.test_patch_npy))
        variance = np.maximum(shape_std ** 2, 1e-5)
        distance = np.mean((query[None, :] - means) ** 2 / variance, axis=1)
        best = int(np.argmin(distance))
        score = -0.5 * distance + np.log(np.maximum(stability, 1e-5))
        responsibility = np.exp(score - score.max())
        responsibility /= responsibility.sum()
        expected_stability = float(np.sum(responsibility * stability))
        cdf = graph["pattern_distance_cdf"]
        percentile = np.searchsorted(cdf, distance.min(), side="right") / max(1, cdf.size)
        novelty = float(np.clip(2.0 * (percentile - 0.5), 0.0, 1.0))
        c_pat = expected_stability * (1.0 - novelty)
        fig, axis = plt.subplots(figsize=(9, 4))
        axis.plot(query, label="current test raw patch")
        axis.plot(means[best], label="best stable raw pattern")
        axis.set_title(
            "best pattern {} | c_pat {:.4f} | novelty {:.4f} | distance {:.4f}".format(
                best, c_pat, novelty, distance[best]
            )
        )
        axis.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, "test_patch_pattern_match.png"), dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
