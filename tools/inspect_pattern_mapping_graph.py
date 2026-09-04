#!/usr/bin/env python3
"""Inspect an FPem-PMG graph artifact and optionally save PNG diagnostics."""

import argparse
import json
import os
from typing import Dict, List

import torch


def _histogram(values: torch.Tensor, bins: int = 10) -> List[int]:
    if values.numel() == 0:
        return [0] * bins
    maximum = float(values.max().item())
    if maximum <= 0:
        return [int(values.numel())] + [0] * (bins - 1)
    return torch.histc(values.float(), bins=bins, min=0.0, max=maximum).long().tolist()


def _distribution(values: torch.Tensor) -> Dict:
    if values.numel() == 0:
        return {"histogram": [], "min": None, "median": None, "p90": None, "max": None}
    ordered = values.float().sort().values
    p90_index = min(ordered.numel() - 1, int(0.9 * ordered.numel()))
    return {
        "histogram": _histogram(ordered, bins=20),
        "min": float(ordered[0]),
        "median": float(ordered[ordered.numel() // 2]),
        "p90": float(ordered[p90_index]),
        "max": float(ordered[-1]),
    }


def summarize(artifact: Dict) -> Dict:
    pattern = artifact["pattern_graph"]
    mapping = artifact["mapping_graph"]
    pattern_order = torch.argsort(pattern["stability"], descending=True)[:10]
    mapping_order = torch.argsort(mapping["stability"], descending=True)[:10]
    top_patterns = [
        {
            "node": int(index),
            "stability": float(pattern["stability"][index]),
            "window_support": float(pattern["window_support"][index]),
            "predictive_support": float(pattern["predictive_support"][index]),
        }
        for index in pattern_order.tolist()
    ]
    top_mappings = [
        {
            "src": int(mapping["src_index"][index]),
            "dst": int(mapping["dst_index"][index]),
            "stability": float(mapping["stability"][index]),
            "window_support": float(mapping["window_support"][index]),
        }
        for index in mapping_order.tolist()
    ]
    return {
        "metadata": artifact.get("metadata", {}),
        "number_of_active_patterns": int(pattern["means"].shape[0]),
        "pattern_support_histogram": _histogram(pattern["window_support"]),
        "top_stable_patterns": top_patterns,
        "number_of_temporal_edges": int(mapping["src_index"].numel()),
        "mapping_support_histogram": _histogram(mapping["window_support"]),
        "top_stable_mappings": top_mappings,
        "pattern_novelty_distance_distribution": _distribution(pattern["distance_cdf"]),
        "mapping_novelty_distance_distribution": _distribution(mapping["distance_cdf"]),
    }


def save_plots(artifact: Dict, output_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; JSON/text diagnostics were still produced")
        return
    os.makedirs(output_dir, exist_ok=True)
    pattern = artifact["pattern_graph"]
    mapping = artifact["mapping_graph"]
    for values, title, filename in [
        (pattern["window_support"], "Pattern window support", "pattern_support.png"),
        (mapping["window_support"], "Mapping window support", "mapping_support.png"),
    ]:
        plt.figure(figsize=(7, 4))
        plt.hist(values.float().numpy(), bins=20)
        plt.title(title)
        plt.xlabel("distinct training windows")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=150)
        plt.close()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(pattern["distance_cdf"].float().numpy(), bins=30)
    axes[0].set_title("Pattern best-distance train CDF")
    axes[1].hist(mapping["distance_cdf"].float().numpy(), bins=30)
    axes[1].set_title("Mapping delta-distance train CDF")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "graph_summary.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("graph", help="path to pattern_mapping_graph.pt")
    parser.add_argument("--output_dir", default="graph_diagnostics")
    args = parser.parse_args()
    artifact = torch.load(args.graph, map_location="cpu")
    summary = summarize(artifact)
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "graph_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    save_plots(artifact, args.output_dir)


if __name__ == "__main__":
    main()
