"""Controlled synthetic shift: diagnostic only, never used for real TEST selection."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from tools.fpem_g1_stability_methods import fit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result_root", type=Path, default=Path("results/fpem_g1_stability_search_20261102"))
    cli = p.parse_args()
    torch.set_num_threads(4)
    rows = []
    for seed in (2021, 2022, 2023):
        gen = torch.Generator().manual_seed(seed)
        n, h = 720, 48
        key = torch.randn(n+300, 1, 64, generator=gen)
        stable = torch.randn(64, 3, generator=gen)/8
        temporal = torch.randn(3, h, generator=gen)/3
        true_map = stable@temporal
        shared = key.squeeze(1)@true_map
        spurious_shape = torch.sin(torch.linspace(0, 6.28, h))
        coefficients = torch.cat([torch.full((n//2,), .7),
                                  torch.full((n-n//2,), .2),
                                  torch.full((300,), -.7)])
        residual = shared+coefficients[:, None]*key[:, 0, 0:1]*spurious_shape
        noise = torch.randn(n+300, h, generator=gen)*.08
        noise[:n//6] *= 4
        residual = (residual+noise)[:, None]
        starts = torch.arange(n)*64
        for method in ("ZERO", "RIDGE", "U01", "U06", "U07", "U11", "U12",
                       "U17", "U20", "U26", "G07", "G08", "G09", "G12"):
            config = {"ridge": .1, "seed": seed, "proxy": "time", "num_groups": 4,
                      "gamma": .2, "rank": 8, "repeats": 8}
            model = fit(method, key[:n], residual[:n], {"starts": starts}, config)
            pred = model.predict(key[n:])
            mse = float((pred-residual[n:]).square().mean())
            true_shared = shared[n:, None]
            shared_mse = float((pred-true_shared).square().mean())
            pred_norm = pred.reshape(-1)
            true_norm = true_shared.reshape(-1)
            cosine = float((pred_norm@true_norm)/(pred_norm.norm()*true_norm.norm()+1e-8))
            rows.append({"seed": seed, "method": method, "test_mse": mse,
                         "stable_component_mse": shared_mse,
                         "stable_prediction_cosine": cosine,
                         "selection_status": "synthetic known-mechanism diagnostic; never fitted to synthetic TEST"})
    cli.result_root.mkdir(parents=True, exist_ok=True)
    with (cli.result_root/"synthetic_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("SYNTHETIC COMPLETE", len(rows), flush=True)


if __name__ == "__main__":
    main()
