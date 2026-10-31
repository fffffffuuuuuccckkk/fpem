"""Add same-checkpoint A0 measurements to the finished G0/G1/G2 pilot."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.run_fpem_shared_factorization import load_experiment, prediction
from tools.run_fpem_s4_losswise_queue import CASES, RESULTS, selected_prior


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = RESULTS / "gradient_surgery_20261020"


@torch.no_grad()
def evaluate_a0(cli):
    args, model, loaders, device, digest = load_experiment(SimpleNamespace(**cli))
    squared = absolute = count = 0.0
    model.eval()
    for x, y, _, cycle in loaders["test"]:
        x, y, cycle = x.float().to(device), y[:, -args.pred_len:].float().to(device), cycle.to(device)
        hidden, mean, std = model.encode(x, cycle)
        forecast = prediction(model, hidden, mean, std, cycle, head="reference")
        squared += (forecast-y).square().sum().item()
        absolute += (forecast-y).abs().sum().item()
        count += y.numel()
    return squared/count, absolute/count, digest


def main():
    rows = []
    for case, _ in CASES:
        _, _, _, prior = selected_prior(case)
        cli = prior["cli"]
        a0_mse, a0_mae, digest = evaluate_a0(cli)
        per_mode = {}
        for mode in ("G0", "G1", "G2"):
            directory = RESULTS / case / "E0" if mode == "G0" else OUTPUT / case / mode
            result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
            if result["reference_sha256"] != digest:
                raise ValueError(f"reference checkpoint mismatch: {case} {mode}")
            per_mode[mode] = result["test"]
        row = {"case": case, "reference_sha256": digest,
               "A0_test_mse": a0_mse, "A0_test_mae": a0_mae}
        for mode in ("G0", "G1", "G2"):
            row[mode+"_test_mse"] = per_mode[mode]["mse"]
            row[mode+"_test_mae"] = per_mode[mode]["mae"]
            row[mode+"_mse_improvement_vs_A0_percent"] = (
                100 * (a0_mse-per_mode[mode]["mse"])/a0_mse)
        rows.append(row)
    with (OUTPUT / "comparison_with_a0.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["", "## Same-checkpoint A0 comparison", "",
             "A0 was evaluated from the exact reference checkpoint/hash loaded by G0/G1/G2,",
             "on the current TEST loader, using the frozen original prediction head.", "",
             "| Case | A0 MSE / MAE | G0 MSE / MAE | G1 MSE / MAE | G2 MSE / MAE |",
             "|---|---:|---:|---:|---:|"]
    for row in rows:
        cells = [row["case"]]
        for mode in ("A0", "G0", "G1", "G2"):
            cells.append(f"{row[mode+'_test_mse']:.6f} / {row[mode+'_test_mae']:.6f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "A0 for PatchTST ETTh1-96 recomputes to 0.410892/0.419296 using the",
              "current loader and the S4-loaded checkpoint; an older archived A0 metrics file",
              "reports 0.410686/0.420848. The same-checkpoint recomputation is used here for",
              "the controlled comparison; the discrepancy remains to be audited."]
    path = OUTPUT / "summary.md"
    original = path.read_text(encoding="utf-8")
    original = original.split("\n## Same-checkpoint A0 comparison")[0].rstrip()+"\n"
    path.write_text(original+"\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
