"""Generate readable point/probability summary once all staged queues settle."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = ("patchtst_ETTh1_720", "itransformer_ETTm1_720", "moderntcn_ETTm2_720")
POINT = ROOT / "results/fpem_learnability/pilot_v4_20261026"
PROB = ROOT / "results/fpem_learnability/residual_pilot_20261026"
OUT = ROOT / "results/fpem_learnability"


def _rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    while True:
        ready = all((POINT / case / "point_pilot_complete").exists() or
                    (POINT / case / "point_pilot_failed.txt").exists()
                    for case in CASES)
        downstream = all((PROB / case / "probability_pilot_complete").exists() or
                         (PROB / case / "probability_skipped.txt").exists() or
                         (PROB / case / "probability_failed.txt").exists()
                         for case in CASES)
        if ready and downstream:
            break
        time.sleep(60)
    point_rows = []
    prob_rows = []
    lines = ["# Learnability-aware memory and local forecast-operator Pilot", "",
             "Checkpoint and local-ridge parameters are selected on the first half of VAL. "
             "The latter half calibrates intervals only. TEST does not select the method.", ""]
    for case in CASES:
        case_root = POINT / case
        lines += [f"## {case}", ""]
        if not (case_root / "point_pilot_complete").exists():
            lines += ["- Point pilot failed; see queue.log.", ""]
            continue
        rows = _rows(case_root / "all_point_results.csv")
        point_rows.extend(rows)
        selected = json.loads((case_root / "point_selection.json").read_text(encoding="utf-8"))
        by_mode = {row["mode"]: row for row in rows}
        best = by_mode[selected["selected_mode"]]
        lines += [f"- A0 TEST MSE/MAE: {float(by_mode['P0']['test_mse']):.6f}/"
                  f"{float(by_mode['P0']['test_mae']):.6f}",
                  f"- VAL-selected real method: {selected['selected_mode']}; TEST MSE/MAE "
                  f"{float(best['test_mse']):.6f}/{float(best['test_mae']):.6f}",
                  f"- P3 Stable protection vs P2 Under-only VAL MSE: "
                  f"{float(by_mode['P3']['val_mse']):.6f} vs {float(by_mode['P2']['val_mse']):.6f}",
                  f"- Matched shuffled-pair/random-retrieval VAL MSE: "
                  f"{float(by_mode['P7']['val_mse']):.6f}/"
                  f"{float(by_mode['P8']['val_mse']):.6f}",
                  "- P3/P6 share the exact same code and checkpoint by definition; "
                  "they are not independent gains."]
        risk_root = PROB / case
        if (risk_root / "probability_pilot_complete").exists():
            risk = _rows(risk_root / "all_interval_results.csv")
            prob_rows.extend(risk)
            by = {row["mode"]: row for row in risk}
            best_probability = min(risk, key=lambda row: float(row["val_crps"]))
            lines += [f"- U0 TEST CRPS/PICP95-trajectory/MPIW95: "
                      f"{float(by['U0']['test_crps']):.6f}/"
                      f"{float(by['U0']['PICP95_trajectory']):.4f}/"
                      f"{float(by['U0']['MPIW95']):.4f}",
                      f"- VAL-selected U mode: {best_probability['mode']} "
                      f"TEST CRPS {float(best_probability['test_crps']):.6f}. "]
        else:
            why = "not completed"
            skip = risk_root / "probability_skipped.txt"
            if skip.exists():
                why = skip.read_text(encoding="utf-8").strip()
            elif (risk_root / "probability_failed.txt").exists():
                why = "failed after three attempts; inspect queue.log"
            lines += [f"- Probability stage: {why}"]
        lines += [""]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("all_point_results.csv", point_rows),
                       ("all_interval_results.csv", prob_rows)):
        if rows:
            with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
    lines += ["## Interpretation limits", "",
              "- Learnability labels are model-error clusters, not true invariant/variant "
              "or environment labels.",
              "- Full A0 TRAIN-fitted error is used to organize the final TRAIN memory; "
              "it is not called A0-OOF. Probability OOF targets use separately "
              "prefix-trained fold A0 models and the full selected local adaptation.",
              "- Fold references share the benchmark scaler fitted on all TRAIN; "
              "this is a remaining strict temporal-OOF caveat. No arbitrary drift "
              "coverage guarantee is claimed.",
              "- P7/P8 are matched negative controls and do not enter method selection. "
              "No failed configurations were dropped.", ""]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
