"""VAL-only shortlist for costlier negative controls (never ranks by TEST)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("case")
    parser.add_argument("--root",default="results/fpem_dual_pattern_memory/rare_learning_pilot")
    args=parser.parse_args()
    case=Path(args.root)/args.case
    with (case/"all_results.csv").open(newline="",encoding="utf-8") as handle:
        rows=list(csv.DictReader(handle))
    good=[]
    for row in rows:
        method=row["method"]
        if method=="R0" or not (case/method/"complete").exists():
            continue
        report=json.loads((case/method/"mapping_oos.json").read_text(encoding="utf-8"))
        evidence=report["metrics"]
        # Segment OOS gain is independent of the diagnostic candidate mixture's
        # NULL fallback convention and is the actual reliability evidence.
        gain=max((item["oof_gain"] for item in report["segment_evidence"]),default=0.)
        if gain>0 and evidence["early_val_reliability_mean"]>0:
            good.append((float(row["val_mse"]),method,gain))
    good.sort()
    chosen=[method for _,method,_ in good[:3]]
    (case/"validation_plan.json").write_text(json.dumps({"selected_by":"later VAL MSE among positive early VAL OOS mapping gain and reliability",
        "selected":chosen,"candidate_count":len(good)},indent=2),encoding="utf-8")
    print(" ".join(chosen))


if __name__=="__main__":
    main()
