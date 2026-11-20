#!/usr/bin/env python3
"""One compact, provenance-preserving table for all frozen-group downstream arms."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.run_fpem_mechanism_fusion import METHODS


def save(path, rows):
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields:fields.append(key)
    if not fields:fields=["case","method","status"]
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fields)
        writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--root",required=True)
    cli=parser.parse_args();root=Path(cli.root);root.mkdir(parents=True,exist_ok=True)
    rows=[];flow=[]
    for case in sorted(root.glob("pilots/*/*/pred_*/seed_*")):
        if not (case/"features.json").exists():continue
        meta=json.loads((case/"features.json").read_text())
        case_name=f'{meta["backbone"]}/{meta["dataset"]}-{meta["horizon"]}'
        for short,method in METHODS.items():
            directory=case/"methods"/method
            status="complete" if (directory/"point_complete").exists() else "pending"
            row={"case":case_name,"method":method,"seed":meta["seed"],"status":status,
                 "a0_sha256":meta["a0_sha256"],"a0_init_error_max":meta["a0_init_error_max"],
                 "normalization":"history_only_per_window"}
            if status=="complete":
                report=json.loads((directory/"point_metrics.json").read_text())
                row.update(source_group_sha256=report["source_group_sha256"],
                           f03_head_sha256=report["f03_head_sha256"],
                           B1_router_top1=report["router_B1"]["top1_agreement"],
                           B1_router_soft_kl=report["router_B1"]["soft_kl"],
                           B2_null_scale=report["selected_null_scale_B2"],
                           C_f03_mse=report["C"]["f03"]["mse"],
                           C_point_mse=report["C"]["f03_r3"]["mse"],
                           TEST_a0_mse=report["test"]["a0"]["mse"],
                           TEST_f03_mse=report["test"]["f03"]["mse"],
                           TEST_point_mse=report["test"]["f03_r3"]["mse"],
                           TEST_point_mae=report["test"]["f03_r3"]["mae"],
                           TEST_active_ratio=report["test"]["active_ratio"])
                row["TEST_gain_vs_f03"]=1-row["TEST_point_mse"]/max(row["TEST_f03_mse"],1e-12)
            if (directory/"flow_status.json").exists():
                row["flow_status"]=json.loads((directory/"flow_status.json").read_text())["status"]
            elif (directory/"flow_complete").exists():row["flow_status"]="complete"
            rows.append(row)
            for mode in ("U0_original_condition","U1_mapping_disagreement",
                         "U2_oof_residual_summary","U3_both"):
                path=directory/(mode+"_flow.json")
                if not path.exists():continue
                report=json.loads(path.read_text())
                flow.append({"case":case_name,"method":method,"mode":mode,"seed":meta["seed"],
                             "source_group_sha256":row.get("source_group_sha256"),
                             "VAL_crps":report["val_crps"],"TEST_crps":report["test"]["CRPS"],
                             "TEST_wis":report["test"]["WIS"],"TEST_picp95":report["test"]["PICP_95"],
                             "TEST_mpiw95":report["test"]["MPIW_95"],
                             "calibration_factor":report["calibration_factor"],
                             "best_epoch":report["best_epoch"],
                             "normalization":"history_only_per_window"})
    save(root/"point_results.csv",rows)
    save(root/"flow_results.csv",flow)
    lines=["# Frozen-group F03/R3/Flow fusion", "",
           f"Point rows: {len(rows)}; completed: {sum(row['status']=='complete' for row in rows)}; Flow rows: {len(flow)}.",
           "", "All point metrics use the same history-only normalized future units. TEST is report-only; C is held out from added maps but not from pretrained A0.",
           "Groups G3/G4/G7 and Raw/Random/ShuffleY are read-only Stage-1 assignments. F03 is A0-head-initialized, fit on TRAIN A only and selected on B1; no Backbone update.",
           "Reliability uses B1, NULL uses B2. Flow uses four purged VAL blocks for residual summaries, training, selection and calibration; TEST is final report. The A0 checkpoint may have used VAL for checkpoint selection, so this is not fully cross-fitted A0 OOF.",
           "If independent-event support is insufficient, Flow is marked skipped, not scored as zero. Historical A0 catalogs may themselves have TEST-selection bias.",
           "", "|Case|Method|C F03 MSE|C Point MSE|TEST F03 MSE|TEST Point MSE|TEST gain|Flow status|", "|---|---|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        if row["status"]!="complete":continue
        lines.append("|"+"|".join([row["case"],row["method"]] +
                     [f"{row[key]:.6f}" for key in ("C_f03_mse","C_point_mse","TEST_f03_mse","TEST_point_mse","TEST_gain_vs_f03")] +
                     [row.get("flow_status","pending")])+"|")
    (root/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"POINT {sum(row['status']=='complete' for row in rows)}/{len(rows)} FLOW {len(flow)}")


if __name__=="__main__":main()
