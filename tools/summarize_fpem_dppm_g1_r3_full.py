"""Consolidate the point-only G1+R3 matrix without choosing on TEST."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"results/fpem_dual_pattern_memory/g1_r3_full_point"
FIELDS=("backbone","dataset","pred_len","a0_mse","a0_mae",
        "g1_mse","g1_mae","g1_r3_mse","g1_r3_mae",
        "g1_r3_minus_a0_mse","g1_r3_minus_g1_mse",
        "g1_ridge","g1_scale","null_offset","reliability_mean",
        "active_sample_ratio","reference_sha256","protocol")

def main():
    rows=[]
    for backbone in ("patchtst","itransformer","moderntcn"):
        for dataset in ("ETTh1","ETTh2","ETTm1","ETTm2","ExchangeRate"):
            for horizon in (96,192,336,720):
                case=BASE/backbone/dataset/f"pred_{horizon}"
                if not (case/"complete").exists(): continue
                data=json.loads((case/"metrics.json").read_text())
                rows.append({"backbone":backbone,"dataset":dataset,"pred_len":horizon,
                             "a0_mse":data["a0"]["mse"],"a0_mae":data["a0"]["mae"],
                             "g1_mse":data["g1"]["mse"],"g1_mae":data["g1"]["mae"],
                             "g1_r3_mse":data["g1_r3"]["mse"],"g1_r3_mae":data["g1_r3"]["mae"],
                             "g1_r3_minus_a0_mse":data["g1_r3"]["mse"]-data["a0"]["mse"],
                             "g1_r3_minus_g1_mse":data["g1_r3"]["mse"]-data["g1"]["mse"],
                             "g1_ridge":data["g1_ridge"],"g1_scale":data["g1_scale"],
                             "null_offset":data["val"]["offset"],
                             "reliability_mean":data["reliability_mean"],
                             "active_sample_ratio":data["active_sample_ratio"],
                             "reference_sha256":data["reference_sha256"],
                             "protocol":data["protocol"]})
    target=BASE/"point_results.csv"
    with target.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=FIELDS)
        writer.writeheader();writer.writerows(rows)
    print(f"{len(rows)}/60 completed; summary {target}")
    for backbone in ("patchtst","itransformer","moderntcn"):
        group=[r for r in rows if r["backbone"]==backbone]
        print(backbone,len(group),"beat_A0",sum(r["g1_r3_mse"]<r["a0_mse"] for r in group))

if __name__=="__main__": main()
