"""Aggregate completed model-free mapping diagnostics without hiding failures."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path("results/history_future_mapping_validation"))
    args=parser.parse_args();rows=[]
    for path in sorted(args.root.glob("*_seed*/summary.csv")):
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                row["source_dir"]=str(path.parent);rows.append(row)
    if rows:
        keys=list(dict.fromkeys(k for r in rows for k in r))
        with (args.root/"all_results.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    lines=["# History–future mapping diagnostics", "",
           "This is raw-data statistical evidence, not a proof of concept drift or an FPEM MSE result.",
           "Permutations are empirical controls only; nonstationary time blocks do not establish exchangeability.",
           "No F03/R3/backbone/deep model was trained or read. TRAIN-only scaler and mapping banks; TEST used for reporting.","",
           "| Setting | Earlier-Z vs future rho | E10 earlier-Z accuracy | E11 Z route MSE | E11 best constant | E11 selected-A fraction | E12 risk rho |", 
           "|---|---:|---:|---:|---:|---:|---:|"]
    grouped={}
    for r in rows:
        key=(r.get("dataset"),r.get("channel"),r.get("L"),r.get("H"),r.get("K"),r.get("seed"))
        grouped.setdefault(key,[]).append(r)
    def get(group,experiment,field,feature=None,outcome=None):
        for r in group:
            if r.get("experiment")==experiment and (feature is None or r.get("feature")==feature) and (outcome is None or r.get("mapping_metric")==outcome):
                try:return f"{float(r.get(field,'')):.3f}"
                except (ValueError,TypeError):return "NA"
        return "NA"
    for key,group in sorted(grouped.items()):
        setting="/".join(str(x) for x in key)
        lines.append(f"| {setting} | {get(group,'E01_E05_E10','rho','older','dy_raw')} | "
            f"{get(group,'E10_triplet','pairwise_accuracy','older')} | "
            f"{get(group,'E11_mapping_swap','mse_match')} | "
            f"{get(group,'E11_mapping_swap','mse_best_constant')} | "
            f"{get(group,'E11_mapping_swap','selected_A_fraction')} | "
            f"{get(group,'E12_causal_early_warning','rho')} |")
    lines += ["", "Interpret E11 as adaptive identification only if selected-A fraction is nondegenerate, "
              "Z-guided beats the VAL-selected constant and permuted-Z controls, and independent time blocks agree.",
              "Every case retains E01–E12 CSVs, configuration, and Figures A–H when supported. "
              "Missing/insufficient-support results remain NA; full details are in all_results.csv."]
    (args.root/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":main()
