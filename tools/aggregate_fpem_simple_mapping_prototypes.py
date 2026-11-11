#!/usr/bin/env python3
"""Aggregate resumable per-combination records; never silently drop failures."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.fpem_simple_mapping_descriptors import NAMES, DIMS, combinations

TABLE = ("case", "seed", "method", "feature_dim", "fve_C", "future_lift_C", "mapping_drift_C",
         "route_agreement_B2", "k_eff_A", "independent_events_C", "global_ridge_mse_C",
         "prototype_ridge_mse_C", "gain_vs_global_C", "gain_vs_random_C",
         "disagreement_error_correlation_C", "train_time_sec", "peak_vram_bytes")


def write_csv(path, records, preferred=()):
    fields = list(preferred)
    for record in records:
        for field in record:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({key: json.dumps(value,ensure_ascii=False) if isinstance(value,(dict,list)) else value
                             for key,value in record.items()})


def numeric(value):
    try:
        return float(value) if value is not None else np.nan
    except (TypeError,ValueError):
        return np.nan


def plots(root, summary, bins, proto, blocks):
    valid=[r for r in summary if r.get("status")=="ok"]
    if not valid:
        return
    fig,ax=plt.subplots(figsize=(8,5))
    cases=sorted(set(r["case"] for r in valid))
    for case in cases:
        rr=[r for r in valid if r["case"]==case and np.isfinite(numeric(r.get("mapping_drift_C")))]
        ax.scatter([numeric(r["mapping_drift_C"]) for r in rr],
                   [numeric(r["gain_vs_global_C"]) for r in rr],s=7,alpha=.45,label=case)
    ax.axhline(0,color="black",lw=.6);ax.set(xlabel="Cross-time mapping drift (lower)",ylabel="Deployable gain vs Global Ridge (higher)")
    ax.legend(fontsize=7);fig.tight_layout();fig.savefig(root/"gain_drift_pareto.png",dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    vals=[numeric(r.get("gain_vs_global_C")) for r in valid]
    ax.hist([v for v in vals if np.isfinite(v)],bins=40)
    ax.set(xlabel="Deployable gain vs Global Ridge",ylabel="Number of combinations")
    fig.tight_layout();fig.savefig(root/"all_combination_gain.png",dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    groups=[[numeric(r.get("gain_vs_global_C")) for r in valid if r.get("descriptor_count")==size]
            for size in range(1,9)]
    ax.boxplot([[v for v in values if np.isfinite(v)] for values in groups],showfliers=False)
    ax.set(xlabel="Number of descriptor groups",ylabel="Deployable gain vs Global Ridge")
    fig.tight_layout();fig.savefig(root/"gain_by_combination_order.png",dpi=150);plt.close(fig)
    # Descriptive occurrence among the best decile within each case; not causal importance.
    frequency={name:0 for name in NAMES}
    for case in cases:
        rr=sorted([r for r in valid if r["case"]==case],key=lambda r:numeric(r.get("gain_vs_global_C")),reverse=True)
        for row in rr[:max(1,int(.1*len(rr)))]:
            for name in row["method"].split("+"):
                frequency[name]+=1
    fig,ax=plt.subplots(figsize=(7,4));ax.bar(list(frequency),list(frequency.values()))
    ax.set(xlabel="Descriptor",ylabel="Count among top 10% by case (descriptive only)")
    fig.tight_layout();fig.savefig(root/"feature_occurrence_top_decile.png",dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for case in cases:
        rr=[r for r in bins if r.get("case")==case]
        if not rr: continue
        ax.scatter([numeric(r["disagreement_mean"]) for r in rr],
                   [numeric(r["error_mse"]) for r in rr],s=3,alpha=.25,label=case)
    ax.set(xlabel="Student-route candidate disagreement",ylabel="True downstream squared error")
    ax.legend(fontsize=7);fig.tight_layout();fig.savefig(root/"disagreement_vs_error.png",dpi=150);plt.close(fig)
    # One heatmap per case for the best deployable-gain combination (selected on C;
    # visualization is exploratory and cannot serve as unbiased selection evidence).
    for case in cases:
        rr=[r for r in valid if r["case"]==case]
        best=max(rr,key=lambda r:numeric(r.get("gain_vs_global_C")))
        records=[r for r in proto if r.get("case")==case and r.get("method")==best["method"]]
        if not records: continue
        matrix=np.full((8,4),np.nan)
        for row in records:
            k=int(row["prototype"])
            j=("A","B1","B2","C").index(row["block"])
            matrix[k,j]=numeric(row.get("mapping_drift_from_B1"))
        fig,ax=plt.subplots(figsize=(6,4));im=ax.imshow(matrix,aspect="auto")
        ax.set(xticks=range(4),xticklabels=("A","B1","B2","C"),yticks=range(8),
               xlabel="Time block",ylabel="Prototype",title=case+" / "+best["method"])
        fig.colorbar(im,ax=ax);fig.tight_layout()
        fig.savefig(root/(case.replace("/","_").replace("-","_")+"_prototype_block_heatmap.png"),dpi=150)
        plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--root",required=True)
    parser.add_argument("--previous_root",default="results/fpem_relation_operator_prototype_pilot")
    args=parser.parse_args();root=Path(args.root);root.mkdir(parents=True,exist_ok=True)
    registry=[dict(index=i,method="+".join(group),descriptor_count=len(group),
                   feature_dim=sum(DIMS[int(name[1:])-1] for name in group),groups=list(group))
              for i,group in enumerate(combinations())]
    assert len(registry)==255 and len({r["method"] for r in registry})==255
    write_csv(root/"feature_registry.csv",registry,("index","method","descriptor_count","feature_dim","groups"))
    outputs={key:[] for key in ("summary","per_prototype","per_time_block","mapping_probe","routing_diagnostics","disagreement_bins")}
    failure=[]
    for case in sorted((root/"pilots").glob("*/*/pred_*/seed_*")):
        config=case/"case_cache.json"
        if not config.exists():
            continue
        case_info=json.loads(config.read_text())
        case_name=case_info["case"];seed=case_info["seed"]
        for item in registry:
            method=item["method"]
            path=case/"combos"/(f'{item["index"]:03d}_'+method+".json")
            if not path.exists():
                reason_file=case/"failures"/(f'{item["index"]:03d}_'+method+".json")
                reason=json.loads(reason_file.read_text()).get("error") if reason_file.exists() else "pending"
                failure.append(dict(case=case_name,seed=seed,method=method,status="failed" if reason_file.exists() else "pending",reason=reason))
                continue
            try:
                result=json.loads(path.read_text())
                for key in outputs:
                    if key=="summary":
                        outputs[key].append(result["summary"])
                        continue
                    for row in result.get(key,[]):
                        outputs[key].append({"case":case_name,"seed":seed,**row})
            except (ValueError,KeyError) as error:
                failure.append(dict(case=case_name,seed=seed,method=method,status="invalid_json",reason=repr(error)))
    # P0/P1 are reused only for identical case, A0 checkpoint, seed, and the
    # same chronological protocol. Their old operator-drift definition is
    # deliberately kept in a separate column, not mixed with the new drift.
    old_path=Path(args.previous_root)/"summary.csv"
    old_probes=Path(args.previous_root)/"mapping_probe.csv"
    if old_path.exists():
        old_probe_rows=list(csv.DictReader(old_probes.open(encoding="utf-8"))) if old_probes.exists() else []
        for old in csv.DictReader(old_path.open(encoding="utf-8")):
            if old.get("method") not in ("P0_Raw","P1_Relation_Fingerprint") or old.get("status")!="ok":
                continue
            case_info=next((json.loads(path.read_text()) for path in (root/"pilots").glob("*/*/pred_*/seed_*/case_cache.json")
                            if json.loads(path.read_text()).get("case")==old.get("case")),None)
            if case_info is None or case_info.get("a0_sha256")!=old.get("a0_sha256") or str(case_info.get("seed"))!=old.get("seed"):
                continue
            prior_global=next((p for p in old_probe_rows if p.get("case")==old.get("case") and
                               p.get("method")==old.get("method") and p.get("probe")=="Global Ridge"),None)
            def f(key):
                try:return float(old[key]) if old.get(key) else None
                except ValueError:return None
            outputs["summary"].append(dict(case=old["case"],seed=int(old["seed"]),method=old["method"],
                source_protocol="reused_previous_identical_A0_and_temporal_split",
                status="historical_reused",feature_dim=64 if old["method"]=="P0_Raw" else 11,
                fve_C=f("FVE"),future_lift_C=f("FutureLift"),future_lift_reason=old.get("FutureLift_reason"),
                mapping_drift_C=None,legacy_operator_drift_C=f("Operator_Drift"),
                route_agreement_B2=f("Route_Agreement"),k_eff_A=f("K_eff"),
                independent_events_C=f("Independent_Events"),
                global_ridge_mse_C=float(prior_global["C_mse"]) if prior_global and prior_global.get("C_mse") else None,
                prototype_ridge_mse_C=f("Ridge_MSE"),gain_vs_global_C=f("Gain_vs_Global"),
                gain_vs_random_C=f("Gain_vs_Random"),random_ridge_mse_C=f("Random_Ridge_MSE"),
                train_time_sec=f("Train_Time"),peak_vram_bytes=f("Peak_VRAM"),
                raw_coverage_mean_C=f("Raw_Coverage_Mean"),raw_coverage_p90_C=f("Raw_Coverage_P90"),
                raw_coverage_p95_C=f("Raw_Coverage_P95")))
    for name,rows in outputs.items():
        write_csv(root/(name+".csv"),rows,TABLE if name=="summary" else ("case","seed","method"))
    write_csv(root/"failure_log.csv",failure,("case","seed","method","status","reason"))
    plots(root,outputs["summary"],outputs["disagreement_bins"],outputs["per_prototype"],outputs["per_time_block"])
    readme="""# Simple mapping prototype quality sweep

All 255 nonempty subsets of M1–M8 are enumerated in feature_registry.csv. Each descriptor is extracted once per case; robust median/IQR scaling is fit on TRAIN A, invalid dimensions are masked, and each group is divided by sqrt(valid dimension count). K=8 MiniBatchKMeans on TRAIN A descriptor vectors creates offline teacher soft labels. A 64→32→8 history-only MLP learns teacher labels on A+B1 (40 epochs). C deployment uses only the student; C teacher labels appear solely in an explicitly offline diagnostic.

Original TRAIN windows are split chronologically A/B/C, with full history+future spans purged at every boundary; B is similarly divided into B1/B2. B1 fits all 64D history→future Ridge probes, B2 selects alpha, C reports holdout. A0 checkpoint is reused frozen; no A0/F03/R3/Flow training. Global and Shared Mapping are mathematically the same objective under normalized soft responsibilities; the result is reused rather than inventing an artificial difference. Random Group permutes student assignment rows within each block. Historical and future values share history-only normalization. M3 period is estimated from TRAIN A only, with persistence fallback if its cycle cannot be represented in history.

MSE and MAE are on full future in historical-normalized units. FVE = 1 − prototype-center prediction MSE / TRAIN-A-global-center MSE on the same block. FutureLift = 1 − within-prototype future pair MSE / different-prototype control pair MSE after matching by original 64D history-distance decile; unavailable when pair support is insufficient. Mapping drift is mean squared difference between B1-fitted and C-fitted four-segment future Ridge responses on the same A reference histories; C fitting is offline diagnosis, not a deployed predictor. Mapping separation compares between-prototype response distance against B1→B2 within-prototype response distance. K_eff = exp(entropy of TRAIN-A soft prototype mass). Independent events are greedy nonoverlapping history+future windows; repeated channels do not create new events. Prototype Ridge gain = 1 − its C MSE / comparison C MSE. Disagreement–error correlation is Pearson correlation between student-weighted variance of the eight candidate forecasts and true squared residual on C.

Missing fields are blank/NA, never zero. failure_log.csv distinguishes pending from failed tasks. Probe selection uses B2; C/TEST ranking is exploratory because repeated method comparison uses C. Hold out a new time block or seed for confirmatory claims. Timing includes per-combination CPU training; peak VRAM is the shared frozen-A0 data extraction peak and is not incremental per combination. CSV preserves all 255 rows per completed case. Heatmap uses the best C gain only for visualization, not unbiased method selection. Top-decile descriptor frequency is descriptive, not causal feature importance.

P0_Raw and P1_Relation_Fingerprint are reused from the previous run only when case, A0 SHA, seed, and time split match. Their previous complex local-operator drift is kept as legacy_operator_drift_C, not compared numerically with new mapping_drift_C. P2_Local_Operator is deliberately excluded.
"""
    (root/"README.md").write_text(readme,encoding="utf-8")
    rows=outputs["summary"]
    headings=("Case","Seed","Method","Feature Dim","FVE","FutureLift","Mapping Drift","Route Agreement","K_eff","Independent Events","Global Ridge MSE","Prototype Ridge MSE","Gain vs Global","Gain vs Random","Disagreement–Error Corr","Train Time(s)","Peak VRAM(bytes)")
    def show(row):
        cells=[row.get(field) for field in TABLE]
        return ["NA" if value is None else f"{value:.6g}" if isinstance(value,float) else str(value) for value in cells]
    text=["# Complete prototype quality table","",f"Completed rows: {len(rows)}; pending/failed: {len(failure)}. No weighted overall score.","",
          "| " + " | ".join(headings) + " |","|"+"---|"*len(headings)]
    text.extend("| " + " | ".join(show(row)) + " |" for row in rows)
    text.extend(["","Selection on C/TEST is exploratory. Refer to README.md and failure_log.csv for definitions and missingness."])
    (root/"summary.md").write_text("\n".join(text)+"\n",encoding="utf-8")
    print(f"RESULTS rows_including_reused_baselines={len(rows)} pending_or_failed={len(failure)} expected_new_per_case=255")
    print(" | ".join(headings))
    for row in rows:
        print(" | ".join(show(row)))


if __name__=="__main__":
    main()
