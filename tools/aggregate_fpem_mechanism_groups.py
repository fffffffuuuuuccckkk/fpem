#!/usr/bin/env python3
"""Final table and handoff for the grouping-only stage."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import numpy as np

METHODS=("G1_Switching_Ridge","G2_HRM_lite","G3_KerHRM_lite","G4_ANM_MM_lite",
         "G5_SSCM_lite","G6_CCSL_proxy","G7_Ridge_Update","G8_A0_Gradient_Direction",
         "G8_A0_Gradient_Amplitude","B0_Raw_History","B1_Random_Balanced","B2_ShuffleY_G1")
TABLE=("case","method","seed","independent_events_C","k_eff","collapsed",
       "within_mapping","between_mapping","cross_time_drift","future_lift",
       "teacher_only_mse_C","student_routed_ridge_mse_C","gain_vs_global_C",
       "gain_vs_random_C","uncertainty_error_spearman_C","total_time_sec","peak_vram_bytes","status")


def save_csv(path,rows,fields=()):
    columns=list(fields)
    for row in rows:
        for key in row:
            if key not in columns:columns.append(key)
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=columns,extrasaction="ignore")
        writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def display(value):
    if value is None or value=="":return "NA"
    if isinstance(value,float):return f"{value:.6g}"
    return str(value)


def pareto(rows):
    valid=[r for r in rows if r.get("status")=="ok" and r.get("cross_time_drift") is not None]
    survivors=[]
    for row in valid:
        if not any(other["case"]==row["case"] and other["method"]!=row["method"] and
                   other["gain_vs_global_C"]>=row["gain_vs_global_C"] and
                   other["cross_time_drift"]<=row["cross_time_drift"] and
                   (other["gain_vs_global_C"]>row["gain_vs_global_C"] or other["cross_time_drift"]<row["cross_time_drift"])
                   for other in valid):
            survivors.append(row)
    return survivors


def ensure_standard_assignment(directory,span):
    """Upgrade an early pilot artifact without recomputing its grouping."""
    path=directory/"assignments_train.npz"
    if not path.exists():return
    with np.load(path) as source:
        if "hard_group" in source.files and "event_start" in source.files:return
        payload={key:source[key] for key in source.files}
    payload["hard_group"]=payload.get("hard_id",payload["q_teacher"].argmax(axis=1)).astype(np.int8)
    starts=payload["window_start"].astype(np.int64)
    payload["event_start"]=starts
    payload["event_end"]=starts+int(span)
    temporary=directory/"assignments_train.upgrade.npz"
    np.savez_compressed(temporary,**payload)
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--root",required=True)
    args=parser.parse_args();root=Path(args.root);root.mkdir(parents=True,exist_ok=True)
    outputs={name:[] for name in ("summary","per_prototype","per_time_block","mapping_probe","router_diagnostic","ambiguity_bins")}
    failures=[];complete_case=[]
    for case in sorted((root/"pilots").glob("*/*/pred_*/seed_*")):
        config=case/"case_cache.json"
        if not config.exists():continue
        case_info=json.loads(config.read_text())
        for method in METHODS:
            directory=case/"methods"/method
            path=directory/"metrics.json"
            if not (directory/"complete").exists() or not path.exists():
                bad=directory/"failed.json"
                reason=json.loads(bad.read_text()).get("error") if bad.exists() else "pending"
                failures.append({"case":case_info["case"],"seed":case_info["seed"],"method":method,
                                 "status":"failed" if bad.exists() else "pending","reason":reason})
                outputs["summary"].append({"case":case_info["case"],"seed":case_info["seed"],
                                            "method":method,"status":"failed" if bad.exists() else "pending",
                                            "reason":reason})
                continue
            try:
                ensure_standard_assignment(directory,case_info["span"])
                result=json.loads(path.read_text())
                for name in outputs:
                    value=result[name]
                    if name=="summary":outputs[name].append(value)
                    else:outputs[name].extend(value)
            except (ValueError,KeyError) as error:
                failures.append({"case":case_info["case"],"seed":case_info["seed"],"method":method,
                                 "status":"invalid_metrics","reason":repr(error)})
        complete_case.append(case_info["case"])
    for name,rows in outputs.items():save_csv(root/(name+".csv"),rows,TABLE if name=="summary" else ("case","method","seed"))
    save_csv(root/"failure_log.csv",failures,("case","seed","method","status","reason"))
    rows=outputs["summary"]
    headings=("Case","Method","Seed","Independent Events","K_eff","Collapsed","Within Mapping","Between Mapping",
              "Cross-Time Drift","FutureLift","Teacher-only MSE (offline)","Student Routed Ridge MSE",
              "Gain vs Global","Gain vs Random","Uncertainty/Error Spearman","Time(s)","VRAM(bytes)","Status")
    lines=["# Mechanism-grouping quality table","",f"Rows: {len(rows)}; failed/pending: {len(failures)}. No weighted score.","",
           "|"+"|".join(headings)+"|","|"+"---|"*len(headings)]
    lines.extend("|"+"|".join(display(row.get(field)) for field in TABLE)+"|" for row in rows)
    lines.extend(["","C future is used only in offline teacher-oracle and diagnostics. Deployed C Ridge always uses the history-only student. Long-horizon independent-event support may be too small for strong stability claims."])
    (root/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    research="""# Grouping-stage handoff only

This result directory contains G1–G8 lite/proxy grouping methods plus B0 raw-history, B1 balanced random and B2 event-block shuffled-future controls. Standardized soft TRAIN assignments are at `pilots/<backbone>/<dataset>/pred_<horizon>/seed_<seed>/methods/<method>/assignments_train.npz`; each corresponding `prototype_meta.json` records K=8, A0 hash, split, support, mapping signature and normalization. This is **not** an F03/R3/Flow implementation or a causal-domain identification claim.

Read `summary.csv` and `summary.md` for all rows, `per_prototype.csv`, `per_time_block.csv`, `mapping_probe.csv`, `router_diagnostic.csv`, `ambiguity_bins.csv`, and `failure_log.csv` for details. C teacher-oracle MSE is cheating/offline and must not be ranked as deployable. C student-routed Ridge is the deployed proxy. Historical windows are embargoed by complete seq+pred span. Distinct channels and overlapping windows are not independent events. Even C belongs to original TRAIN: it is a held-out time block within TRAIN, not official TEST. Repeated C-based method choice would bias any claimed confirmatory result. G1–G6 approximate conditional mapping families; G7/G8 are update-fingerprint clustering. G6 is only a coefficient-support proxy, not CCSL causal graph learning. G3 uses fixed RFF Ridge, not full KerHRM. G4 uses bounded shrunk Gaussian variance, not full ANM-MM. G8 uses frozen A0 function gradients, not an original paper method.

For follow-up work, use only `assignments_train.npz` A/B1/B2 soft labels; C oracle labels are intentionally absent. Do not start downstream fusion automatically.
"""
    frontier=pareto(rows)
    research+="\n## Exploratory Pareto candidates (deployable gain up, drift down)\n\n"
    for row in frontier:research+=f"- {row['case']} / {row['method']}: gain={display(row.get('gain_vs_global_C'))}, drift={display(row.get('cross_time_drift'))}, independent C events={display(row.get('independent_events_C'))}\n"
    if not frontier:research+="- None estimable.\n"
    (root/"handoff_for_new_codex_window.md").write_text(research,encoding="utf-8")
    readme="""# Metric definitions

All grouping has K=8. A fits the grouping model, A+B1 teaches the diagnostic student router, A fits all full-horizon Ridge probes, B2 selects fixed alpha, and C is a later TRAIN holdout. Complete input+future windows are embargoed at boundaries. z is frozen-A0 H fitted with TRAIN-A StandardScaler and PCA16 (PCA8 fallback); v is the first eight future DCT coefficients standardized on TRAIN A. Full-future Ridge uses identical 64D historical key across methods. q_teacher in the standard artifact covers A/B1/B2 only and is available because these are TRAIN samples. C teacher is an offline oracle metric, not exported for downstream use. Student q_C depends only on z_C, hence only historical X.

FVE = 1 − prototype future-center MSE / TRAIN-A global future-center MSE. FutureLift = 1 − same-group future pair MSE / different-group random-control pair MSE in matched original-history-distance bins; NA when pair support is insufficient. K_eff = exp(entropy of A mean soft assignment). Within Mapping is weighted TRAIN-A z→v expert residual MSE in standardized DCT space. Between Mapping is the mean pairwise expert function-response squared distance on a common A z reference grid. Cross-Time Drift compares A and B2 expert responses after Hungarian alignment. Independent Events are greedy nonoverlapping seq+pred windows, not count of windows or channels; the exported event_id is only a coarse nonoverlapping-span bin, and event_start/end record the actual interval. Gain vs Global/Random = 1 − student-routed C MSE / baseline C MSE. Uncertainty/Error Correlation is Spearman correlation of candidate Ridge disagreement with deployed C squared error. Student/teacher agreement is diagnostic; low agreement alone does not invalidate a grouping when the mechanism is genuinely invisible from history. `failure_log.csv` gives concrete reasons for missing metrics. Peak VRAM is shared frozen-A0 data extraction peak, not per-group extra VRAM. TEST is not used to train, route or select alpha.

Method inspiration: HRM (PMLR 2021), KerHRM (NeurIPS 2021), Switching Regression (JMLR 2020), ANM-MM (NeurIPS 2018), SSCM (NeurIPS 2019), CCSL (arXiv 2111.09666). All are lite/proxy adaptations and make no causal-identification guarantee.
"""
    (root/"README.md").write_text(readme,encoding="utf-8")
    print(f"RESULTS rows={len(rows)} pending_or_failed={len(failures)}")
    print(" | ".join(headings))
    for row in rows:print(" | ".join(display(row.get(field)) for field in TABLE))


if __name__=="__main__":main()
