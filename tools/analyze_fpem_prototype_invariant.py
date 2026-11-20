"""Regenerate per-case summaries from actual completed markers; no invented rows."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_fpem_prototype_invariant_search import METHODS


def write_csv(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w",newline="",encoding="utf-8") as handle:
        w=csv.DictWriter(handle,fieldnames=fields);w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument("--case_root",required=True)
    cli=p.parse_args();root=Path(cli.case_root)
    shared=root/"shared";forward=root/"forward";fusion=root/"fusion";finetune=root/"finetune"
    methods=[];failures=[]
    if shared.exists():
        for folder in sorted(shared.iterdir()):
            if not folder.is_dir():continue
            path=folder/"metrics.json"
            if (folder/"complete").exists() and path.exists():
                data=json.loads(path.read_text())
                methods.append({"method":folder.name,"val_mse":data["val"]["mse"],
                    "val_mae":data["val"]["mae"],"test_mse":data["test"]["mse"],
                    "test_mae":data["test"]["mae"],
                    "trainable_parameters":data.get("trainable_parameters"),
                    "reference_sha256":data.get("reference_sha256")})
            elif path.exists() and not (folder/"complete").exists():
                failures.append({"stage":"shared","method":folder.name,"reason":"missing complete marker"})
    categories={
        "prototype_invariance_results.csv":lambda n:n.startswith(("risk_","shared_","conditional","adversarial_","pairwise")),
        "mapping_consensus_results.csv":lambda n:n.startswith("map_"),
        "representation_results.csv":lambda n:n in ("p01","p02","subspace","ref_global","ref_prototype","shared_private","shared_private_irm"),
        "ablation_results.csv":lambda n:n.endswith(("shuffled","random","time","uniform","hcluster","eiil","sample")),
    }
    write_csv(root/"shared_forecast_results.csv",methods)
    completed={row["method"] for row in methods}
    registry=[{"method":name,"stage":"J1 shared forecast",
               "status":"complete" if name in completed else "not_complete"}
              for name in METHODS]
    registry.extend([{"method":name,"stage":stage,
                      "status":"complete" if (root/stage/"complete").exists() else "not_complete"}
                     for name,stage in (("forward_oof","forward"),("partial_backbone","finetune/partial"),
                                        ("full_backbone","finetune/full"))])
    write_csv(root/"method_registry.csv",registry)
    for filename,predicate in categories.items():
        write_csv(root/filename,[r for r in methods if predicate(r["method"])])
    rows=[]
    for mode in ("partial","full"):
        path=finetune/mode/"metrics.json"
        if path.exists() and (finetune/mode/"complete").exists():
            data=json.loads(path.read_text())
            rows.append({"mode":mode,"method":data["method"],
                 "val_mse":data["val"]["mse"],"test_mse":data["test"]["mse"],
                 "test_mae":data["test"]["mae"],
                 "trainable_parameters":data["trainable_backbone_parameters"]+data["trainable_head_parameters"]})
    write_csv(root/"backbone_finetune_results.csv",rows)
    forward_data=json.loads((forward/"metrics.json").read_text()) if (forward/"complete").exists() else None
    if forward_data:
        write_csv(root/"time_forward_results.csv",[{"method":forward_data["method"],
             "a0_mse":forward_data["a0"]["mse"],"shared_mse":forward_data["shared"]["mse"],
             "r3_mix_mse":forward_data["r3_mix"]["mse"],"block_gain":forward_data["block_gain"],
             "embargo":forward_data["embargo"],"counts":forward_data["counts"]}])
    seed_rows=[]
    for folder in sorted(root.glob("seed_*/forward")):
        if (folder/"complete").exists() and (folder/"metrics.json").exists():
            data=json.loads((folder/"metrics.json").read_text())
            seed_rows.append({"seed":folder.parent.name,"method":data["method"],
                 "a0_oof_mse":data["a0"]["mse"],"shared_oof_mse":data["shared"]["mse"],
                 "r3_mix_oof_mse":data["r3_mix"]["mse"],
                 "worst_block_gain":min(data["block_gain"]) if data["block_gain"] else None})
    write_csv(root/"seed_forward_results.csv",seed_rows)
    proto_path=shared/"prototype_diagnostics.json"
    if proto_path.exists():
        proto=json.loads(proto_path.read_text())
        write_csv(root/"prototype_diagnostics.csv",
            [{"prototype":i,"mass":mass,"effective":proto["effective"][i],
              "null_train":proto["null_train"],"null_val":proto["null_val"],
              "null_test":proto["null_test"]}
             for i,mass in enumerate(proto["mass"])])
    fusion_rows=[]
    fusion_ablations=[]
    if fusion.exists():
        for folder in fusion.iterdir():
            path=folder/"metrics.json"
            if path.exists() and (folder/"complete").exists():
                data=json.loads(path.read_text())
                fusion_rows.append({"method":folder.name,"shared_mse":data["shared"]["mse"],
                    "ungated_mse":data["r3_ungated"]["mse"],
                    "recalibrated_mse":data["r3_recalibrated"]["mse"],
                    "protected_mse":data.get("protected_shared_benefit",{}).get("mse"),
                    "effect":data["mean_effect"],"active_ratio":data["active_ratio"],
                    "null_mean":data["null_mean"]})
                for key,value in data.items():
                    if key.startswith("ablation_") and isinstance(value,dict):
                        fusion_ablations.append({"shared_method":folder.name,
                            "control":key[len("ablation_"):] if key.startswith("ablation_") else key,"test_mse":value["mse"],
                            "test_mae":value["mae"]})
    write_csv(root/"fusion_results.csv",fusion_rows)
    write_csv(root/"fusion_ablation_results.csv",fusion_ablations)
    write_csv(root/"failure_log.csv",failures)
    if methods:
        base=next((r for r in methods if r["method"]=="A0"),None)
        p01=next((r for r in methods if r["method"]=="p01"),None)
        best_test=min(methods,key=lambda r:r["test_mse"])
        best_val=min(methods,key=lambda r:r["val_mse"])
        proto=[r for r in methods if r["method"] not in ("A0","linear","head_init","p01","p02")]
        best_proto=min(proto,key=lambda r:r["val_mse"]) if proto else None
        if best_val["method"] not in ("A0",) and (shared/best_val["method"]/"model.pt").exists():
            best_dir=root/"best_models"
            best_dir.mkdir(exist_ok=True)
            shutil.copy2(shared/best_val["method"]/"model.pt",best_dir/"best_val_shared.pt")
            (best_dir/"selection.json").write_text(json.dumps(best_val,indent=2)+"\n")
        lines=["# Prototype-guided shared forecasting — current completed evidence","",
               "TEST-based method ranking is exploratory development, not independent generalization evidence.","",
               f"Completed shared methods: {len(methods)}.",
               f"A0 TEST MSE: {base['test_mse'] if base else 'missing'}.",
               f"P01 TEST MSE: {p01['test_mse'] if p01 else 'missing'}.",
               f"Best VAL method: {best_val['method']} (VAL {best_val['val_mse']:.6f}, TEST {best_val['test_mse']:.6f}).",
               f"Exploratory best TEST: {best_test['method']} ({best_test['test_mse']:.6f})."]
        if best_proto:
            lines.append(f"Best prototype method by VAL: {best_proto['method']} (VAL {best_proto['val_mse']:.6f}, TEST {best_proto['test_mse']:.6f}).")
            if p01:
                lines.append("Prototype constraint improves P01 on VAL: "+str(best_proto["val_mse"]<p01["val_mse"])+".")
        if forward_data:
            lines.append(f"Strict TRAIN forward-OOF: A0 {forward_data['a0']['mse']:.6f}, shared {forward_data['shared']['mse']:.6f}; block gains {forward_data['block_gain']}.")
        if fusion_rows:
            best_fusion=min(fusion_rows,key=lambda r:r["recalibrated_mse"])
            lines.append(f"Best recalibrated fusion: {best_fusion['method']} ({best_fusion['recalibrated_mse']:.6f}); shared anchor {best_fusion['shared_mse']:.6f}.")
        if rows:
            lines.append("Fine-tune results: "+", ".join(f"{r['mode']} TEST {r['test_mse']:.6f}" for r in rows)+".")
        lines += ["","No causal invariance claim follows from prototype risk balance alone.",
                  "Unrun contingent methods/matrices must not be presented as completed."]
        (root/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":main()
