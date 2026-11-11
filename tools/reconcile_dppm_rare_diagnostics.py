"""Recompute NULL-aware rare candidate OOS diagnostics after a resumable pilot.

This corrects early pilot rows that treated NULL candidate mass as zero rather
than the frozen stable forecast. It never retrains or changes chosen models.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from models.predictive_env.dual_pattern_memory.prototype_bank import PatternBank
from models.predictive_env.dual_pattern_memory.rare_learning import RarePatternLearner
from tools.diagnose_predictive_rescue import sparse_record
from tools.run_fpem_dppm import BASE, CASES, normalized_record
from tools.run_fpem_dppm_rare import rare_query, score, stable_prediction, subset
from tools.run_fpem_shared_factorization import load_experiment, write_json

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("case",choices=CASES)
    parser.add_argument("--gpu",type=int,default=0)
    parser.add_argument("--seed",type=int,default=2021)
    parser.add_argument("--output",default="results/fpem_dual_pattern_memory/rare_learning_pilot")
    cli=parser.parse_args()
    torch.set_num_threads(4)
    out=ROOT/cli.output/cli.case
    cli.base_run_config=str(BASE/(f"{cli.case}_A0_complementary_gate_predictive_"
                        "sub0_orth0_refnone_fp0_rank8_bias-4/run_config.json"))
    cli.allow_legacy_reference_signature=cli.case.startswith("moderntcn")
    args,model,loaders,device,sha=load_experiment(cli)
    old=ROOT/"results/fpem_dual_pattern_memory/point_pilot"/cli.case/"dual_pattern_memory.pt"
    bundle=torch.load(old,map_location="cpu",weights_only=False)
    cache=torch.load(ROOT/"results/fpem_predictive_memory/pilot_20261024"/cli.case/
                     "phase0_cache.pt",map_location="cpu",weights_only=False)
    if sha!=bundle["reference_sha256"] or sha!=cache["reference_sha256"]:
        raise ValueError("frozen reference mismatch")
    config=json.loads((out/"run_config.json").read_text(encoding="utf-8"))
    stride=int(config["eval_stride"])
    stable=PatternBank.from_state(bundle["stable"])
    basis=bundle["basis"]
    eta=float(bundle["selection"]["eta"])
    raw=cache["records"]
    train=normalized_record({"x":raw["x"],"target":raw["target"],
                             "base":raw["a0_pred"]},bundle["floor"])
    starts=raw["starts"].long()
    first_val_history=len(loaders["train"].dataset.dataset.data_x)-args.seq_len
    valid=starts+args.seq_len+args.pred_len<=first_val_history
    train=subset(train,valid)
    val,val_ids=sparse_record(model,loaders["val"].dataset.dataset,
                              args,device,stride,bundle["floor"])
    cut=len(loaders["val"].dataset.dataset.data_x)//2
    first_test_history=len(loaders["val"].dataset.dataset.data_x)-args.seq_len
    early_mask=val_ids+args.seq_len+args.pred_len<=cut
    select_mask=(val_ids>=cut)&(val_ids+args.seq_len+args.pred_len<=first_test_history)
    records={"train":train,"early_val":subset(val,early_mask),
             "selection":subset(val,select_mask)}
    rows=[]
    for method in [f"R{i}" for i in range(8)]:
        target=out/method
        if not (target/"complete").exists():
            continue
        saved=torch.load(target/"rare_model.pt",map_location="cpu",weights_only=False)
        learner=PatternBank.from_state(saved["rare"]) if method=="R0" else RarePatternLearner.from_export(saved["rare"])
        offset=float(saved["null_offset"])
        candidate={}
        for name,record in records.items():
            anchor,sq=stable_prediction(record,stable,basis,eta)
            q=rare_query(method,learner,record["key"],sq,offset)
            pred=(q["weight"][...,None]*q["candidate"]).sum(2)+q["null_weight"][...,None]*anchor
            candidate[name]=score(record,pred)["mse"]
        mapping_path=target/"mapping_oos.json"
        content=json.loads(mapping_path.read_text(encoding="utf-8"))
        metrics=content["metrics"]
        metrics["train_candidate_mse_normalized"] = candidate["train"]
        metrics["early_val_candidate_mse"] = candidate["early_val"]
        metrics["selection_candidate_mse"] = candidate["selection"]
        metrics["early_val_rare_candidate_mse"] = candidate["early_val"]
        content["null_fallback"]="frozen DPPM stable forecast; diagnostics reconciled post-run"
        write_json(mapping_path,content)
        rows.append({"case":cli.case,"method":method,**metrics})
    if rows:
        columns=[]
        for row in rows:
            for key in row:
                if key not in columns:columns.append(key)
        with (out/"rare_mapping_oof.csv").open("w",newline="",encoding="utf-8") as handle:
            writer=csv.DictWriter(handle,fieldnames=columns,extrasaction="ignore")
            writer.writeheader();writer.writerows(rows)
    print(f"RECONCILED {cli.case}: {len(rows)} methods",flush=True)


if __name__=="__main__":
    main()
