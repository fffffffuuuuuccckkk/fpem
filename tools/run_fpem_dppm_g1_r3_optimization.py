"""Cached, chronological G1+R3 optimizer pilot. TEST never selects a candidate."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import sys
import time
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models.predictive_env.dual_pattern_memory.prototype_bank import PatternBank
from models.predictive_env.dual_pattern_memory.forecast_fusion import prototype_segment_reliability
from models.predictive_env.dual_pattern_memory.rare_learning import RarePatternLearner
from tools.diagnose_predictive_rescue import residual_ridge,ridge_query,sparse_record
from tools.diagnose_global_local_mapping import GLOBAL_SCALES
from tools.run_fpem_dppm import BASE,CASES,normalized_record
from tools.run_fpem_dppm_rare import subset,score,rare_forecast,rare_query,activation_stats
from tools.run_fpem_shared_factorization import load_experiment,write_json

RIDGES=(1e-4,1e-3,1e-2,1e-1,1.,10.,100.,1000.,10000.)
OUTPUT="results/fpem_dual_pattern_memory/g1_r3_optimization_pilot"

def args_parse():
    p=argparse.ArgumentParser()
    p.add_argument("case",choices=CASES)
    p.add_argument("--gpu",type=int,default=0)
    p.add_argument("--seed",type=int,default=2021)
    p.add_argument("--eval_stride",type=int,default=4)
    p.add_argument("--max_lr_trials",type=int,default=24)
    p.add_argument("--output",default=OUTPUT)
    return p.parse_args()

def append(path,row):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(row))
        if handle.tell()==0: writer.writeheader()
        writer.writerow(row)

def empty_csv(path,columns):
    if not path.exists():
        with path.open("w",newline="",encoding="utf-8") as handle:
            csv.writer(handle).writerow(columns)

def load_records(args):
    args.base_run_config=str(BASE/(f"{args.case}_A0_complementary_gate_predictive_"
                              "sub0_orth0_refnone_fp0_rank8_bias-4/run_config.json"))
    args.allow_legacy_reference_signature=args.case.startswith("moderntcn")
    config,model,loaders,device,sha=load_experiment(args)
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise AssertionError("Backbone must be frozen")
    args.seq_len,args.pred_len=config.seq_len,config.pred_len
    old=ROOT/"results/fpem_dual_pattern_memory/point_pilot"/args.case/"dual_pattern_memory.pt"
    bundle=torch.load(old,map_location="cpu",weights_only=False)
    cache=torch.load(ROOT/"results/fpem_predictive_memory/pilot_20261024"/args.case/
                     "phase0_cache.pt",map_location="cpu",weights_only=False)
    if bundle["reference_sha256"]!=sha or cache["reference_sha256"]!=sha:
        raise ValueError("A0 / memory cache checkpoint mismatch")
    raw=cache["records"]
    train=normalized_record({"x":raw["x"],"target":raw["target"],
                             "base":raw["a0_pred"]},bundle["floor"])
    starts=raw["starts"].long()
    first_val_history=len(loaders["train"].dataset.dataset.data_x)-args.seq_len
    valid=starts+args.seq_len+args.pred_len<=first_val_history
    train,starts=subset(train,valid),starts[valid]
    val,val_ids=sparse_record(model,loaders["val"].dataset.dataset,
                              config,device,args.eval_stride,bundle["floor"])
    test,_=sparse_record(model,loaders["test"].dataset.dataset,
                         config,device,args.eval_stride,bundle["floor"])
    cut=len(loaders["val"].dataset.dataset.data_x)//2
    first_test_history=len(loaders["val"].dataset.dataset.data_x)-args.seq_len
    early_mask=val_ids+args.seq_len+args.pred_len<=cut
    selection_mask=(val_ids>=cut)&(val_ids+args.seq_len+args.pred_len<=first_test_history)
    early,selection=subset(val,early_mask),subset(val,selection_mask)
    if min(len(early["key"]),len(selection["key"]))<8:
        raise ValueError("purged VAL support is too small")
    if int((val_ids[early_mask]+args.seq_len+args.pred_len).max())>int(val_ids[selection_mask].min()):
        raise AssertionError("early VAL future overlaps selection histories")
    if int((val_ids[selection_mask]+args.seq_len+args.pred_len).max())>first_test_history:
        raise AssertionError("selection VAL future overlaps TEST history")
    return train,starts,early,selection,test,sha

def select_g1(early,selection,test,output):
    path=output/"g1_ridge_results.csv"
    rows=[]
    residual=early["truth"]-early["base"]
    for ridge in RIDGES:
        weight=residual_ridge(early["key"],residual,ridge)
        sel_delta=ridge_query(selection["key"],weight)
        test_delta=ridge_query(test["key"],weight)
        for scale in GLOBAL_SCALES:
            val=score(selection,selection["base"]+scale*sel_delta)
            tst=score(test,test["base"]+scale*test_delta)
            rows.append({"ridge":ridge,"scale":scale,"val_mse":val["mse"],
                         "val_mae":val["mae"],"test_mse_exploratory":tst["mse"],
                         "test_mae_exploratory":tst["mae"]})
    if not path.exists():
        for row in rows: append(path,row)
    best=min(rows,key=lambda row:row["val_mse"])
    best_weight=residual_ridge(early["key"],residual,best["ridge"])
    return {"kind":"g1","weight":best_weight,"scale":best["scale"]},best

def lr_grid(n,seed):
    product=list(itertools.product((1e-4,3e-4,1e-3,3e-3),
                                  (.1,.3,1.),(0.,.1,.3,1.)))
    rng=random.Random(seed)
    selected=[]
    for lr in (1e-4,3e-4,1e-3,3e-3):
        sub=[item for item in product if item[0]==lr]
        rng.shuffle(sub)
        selected.extend(sub[:max(1,n//4)])
    for item in product:
        if len(selected)>=n: break
        if item not in selected: selected.append(item)
    return selected[:n]

def fit_r3(train,starts,early,args,reference,config):
    context={"reference_map":reference,"starts":starts,
             "span":args.seq_len+args.pred_len,"device":"cpu",
             "validation":(early["key"],early["truth"])}
    learner=RarePatternLearner("R3",patterns=8,rank=8,seed=int(config.get("seed",args.seed)),
                              epochs=int(config.get("epochs",10)),
                              optimizer_config={k:v for k,v in config.items() if k not in {"seed","epochs"}})
    return learner.fit(train["key"],train["truth"],context)

def evaluate(learner,stable,early,selection,starts,train,args):
    zero=early["truth"].new_zeros(early["truth"].shape[-1],0)
    anchor=early["base"]+stable["scale"]*ridge_query(early["key"],stable["weight"])
    eq=rare_query("R3",learner,early["key"],{},0.)
    reliability,evidence=prototype_segment_reliability(
        anchor,eq["candidate"],early["truth"],eq["weight"],8)
    # Candidate quality is R3-out-of-sample, but the G1 component was fitted
    # on early VAL. This is not a full-system OOF metric.
    candidate=(eq["weight"][...,None]*eq["candidate"]).sum(2)+eq["null_weight"][...,None]*anchor
    candidate_mse=score(early,candidate)["mse"]
    anchor_mse=score(early,anchor)["mse"]
    best=None
    selection_candidate_gain=None
    for offset in (-3.,-1.,0.,1.,3.):
        pred,diag=rare_forecast(selection,"R3",learner,stable,zero,0.,reliability,offset)
        metric=score(selection,pred)
        if best is None or metric["mse"]<best["val_mse"]:
            best={"offset":offset,"val_mse":metric["mse"],"val_mae":metric["mae"],
                  "val_effect":diag["effect"],"val_null":diag["null"]}
            sq=rare_query("R3",learner,selection["key"],{},offset)
            selection_anchor=selection["base"]+stable["scale"]*ridge_query(selection["key"],stable["weight"])
            selection_candidate=(sq["weight"][...,None]*sq["candidate"]).sum(2)+sq["null_weight"][...,None]*selection_anchor
            selection_candidate_gain=score(selection,selection_anchor)["mse"]-score(selection,selection_candidate)["mse"]
    stats=activation_stats("R3",learner,train,starts,args.seq_len+args.pred_len,stable)
    return best,reliability,evidence,{"early_candidate_mse":candidate_mse,
        "early_g1_mse":anchor_mse,"early_candidate_gain":anchor_mse-candidate_mse,
        "selection_candidate_gain":selection_candidate_gain,
        "reliability_mean":float(reliability.mean()),"active_per_series":stats["active_per_series"],
        "null_mean":stats["null_mean"],"independent_event_support":stats["independent_event_support"],
        "grad_last":learner.stats["loss_trace"][-1]["grad_norm"]}

def main():
    args=args_parse()
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    output=ROOT/args.output/args.case
    output.mkdir(parents=True,exist_ok=True)
    train,starts,early,selection,test,sha=load_records(args)
    stable,g1_best=select_g1(early,selection,test,output)
    zero=test["truth"].new_zeros(test["truth"].shape[-1],0)
    g1_val=score(selection,selection["base"]+stable["scale"]*ridge_query(selection["key"],stable["weight"]))
    g1_test=score(test,test["base"]+stable["scale"]*ridge_query(test["key"],stable["weight"]))
    prior=ROOT/"results/fpem_dual_pattern_memory/g1_plus_r3_pilot"/args.case/"R3/test_metrics.json"
    prior_metrics=json.loads(prior.read_text(encoding="utf-8")) if prior.exists() else None
    write_json(output/"run_config.json",{"case":args.case,"seed":args.seed,"reference_sha256":sha,
        "eval_stride":args.eval_stride,"max_lr_trials":args.max_lr_trials,
        "g1_selected_by":"later purged VAL MSE","g1":g1_best,"g1_val":g1_val,"g1_test":g1_test,
        "previous_g1_r3_test":prior_metrics["test"]["full"] if prior_metrics else None,
        "source":"TRAIN R3 targets; early purged VAL fits G1 and estimates reliability; "
                 "later purged VAL selects configuration and NULL offset; TEST is report only",
        "caveat":"early candidate diagnostic is R3-OOS but G1-in-sample, not full-system OOF"})
    reference=residual_ridge(train["key"],train["truth"],100.)
    result=output/"lr_group_results.csv"
    existing=set()
    if result.exists():
        with result.open(newline="",encoding="utf-8") as handle:
            existing={row["config_id"] for row in csv.DictReader(handle)}
    configs=[]
    for index,(head,dr,br) in enumerate(lr_grid(args.max_lr_trials,args.seed)):
        configs.append((f"L{index:02d}",{"head_lr":head,"dictionary_ratio":dr,
                                        "basis_ratio":br,"route_ratio":dr,
                                        "optimizer":"adam","schedule":"constant"}))
    for config_id,config in configs:
        if config_id in existing: continue
        print("FIT",args.case,config_id,config,flush=True)
        started=time.monotonic()
        learner=fit_r3(train,starts,early,args,reference,config)
        selected,reliability,evidence,diag=evaluate(learner,stable,early,selection,starts,train,args)
        ckpt=output/"checkpoints"/f"{config_id}.pt"
        ckpt.parent.mkdir(exist_ok=True)
        torch.save({"rare":learner.export(),"config":config,
                    "reliability":reliability,"offset":selected["offset"],
                    "g1_weight":stable["weight"],"g1_scale":stable["scale"]},ckpt)
        row={"case":args.case,"config_id":config_id,"head_lr":config["head_lr"],
             "dictionary_ratio":config["dictionary_ratio"],"basis_ratio":config["basis_ratio"],
             "route_ratio":config["route_ratio"],"val_mse":selected["val_mse"],
             "val_gain_vs_g1":g1_val["mse"]-selected["val_mse"],
             "early_candidate_gain":diag["early_candidate_gain"],
             "selection_candidate_gain":diag["selection_candidate_gain"],
             "reliability_mean":diag["reliability_mean"],"null_mean":selected["val_null"],
             "train_seconds":time.monotonic()-started}
        append(result,row)
        append(output/"rare_gradient_and_activation.csv",{"case":args.case,"config_id":config_id,
            "grad_mapping":diag["grad_last"]["mapping"],"grad_dictionary":diag["grad_last"]["dictionary"],
            "grad_basis":diag["grad_last"]["basis"],"grad_routing":diag["grad_last"]["routing"],
            "early_candidate_mse":diag["early_candidate_mse"],"early_g1_mse":diag["early_g1_mse"],
            "early_candidate_gain":diag["early_candidate_gain"],"reliability_mean":diag["reliability_mean"],
            "selection_candidate_gain":diag["selection_candidate_gain"],
            "active_per_series":diag["active_per_series"],"null_mean":diag["null_mean"],
            "independent_event_support":json.dumps(diag["independent_event_support"])})
        print("DONE",config_id,"val_gain",row["val_gain_vs_g1"],
              "candidate_gain",row["early_candidate_gain"],flush=True)
    with result.open(newline="",encoding="utf-8") as handle:
        rows=list(csv.DictReader(handle))
    best=sorted(rows,key=lambda row:float(row["val_mse"]))[:3]
    # Search the next family only when the rare candidate itself is positive
    # and the final VAL forecast beats the G1 anchor.
    eligible=[row for row in rows if float(row["selection_candidate_gain"])>0 and
              float(row["val_gain_vs_g1"])>0 and float(row["reliability_mean"])>0]
    schedules=output/"training_schedule_results.csv"
    optimizers=output/"optimizer_comparison.csv"
    comparison_columns=["case","config_id","optimizer","order","schedule","val_mse","val_gain_vs_g1"]
    empty_csv(schedules,comparison_columns)
    empty_csv(optimizers,comparison_columns)
    advanced=[]
    if eligible:
        anchor=min(eligible,key=lambda row:float(row["val_mse"]))
        base_config=next(cfg for name,cfg in configs if name==anchor["config_id"])
        variants=[("S1",{"order":"S1"}),("S2",{"order":"S2"}),("S3",{"order":"S3"}),
                  ("adamw",{"optimizer":"adamw"}),("adamw_lbfgs",{"optimizer":"adamw_lbfgs"}),
                  ("cosine",{"schedule":"cosine"}),("warmup_cosine",{"schedule":"warmup_cosine"}),
                  ("plateau",{"schedule":"plateau"}),
                  ("wd",{"weight_decay":1e-4}),("ema",{"ema_decay":.99}),
                  ("early_stop",{"early_stop_patience":3,"restore_best":True}),
                  ("basis_init",{"basis_init_scale":.5})]
        for name,change in variants:
            config={**base_config,**change}
            learner=fit_r3(train,starts,early,args,reference,config)
            selected,reliability,evidence,diag=evaluate(learner,stable,early,selection,starts,train,args)
            row={"case":args.case,"config_id":name,"val_mse":selected["val_mse"],
                 "val_gain_vs_g1":g1_val["mse"]-selected["val_mse"],
                 "early_candidate_gain":diag["early_candidate_gain"],
                 "selection_candidate_gain":diag["selection_candidate_gain"],
                 "reliability_mean":diag["reliability_mean"]}
            advanced.append((row,learner,reliability,selected,diag,config))
            target=optimizers if name in {"adamw","adamw_lbfgs","wd","ema"} else schedules
            with target.open("a",newline="",encoding="utf-8") as handle:
                writer=csv.writer(handle)
                writer.writerow([args.case,name,config.get("optimizer","adam"),
                                 config.get("order","S0"),config.get("schedule","constant"),
                                 row["val_mse"],row["val_gain_vs_g1"]])
        # The top three VAL-ranked settings receive independent rare-module
        # seeds. The A0 checkpoint, G1 fit and split stay fixed.
        ranked=[(name,cfg) for name,cfg in configs if name in {row["config_id"] for row in best}]
        for name,cfg in ranked[:3]:
            for seed in (2021,2022,2023):
                trial={**cfg,"seed":seed}
                learner=fit_r3(train,starts,early,args,reference,trial)
                selected,reliability,evidence,diag=evaluate(learner,stable,early,selection,starts,train,args)
                append(output/"seed_results.csv",{"case":args.case,"config_id":name,
                    "seed":seed,"val_mse":selected["val_mse"],
                    "val_gain_vs_g1":g1_val["mse"]-selected["val_mse"],
                    "selection_candidate_gain":diag["selection_candidate_gain"],
                    "reliability_mean":diag["reliability_mean"]})
    # TEST is read for the VAL-selected candidate only, plus two pre-existing anchors.
    selected_id=best[0]["config_id"]
    selected_ckpt=torch.load(output/"checkpoints"/f"{selected_id}.pt",map_location="cpu",weights_only=False)
    if advanced:
        candidate=min(advanced,key=lambda item:item[0]["val_mse"])
        if candidate[0]["val_mse"]<float(best[0]["val_mse"]):
            selected_id=candidate[0]["config_id"]
            _,learner,reliability,selected,diag,config=candidate
            selected_ckpt={"rare":learner.export(),"reliability":reliability,
                           "offset":selected["offset"],"config":config}
    selected_learner=RarePatternLearner.from_export(selected_ckpt["rare"])
    pred,_=rare_forecast(test,"R3",selected_learner,stable,zero,0.,
                         selected_ckpt["reliability"],selected_ckpt["offset"])
    final_test=score(test,pred)
    all_row={"case":args.case,"selected_config":selected_id,
             "a0_mse":score(test,test["base"])["mse"],"g1_mse":g1_test["mse"],
             "g1_mae":g1_test["mae"],
             "original_g1_r3_mse":prior_metrics["test"]["full"]["mse"] if prior_metrics else "",
             "optimized_g1_r3_mse":final_test["mse"],"optimized_g1_r3_mae":final_test["mae"],
             "incremental_gain_vs_g1":g1_test["mse"]-final_test["mse"],
             "selected_val_mse":float(next(row["val_mse"] for row in rows if row["config_id"]==best[0]["config_id"]))
             if selected_id==best[0]["config_id"] else candidate[0]["val_mse"],
             "advanced_search_started":bool(eligible)}
    append(output/"all_results.csv",all_row)
    write_json(output/"selected_checkpoint.json",{"config_id":selected_id,
        "g1":g1_best,"test":final_test,"g1_test":g1_test,
        "advanced_search_started":bool(eligible)})
    (output/"summary.md").write_text(
        f"# G1 + R3 optimization pilot: {args.case}\n\n"
        f"G1 VAL-selected ridge={g1_best['ridge']}, scale={g1_best['scale']}; "
        f"TEST MSE={g1_test['mse']:.6f}.\n\n"
        f"Selected R3={selected_id}; TEST MSE={final_test['mse']:.6f}, "
        f"MAE={final_test['mae']:.6f}; gain vs G1="
        f"{g1_test['mse']-final_test['mse']:.6f}.\n\n"
        f"Stage 3 optimizer/schedule search {'ran' if eligible else 'stopped'}: "
        f"{'positive OOS candidate, VAL and reliability evidence' if eligible else 'no jointly positive candidate/VAL/reliability evidence'}.\n\n"
        "R3 TRAIN cache uses in-sample A0; early VAL is R3-out-of-sample but "
        "G1-in-sample; TEST was not used to select settings.\n",encoding="utf-8")
    (output/"complete").write_text("complete\n",encoding="utf-8")
    print("COMPLETE",args.case,selected_id,final_test,flush=True)

if __name__=="__main__":
    main()
