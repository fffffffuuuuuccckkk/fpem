"""Sequential DPPM V -> C -> R3 pilot with purged chronological memory OOF.

Backbone is frozen. V and C select only on first-half VAL. R3 scales are
selected on the same VAL subset; TEST is retrospective reporting, never fit.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models.predictive_env.dual_pattern_memory import PatternBank
from tools.run_fpem_dppm import (BASE,CASES,normalized_record,variation_key)
from tools.diagnose_fpem_dppm import purged_source_mask
from tools.run_fpem_shared_factorization import load_experiment,write_json
from tools.run_predictive_env_iv_patchtst import IndexedDataset
from tools.run_fpem_smrva import collect_split

ALPHAS=(0.,.01,.025,.05,.1,.2,.5,1.)


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("case",choices=CASES)
    p.add_argument("--gpu",type=int,default=0)
    p.add_argument("--seed",type=int,default=2021)
    p.add_argument("--output",default="results/fpem_dual_pattern_memory/next_pilot")
    p.add_argument("--max_batches",type=int,default=0,help="smoke only")
    return p.parse_args()


def history_description(u,stable_bank,mode):
    if mode=="V2":return u
    v=variation_key(stable_bank,u)
    if mode in ("V0","V3"):return v
    if mode=="V1":
        n,c,d=v.shape
        difference=F.interpolate((v[...,1:]-v[...,:-1]).reshape(-1,1,d-1),
                                  size=d//2,mode="linear",align_corners=False)
        trend=F.avg_pool1d(F.pad(v.reshape(-1,1,d),(2,2),mode="replicate"),
                            kernel_size=5,stride=1)
        trend=F.interpolate(trend,size=d-d//2,mode="linear",align_corners=False)
        return torch.cat((difference,trend),-1).reshape(n,c,d).clamp(-5,5)
    raise ValueError(mode)


def mapping_feature(u,description,input_mode):
    zero=torch.zeros_like(u)
    if input_mode=="description":return torch.cat((description,zero),-1)
    if input_mode=="concat":return torch.cat((u,description),-1)
    if input_mode=="history":return torch.cat((u,zero),-1)
    raise ValueError(input_mode)


def block_shuffle(feature,blocks,seed):
    # Permute entire [C,D] sample rows within each TRAIN time block.
    shuffled=feature.clone()
    generator=torch.Generator().manual_seed(seed)
    for block in blocks.unique():
        rows=(blocks==block).nonzero(as_tuple=True)[0]
        if len(rows)>1:
            shuffled[rows]=feature[rows[torch.randperm(len(rows),generator=generator)]]
    return shuffled


def shared_future_basis(target,rank):
    y=target.reshape(-1,target.shape[-1]).double()
    gram=y.T@y
    _,vectors=torch.linalg.eigh(gram)
    return vectors[:,-rank:].float().contiguous()


def fit_memory(source,all_data,blocks,settings,seed,stage="variation"):
    key,truth=all_data["key"],all_data["truth"]
    ridge=float(settings.get("ridge",.05))
    ms=int(settings.get("ms",16));mv=int(settings.get("mv",8))
    stable=PatternBank(ms,truth.shape[-1],key.shape[-1],ridge=ridge,seed=seed)
    stable.fit(key[source],truth[source],blocks[source],
               output_basis=(shared_future_basis(truth[source],settings["rank"])
                             if settings.get("rank") else None))
    if stage=="stable":return stable,None
    representation=settings.get("representation","V0")
    input_mode=settings.get("input","concat")
    description=history_description(key[source],stable,representation)
    if representation=="V3":description=block_shuffle(description,blocks[source],seed+109)
    feature=mapping_feature(key[source],description,input_mode)
    if settings.get("control")=="global":
        description=key[source]
        feature=key[source]
        mv=1
    elif settings.get("control")=="single_bank":
        description=key[source]
        feature=key[source]
    if settings.get("control")=="shuffle_xy":
        # Keep the same temporal-block future marginal but break X->Y pairs.
        target=block_shuffle(truth[source],blocks[source],seed+211)
    else:target=truth[source]
    future_basis=(shared_future_basis(target,settings["rank"])
                  if settings.get("rank") else None)
    variation=PatternBank(mv,truth.shape[-1],description.shape[-1],ridge=ridge,
                          seed=seed+11)
    variation.fit(description,target,blocks[source],feature,output_basis=future_basis)
    return stable,variation


def query_memory(data,stable,variation,settings):
    key=data["key"]
    control=settings.get("control")
    if control in ("global","single_bank"):
        description=key;feature=key
    else:
        mode=settings.get("representation","V0")
        if mode=="V3":mode="V0"  # source features alone are shuffled
        description=history_description(key,stable,mode)
        feature=mapping_feature(key,description,settings.get("input","concat"))
    query=variation.query(description,feature)
    return query


def mse_mae(pred,truth,data):
    raw=(pred-truth)*data["scale"].transpose(1,2)
    return float(raw.square().mean()),float(raw.abs().mean())


def entropy_and_support(bank,train_data,starts,seq_len,pred_len,settings):
    key=train_data["key"]
    rep=settings.get("representation","V0")
    if settings.get("control") in ("global","single_bank"):description=key
    else:description=history_description(key,bank[0],"V0" if rep=="V3" else rep)
    matched=bank[1].query(description, key if settings.get("control") in
                          ("global","single_bank") else
                          mapping_feature(key,description,settings.get("input","concat")))
    weights=matched["weight"]
    entropy=-(weights*weights.clamp_min(1e-9).log()).sum(-1)
    assignment=weights.argmax(-1)
    effective=[];dominant=[]
    for proto in range(bank[1].n_patterns):
        owned=(assignment==proto)
        dominant.append(int(owned.sum()))
        # One event cannot contribute two overlapping future windows.
        anchors=owned.any(1).nonzero(as_tuple=True)[0].tolist()
        last_end=-10**12;independent=0
        for idx in anchors:
            if int(starts[idx])>=last_end:
                independent+=1;last_end=int(starts[idx])+seq_len+pred_len
        effective.append(independent)
    return {"match_entropy_mean":float(entropy.mean()),
            "match_entropy_normalized":float((entropy.mean()/torch.tensor(bank[1].n_patterns).log()).item()) if bank[1].n_patterns>1 else 0.,
            "positive_weight_count_mean":float((weights>0).float().sum(-1).mean()),
            "dominant_pairs_per_prototype":dominant,
            "independent_events_per_prototype":effective,
            "map_parameters":int(bank[1].mapping.numel()),
            "stable_parameters":int(bank[0].mapping.numel())}


def evaluate_candidate(data,stable,variation,settings,alpha=None):
    query=query_memory(data,stable,variation,settings)
    base=data["base"]
    candidate=query["prediction"]
    if alpha is None:alpha=1.
    final=base+float(alpha)*(candidate-base)
    base_mse,base_mae=mse_mae(base,data["truth"],data)
    candidate_mse,candidate_mae=mse_mae(candidate,data["truth"],data)
    final_mse,final_mae=mse_mae(final,data["truth"],data)
    return {"a0_mse":base_mse,"a0_mae":base_mae,
            "candidate_mse":candidate_mse,"candidate_mae":candidate_mae,
            "mse":final_mse,"mae":final_mae,
            "gain":base_mse-final_mse,
            "correction_abs_mean":float((candidate-base).abs().mean())}


def choose_alpha(val,stable,variation,settings):
    trials={str(a):evaluate_candidate(val,stable,variation,settings,a)["mse"]
            for a in ALPHAS}
    selected=min(ALPHAS,key=lambda a:trials[str(a)])
    return selected,trials


def chronological_oof(train,starts,ends,settings,seed,seq_len,pred_len):
    n=len(train["key"])
    blocks=(torch.arange(n)*6//n).clamp_max(5)
    base=train["base"]
    candidate=torch.zeros_like(base)
    stable_candidate=torch.zeros_like(base)
    constant_candidate=torch.zeros_like(base)
    selected=torch.zeros(n,dtype=torch.bool)
    fold_audit=[]
    for fold in (2,3,4,5):
        source=purged_source_mask(blocks,starts,ends,fold)
        hold=blocks==fold
        stable,variation=fit_memory(source,train,blocks,settings,seed)
        subset={key:value[hold] for key,value in train.items()}
        prediction=query_memory(subset,stable,variation,settings)["prediction"]
        candidate[hold]=prediction
        stable_prediction=stable.query(subset["key"])["prediction"]
        stable_candidate[hold]=stable_prediction
        constant_prediction=train["truth"][source].mean((0,1),keepdim=True)
        constant_candidate[hold]=constant_prediction
        selected|=hold
        fold_audit.append({"fold":fold,"source_rows":int(source.sum()),
            "held_rows":int(hold.sum()),"max_source_future_end":int(ends[source].max()),
            "min_query_history_start":int(starts[hold].min()),
            "a0_mse":mse_mae(base[hold],train["truth"][hold],subset)[0],
            "candidate_mse":mse_mae(prediction,train["truth"][hold],subset)[0],
            "stable_memory_mse":mse_mae(stable_prediction,train["truth"][hold],subset)[0],
            "constant_future_mse":mse_mae(constant_prediction.expand_as(prediction),
                                            train["truth"][hold],subset)[0]})
    subset={key:value[selected] for key,value in train.items()}
    return {"candidate":candidate[selected],"stable_candidate":stable_candidate[selected],
            "constant_candidate":constant_candidate[selected],"record":subset,
            "fold_audit":fold_audit,"scope":"TRAIN memory-module OOF only; A0 TRAIN checkpoint may be in-sample"}


def candidate_oof_metrics(oof,alpha):
    record=oof["record"]
    base=record["base"];candidate=oof["candidate"]
    a0=mse_mae(base,record["truth"],record)[0]
    raw=mse_mae(candidate,record["truth"],record)[0]
    scaled=mse_mae(base+alpha*(candidate-base),record["truth"],record)[0]
    stable=mse_mae(oof["stable_candidate"],record["truth"],record)[0]
    constant=mse_mae(oof["constant_candidate"],record["truth"],record)[0]
    return {"a0_mse":a0,"candidate_mse":raw,"candidate_gain":a0-raw,
            "scaled_mse":scaled,"scaled_gain":a0-scaled,
            "stable_memory_mse":stable,"constant_future_mse":constant}


def effect_diagnostics(data,stable,variation,settings,alpha):
    candidate=query_memory(data,stable,variation,settings)["prediction"]
    base=data["base"];truth=data["truth"]
    scaled=base+alpha*(candidate-base)
    scale=data["scale"].transpose(1,2)
    gain=((base-truth)*scale).square().mean((1,2))-((scaled-truth)*scale).square().mean((1,2))
    horizon=truth.shape[-1]
    segments=[]
    for p in range(4):
        a=p*horizon//4;b=(p+1)*horizon//4
        before=(((base-truth)*scale)[:,:,a:b]).square().mean()
        after=(((scaled-truth)*scale)[:,:,a:b]).square().mean()
        segments.append(float(before-after))
    quartiles=[]
    for p in range(4):
        a=p*len(gain)//4;b=(p+1)*len(gain)//4
        quartiles.append(float(gain[a:b].mean()))
    best=gain.topk(min(5,len(gain))).values
    return {"gain_mean":float(gain.mean()),"gain_median":float(gain.median()),
       "positive_gain_fraction":float((gain>0).float().mean()),
       "gain_p10_p50_p90":torch.quantile(gain,torch.tensor([.1,.5,.9])).tolist(),
       "best_5_gain_sum_fraction":float(best.sum()/gain.sum().abs().clamp_min(1e-8)),
       "future_segment_gains":segments,"chronological_query_quartile_gains":quartiles}


def save_rows(path,rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)


def main():
    cli=parse_args()
    torch.manual_seed(cli.seed)
    torch.set_num_threads(4)
    out=ROOT/cli.output/cli.case
    out.mkdir(parents=True,exist_ok=True)
    if (out/"next_pilot_complete").exists():print("already complete");return
    cli.base_run_config=str(BASE/(f"{cli.case}_A0_complementary_gate_predictive_"
                             "sub0_orth0_refnone_fp0_rank8_bias-4/run_config.json"))
    cli.allow_legacy_reference_signature=cli.case.startswith("moderntcn")
    args,model,loaders,device,sha=load_experiment(cli)
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise AssertionError("frozen A0 required")
    old=ROOT/"results/fpem_dual_pattern_memory/point_pilot"/cli.case
    bundle=torch.load(old/"dual_pattern_memory.pt",map_location="cpu",weights_only=False)
    if bundle["reference_sha256"]!=sha:raise ValueError("A0 SHA mismatch")
    cache=torch.load(ROOT/"results/fpem_predictive_memory/pilot_20261024"/cli.case/
                     "phase0_cache.pt",map_location="cpu",weights_only=False)
    if cache["reference_sha256"]!=sha:raise ValueError("TRAIN cache SHA mismatch")
    records=cache["records"]
    train=normalized_record({"x":records["x"],"target":records["target"],
                             "base":records["a0_pred"]},bundle["floor"])
    starts=records["starts"].long().cpu()
    ends=starts+args.seq_len+args.pred_len
    blocks=(torch.arange(len(starts))*6//len(starts)).clamp_max(5)
    val_data=loaders["val"].dataset.dataset
    val_loader=DataLoader(Subset(IndexedDataset(val_data,cycle_len=args.cyclenet_cycle_len),
                    list(range(max(1,len(val_data)//2)))),batch_size=args.batch_size,
                    shuffle=False,num_workers=0)
    val=normalized_record(collect_split(model,val_loader,device,cli.max_batches),bundle["floor"])
    test=normalized_record(collect_split(model,loaders["test"],device,cli.max_batches),bundle["floor"])
    write_json(out/"run_config.json",{"case":cli.case,"reference_sha256":sha,
        "source_cache":str(ROOT/"results/fpem_predictive_memory/pilot_20261024"/cli.case/
                          "phase0_cache.pt"),"backbone_frozen":True,
        "selection":"first-half VAL only; TEST retrospective reporting",
        "oof":"6 chronological blocks, source future end <= first held history start",
        "stages":["V0-V3","C0-C3","R3-A-D"],"flow":"not trained"})
    all_rows=[]
    def run_one(stage,name,settings):
        result_path=out/f"{name}.json"
        if result_path.exists():
            cached=json.loads(result_path.read_text(encoding="utf-8"))
            if cached.get("settings")!=settings or cached["row"]["case"]!=cli.case:
                raise ValueError(f"existing variant mismatch: {result_path}")
            all_rows.append(cached["row"])
            print(f"resumed {name}",flush=True)
            return cached["row"]
        bank=fit_memory(torch.ones(len(starts),dtype=torch.bool),train,blocks,settings,cli.seed)
        alpha,trials=choose_alpha(val,*bank,settings)
        trained=evaluate_candidate(train,*bank,settings,alpha)
        validated=evaluate_candidate(val,*bank,settings,alpha)
        tested=evaluate_candidate(test,*bank,settings,alpha)
        stable_train=mse_mae(bank[0].query(train["key"])["prediction"],
                              train["truth"],train)[0]
        stable_val=mse_mae(bank[0].query(val["key"])["prediction"],
                            val["truth"],val)[0]
        oof=chronological_oof(train,starts,ends,settings,cli.seed,args.seq_len,args.pred_len)
        oof_metric=candidate_oof_metrics(oof,alpha)
        support=entropy_and_support(bank,train,starts,args.seq_len,args.pred_len,settings)
        row={"case":cli.case,"stage":stage,"variant":name,
             "representation":settings.get("representation","V0"),
             "mapping_input":settings.get("input","concat"),
             "ms":settings.get("ms",16),"mv":settings.get("mv",8),
             "ridge":settings.get("ridge",.05),"rank":settings.get("rank",0),
             "control":settings.get("control","none"),"val_alpha":alpha,
             "train_candidate_mse":trained["candidate_mse"],
             "train_stable_memory_mse":stable_train,
             "oof_candidate_mse":oof_metric["candidate_mse"],
             "oof_stable_memory_mse":oof_metric["stable_memory_mse"],
             "oof_constant_future_mse":oof_metric["constant_future_mse"],
             "oof_a0_mse":oof_metric["a0_mse"],
             "oof_candidate_gain":oof_metric["candidate_gain"],
             "oof_scaled_gain":oof_metric["scaled_gain"],
             "val_mse":validated["mse"],"val_mae":validated["mae"],
             "val_candidate_mse":validated["candidate_mse"],
             "val_stable_memory_mse":stable_val,
             "test_mse":tested["mse"],"test_mae":tested["mae"],
             "a0_test_mse":tested["a0_mse"],
             "match_entropy":support["match_entropy_normalized"],
             "mean_active_prototypes":support["positive_weight_count_mean"],
             "min_independent_events":min(support["independent_events_per_prototype"]),
             "stable_parameters":support["stable_parameters"],
             "variation_parameters":support["map_parameters"]}
        write_json(result_path,{"row":row,"settings":settings,"val_alpha_grid":trials,
               "support":support,"oof_folds":oof["fold_audit"],
               "val_effect":effect_diagnostics(val,*bank,settings,alpha),
               "test_effect_exploratory":effect_diagnostics(test,*bank,settings,alpha)})
        all_rows.append(row)
        print(json.dumps({"stage":stage,"variant":name,"val":row["val_mse"],
                          "oof_gain":row["oof_candidate_gain"],"test":row["test_mse"]}),flush=True)
        return row
    # Stage V: preserve map dimensions through zero padding for one-input modes.
    vrows=[]
    for representation in ("V0","V1","V2","V3"):
        for input_mode in ("description","concat","history"):
            settings={"representation":representation,"input":input_mode,
                      "ms":16,"mv":8,"ridge":.05}
            vrows.append(run_one("V",f"{representation}_{input_mode}",settings))
    eligible=[r for r in vrows if r["representation"]!="V3"]
    # When VAL chooses exact A0 fallback for multiple variants, break the
    # practical tie using TRAIN purged OOF candidate MSE, never TEST.
    best_v=min(eligible,key=lambda r:(round(r["val_mse"],6),r["oof_candidate_mse"]))
    selected_v={"representation":best_v["representation"],"input":best_v["mapping_input"]}
    write_json(out/"stage_v_selected.json",{"choice":selected_v,"rule":"VAL MSE (1e-6 tie: purged TRAIN OOF candidate MSE)",
                                         "negative_control_excluded":True})
    print("stage V complete",flush=True)
    # Stage C: sequential, bounded capacity comparisons; no Cartesian product.
    csettings=[("C0",{"ms":16,"mv":8,"ridge":.05}),
               ("C1_ridge0.5",{"ms":16,"mv":8,"ridge":.5}),
               ("C1_ridge5",{"ms":16,"mv":8,"ridge":5.}),
               ("C2_small",{"ms":8,"mv":4,"ridge":.05})]
    csettings += [(f"C3_rank{rank}",{"ms":16,"mv":8,"ridge":.05,"rank":rank})
                  for rank in (8,16,32)]
    crows=[run_one("C",name,{**selected_v,**setting}) for name,setting in csettings]
    best_c=min(crows,key=lambda r:(round(r["val_mse"],6),r["oof_candidate_mse"]))
    write_json(out/"stage_c_selected.json",{"choice":best_c["variant"],
                                            "rule":"VAL MSE (1e-6 tie: purged TRAIN OOF candidate MSE)"})
    print("stage C complete",flush=True)
    # R3 controls all use the *original* bank count/ridge and same VAL scaling.
    base={"representation":"V0","input":"concat","ms":16,"mv":8,"ridge":.05}
    rsettings=[("R3_A_current",base),("R3_B_shuffle_xy",{**base,"control":"shuffle_xy"}),
               ("R3_C_global",{**base,"control":"global","mv":1}),
               ("R3_D_single_bank",{**base,"control":"single_bank"})]
    for name,setting in rsettings:run_one("R3",name,setting)
    save_rows(out/"all_results.csv",all_rows)
    best_v_row=min(vrows,key=lambda r:r["val_mse"])
    best_c_row=min(crows,key=lambda r:r["val_mse"])
    write_json(out/"summary.json",{"case":cli.case,"reference_sha256":sha,
       "a0_test_mse":all_rows[0]["a0_test_mse"],
       "best_v_by_val":best_v_row,"best_c_by_val":best_c_row,
       "r3":{r["variant"]:r for r in all_rows if r["stage"]=="R3"},
       "selection_note":"TEST not used for alpha or method selection; all TEST rankings exploratory",
       "probability_status":"not run"})
    (out/"next_pilot_complete").write_text("complete\n",encoding="utf-8")


if __name__=="__main__":main()
