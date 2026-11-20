"""Strict chronological TRAIN forward-OOF mechanism and basis evidence."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models.predictive_env.dual_pattern_memory.forecast_fusion import stable_direction_basis
from models.predictive_env.prototype_guided_invariant import PrototypeGuidedSharedHead,prototype_h_centers
from tools.run_fpem_prototype_invariant_search import (
    stage_cache,fit_r3,train_method,method_spec,build_head,predict_head,mse_mae,
    normalized_to_raw,write_json)
from tools.run_fpem_prototype_invariant_fusion import shared_candidates,to_r3_space,from_r3_space


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--base_run_config",required=True)
    p.add_argument("--shared_root",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--gpu",type=int,default=0)
    p.add_argument("--seed",type=int,default=2021)
    p.add_argument("--epochs",type=int,default=10)
    p.add_argument("--train_stride",type=int,default=16)
    p.add_argument("--val_stride",type=int,default=4)
    p.add_argument("--test_stride",type=int,default=1)
    p.add_argument("--allow_legacy_reference_signature",action="store_true")
    return p.parse_args()


def subset(data,mask):
    return {key:value[mask] if isinstance(value,torch.Tensor) and len(value)==len(mask)
            else value for key,value in data.items()}


def main():
    cli=parse_args();root=Path(cli.shared_root);out=Path(cli.output)
    if (out/"complete").exists():return
    out.mkdir(parents=True,exist_ok=True)
    args,model,data,device,sha,_,_,_=stage_cache(cli,root)
    full=data["train"]
    starts=full["ids"].long();end=starts+args.seq_len+args.pred_len
    last=int(end.max())+1
    cut1=int(last*.48);cut2=int(last*.74)
    masks=(end<=cut1,(starts>=cut1)&(end<=cut2),starts>=cut2)
    if min(int(mask.sum()) for mask in masks)<12:
        write_json(out/"failure.json",{"reason":"insufficient embargo-separated TRAIN windows",
                  "counts":[int(mask.sum()) for mask in masks]})
        raise ValueError("insufficient embargo-separated TRAIN windows for forward OOF")
    early,middle,late=[subset(full,mask) for mask in masks]
    if int(early["ids"].max()+args.seq_len+args.pred_len)>int(middle["ids"].min()):
        raise AssertionError("early TRAIN future overlaps middle history")
    if int(middle["ids"].max()+args.seq_len+args.pred_len)>int(late["ids"].min()):
        raise AssertionError("middle TRAIN future overlaps late history")
    oof_r3=fit_r3(early,middle,args,cli.seed,out)
    for split in (early,middle,late):
        query=oof_r3.query(split["key"])
        split.update({"q":query["weight"],"null":query["null_weight"],
                      "candidate":query["candidate"]})
    centers=prototype_h_centers(early["hidden"],early["q"])
    method=shared_candidates(root,1)[0]
    _,state=train_method(method,early,middle,late,model,args,device,cli,centers[:2])
    rank,reference,_,residual=method_spec(method)
    tokens=state["forecaster.head.weight"].shape[1]//(rank or model.d_model)
    head=build_head(method,tokens,model.d_model,args.pred_len)
    head.load_state_dict(state);head.to(device).eval()
    shared=predict_head(head,late,device)
    base=late["base"]
    candidate=late["candidate"]
    q=late["q"]
    shared_r3=to_r3_space(shared,late)
    memory_r3=(q[...,None]*candidate).sum(2)+(1-q.sum(-1))[...,None]*shared_r3
    memory=from_r3_space(memory_r3,late)
    block=torch.bucketize(late["ids"].float(),torch.tensor([float(late["ids"].median())]))
    source_a,audit_a=stable_direction_basis(shared.transpose(1,2),base.transpose(1,2),
            late["truth"].transpose(1,2),block,max_rank=8)
    source_b,audit_b=stable_direction_basis(memory.transpose(1,2),shared.transpose(1,2),
            late["truth"].transpose(1,2),block,max_rank=8)
    torch.save({"shared_benefit":source_a,"r3_benefit":source_b,
                "reference_sha256":sha,"method":method},out/"basis.pt")
    result={"method":method,"counts":[len(early["ids"]),len(middle["ids"]),len(late["ids"])],
            "cutpoints":[cut1,cut2],"embargo":args.seq_len+args.pred_len,
            "a0":mse_mae(base,late["truth"]),
            "shared":mse_mae(shared,late["truth"]),
            "r3_mix":mse_mae(memory,late["truth"]),
            "block_gain":[float((base[block==i]-late["truth"][block==i]).square().mean()
                         -(shared[block==i]-late["truth"][block==i]).square().mean())
                         for i in (0,1) if (block==i).any()],
            "basis_shared_benefit_rank":source_a.shape[-1],
            "basis_r3_benefit_rank":source_b.shape[-1],
            "basis_shared_audit":audit_a,"basis_r3_audit":audit_b,
            "provenance":"TRAIN early fit; disjoint middle VAL checkpoint; late TRAIN forward-OOF report",
            "reference_sha256":sha}
    write_json(out/"metrics.json",result)
    (out/"complete").write_text("complete\n")
    print("FORWARD OOF COMPLETE",result["shared"],flush=True)


if __name__=="__main__":main()
