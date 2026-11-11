"""Stage J3: recalibrate existing R3 against a direct shared anchor."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models.predictive_env.dual_pattern_memory.forecast_fusion import (
    prototype_segment_reliability,variation_forecast)
from models.predictive_env.prototype_guided_invariant import PrototypeGuidedSharedHead
from tools.run_fpem_prototype_invariant_search import (
    stage_cache,method_spec,build_head,predict_head,mse_mae,write_json)


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--base_run_config",required=True)
    p.add_argument("--shared_root",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--gpu",type=int,default=0)
    p.add_argument("--seed",type=int,default=2021)
    p.add_argument("--allow_legacy_reference_signature",action="store_true")
    p.add_argument("--train_stride",type=int,default=16)
    p.add_argument("--val_stride",type=int,default=4)
    p.add_argument("--test_stride",type=int,default=1)
    p.add_argument("--top_k",type=int,default=3)
    p.add_argument("--oof_root",default="")
    return p.parse_args()


def shared_candidates(root,top_k):
    names=("p01","p02","risk_var","risk_dro","shared_grad_norm",
           "shared_grad_variance","shared_irm","conditional","subspace",
           "ref_global","ref_prototype","risk_incremental","pairwise",
           "shared_private","shared_private_irm","adversarial_weak",
           "adversarial_strong","map_projection","map_shared_irm")
    scores=[]
    for name in names:
        path=root/name/"metrics.json"
        if path.exists() and (root/name/"model.pt").exists():
            score=json.loads(path.read_text())["val"]["mse"]
            scores.append((score,name))
    if not scores:raise RuntimeError("no completed shared candidates")
    return [name for _,name in sorted(scores)[:top_k]]


def load_head(root,method,model,args,device):
    rank,reference,_,residual=method_spec(method)
    state=torch.load(root/method/"model.pt",map_location="cpu",weights_only=False)
    tokens=state["forecaster.head.weight"].shape[1]//(rank or model.d_model)
    head=build_head(method,tokens,model.d_model,args.pred_len)
    head.load_state_dict(state)
    return head.to(device).eval()


def to_r3_space(raw,data):
    return ((raw-data["hist_mean"])/data["hist_scale"]).transpose(1,2)


def from_r3_space(norm,data):
    return norm.transpose(1,2)*data["hist_scale"]+data["hist_mean"]


def evaluate_fusion(anchor_raw,calib,test,method,bases=None):
    anchor_cal=to_r3_space(anchor_raw["val"],calib)
    anchor_test=to_r3_space(anchor_raw["test"],test)
    reliability,audit=prototype_segment_reliability(
        anchor_cal,calib["candidate"],calib["r3_truth"],calib["q"],8)
    empty=anchor_test.new_zeros(anchor_test.shape[-1],0)
    fused,beta=variation_forecast(anchor_test,test["candidate"],test["q"],
                                   reliability,empty,True,8)
    ones=torch.ones_like(reliability)
    ungated,_=variation_forecast(anchor_test,test["candidate"],test["q"],ones,empty,True,8)
    # Null is explicitly retained by variation_forecast because sum(q)<=1.
    query_unconditional=(test["q"][...,None]*test["candidate"]).sum(2)
    qmass=test["q"].sum(-1).clamp(0,1)
    r3_only=query_unconditional + (1-qmass)[...,None]*anchor_test
    result={"method":method,"shared":mse_mae(anchor_raw["test"],test["truth"]),
            "r3_ungated":mse_mae(from_r3_space(ungated,test),test["truth"]),
            "r3_recalibrated":mse_mae(from_r3_space(fused,test),test["truth"]),
            "r3_null_mix":mse_mae(from_r3_space(r3_only,test),test["truth"]),
            "mean_effect":float((fused-anchor_test).abs().mean()),
            "active_ratio":float((beta.abs().sum((1,2))>0).float().mean()),
            "reliability_mean":float(reliability.mean()),
            "null_mean":float(test["null"].mean()),
            "calibration":"held-out VAL only, independent of shared/R3 TRAIN fit",
            "protected":bool(bases)}
    # H2 source controls: every control receives its own VAL-only reliability
    # calibration; none is allowed to borrow the real-route reliability.
    controls={
        "uniform_route":(calib["q"].sum(-1,keepdim=True)/calib["q"].shape[-1],
                         test["q"].sum(-1,keepdim=True)/test["q"].shape[-1],
                         calib["candidate"],test["candidate"]),
        "shuffled_route":(calib["q"].roll(1,0),test["q"].roll(1,0),
                          calib["candidate"],test["candidate"]),
        "shuffled_future_mapping":(calib["q"],test["q"],
                                    calib["candidate"].roll(1,2),test["candidate"].roll(1,2)),
    }
    for label,(q_cal,q_test,cand_cal,cand_test) in controls.items():
        if q_cal.shape[-1]==1:q_cal=q_cal.expand_as(calib["q"])
        if q_test.shape[-1]==1:q_test=q_test.expand_as(test["q"])
        control_rel,_=prototype_segment_reliability(anchor_cal,cand_cal,
                          calib["r3_truth"],q_cal,8)
        control_pred,_=variation_forecast(anchor_test,cand_test,q_test,control_rel,empty,True,8)
        result["ablation_"+label]=mse_mae(from_r3_space(control_pred,test),test["truth"])
    if bases:
        for label,basis in bases.items():
            protected,effect=variation_forecast(anchor_test,test["candidate"],test["q"],
                                                  reliability,basis,True,8)
            result["protected_"+label]=mse_mae(from_r3_space(protected,test),test["truth"])
            result["protected_"+label+"_effect"]=float(effect.mean())
    return result,reliability,audit


def main():
    cli=parse_args();root=Path(cli.shared_root);output=Path(cli.output)
    output.mkdir(parents=True,exist_ok=True)
    args,model,data,device,sha,_,_,_=stage_cache(cli,root)
    bases=None
    if cli.oof_root:
        basis_file=Path(cli.oof_root)/"basis.pt"
        if basis_file.exists():
            loaded=torch.load(basis_file,map_location="cpu",weights_only=False)
            if loaded["reference_sha256"]!=sha:
                raise ValueError("forward-OOF basis uses a different A0 checkpoint")
            bases={"shared_benefit":loaded["shared_benefit"],
                   "r3_benefit":loaded["r3_benefit"]}
    names=shared_candidates(root,cli.top_k)
    rows=[]
    for name in names:
        case=output/name
        if (case/"complete").exists():
            rows.append(json.loads((case/"metrics.json").read_text()));continue
        case.mkdir(parents=True,exist_ok=True)
        head=load_head(root,name,model,args,device)
        anchor={part:predict_head(head,data[part],device) for part in ("val","test")}
        result,reliability,audit=evaluate_fusion(anchor,data["val"],data["test"],name,bases)
        result.update({"backbone":args.backbone,"dataset":args.dataset_name,
                       "pred_len":args.pred_len,"reference_sha256":sha,
                       "route_source":"same frozen R3 used in shared training",
                       "selection":"VAL head checkpoint; TEST exploratory report"})
        torch.save({"reliability":reliability,"shared_method":name},case/"calibration.pt")
        write_json(case/"reliability_audit.json",audit)
        write_json(case/"metrics.json",result)
        (case/"complete").write_text("complete\n")
        rows.append(result)
    with (output/"fusion_results.csv").open("w",newline="") as handle:
        keys=("method","shared_mse","ungated_mse","recalibrated_mse","null_mix_mse",
              "active_ratio","reliability_mean","null_mean","reference_sha256")
        writer=csv.DictWriter(handle,fieldnames=keys);writer.writeheader()
        for row in rows:
            writer.writerow({"method":row["method"],"shared_mse":row["shared"]["mse"],
                "ungated_mse":row["r3_ungated"]["mse"],
                "recalibrated_mse":row["r3_recalibrated"]["mse"],
                "null_mix_mse":row["r3_null_mix"]["mse"],
                "active_ratio":row["active_ratio"],"reliability_mean":row["reliability_mean"],
                "null_mean":row["null_mean"],"reference_sha256":sha})
    print("FUSION COMPLETE",output,flush=True)


if __name__=="__main__":main()
