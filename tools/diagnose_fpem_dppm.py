"""DPPM D0-D6/D9-D11 offline diagnosis; never fits on VAL/TEST targets.

R3's single coefficient is selected on first-half VAL. TEST oracle
coefficients are printed only as retrospective diagnostics, not deployed.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.predictive_env.dual_pattern_memory import PatternBank
from models.predictive_env.dual_pattern_memory.forecast_fusion import (
    stable_forecast, variation_forecast, stable_direction_basis,
    prototype_segment_reliability)
from models.predictive_env.dual_pattern_memory.prototype_bank import normalized_pairs
from models.predictive_env.dual_pattern_memory.uncertainty_memory import fit_uncertainty_memory
from models.predictive_env.paired_memory.memory_bank import history_key
from tools.run_fpem_dppm import (CASES, BASE, normalized_record, blocks_for,
                                  train_evidence, stable_oof, variation_key, fit_bank)
from tools.run_fpem_shared_factorization import load_experiment, write_json
from tools.run_predictive_env_iv_patchtst import IndexedDataset
from tools.run_fpem_smrva import collect_split


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("case", choices=CASES)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--output", default="results/fpem_dual_pattern_memory/diagnostics")
    p.add_argument("--max_batches", type=int, default=0)
    return p.parse_args()


def stats(tensor):
    x = tensor.detach().float().cpu()
    valid = x[torch.isfinite(x)]
    return {"shape": list(x.shape), "mean": float(valid.mean()) if valid.numel() else None,
            "std": float(valid.std(unbiased=False)) if valid.numel() else None,
            "min": float(valid.min()) if valid.numel() else None,
            "max": float(valid.max()) if valid.numel() else None,
            "nonfinite": int((~torch.isfinite(x)).sum())}


def metric(pred, truth, record):
    scale = record["scale"].transpose(1, 2)
    diff = (pred-truth)*scale
    return {"mse": float(diff.square().mean()), "mae": float(diff.abs().mean())}


def correction_direction(base, change, truth):
    diff = (change-base).float()
    error = (truth-base).float()
    numerator = (diff*error).sum((1,2))
    denominator = diff.square().sum((1,2))
    oracle = (numerator/(denominator+1e-8)).clamp(0,1)
    active = denominator > 1e-8
    cosine = numerator/(denominator.sqrt()*error.square().sum((1,2)).sqrt()+1e-8)
    return {"dot_mean":float(numerator.mean()),
            "positive_fraction":float((numerator[active]>0).float().mean()) if bool(active.any()) else 0.,
            "oracle_eta_mean_diagnostic_only":float(oracle.mean()),
            "cosine_mean":float(cosine.mean()),
            "effect_norm_mean":float(denominator.sqrt().mean()),
            "active_fraction":float(active.float().mean())}


def _component(record, stable_bank, variation_bank, basis, eta, reliability,
               null_score, segments=8):
    key, base = record["key"], record["base"]
    sq = stable_bank.query(key)
    stable = stable_forecast(base, sq["prediction"], basis, eta)
    vk = variation_key(stable_bank,key)
    vq = variation_bank.query(vk,torch.cat((key,vk),-1))
    vq_null = variation_bank.query(vk,torch.cat((key,vk),-1),null_score)
    candidates = vq["candidate"]
    if basis.shape[-1]:
        diff = candidates-stable[:,:,None]
        projected = stable[:,:,None]+diff-(diff@basis)@basis.T
    else:
        projected = candidates
    ones = torch.ones_like(reliability)
    b3,beta3=variation_forecast(stable,candidates,vq["weight"],ones,basis,
                                use_segments=False,segment_count=segments)
    r2,beta2=variation_forecast(stable,candidates,vq_null["weight"],ones,basis,
                                use_segments=False,segment_count=segments)
    b6,beta6=variation_forecast(stable,candidates,vq_null["weight"],reliability,basis,
                                use_segments=True,segment_count=segments)
    return {"base":base,"stable_memory":sq["prediction"],"stable":stable,
            "stable_weight":sq["weight"],"variation_key":vk,
            "variation_candidates":candidates,"protected_candidates":projected,
            "variation_weight":vq["weight"],"null_weight":vq_null["null_weight"],
            "b3":b3,"r2":r2,"b6":b6,"beta3":beta3,"beta2":beta2,
            "beta6":beta6,"stable_match":sq["matched_history"]}


def split_diagnostics(name, record, components, basis, eta, selected_alpha=None):
    truth=record["truth"]
    result={"split":name,"sample_count":len(truth),"metrics":{},"direction":{},"tensors":{}}
    for item in ("base","stable_memory","stable","b3","r2","b6"):
        result["metrics"][item]=metric(components[item],truth,record)
        result["tensors"][item]=stats(components[item])
    candidate=components["variation_candidates"]
    protected=components["protected_candidates"]
    result["metrics"]["variation_candidate_weighted"] = metric(
        (candidate*components["variation_weight"][:,:,:,None]).sum(2),truth,record)
    result["metrics"]["variation_protected_weighted"] = metric(
        (protected*components["variation_weight"][:,:,:,None]).sum(2),truth,record)
    result["direction"]["stable"] = correction_direction(components["base"],
        stable_forecast(components["base"],components["stable_memory"],basis,1.),truth)
    result["direction"]["variation"] = correction_direction(components["stable"],
        components["b3"],truth)
    result["tensors"].update({name:stats(components[name]) for name in
        ("stable_weight","variation_key","variation_weight","null_weight",
         "beta3","beta2","beta6","variation_candidates","protected_candidates")})
    result["fusion"]={"beta3_min":float(components["beta3"].min()),
        "beta3_max":float(components["beta3"].max()),
        "beta6_min":float(components["beta6"].min()),
        "beta6_max":float(components["beta6"].max()),
        "null_mean":float(components["null_weight"].mean()),
        "boundary_jump_mean":float((components["b3"][:,:,1:]-components["b3"][:,:,:-1]).abs().mean())}
    assert bool((components["beta3"]>=-1e-6).all() and (components["beta3"]<=1+1e-6).all())
    assert bool((components["beta6"]>=-1e-6).all() and (components["beta6"]<=1+1e-6).all())
    assert bool(torch.isfinite(components["b3"]).all())
    if basis.shape[-1]:
        projector=basis@basis.T
        result["projection"]={"symmetry_error":float((projector-projector.T).abs().max()),
            "idempotence_error":float((projector@projector-projector).abs().max()),
            "base_error_inside":float(((truth-components["base"])@projector).square().mean()),
            "base_error_outside":float(((truth-components["base"])@(torch.eye(len(projector))-projector)).square().mean())}
        assert result["projection"]["symmetry_error"]<1e-4
        assert result["projection"]["idempotence_error"]<1e-4
    else: result["projection"]={"rank":0}
    if selected_alpha is not None:
        final=components["stable"]+selected_alpha*(components["b3"]-components["stable"])
        result["metrics"]["r3_val_scaled_b3"]=metric(final,truth,record)
    return result


def worst20(record, comp):
    score=(((comp["b3"]-record["truth"])*record["scale"].transpose(1,2)).square()
           -((comp["base"]-record["truth"])*record["scale"].transpose(1,2)).square()).mean((1,2))
    out=[]
    for i in score.topk(min(20,len(score))).indices.tolist():
        out.append({"index":i,"b3_minus_a0_mse":float(score[i]),
            "base":comp["base"][i].T.tolist(),
            "stable":comp["stable"][i].T.tolist(),
            "b3":comp["b3"][i].T.tolist(),"b6":comp["b6"][i].T.tolist(),
            "truth_offline_diagnostic_only":record["truth"][i].T.tolist(),
            "beta3":comp["beta3"][i].T.tolist(),
            "null_weight":comp["null_weight"][i].tolist()})
    return out


def evidence_table(reliability_audit, held, comp):
    truth=held["truth"]
    stable=comp["stable"]
    cand=comp["protected_candidates"]
    weight=comp["variation_weight"]
    n,c,m,t=cand.shape
    rows=[]
    for record in reliability_audit:
        j,p=record["pattern"],record["segment"]
        a=p*t//8;b=(p+1)*t//8
        w=weight[:,:,j]
        sample_gain=((stable[:,:,a:b]-truth[:,:,a:b]).square()
          -(cand[:,:,j,a:b]-truth[:,:,a:b]).square()).mean(-1)
        mask=w>=w.flatten().median()
        high=float(sample_gain[mask].mean()) if bool(mask.any()) else None
        rows.append({**record,"high_match_oof_gain":high,
            "positive_raw_gain":record["oof_gain"]>0,
            "zero_reason":("nonpositive_oof_gain" if record["oof_gain"]<=0 else
                "insufficient_effective_support" if record["effective_support"]<=1 else
                "nonzero")})
    return rows


def _git_audit():
    def command(*argv):
        result=subprocess.run(argv,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              text=True,check=False)
        return result.stdout
    return {"head":command("git","rev-parse","HEAD").strip(),
            "branch":command("git","branch","-avv"),
            "remote":command("git","remote","-v"),
            "status":command("git","status","--short")}


def purged_source_mask(groups, starts, ends, fold):
    """A source's entire target must end before first held-out history starts."""
    hold = groups == fold
    if not bool(hold.any()):
        raise ValueError("empty held-out block")
    mask = (groups < fold) & (ends <= starts[hold].min())
    if bool((ends[mask] > starts[hold].min()).any()):
        raise AssertionError("source future overlaps held-out history")
    return mask


def purged_evidence(training, groups, starts, ends, ms, mv, ridge, seed):
    key, truth, base = training["key"], training["truth"], training["base"]
    oof_memory = torch.zeros_like(truth)
    causality = []
    for fold in (2,3,4,5):
        source = purged_source_mask(groups,starts,ends,fold)
        hold = groups == fold
        bank,_=fit_bank(key[source],truth[source],groups[source],ms,ridge,seed)
        oof_memory[hold]=bank.query(key[hold])["prediction"]
        causality.append({"fold":fold,"source_rows":int(source.sum()),
            "held_rows":int(hold.sum()),"last_source_target_end":int(ends[source].max()),
            "first_held_history_start":int(starts[hold].min()),
            "unpurged_overlapping_rows":int(((groups<fold)&(ends>starts[hold].min())).sum())})
    fit_basis = (groups==2)|(groups==3)
    basis,basis_audit=stable_direction_basis(oof_memory[fit_basis],base[fit_basis],
                                             truth[fit_basis],groups[fit_basis],8)
    def fold_banks(fold):
        source=purged_source_mask(groups,starts,ends,fold)
        hold=groups==fold
        stable,_=fit_bank(key[source],truth[source],groups[source],ms,ridge,seed)
        vk_source=variation_key(stable,key[source])
        mapping_source=torch.cat((key[source],vk_source),-1)
        variation,_=fit_bank(vk_source,truth[source],groups[source],mv,ridge,
                              seed+11,mapping_source)
        held={"key":key[hold],"truth":truth[hold],"base":base[hold],
              "scale":training["scale"][hold],"mean":training["mean"][hold]}
        return stable,variation,held
    rel_stable,rel_variation,rel_held=fold_banks(4)
    eval_stable,eval_variation,eval_held=fold_banks(5)
    return (basis,basis_audit,rel_stable,rel_variation,rel_held,
            eval_stable,eval_variation,eval_held,causality)


def purged_reliability(held,stable,variation,basis,eta):
    key=held["key"]
    sq=stable.query(key)
    stable_pred=stable_forecast(held["base"],sq["prediction"],basis,eta)
    vk=variation_key(stable,key)
    vq=variation.query(vk,torch.cat((key,vk),-1))
    cand=vq["candidate"]
    if basis.shape[-1]:
        diff=cand-stable_pred[:,:,None]
        cand=stable_pred[:,:,None]+diff-(diff@basis)@basis.T
    return prototype_segment_reliability(stable_pred,cand,held["truth"],
                                          vq["weight"],8)


def main():
    cli=parse_args()
    torch.set_num_threads(4)
    out=ROOT/cli.output/cli.case
    out.mkdir(parents=True,exist_ok=True)
    if (out/"diagnostic_complete").exists():
        print("already complete",flush=True);return
    point_dir=ROOT/"results/fpem_dual_pattern_memory/point_pilot"/cli.case
    config=json.loads((point_dir/"run_config.json").read_text())
    selected=json.loads((point_dir/"val_selection.json").read_text())
    bundle=torch.load(point_dir/"dual_pattern_memory.pt",map_location="cpu",weights_only=False)
    stable_bank=PatternBank.from_state(bundle["stable"])
    variation_bank=PatternBank.from_state(bundle["variation"])
    basis=bundle["basis"]
    reliability=bundle["reliability"]
    eta=selected["eta"];null=selected["null_score"]
    cli.base_run_config=str(BASE/(f"{cli.case}_A0_complementary_gate_predictive_"
                 "sub0_orth0_refnone_fp0_rank8_bias-4/run_config.json"))
    cli.allow_legacy_reference_signature=cli.case.startswith("moderntcn")
    args,model,loaders,device,sha=load_experiment(cli)
    if sha!=bundle["reference_sha256"]:raise ValueError("reference mismatch")
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise AssertionError("A0 must be frozen eval")
    cache=torch.load(ROOT/"results/fpem_predictive_memory/pilot_20261024"/cli.case/
                     "phase0_cache.pt",map_location="cpu",weights_only=False)
    if cache["reference_sha256"]!=sha:raise ValueError("TRAIN cache mismatch")
    records=cache["records"]
    training=normalized_record({"x":records["x"],"target":records["target"],
                                "base":records["a0_pred"]},bundle["floor"])
    group=blocks_for(len(training["key"]))
    val_data=loaders["val"].dataset.dataset
    val_loader=DataLoader(Subset(IndexedDataset(val_data,cycle_len=args.cyclenet_cycle_len),
                           list(range(max(1,len(val_data)//2)))),batch_size=args.batch_size,
                           shuffle=False,num_workers=0)
    val=normalized_record(collect_split(model,val_loader,device,cli.max_batches),bundle["floor"])
    test=normalized_record(collect_split(model,loaders["test"],device,cli.max_batches),bundle["floor"])
    print("loaded A0/VAL/TEST; generating TRAIN memory-OOF",flush=True)
    _,direction_audit=stable_oof(training,group,stable_bank.n_patterns,
                                 stable_bank.ridge,stable_bank.seed)[1:]
    _,evidence_audit,held,prefix_s,prefix_v=train_evidence(training,group,
        stable_bank.n_patterns,variation_bank.n_patterns,stable_bank.ridge,
        stable_bank.seed,basis,eta)
    oof={"key":held["key"],"base":held["base"],"truth":held["truth"],
         "scale":training["scale"][group==4],"mean":training["mean"][group==4]}
    splits={"TRAIN_fitted":(training,stable_bank,variation_bank),
            "TRAIN_memory_OOF_fold4":(oof,prefix_s,prefix_v),
            "VAL_selection":(val,stable_bank,variation_bank),
            "TEST_diagnostic_only":(test,stable_bank,variation_bank)}
    components={name:_component(rec,s,v,basis,eta,reliability,null) for name,(rec,s,v) in splits.items()}
    # R3 is a *single* nonnegative global scale, chosen only on VAL.
    alpha_candidates=(0.,.01,.025,.05,.1,.2,.5,1.)
    r3_val={str(a):metric(components["VAL_selection"]["stable"]+
        a*(components["VAL_selection"]["b3"]-components["VAL_selection"]["stable"]),
        val["truth"],val)["mse"] for a in alpha_candidates}
    alpha=min(alpha_candidates,key=lambda a:r3_val[str(a)])
    reports={name:split_diagnostics(name,rec,components[name],basis,eta,alpha)
             for name,(rec,_,_) in splits.items()}
    write_json(out/"module_metrics.json",reports)
    starts=records["starts"].long().cpu()
    ends=starts+args.seq_len+args.pred_len
    old_overlap=sum(int(((group<fold)&(ends>starts[group==fold].min())).sum())
                    for fold in (2,3,4))
    corrected_group=(torch.arange(len(training["key"]))*6//len(training["key"])).clamp_max(5)
    (purged_basis,purged_basis_audit,rel_stable,rel_variation,rel_held,
     eval_stable,eval_variation,eval_held,causality)=purged_evidence(training,corrected_group,starts,ends,
             stable_bank.n_patterns,variation_bank.n_patterns,
             stable_bank.ridge,stable_bank.seed)
    write_json(out/"chronological_purge_audit.json",causality)
    write_json(out/"purged_stable_direction_audit.json",purged_basis_audit)
    # Full TRAIN banks remain unchanged at deployment. Only genuinely purged
    # TRAIN OOF evidence and VAL-selected fusion scalars are repaired.
    eta_candidates=(0.,.25,.5,1.)
    eta_val={str(a):metric(stable_forecast(val["base"],
          stable_bank.query(val["key"])["prediction"],purged_basis,a),
          val["truth"],val)["mse"] for a in eta_candidates}
    purged_eta=min(eta_candidates,key=lambda a:eta_val[str(a)])
    purged_reliability_matrix,purged_reliability_audit=purged_reliability(
          rel_held,rel_stable,rel_variation,purged_basis,purged_eta)
    null_candidates=(-4.,-2.,-1.,0.,1.)
    null_val={}
    for candidate_null in null_candidates:
        val_component=_component(val,stable_bank,variation_bank,purged_basis,
             purged_eta,purged_reliability_matrix,candidate_null)
        null_val[str(candidate_null)]=metric(val_component["b6"],val["truth"],val)["mse"]
    purged_null=min(null_candidates,key=lambda a:null_val[str(a)])
    corrected_components={name:_component(rec,s,v,purged_basis,purged_eta,
          purged_reliability_matrix,purged_null) for name,(rec,s,v) in
          {"TRAIN_memory_OOF_fold5":(eval_held,eval_stable,eval_variation),
           "VAL_selection":(val,stable_bank,variation_bank),
           "TEST_diagnostic_only":(test,stable_bank,variation_bank)}.items()}
    corrected_reports={name:split_diagnostics(name,rec,corrected_components[name],
          purged_basis,purged_eta) for name,rec in
          {"TRAIN_memory_OOF_fold5":eval_held,"VAL_selection":val,
           "TEST_diagnostic_only":test}.items()}
    write_json(out/"purged_module_metrics.json",corrected_reports)
    write_json(out/"purged_reliability_64.json",purged_reliability_audit)
    write_json(out/"purged_val_selection.json",{"eta":purged_eta,"eta_val_mse":eta_val,
         "null_score":purged_null,"null_val_mse":null_val,
         "rule":"TRAIN future-end purge; first-half VAL only; TEST report only"})
    write_json(out/"r3_val_selection.json",{"alpha":alpha,"val_mse":r3_val,
              "rule":"first-half VAL only; TEST oracle never used"})
    write_json(out/"worst20_test.json",worst20(test,components["TEST_diagnostic_only"]))
    rows=evidence_table(evidence_audit,held,components["TRAIN_memory_OOF_fold4"])
    with (out/"reliability_64.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    support={"stable":bundle["stable"]["counts"].tolist(),
             "variation":bundle["variation"]["counts"].tolist(),
             "stable_by_block":bundle["stable"]["block_support"].tolist(),
             "variation_by_block":bundle["variation"]["block_support"].tolist(),
             "stable_head_parameters":int(bundle["stable"]["mapping"].numel()),
             "variation_head_parameters":int(bundle["variation"]["mapping"].numel()),
             "train_anchor_rows":len(training["key"]),
             "train_anchor_channel_pairs":training["key"].shape[0]*training["key"].shape[1]}
    write_json(out/"prototype_capacity.json",support)
    audit={"git":_git_audit(),"point_run_config":config,"point_selection":selected,
        "code_paths":{"stable_bank":"models/predictive_env/dual_pattern_memory/prototype_bank.py:PatternBank.fit/query",
         "future_projection":"models/predictive_env/dual_pattern_memory/forecast_fusion.py:stable_direction_basis/stable_forecast",
         "variation_reliability":"models/predictive_env/dual_pattern_memory/forecast_fusion.py:prototype_segment_reliability",
         "variation_fusion":"models/predictive_env/dual_pattern_memory/forecast_fusion.py:variation_forecast",
         "normalization":"models/predictive_env/dual_pattern_memory/prototype_bank.py:normalized_pairs"},
        "design_gaps":["Original stable/variation maps use weighted Ridge, not Huber or gradient training; there is no map LR.",
          "Only TRAIN fold4 provides variation reliability; this is memory-module OOF, not full-system OOF because A0 was trained on TRAIN.",
          "First point pilot evaluated B0-B6 only; probability Flow and B7-B13 are not yet run.",
          "Original no-segment B4 path averages segment reliability; all reliabilities are currently zero."],
        "test_selection_policy":"TEST outputs here are retrospective diagnostics; no coefficient is fit to TEST",
        "causality_finding":"Original chronological blocks did not purge overlapping source target windows. Repaired 6-block protocol uses fold2/3 for basis, fold4 for reliability, fold5 for independent OOF evaluation, with source-target-end purging."}
    write_json(out/"code_audit.json",audit)
    short={"case":cli.case,"reference_sha256":sha,"eta":eta,"stable_rank":basis.shape[-1],
           "val_selected_r3_alpha":alpha,"reliability_positive_gain_count":sum(r["positive_raw_gain"] for r in rows),
           "reliability_nonzero_count":sum(r["reliability"]>0 for r in rows),
           "reliability_zero_reasons":{k:sum(r["zero_reason"]==k for r in rows) for k in
               {r["zero_reason"] for r in rows}},
           "a0_test_mse":reports["TEST_diagnostic_only"]["metrics"]["base"]["mse"],
           "stable_memory_test_mse":reports["TEST_diagnostic_only"]["metrics"]["stable_memory"]["mse"],
           "b3_test_mse":reports["TEST_diagnostic_only"]["metrics"]["b3"]["mse"],
           "b6_test_mse":reports["TEST_diagnostic_only"]["metrics"]["b6"]["mse"],
           "r3_val_scaled_test_mse":reports["TEST_diagnostic_only"]["metrics"]["r3_val_scaled_b3"]["mse"],
           "purged_eta":purged_eta,"purged_rank":purged_basis.shape[-1],
           "purged_reliability_positive_gain_count":sum(r["oof_gain"]>0 for r in purged_reliability_audit),
           "purged_reliability_nonzero_count":sum(r["reliability"]>0 for r in purged_reliability_audit),
           "purged_b3_test_mse":corrected_reports["TEST_diagnostic_only"]["metrics"]["b3"]["mse"],
           "purged_b6_test_mse":corrected_reports["TEST_diagnostic_only"]["metrics"]["b6"]["mse"],
           "overlapping_source_windows_in_old_oof":old_overlap,
           "probability_status":"not tested; these are point-only diagnostics"}
    write_json(out/"summary.json",short)
    (out/"diagnostic_complete").write_text("complete\n")
    print(json.dumps(short),flush=True)


if __name__=="__main__":
    main()
