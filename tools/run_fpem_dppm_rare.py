"""Independent R0–R7 rare-pattern pilot; frozen A0/stable bank and strict VAL OOS.

TRAIN labels learn mappings. Early, non-overlapping VAL labels calibrate future-
segment reliability. Later VAL selects NULL offset; TEST is report-only.
The frozen A0 TRAIN cache is in-sample, so TRAIN losses are *not* full-system OOF.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.predictive_env.dual_pattern_memory.prototype_bank import PatternBank
from models.predictive_env.dual_pattern_memory.forecast_fusion import (
    prototype_segment_reliability, stable_forecast, variation_forecast)
from models.predictive_env.dual_pattern_memory.rare_learning import RarePatternLearner
from tools.diagnose_predictive_rescue import per_sample_mae, per_sample_mse, residual_ridge, ridge_query, sparse_record
from tools.diagnose_global_local_mapping import fit_g1
from tools.run_fpem_dppm import BASE, CASES, normalized_record, clamp_variation
from tools.run_fpem_shared_factorization import load_experiment, write_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("case", choices=CASES)
    p.add_argument("--methods", default="R0,R1,R2,R3,R4,R5,R6,R7")
    p.add_argument("--stable_mode", choices=("dppm", "g1"), default="dppm",
                   help="Use the frozen DPPM stable bank or the existing G1 shared map")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--patterns", type=int, default=8)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--eval_stride", type=int, default=4)
    p.add_argument("--controls", action="store_true")
    p.add_argument("--lambda_sparse", type=float, default=.002)
    p.add_argument("--freeze_router", action="store_true",
                   help="history-only router; future labels still fit candidate maps")
    p.add_argument("--shuffle_train_seed", type=int, default=0,
                   help="shuffle TRAIN X-Y pairing within contiguous blocks")
    p.add_argument("--output", default="results/fpem_dual_pattern_memory/rare_learning_pilot")
    return p.parse_args()


def subset(record, mask):
    return {k: v[mask] for k, v in record.items()}


def score(record, prediction):
    return {"mse": float(per_sample_mse(record, prediction).mean()),
            "mae": float(per_sample_mae(record, prediction).mean())}


def stable_prediction(record, stable, basis, eta):
    if isinstance(stable, dict) and stable.get("kind") == "g1":
        prediction = record["base"] + stable["scale"] * ridge_query(
            record["key"], stable["weight"])
        return prediction, {}
    sq = stable.query(record["key"])
    return stable_forecast(record["base"], sq["prediction"], basis, eta), sq


def rare_query(method, learner, key, stable_query, offset):
    if method == "R0":
        v = clamp_variation(key-stable_query["matched_history"])
        return learner.query(v, torch.cat((key,v), -1), offset)
    return learner.query(key, offset)


def rare_forecast(record, method, learner, stable, basis, eta, reliability, offset,
                  variant="full"):
    anchor, sq = stable_prediction(record, stable, basis, eta)
    if variant == "a0":
        return record["base"], {"null": 1., "effect": 0.}
    if variant == "stable":
        return anchor, {"null": 1., "effect": 0.}
    q = rare_query(method, learner, record["key"], sq, offset)
    candidate = q["candidate"]
    use_basis = basis
    if variant == "rare_only":
        anchor, use_basis = record["base"], basis[:, :0]
    if variant == "shared_only":
        return anchor, {"null": 1., "effect": 0.}
    forecast, beta = variation_forecast(anchor, candidate, q["weight"],
                                        reliability, use_basis, True, 8)
    return forecast, {"null": float(q["null_weight"].mean()),
                      "effect": float((forecast-anchor).abs().mean()),
                      "beta": float(beta.mean())}


def gradient_teacher(key, truth, reference, patterns, seed):
    # Lightweight reference-map gradient signature, never Backbone gradient.
    from sklearn.cluster import MiniBatchKMeans
    flat = key.reshape(-1,64)
    design = torch.cat((flat, flat.new_ones(len(flat),1)), -1)
    residual = truth.reshape(len(flat),-1) - design @ reference
    signature = torch.cat((flat[:, :4]*residual.mean(-1,keepdim=True),
                           flat[:, -4:]*residual.std(-1,keepdim=True)), -1)
    km = MiniBatchKMeans(n_clusters=patterns, random_state=seed, n_init=3,
                         batch_size=min(256,len(flat)))
    km.fit(signature.numpy())
    center = torch.from_numpy(km.cluster_centers_.copy()).float()
    dist = torch.cdist(signature, center).square()/signature.shape[-1]
    # Centered teacher logit gives a supervised gradient consensus target.
    return (-dist + dist.mean(-1,keepdim=True)).reshape(*key.shape[:2],patterns)


def activation_stats(method, learner, record, starts, span, stable):
    _, sq = stable_prediction(record, stable, torch.zeros(record["truth"].shape[-1],0), 0.)
    q = rare_query(method, learner, record["key"], sq, 0.)
    w = q["weight"].reshape(-1,q["weight"].shape[-1])
    events = torch.div(starts.long(), span, rounding_mode="floor").repeat_interleave(record["key"].shape[1])
    independent = [int(torch.unique(events[w[:,k]>0]).numel()) for k in range(w.shape[-1])]
    return {"active_per_series": float((w>0).sum(-1).float().mean()),
            "all_null_ratio": float((w.sum(-1)<1e-8).float().mean()),
            "null_mean": float(q["null_weight"].mean()),
            "independent_event_support": independent,
            "activation_mass": w.sum(0).tolist(),
            "mapping_diversity": float(q["candidate"].std(2).mean())}


def record_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train_one(method, train, raw_starts, reference, original_rare, stable, args, shuffled=False):
    if method == "R0":
        if not shuffled and not args.shuffle_train_seed:
            return original_rare
        # Same original capacity and fitting, but only TRAIN targets permuted.
        from tools.run_fpem_dppm import fit_bank, variation_key, blocks_for
        groups = blocks_for(len(train["key"]))
        v = variation_key(stable, train["key"])
        mapping = torch.cat((train["key"],v),-1)
        return fit_bank(v, train["truth"], groups, original_rare.n_patterns,
                        original_rare.ridge, args.seed+11, mapping)[0]
    context = {"reference_map": reference, "starts": raw_starts,
               "span": args.seq_len+args.pred_len, "device": "cpu"}
    if method == "R5":
        context["teacher_gradient"] = gradient_teacher(train["key"], train["truth"],
                                                   reference, args.patterns,args.seed)
    return RarePatternLearner(method, args.patterns, args.rank, args.seed,
                             args.epochs, lambda_sparse=args.lambda_sparse,
                             freeze_router=args.freeze_router).fit(train["key"],train["truth"],context)


def evaluate_one(method, learner, stable, basis, eta, early, selection, test,
                 train, train_starts, args, output):
    output.mkdir(parents=True, exist_ok=True)
    # Reliability is estimated from future-disjoint early VAL, never TRAIN fit.
    early_stable, early_sq = stable_prediction(early, stable, basis, eta)
    early_q = rare_query(method, learner, early["key"], early_sq, 0.)
    reliability, evidence = prototype_segment_reliability(
        early_stable, early_q["candidate"],early["truth"],early_q["weight"],8)
    choices = (-3., -1., 0., 1., 3.) if method != "R0" else (-4.,-2.,-1.,0.,1.)
    best = None
    for offset in choices:
        pred,_ = rare_forecast(selection,method,learner,stable,basis,eta,reliability,offset)
        m = score(selection,pred)
        if best is None or m["mse"] < best["val_mse"]:
            best={"offset":offset,"val_mse":m["mse"],"val_mae":m["mae"]}
    offset=best["offset"]
    modes = {}
    for mode in ("a0","stable","rare_only","full"):
        pred,diag=rare_forecast(test,method,learner,stable,basis,eta,reliability,offset,mode)
        modes[mode]={**score(test,pred),**diag}
    train_stable,train_sq=stable_prediction(train,stable,basis,eta)
    train_q=rare_query(method,learner,train["key"],train_sq,offset)
    early_q=rare_query(method,learner,early["key"],early_sq,offset)
    select_stable,select_sq=stable_prediction(selection,stable,basis,eta)
    select_q=rare_query(method,learner,selection["key"],select_sq,offset)
    def candidate_mix(query, fallback):
        return ((query["weight"][...,None]*query["candidate"]).sum(2)
                + query["null_weight"][...,None]*fallback)
    mapping={"train_candidate_mse":float((candidate_mix(train_q,train_stable)-train["truth"]).square().mean()),
             "early_val_candidate_mse":float((candidate_mix(early_q,early_stable)-early["truth"]).square().mean()),
             "selection_candidate_mse":float((candidate_mix(select_q,select_stable)-selection["truth"]).square().mean()),
             "early_val_stable_mse":score(early,early_stable)["mse"],
             "early_val_rare_candidate_mse":score(early,candidate_mix(early_q,early_stable))["mse"],
             "early_val_reliability_mean":float(reliability.mean())}
    stats=activation_stats(method,learner,train,train_starts,args.seq_len+args.pred_len,stable)
    write_json(output/"mapping_oos.json", {"metrics":mapping,"segment_evidence":evidence,
                                         "scope":"rare TRAIN-fit vs early VAL A0-OOS, not TRAIN full-system OOF"})
    write_json(output/"activation.json",stats)
    write_json(output/"test_metrics.json", {"selection":best,"test":modes,
                                            "reference_sha256":args.reference_sha256})
    checkpoint={"rare":learner.state() if method=="R0" else learner.export(),
                "reliability":reliability,"null_offset":offset,
                "stable_reference":str(args.stable_reference)}
    if isinstance(stable, dict) and stable.get("kind") == "g1":
        checkpoint["g1_stable"]={"weight":stable["weight"],"scale":stable["scale"]}
    torch.save(checkpoint,output/"rare_model.pt")
    return best,modes,mapping,stats,reliability


def main():
    cli=parse_args()
    torch.manual_seed(cli.seed)
    torch.set_num_threads(4)
    methods=[s.strip() for s in cli.methods.split(",") if s.strip()]
    if not methods or any(s not in {f"R{i}" for i in range(8)} for s in methods):
        raise ValueError("methods must be R0–R7")
    if cli.stable_mode == "g1" and methods != ["R3"]:
        raise ValueError("G1 stable-mode pilot is restricted to R3")
    out=ROOT/cli.output/cli.case
    out.mkdir(parents=True,exist_ok=True)
    cli.base_run_config=str(BASE/(f"{cli.case}_A0_complementary_gate_predictive_"
                             "sub0_orth0_refnone_fp0_rank8_bias-4/run_config.json"))
    cli.allow_legacy_reference_signature=cli.case.startswith("moderntcn")
    config,model,loaders,device,sha=load_experiment(cli)
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise AssertionError("Backbone must remain frozen")
    cli.seq_len,cli.pred_len=config.seq_len,config.pred_len
    cli.reference_sha256=sha
    old=ROOT/"results/fpem_dual_pattern_memory/point_pilot"/cli.case/"dual_pattern_memory.pt"
    bundle=torch.load(old,map_location="cpu",weights_only=False)
    cache=torch.load(ROOT/"results/fpem_predictive_memory/pilot_20261024"/cli.case/
                     "phase0_cache.pt",map_location="cpu",weights_only=False)
    if bundle["reference_sha256"]!=sha or cache["reference_sha256"]!=sha:
        raise ValueError("A0 / cache / stable checkpoint hash mismatch")
    cli.stable_reference=old
    stable=PatternBank.from_state(bundle["stable"])
    original_rare=PatternBank.from_state(bundle["variation"])
    basis=bundle["basis"]
    eta=float(bundle["selection"]["eta"])
    raw=cache["records"]
    train=normalized_record({"x":raw["x"],"target":raw["target"],
                             "base":raw["a0_pred"]},bundle["floor"])
    train_starts=raw["starts"].long()
    # Last TRAIN target must end before first VAL future; no border leakage.
    first_val_history=len(loaders["train"].dataset.dataset.data_x)-cli.seq_len
    valid=train_starts+cli.seq_len+cli.pred_len<=first_val_history
    train=subset(train,valid)
    train_starts=train_starts[valid]
    if cli.shuffle_train_seed:
        generator=torch.Generator().manual_seed(cli.shuffle_train_seed)
        permutation=torch.arange(len(train_starts))
        for part in torch.tensor_split(permutation,4):
            indices=part.clone()
            permutation[indices]=indices[torch.randperm(len(indices),generator=generator)]
        train["truth"]=train["truth"][permutation]
    val,val_ids=sparse_record(model,loaders["val"].dataset.dataset,
                              config,device,cli.eval_stride,bundle["floor"])
    test,_=sparse_record(model,loaders["test"].dataset.dataset,
                         config,device,cli.eval_stride,bundle["floor"])
    cut=len(loaders["val"].dataset.dataset.data_x)//2
    first_test_history=len(loaders["val"].dataset.dataset.data_x)-cli.seq_len
    early_mask=val_ids+cli.seq_len+cli.pred_len<=cut
    select_mask=(val_ids>=cut)&(val_ids+cli.seq_len+cli.pred_len<=first_test_history)
    early,selection=subset(val,early_mask),subset(val,select_mask)
    if min(len(early["key"]),len(selection["key"]))<8:
        raise ValueError("insufficient purged VAL OOS / selection support")
    if int((val_ids[early_mask]+cli.seq_len+cli.pred_len).max())>int(val_ids[select_mask].min()):
        raise AssertionError("early VAL future overlaps selection history")
    if int((val_ids[select_mask]+cli.seq_len+cli.pred_len).max())>first_test_history:
        raise AssertionError("selection future overlaps TEST history")
    # Shared full-future reference is internal to rare candidate generation;
    # it never replaces the frozen DPPM stable prediction.
    reference=residual_ridge(train["key"],train["truth"],100.)
    g1=fit_g1(early,selection)
    g1_test=score(test,test["base"]+g1["scale"]*ridge_query(test["key"],g1["weight"]))
    if cli.stable_mode == "g1":
        # Replace only R3's stable prediction. Its rare learner, candidate
        # maps, reliability, NULL selection and training budget stay unchanged.
        stable={"kind":"g1","weight":g1["weight"],"scale":g1["scale"]}
        basis=basis[:, :0]  # The old DPPM stable subspace does not describe G1.
        eta=0.
        cli.stable_reference=out/"g1_reference.json"
    write_json(out/"run_config.json", {"case":cli.case,"methods":methods,"seed":cli.seed,
        "patterns":cli.patterns,"rank":cli.rank,"epochs":cli.epochs,
        "stable_mode":cli.stable_mode,
        "reference_sha256":sha,"stable_reference":str(old),"stable_eta":eta,
        "eval_stride":cli.eval_stride,"controls":cli.controls,
        "lambda_sparse":cli.lambda_sparse,"freeze_router":cli.freeze_router,
        "shuffle_train_seed":cli.shuffle_train_seed,
        "source":"TRAIN patterns; early purged VAL reliability; later purged VAL selection; TEST report",
        "g1_protocol":"G1 map fitted on early purged VAL and tuned on later purged VAL; "
                      "G1 early-VAL reliability is in-sample for that map" if cli.stable_mode=="g1" else None,
        "a0_train_cache":"in-sample A0; never called full-system OOF"})
    write_json(out/"g1_reference.json", {"selection_mse":g1["selection_mse"],
        "ridge":g1["ridge"],"scale":g1["scale"],"test":g1_test,
        "source":"early VAL, future-disjoint from later VAL selection"})
    print(f"READY {cli.case}: TRAIN={len(train['key'])} early_val={len(early['key'])} "
          f"selection={len(selection['key'])} TEST={len(test['key'])}",flush=True)
    for method in methods:
        target=out/method
        target.mkdir(exist_ok=True)
        if (target/"complete").exists():
            print(f"SKIP {cli.case} {method} complete",flush=True);continue
        started=time.monotonic()
        learner=train_one(method,train,train_starts,reference,original_rare,stable,cli)
        best,modes,mapping,stats,rel=evaluate_one(method,learner,stable,basis,eta,
                        early,selection,test,train,train_starts,cli,target)
        row={"case":cli.case,"method":method,"val_mse":best["val_mse"],
             "test_a0_mse":modes["a0"]["mse"],"test_stable_mse":modes["stable"]["mse"],
             "test_g1_mse":g1_test["mse"],
             "test_rare_only_mse":modes["rare_only"]["mse"],
             "test_full_mse":modes["full"]["mse"],"test_full_mae":modes["full"]["mae"],
             "early_val_rare_candidate_mse":mapping["early_val_rare_candidate_mse"],
             "early_val_stable_mse":mapping["early_val_stable_mse"],
             "reliability_mean":mapping["early_val_reliability_mean"],
             "null_mean":modes["full"]["null"],
             "train_seconds":time.monotonic()-started}
        record_row(out/"all_results.csv",row)
        record_row(out/"rare_mapping_oof.csv",{"case":cli.case,"method":method,**mapping})
        record_row(out/"rare_activation_statistics.csv",{"case":cli.case,"method":method,
                 "active_per_series":stats["active_per_series"],
                 "all_null_ratio":stats["all_null_ratio"],
                 "independent_event_support":json.dumps(stats["independent_event_support"]),
                 "mapping_diversity":stats["mapping_diversity"]})
        record_row(out/"common_rare_contributions.csv",{"case":cli.case,"method":method,
            "a0":modes["a0"]["mse"],"stable":modes["stable"]["mse"],
            "rare_only":modes["rare_only"]["mse"],"joint":modes["full"]["mse"]})
        record_row(out/"efficiency.csv",{"case":cli.case,"method":method,
            "seconds":time.monotonic()-started,
            "parameter_count":0 if method=="R0" else learner.stats["parameters"]})
        if cli.controls:
            # Within contiguous TRAIN quarters; future targets alone are permuted.
            shuffled=dict(train)
            perm=torch.arange(len(train_starts))
            generator=torch.Generator().manual_seed(cli.seed+981)
            for part in torch.tensor_split(torch.arange(len(perm)),4):
                perm[part]=part[torch.randperm(len(part),generator=generator)]
            shuffled["truth"]=train["truth"][perm]
            shuffled_learner=train_one(method,shuffled,train_starts,reference,
                                      original_rare,stable,cli,True)
            _,null_modes,_,_,_=evaluate_one(method,shuffled_learner,stable,basis,eta,
                           early,selection,test,shuffled,train_starts,cli,target/"shuffled")
            record_row(out/"rare_pairing_controls.csv",{"case":cli.case,"method":method,
                "true_test_mse":modes["full"]["mse"],
                "shuffled_test_mse":null_modes["full"]["mse"],
                "true_minus_shuffled":modes["full"]["mse"]-null_modes["full"]["mse"]})
        (target/"complete").write_text("complete\n",encoding="utf-8")
        print(f"DONE {cli.case} {method} A0={modes['a0']['mse']:.6f} "
              f"stable={modes['stable']['mse']:.6f} full={modes['full']['mse']:.6f}",flush=True)


if __name__=="__main__":
    main()
