#!/usr/bin/env python3
"""Frozen A0 -> EIIL3 history router -> Soft-IRM F03 -> three weighted residual Ridges.

Legacy q is accepted only after the asset audit matches its source A0 SHA.
Missing q may be generated offline from TRAIN A0 probes, never VAL/TEST Y.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.predictive_env.eiil_environment import PredictiveConflictEnvironment
from models.predictive_env.eiil3_modules import (
    global_residual_ridge, predict_residual_candidates,
    soft_risks_and_full_head_penalty, weighted_residual_ridge,
)
from models.predictive_env.relation_prototypes import route, train_router, teacher_quality
from tools.run_fpem_f03_r3_invariance import collect, prepare
from tools.run_fpem_mechanism_fusion import flat_record, score
from tools.run_fpem_shared_factorization_g1r3 import load_experiment

SPACES = ("A", "B1", "B2", "C", "val", "test")
VARIANTS = ("A0", "I3_ERM", "I3_SoftIRM_EIIL3", "I3_SoftIRM_ShuffleQ",
            "I3_GlobalRidge", "I3_R3_EIIL3", "I3_R3_ShuffleQ")


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def _take(record, mask):
    return {name: value[mask] for name, value in record.items()}


def split_train(data, span):
    start = data["start"]
    lo, hi = int(start.min()), int(start.max())
    cuts = [lo + int((hi-lo)*fraction) for fraction in (.50, .67, .84)]
    masks = (start+span <= cuts[0],
             (start >= cuts[0]) & (start+span <= cuts[1]),
             (start >= cuts[1]) & (start+span <= cuts[2]),
             start >= cuts[2])
    spaces = {name: _take(data, mask) for name, mask in zip(SPACES[:4], masks)}
    for left, right in zip(SPACES[:3], SPACES[1:4]):
        if not len(spaces[left]["start"]) or not len(spaces[right]["start"]):
            raise ValueError(f"empty purged TRAIN block {left}/{right}")
        if spaces[left]["start"].max()+span > spaces[right]["start"].min():
            raise AssertionError(f"TRAIN blocks {left}/{right} overlap")
    return spaces, cuts


def event_shuffle(q, starts, seed):
    """Permute whole time-window q bundles, never split its channels."""
    unique, inverse = np.unique(starts, return_inverse=True)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(unique))
    first = np.unique(inverse, return_index=True)[1]
    event_q = q[first]
    return event_q[perm][inverse]


def valid_teacher(cli, train_ids, raw_train):
    cache = cli.output / "teacher_q.npz"
    if cache.exists():
        with np.load(cache) as src:
            if np.array_equal(src["sample_id"], train_ids):
                return src["q"].astype(np.float32), json.loads((cli.output/"teacher_q.json").read_text())
    if cli.asset.get("status") == "READY":
        source = Path(cli.asset["env_artifact"])
        if file_hash(source) != cli.asset["env_artifact_sha256"]:
            raise ValueError("existing teacher q SHA changed after audit")
        with source.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        lookup = {int(row["sample_id"]): np.array([float(row[f"q_env_{k}"]) for k in range(3)], np.float32)
                  for row in rows}
        if not set(train_ids.tolist()).issubset(lookup):
            raise ValueError("legacy q does not cover every sampled TRAIN window")
        q = np.stack([lookup[int(sample)] for sample in train_ids])
        meta = {"source": "legacy_predictive_env_iv_final_stage", "path": str(source),
                "sha256": cli.asset["env_artifact_sha256"],
                "provenance": cli.asset["q_provenance"], "sample_count": len(q)}
    elif cli.generate_missing_q:
        inference = PredictiveConflictEnvironment(
            len(train_ids), env_num=3, steps=cli.eiil_steps, lr=.1,
            seed=cli.seed, diagnostic_samples=min(256,len(train_ids)),
            random_partition_repeats=10, quality_diagnostics=True)
        q, diag = inference.update(raw_train["base"], raw_train["truth"],
                                   np.arange(len(train_ids)), source_split="train")
        meta = {"source": "offline_frozen_A0_TRAIN_stride_subset", "sample_count": len(q),
                "train_stride": cli.train_stride, "diagnostics": diag,
                "provenance": "A0 fixed; EIIL optimized only on sampled TRAIN windows"}
    else:
        raise RuntimeError("MISSING_ENV_Q: use --generate-missing-q only for exact A0 case")
    if q.shape != (len(train_ids),3) or not np.isfinite(q).all() or not np.allclose(q.sum(1),1,atol=2e-4):
        raise ValueError("invalid TRAIN teacher q")
    np.savez_compressed(cache, sample_id=train_ids, q=q)
    dump(cli.output/"teacher_q.json",meta)
    return q, meta


def get_data(cli):
    cache = cli.output / "features.npz"
    meta_file = cli.output / "features.json"
    if cache.exists() and meta_file.exists():
        meta = json.loads(meta_file.read_text())
        if meta["A0_sha256"] != cli.asset["A0_sha256"]:
            raise ValueError("cached A0 SHA mismatch")
        with np.load(cache) as source:
            data = {space:{field:source[f"{space}_{field}"] for field in
                    ("design","key","y","a0","ratio","start","channel")}
                    for space in SPACES}
            teacher={space:source[f"q_teacher_{space}"] for space in SPACES[:4]}
        state = torch.load(cli.output/"a0_head.pt",map_location="cpu",weights_only=False)
        return data, meta, state, teacher
    loader_cli = argparse.Namespace(base_run_config=cli.asset["base_run_config"],
                                    gpu=cli.gpu,seed=cli.seed,
                                    allow_legacy_reference_signature=True)
    args, model, loaders, device, a0_sha = load_experiment(loader_cli)
    if a0_sha != cli.asset["A0_sha256"]:
        raise ValueError("A0 checkpoint SHA differs from audit")
    datasets = {space:loader.dataset.dataset for space,loader in loaders.items()}
    raw_train = collect(model,datasets["train"],args,device,cli.train_stride)
    train_ids = raw_train["ids"].numpy().astype(np.int64)
    q_window,q_meta = valid_teacher(cli,train_ids,raw_train)
    floor = raw_train["x"].float().var(1,unbiased=False).add(1e-5).sqrt().median(0).values.mul(.25).clamp_min(.01)
    train = flat_record(prepare(raw_train,floor))
    spaces,cuts = split_train(train,args.seq_len+args.pred_len)
    q_by_start = dict(zip(train_ids.tolist(),q_window))
    teacher = {space:np.stack([q_by_start[int(s)] for s in spaces[space]["start"]])
               for space in SPACES[:4]}
    for space,stride in (("val",cli.val_stride),("test",cli.test_stride)):
        raw = collect(model,datasets[space],args,device,stride)
        spaces[space]=flat_record(prepare(raw,floor))
    head=(model.h_forecast_head.weight.detach().cpu().numpy().astype(np.float32),
          model.h_forecast_head.bias.detach().cpu().numpy().astype(np.float32))
    a=spaces["A"]
    init=((a["design"][:64]@head[0].T+head[1])*a["ratio"][:64])
    error=float(np.max(np.abs(init-a["a0"][:64])))
    if error>1e-3:raise AssertionError(f"A0 head initialization mismatch {error}")
    if cli.persist_features:
        np.savez_compressed(cache,**{f"{space}_{field}":value for space in SPACES
                                      for field,value in spaces[space].items()},
                            **{f"q_teacher_{space}":teacher[space] for space in SPACES[:4]})
    torch.save(head,cli.output/"a0_head.pt")
    meta={"A0_sha256":a0_sha,"q_sha256":file_hash(cli.output/"teacher_q.npz"),
          "dataset":args.dataset_name,"backbone":args.backbone,"horizon":args.pred_len,
          "seed":cli.seed,"span":args.seq_len+args.pred_len,"train_cuts":cuts,
          "init_error_max":error,"teacher_source":q_meta["source"],
          "strides":{"train":cli.train_stride,"val":cli.val_stride,"test":cli.test_stride},
          "normalization":"history_only_per_window; same A0 head ratio",
          "features_persisted":bool(cli.persist_features)}
    dump(meta_file,meta)
    return spaces,meta,head,teacher


def fit_router(cli,data,teacher,meta,shuffled=False):
    name="router_shuffled.pt" if shuffled else "router.pt"
    path=cli.output/name
    a_teacher=event_shuffle(teacher["A"],data["A"]["start"],cli.seed+91) if shuffled else teacher["A"]
    best=None
    for lr in ((.001,.003,.01) if cli.pilot_search else (cli.router_lr,)):
        net,seconds,parameters=train_router(data["A"]["key"],a_teacher,
                                          num_env=3,lr=lr,seed=cli.seed,epochs=cli.router_epochs)
        actual=route(net,data["B1"]["key"])
        diagnostic=teacher_quality(teacher["B1"],actual)
        from tools.storage.retention import record_trial
        record_trial(cli.output,{"trial_id":f"{'shuffle' if shuffled else 'real'}_router_lr_{lr:g}",
                     "comparison_key":f"{cli.case_key}/{'shuffle' if shuffled else 'real'}_router",
                     "params":{"lr":lr},"seed":cli.seed,"epoch":cli.router_epochs,
                     "status":"complete","objective":"B1_soft_kl","selected_by":"VAL_B1",
                     "B1_soft_kl":diagnostic["soft_kl"],"seconds":seconds})
        if best is None or diagnostic["soft_kl"]<best[0]:
            best=(diagnostic["soft_kl"],lr,net,seconds,parameters,diagnostic)
    _,lr,net,seconds,parameters,diagnostic=best
    q={space:route(net,data[space]["key"]) for space in SPACES}
    torch.save(net.state_dict(),path)
    result={"lr":lr,"B1_quality":diagnostic,"seconds":seconds,
            "parameters":parameters,"shuffled_teacher":shuffled,
            "router_sha256":file_hash(path)}
    dump(cli.output/("router_shuffled.json" if shuffled else "router.json"),result)
    return q,result


def fit_head(cli,data,q,initial,variant,lr,lam):
    torch.manual_seed(cli.seed)
    device=torch.device(f"cuda:{cli.gpu}" if torch.cuda.is_available() else "cpu")
    weight,bias=initial
    head=torch.nn.Linear(weight.shape[1],weight.shape[0]).to(device)
    with torch.no_grad():
        head.weight.copy_(torch.from_numpy(weight).to(device));head.bias.copy_(torch.from_numpy(bias).to(device))
    opt=torch.optim.Adam(head.parameters(),lr=lr)
    tensors={space:{key:torch.from_numpy(data[space][key]).to(device) for key in ("design","y","ratio")}
             for space in ("A","B1")}
    qt=torch.from_numpy(q["A"]).to(device)
    best=float("inf");chosen_epoch=0;chosen=None;trace=[]
    for epoch in range(cli.epochs+1):
        head.eval()
        with torch.no_grad():
            b=tensors["B1"]
            mse=float(((head(b["design"])*b["ratio"]-b["y"])**2).mean())
        trace.append({"epoch":epoch,"B1_mse":mse})
        if mse<best:
            best=mse;chosen_epoch=epoch
            chosen={name:value.detach().cpu().clone() for name,value in head.state_dict().items()}
        if epoch==cli.epochs:break
        head.train()
        a=tensors["A"]
        for indices in torch.randperm(len(a["design"]),device=device).split(cli.batch_size):
            risks,penalty,_,_=soft_risks_and_full_head_penalty(
                head,a["design"][indices],a["y"][indices],a["ratio"][indices],qt[indices])
            loss=risks.mean()+lam*penalty
            if not torch.isfinite(loss):raise FloatingPointError("nonfinite I3 loss")
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    head.load_state_dict(chosen);head.eval();head.requires_grad_(False)
    preds={}
    with torch.no_grad():
        for space in SPACES:
            chunks=[]
            for start in range(0,len(data[space]["design"]),512):
                x=torch.from_numpy(data[space]["design"][start:start+512]).to(device)
                ratio=torch.from_numpy(data[space]["ratio"][start:start+512]).to(device)
                chunks.append((head(x)*ratio).cpu().numpy())
            preds[space]=np.concatenate(chunks).astype(np.float32)
    return preds,chosen,{"lr":lr,"lambda_inv":lam,"B1_mse":best,
                         "best_epoch":chosen_epoch,"trace":trace,
                         "weight_relative_drift":float(np.linalg.norm(chosen["weight"].numpy()-weight)/max(np.linalg.norm(weight),1e-8))}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",type=Path,default=Path("results/fpem_eiil3_20261110/manifest.jsonl"))
    p.add_argument("--case-key",required=True)
    p.add_argument("--output-root",type=Path,default=Path("results/fpem_eiil3_20261110"))
    p.add_argument("--gpu",type=int,default=0);p.add_argument("--seed",type=int,default=2021)
    p.add_argument("--train-stride",type=int,default=8)
    p.add_argument("--val-stride",type=int,default=8)
    p.add_argument("--test-stride",type=int,default=1)
    p.add_argument("--router-epochs",type=int,default=30)
    p.add_argument("--router-lr",type=float,default=.01)
    p.add_argument("--epochs",type=int,default=10)
    p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--lr-i3",type=float,default=3e-4)
    p.add_argument("--lambda-inv",type=float,default=1e-4)
    p.add_argument("--ridge-alpha",type=float,default=100.)
    p.add_argument("--eiil-steps",type=int,default=50)
    p.add_argument("--generate-missing-q",action="store_true")
    p.add_argument("--pilot-search",action="store_true")
    p.add_argument("--persist-features",action="store_true",
                   help="Opt in to a persistent large feature NPZ; default is in-memory only")
    p.add_argument("--resume",action="store_true")
    args=p.parse_args();torch.set_num_threads(4)
    row=next((json.loads(line) for line in args.manifest.read_text().splitlines()
              if json.loads(line)["case_key"]==args.case_key),None)
    if row is None:raise KeyError(args.case_key)
    if row["status"] not in ("READY","MISSING_ENV_Q"):
        print(json.dumps({"case":args.case_key,"status":"BLOCKED","reason":row["missing_reason"]}));return
    if row["status"]=="MISSING_ENV_Q" and not args.generate_missing_q:
        print(json.dumps({"case":args.case_key,"status":"BLOCKED","reason":row["missing_reason"]}));return
    args.asset=row
    args.output=args.output_root/args.case_key
    complete=args.output/"point_complete"
    if args.resume and complete.exists():
        saved=json.loads((args.output/"point_metrics.json").read_text())
        if saved["A0_sha256"]==row["A0_sha256"]:
            print(json.dumps({"case":args.case_key,"status":"REUSED"}));return
    from tools.storage.retention import require_free_space, record_trial
    require_free_space(args.output_root, policy_path=Path("config/storage_policy.yaml"))
    args.output.mkdir(parents=True,exist_ok=True)
    data,meta,initial,teacher=get_data(args)
    q,router_info=fit_router(args,data,teacher,meta)
    shuffled_q,shuffled_info=fit_router(args,data,teacher,meta,shuffled=True)
    lr_candidates=(2e-5,1e-4,3e-4) if args.pilot_search else (args.lr_i3,)
    trials=[]
    erm_runs=[]
    for lr in lr_candidates:
        pred,state,info=fit_head(args,data,q,initial,"I3_ERM",lr,0.)
        erm_runs.append((info["B1_mse"],lr,pred,state,info))
        trials.append({"module":"I3_ERM",**info})
        val_score=score(pred["val"],data["val"]["y"])
        test_score=score(pred["test"],data["test"]["y"])
        record_trial(args.output,{"trial_id":f"I3_ERM_lr_{lr:g}",
                     "comparison_key":f"{args.case_key}/I3_ERM", "params":{"lr":lr},
                     "seed":args.seed,"epoch":info["best_epoch"],"status":"complete",
                     "objective":"B1_mse","selected_by":"VAL_B1",
                     "B1_mse":info["B1_mse"],"val":val_score,"test":test_score,
                     "trace":info["trace"],"test_used_for_selection":False})
    _,chosen_lr,erm,erm_state,erm_info=min(erm_runs,key=lambda r:r[0])
    lambda_candidates=(0.,1e-4,1e-3) if args.pilot_search else (args.lambda_inv,)
    soft_runs=[]
    for lam in lambda_candidates:
        pred,state,info=fit_head(args,data,q,initial,"I3_SoftIRM_EIIL3",chosen_lr,lam)
        soft_runs.append((info["B1_mse"],lam,pred,state,info))
        trials.append({"module":"I3_SoftIRM_EIIL3",**info})
        val_score=score(pred["val"],data["val"]["y"])
        test_score=score(pred["test"],data["test"]["y"])
        record_trial(args.output,{"trial_id":f"I3_SoftIRM_lr_{chosen_lr:g}_lam_{lam:g}",
                     "comparison_key":f"{args.case_key}/I3_SoftIRM_EIIL3",
                     "params":{"lr":chosen_lr,"lambda_inv":lam},
                     "seed":args.seed,"epoch":info["best_epoch"],"status":"complete",
                     "objective":"B1_mse","selected_by":"VAL_B1",
                     "B1_mse":info["B1_mse"],"val":val_score,"test":test_score,
                     "trace":info["trace"],"test_used_for_selection":False})
    _,chosen_lam,soft,soft_state,soft_info=min(soft_runs,key=lambda r:r[0])
    shuffled,shuffled_state,shuffled_head_info=fit_head(args,data,shuffled_q,initial,
                                      "I3_SoftIRM_ShuffleQ",chosen_lr,chosen_lam)
    torch.save(erm_state,args.output/"I3_ERM_head.pt")
    torch.save(soft_state,args.output/"I3_SoftIRM_EIIL3_head.pt")
    torch.save(shuffled_state,args.output/"I3_SoftIRM_ShuffleQ_head.pt")
    dump(args.output/"lr_search_trials.json",trials)
    chosen_alpha=None;alpha_trials=[]
    for alpha in ((1.,10.,100.,1000.) if args.pilot_search else (args.ridge_alpha,)):
        r1=data["B1"]["y"]-soft["B1"]
        model=weighted_residual_ridge(data["B1"]["key"],r1,q["B1"],alpha)
        global_model=global_residual_ridge(data["B1"]["key"],r1,alpha)
        _,corr,_=predict_residual_candidates(model,data["B2"]["key"],q["B2"])
        mse=score(soft["B2"]+corr,data["B2"]["y"])["mse"]
        _,test_corr,_=predict_residual_candidates(model,data["test"]["key"],q["test"])
        test_score=score(soft["test"]+test_corr,data["test"]["y"])
        alpha_trials.append({"alpha":alpha,"B2_mse":mse})
        record_trial(args.output,{"trial_id":f"R3_ridge_alpha_{alpha:g}",
                     "comparison_key":f"{args.case_key}/R3_ridge",
                     "params":{"alpha":alpha},"seed":args.seed,"epoch":0,
                     "status":"complete","objective":"B2_mse","selected_by":"VAL_B2",
                     "B2_mse":mse,"test":test_score,"test_used_for_selection":False})
        if chosen_alpha is None or mse<chosen_alpha[0]:
            chosen_alpha=(mse,alpha,model,global_model)
    _,alpha,model,global_model=chosen_alpha
    model_shuffle=weighted_residual_ridge(data["B1"]["key"],
                              data["B1"]["y"]-soft["B1"],shuffled_q["B1"],alpha)
    np.savez_compressed(args.output/"r3_ridge.npz",weight=model["weight"],bias=model["bias"],
                        mass=np.asarray(model["mass"]),alpha=np.asarray(alpha))
    metrics={"case":args.case_key,"status":"complete","A0_sha256":row["A0_sha256"],
             "q_sha256":meta["q_sha256"],"router_sha256":router_info["router_sha256"],
             "I3_sha256":file_hash(args.output/"I3_SoftIRM_EIIL3_head.pt"),
             "R3_sha256":file_hash(args.output/"r3_ridge.npz"),
             "router":router_info,"router_shuffled":shuffled_info,
             "I3_ERM":erm_info,"I3_SoftIRM_EIIL3":soft_info,
             "I3_SoftIRM_ShuffleQ":shuffled_head_info,
             "ridge_alpha_trials":alpha_trials,"ridge_alpha_selected":alpha,
             "normalization":meta["normalization"],"teacher_source":meta["teacher_source"],
             "scores":{}}
    flow_cache={}
    for space in SPACES:
        x=data[space]["key"]
        candidate,correction,u=predict_residual_candidates(model,x,q[space])
        _,correction_shuffled,_=predict_residual_candidates(model_shuffle,x,shuffled_q[space])
        global_corr=x@global_model["weight"]+global_model["bias"]
        predictions={"A0":data[space]["a0"],"I3_ERM":erm[space],
                     "I3_SoftIRM_EIIL3":soft[space],
                     "I3_SoftIRM_ShuffleQ":shuffled[space],
                     "I3_GlobalRidge":soft[space]+global_corr,
                     "I3_R3_EIIL3":soft[space]+correction,
                     "I3_R3_ShuffleQ":soft[space]+correction_shuffled}
        metrics["scores"][space]={variant:score(prediction,data[space]["y"])
                                   for variant,prediction in predictions.items()}
        if space in ("val","test"):
            flow_cache[f"{space}_key"]=x
            flow_cache[f"{space}_y"]=data[space]["y"]
            flow_cache[f"{space}_start"]=data[space]["start"]
            flow_cache[f"{space}_point"]=predictions["I3_R3_EIIL3"]
            flow_cache[f"{space}_u"]=u
        if space=="test":
            metrics["u_test"]={"mean":float(u.mean()),"std":float(u.std()),
                               "p10_p50_p90":np.quantile(u,[.1,.5,.9]).tolist(),
                               "candidate_spread":float(candidate.std(axis=1).mean())}
    np.savez_compressed(args.output/"flow_input.npz",**flow_cache)
    dump(args.output/"point_metrics.json",metrics)
    dump(args.output/"run_config.json",{key:value for key,value in vars(args).items()
                                        if key not in ("asset",)})
    from tools.storage.common import atomic_json
    try:
        commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
        diff=subprocess.check_output(["git","diff","--","tools","models","config"],cwd=ROOT)
        dirty_hash=hashlib.sha256(diff).hexdigest()
    except (OSError,subprocess.CalledProcessError):
        commit="unknown";dirty_hash="unknown"
    atomic_json(args.output/"manifest.json",{
        "run_id":args.case_key+f"/seed_{args.seed}","type":"EIIL3 formal point study",
        "status":"complete","git_commit":commit,"tracked_code_diff_sha256":dirty_hash,
        "A0_checkpoint_sha256":row["A0_sha256"],"teacher_q_sha256":meta["q_sha256"],
        "router_sha256":router_info["router_sha256"],
        "I3_sha256":metrics["I3_sha256"],"R3_sha256":metrics["R3_sha256"],
        "backbone":meta["backbone"],"dataset":meta["dataset"],
        "pred_len":meta["horizon"],"seed":args.seed,"environment_k":3,
        "splits":meta["train_cuts"],"normalization":meta["normalization"],
        "feature_cache_persistent":meta.get("features_persisted",True),
        "feature_cache_key":{"A0_sha256":row["A0_sha256"],"q_sha256":meta["q_sha256"],
                             "dataset":meta["dataset"],"pred_len":meta["horizon"],
                             "seed":args.seed,"strides":meta["strides"]},
        "selection":{"I3":"VAL_B1_MSE","R3":"VAL_B2_MSE",
                     "test_used_for_selection":False},
        "metrics_file":"point_metrics.json","config_file":"run_config.json",
        "shared_A0_config":row["base_run_config"],
        "versions":{"torch":torch.__version__,"numpy":np.__version__}})
    complete.write_text("complete\n")
    print(json.dumps({"case":args.case_key,"status":"complete","test":metrics["scores"]["test"],
                      "u_test":metrics["u_test"]}),flush=True)


if __name__=="__main__":main()
