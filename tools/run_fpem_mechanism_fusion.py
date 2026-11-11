#!/usr/bin/env python3
"""Frozen Stage-1 groups -> common F03 anchor -> full-future experts -> Flow.

The group artifact is read-only. A/B1/B2/C are chronological TRAIN blocks:
A fits F03, router and experts; B1 selects F03 epoch and calibrates expert
reliability; B2 selects NULL; C audits later TRAIN transfer for the added maps.
The frozen A0 saw TRAIN C, so C residuals are not full-model OOF. TEST is report-only.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.predictive_env.mechanism_fusion import (FrozenGrouping, K, candidates,
                                                    fit_full_future_experts,
                                                    independent_support, sha256)
from models.predictive_env.relation_prototypes import train_router, route, teacher_quality
from models.predictive_env.dual_pattern_memory.forecast_fusion import (
    prototype_segment_reliability, variation_forecast)
from models.predictive_env.residual_difference.condition import ResidualDifferenceCondition
from models.predictive_env.probability_heads import ProbabilityHeadBank
from tools.evaluate_fpem_probabilistic import probabilistic_metrics
from tools.run_fpem_f03_r3_invariance import collect, prepare
from tools.run_fpem_shared_factorization_g1r3 import load_experiment

METHODS = {"G3": "G3_KerHRM_lite", "G4": "G4_ANM_MM_lite", "G7": "G7_Ridge_Update",
           "Raw": "B0_Raw_History", "Random": "B1_Random_Balanced", "ShuffleY": "B2_ShuffleY_G1"}
SPACES = ("A", "B1", "B2", "C", "val", "test")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def score(pred, truth):
    return {"mse": float(np.mean((pred - truth) ** 2)),
            "mae": float(np.mean(np.abs(pred - truth)))}


def flat_record(raw):
    n, c, p, d = raw["hidden"].shape
    return {"design": raw["hidden"].permute(0, 1, 3, 2).reshape(n*c, p*d).numpy().astype(np.float32),
            "key": raw["key"].reshape(n*c, 64).numpy().astype(np.float32),
            "y": raw["r3_truth"].reshape(n*c, -1).numpy().astype(np.float32),
            "a0": raw["base_hist_norm"].reshape(n*c, -1).numpy().astype(np.float32),
            "ratio": (raw["std"] / raw["scale"]).transpose(1, 2).reshape(n*c, 1).numpy().astype(np.float32),
            "start": raw["ids"].long()[:, None].expand(n, c).reshape(-1).numpy(),
            "channel": torch.arange(c)[None].expand(n, c).reshape(-1).numpy()}


def take(data, index):
    return {key: value[index] for key, value in data.items()}


def prepare_shared(cli, case_dir):
    cache = case_dir / "features.npz"
    manifest = case_dir / "features.json"
    if cache.exists() and manifest.exists():
        with np.load(cache) as contents:
            data = {space: {key: contents[f"{space}_{key}"] for key in
                    ("design", "key", "y", "a0", "ratio", "start", "channel")}
                    for space in SPACES}
        state=torch.load(case_dir/"a0_head.pt",map_location="cpu",weights_only=False)
        return data, json.loads(manifest.read_text()), state
    base = Path(cli.base_run_config)
    config = json.loads(base.read_text())
    load_cli = argparse.Namespace(**vars(cli))
    load_cli.seed = config["seed"]
    args, model, loaders, device, a0_sha = load_experiment(load_cli)
    if any(p.requires_grad for p in model.parameters()):
        raise AssertionError("A0 was not frozen")
    stage = json.loads((Path(cli.stage1_case)/"case_cache.json").read_text())
    if (stage["a0_sha256"] != a0_sha or stage["dataset"] != args.dataset_name or
            stage["pred_len"] != args.pred_len or stage["stride"] != cli.train_stride):
        raise ValueError("Stage1 case/A0 identity or TRAIN stride mismatch")
    datasets = {name: loader.dataset.dataset for name, loader in loaders.items()}
    raw_train = collect(model, datasets["train"], args, device, cli.train_stride)
    floor = raw_train["x"].float().var(1, unbiased=False).add(1e-5).sqrt().median(0).values.mul(.25).clamp_min(.01)
    full_train = flat_record(prepare(raw_train, floor))
    reference = FrozenGrouping(Path(cli.stage1_case)/"methods"/METHODS["G3"],
                               a0_sha=a0_sha, dataset=args.dataset_name,
                               horizon=args.pred_len, seed=cli.seed)
    joined = reference.join(args.dataset_name, full_train["start"], full_train["channel"])
    index_file=Path(cli.stage1_case)/"case_indices.npz"
    if not index_file.exists():index_file=Path(cli.stage1_case)/"case_cache.npz"
    stage_cache = np.load(index_file)
    c_keys = set(zip(stage_cache["C_ids"].tolist(), stage_cache["C_channels"].tolist()))
    c_index = np.array([i for i, key in enumerate(zip(full_train["start"], full_train["channel"]))
                        if key in c_keys], dtype=np.int64)
    if len(c_index) != len(c_keys):
        raise ValueError("Stage1 C rows do not match recomputed TRAIN cache")
    data = {space: take(full_train, joined[space][0] if space != "C" else c_index)
            for space in ("A", "B1", "B2", "C")}
    for space in ("val", "test"):
        raw = collect(model, datasets[space], args, device,
                      cli.val_stride if space == "val" else cli.test_stride)
        data[space] = flat_record(prepare(raw, floor))
    # A0's direct head must reproduce the cached history-normalized forecast.
    weight = model.h_forecast_head.weight.detach().cpu().numpy().astype(np.float32)
    bias = model.h_forecast_head.bias.detach().cpu().numpy().astype(np.float32)
    predicted = (data["A"]["design"][:128] @ weight.T + bias) * data["A"]["ratio"][:128]
    error = float(np.max(np.abs(predicted - data["A"]["a0"][:128])))
    if error > 1e-3:
        raise AssertionError(f"F03=A0 direct head initialization mismatch: {error}")
    np.savez_compressed(cache, **{f"{space}_{key}": value for space in SPACES
                                  for key, value in data[space].items()})
    torch.save((weight,bias),case_dir/"a0_head.pt")
    metadata = {"a0_sha256": a0_sha, "dataset": args.dataset_name, "backbone": args.backbone,
                "horizon": args.pred_len, "seed": cli.seed, "span": stage["span"],
                "stage1_case": str(cli.stage1_case), "a0_init_error_max": error,
                "train_stride": cli.train_stride, "val_stride": cli.val_stride,
                "test_stride": cli.test_stride, "A0_training": "frozen, reused",
                "f03_fit": "A only, B1 epoch selection; C and later VAL/TEST never fit F03"}
    write_json(manifest, metadata)
    return data, metadata, (weight, bias)


def fit_f03(data, case_dir, a0_weights, cli):
    cache = case_dir / "f03_predictions.npz"
    if cache.exists():
        with np.load(cache) as contents:
            return {space: contents[space] for space in SPACES}
    if a0_weights is None:
        raise RuntimeError("F03 missing and A0 weights not loaded")
    weight, bias = a0_weights
    torch.manual_seed(cli.seed)
    device = torch.device(f"cuda:{cli.gpu}" if torch.cuda.is_available() else "cpu")
    head = torch.nn.Linear(weight.shape[1], weight.shape[0]).to(device)
    with torch.no_grad():
        head.weight.copy_(torch.from_numpy(weight).to(device))
        head.bias.copy_(torch.from_numpy(bias).to(device))
    optimizer = torch.optim.Adam(head.parameters(), lr=cli.f03_lr)
    x = torch.from_numpy(data["A"]["design"]).to(device)
    y = torch.from_numpy(data["A"]["y"]).to(device)
    ratio = torch.from_numpy(data["A"]["ratio"]).to(device)
    val_x = torch.from_numpy(data["B1"]["design"]).to(device)
    val_y = torch.from_numpy(data["B1"]["y"]).to(device)
    val_ratio = torch.from_numpy(data["B1"]["ratio"]).to(device)
    best = float("inf"); trace = []
    selected = {name: tensor.detach().cpu().clone() for name, tensor in head.state_dict().items()}
    for epoch in range(cli.f03_epochs + 1):
        head.eval()
        with torch.no_grad():
            val_mse = float(((head(val_x) * val_ratio - val_y) ** 2).mean())
        trace.append({"epoch": epoch, "B1_mse": val_mse})
        if val_mse < best:
            best = val_mse
            selected = {name: tensor.detach().cpu().clone() for name, tensor in head.state_dict().items()}
            chosen_epoch = epoch
        if epoch == cli.f03_epochs: break
        head.train()
        for indices in torch.randperm(len(x), device=device).split(256):
            loss = ((head(x[indices]) * ratio[indices] - y[indices]) ** 2).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    head.load_state_dict(selected); head.eval(); head.requires_grad_(False)
    preds = {}
    with torch.no_grad():
        for space in SPACES:
            chunks = []
            for start in range(0, len(data[space]["design"]), 512):
                stop = start + 512
                design = torch.from_numpy(data[space]["design"][start:stop]).to(device)
                scale = torch.from_numpy(data[space]["ratio"][start:stop]).to(device)
                chunks.append((head(design) * scale).cpu().numpy())
            preds[space] = np.concatenate(chunks).astype(np.float32)
    torch.save(selected, case_dir/"f03_head.pt")
    np.savez_compressed(cache, **preds)
    write_json(case_dir/"f03_selection.json", {"epoch": chosen_epoch, "B1_mse": best,
                "trace": trace, "checkpoint": str(case_dir/"f03_head.pt"),
                "note": "A0-initialized direct full-H head; A fit, B1 selected, then frozen for all group methods"})
    return preds


def torch_fuse(anchor, candidate, q, reliability):
    base = torch.from_numpy(anchor[:, None, :])
    cand = torch.from_numpy(candidate[:, None, :, :])
    weights = torch.from_numpy(q[:, None, :].astype(np.float32))
    rho = torch.from_numpy(reliability.astype(np.float32))
    empty = base.new_zeros(base.shape[-1], 0)
    full, beta = variation_forecast(base, cand, weights, rho, empty, True, 8)
    return full[:, 0].numpy(), beta[:, 0].numpy()


def point(cli, case_dir, data, meta, f03):
    method = METHODS[cli.method]
    output = case_dir / "methods" / method
    output.mkdir(parents=True, exist_ok=True)
    if (output/"point_complete").exists():
        return json.loads((output/"point_metrics.json").read_text())
    group = FrozenGrouping(Path(cli.stage1_case)/"methods"/method,
                           a0_sha=meta["a0_sha256"], dataset=meta["dataset"],
                           horizon=meta["horizon"], seed=cli.seed)
    # The join is checked per artifact. Same A/B1/B2 rows across all methods.
    with np.load(case_dir/"features.npz") as source:
        train_start=source["A_start"]
        train_channel=source["A_channel"]
    group_positions = [group.lookup[f'{meta["dataset"]}:TRAIN:{int(s)}:{int(c)}']
                       for s,c in zip(train_start,train_channel)]
    q_a = group.q[group_positions]
    router, route_time, parameters = train_router(data["A"]["key"], q_a,
                                                    seed=cli.seed, epochs=40)
    q = {space: route(router, data[space]["key"]) for space in SPACES}
    teacher_b1 = np.array([group.q[group.lookup[f'{meta["dataset"]}:TRAIN:{int(s)}:{int(c)}']]
                           for s,c in zip(data["B1"]["start"],data["B1"]["channel"])])
    router_quality = teacher_quality(teacher_b1, q["B1"])
    experts = fit_full_future_experts(data["A"]["key"], data["A"]["y"], q_a,
                                      alpha=cli.ridge_alpha, rank=8)
    candidate = {space: candidates(experts, data[space]["key"]) for space in SPACES}
    early = data["B1"]
    rho, evidence = prototype_segment_reliability(
        torch.from_numpy(f03["B1"][:, None]), torch.from_numpy(candidate["B1"][:, None]),
        torch.from_numpy(early["y"][:, None]), torch.from_numpy(q["B1"][:, None]), 8)
    rho = rho.numpy()
    support_a=independent_support(data["A"]["start"], q_a, meta["span"])
    support_b=independent_support(data["B1"]["start"], q["B1"], meta["span"])
    # Neither window count nor channel count substitutes for independent events.
    rho[(support_a < 2) | (support_b < 2)] = 0
    choice=[]
    for scale in (0., .25, .5, .75, 1.):
        trial,_=torch_fuse(f03["B2"],candidate["B2"],q["B2"],rho*scale)
        choice.append((score(trial,data["B2"]["y"])["mse"],scale))
    _, selected_scale=min(choice)
    rho *= selected_scale
    metrics={"case": f'{meta["backbone"]}/{meta["dataset"]}-{meta["horizon"]}',
             "method": method,"seed":cli.seed,"source_group_artifact":str(group.path),
             "source_group_sha256":group.hash,"a0_sha256":meta["a0_sha256"],
             "f03_head_sha256":sha256(case_dir/"f03_head.pt"),
             "router_B1":router_quality,"router_seconds":route_time,
             "router_parameters":parameters,"support_A":support_a.tolist(),
             "support_B1":support_b.tolist(),"reliability":rho.tolist(),
             "selected_null_scale_B2":selected_scale,"point_selection":"B1 reliability; B2 NULL scale; C held out from added maps but seen by A0; TEST report-only"}
    point_cache={}
    for space in SPACES:
        prediction,beta=torch_fuse(f03[space],candidate[space],q[space],rho)
        point_cache[space]=prediction
        metrics[space]={"a0":score(data[space]["a0"],data[space]["y"]),
                        "f03":score(f03[space],data[space]["y"]),
                        "f03_r3":score(prediction,data[space]["y"]),
                        "null_only":score(f03[space],data[space]["y"]),
                        "mean_beta":float(beta.mean()),
                        "active_ratio":float(np.mean(beta.max(axis=1)>1e-8))}
        if space=="test":
            hard=np.eye(K,dtype=np.float32)[q[space].argmax(1)]
            hard_pred,_=torch_fuse(f03[space],candidate[space],hard,rho)
            metrics[space]["hard_route"]=score(hard_pred,data[space]["y"])
    # Store compact normalized residuals only from chronologically held-out C.
    np.savez_compressed(output/"point_cache.npz", **{f"{space}_pred":point_cache[space] for space in SPACES},
                        **{f"{space}_q":q[space] for space in SPACES},
                        **{f"{space}_candidate":candidate[space] for space in ("C","val","test")})
    torch.save({"router":router.state_dict(),"experts":experts,"reliability":rho},output/"model.pt")
    write_json(output/"reliability_evidence.json",evidence)
    write_json(output/"point_metrics.json",metrics)
    (output/"point_complete").write_text("complete\n")
    return metrics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--base_run_config",required=True)
    parser.add_argument("--stage1_case",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--method",choices=METHODS,required=True)
    parser.add_argument("--gpu",type=int,default=0)
    parser.add_argument("--seed",type=int,default=2022)
    parser.add_argument("--train_stride",type=int,default=8)
    parser.add_argument("--val_stride",type=int,default=8)
    parser.add_argument("--test_stride",type=int,default=1)
    parser.add_argument("--f03_epochs",type=int,default=10)
    parser.add_argument("--f03_lr",type=float,default=3e-4)
    parser.add_argument("--ridge_alpha",type=float,default=100.)
    parser.add_argument("--allow_legacy_reference_signature",action="store_true")
    cli=parser.parse_args()
    torch.set_num_threads(4)
    case_dir=Path(cli.output)
    case_dir.mkdir(parents=True,exist_ok=True)
    data,meta,a0_weights=prepare_shared(cli,case_dir)
    f03=fit_f03(data,case_dir,a0_weights,cli)
    result=point(cli,case_dir,data,meta,f03)
    print(json.dumps({"case":result["case"],"method":result["method"],
                      "C":result["C"],"TEST":result["test"]}),flush=True)


if __name__=="__main__":main()
