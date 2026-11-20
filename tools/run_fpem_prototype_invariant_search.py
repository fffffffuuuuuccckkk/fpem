"""R3-shared prototype pilot: direct full-future TRAIN supervision only.

Stages are independently restartable. TEST is read only for exploratory
ranking after VAL checkpoint selection; R3 router and shared heads never see
TEST labels during fitting or reliability calibration.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.predictive_env.dual_pattern_memory.prototype_bank import normalized_pairs
from models.predictive_env.dual_pattern_memory.rare_learning import RarePatternLearner
from models.predictive_env.direct_invariant_forecaster import direct_ridge
from models.predictive_env.prototype_guided_invariant import (
    PrototypeGuidedSharedHead, prototype_h_centers, routing_control)
from models.predictive_env.prototype_invariance_losses import (
    prototype_risk_penalty, shared_optimum_penalty, conditional_agreement,
    prototype_risks)
from models.predictive_env.prototype_mapping_consensus import (
    fit_prototype_mappings, predict_mapping)
from models.predictive_env.prototype_shared_subspace import (
    PrototypeAdversarialHead, PrototypeSharedPrivateHead,
    pairwise_prototype_error_loss)
from models.predictive_env.eiil_environment import PredictiveConflictEnvironment
from tools.run_fpem_dppm_g1_r3_full import history_key, residual_ridge
from tools.run_fpem_direct_invariant_search import cluster_groups
from tools.run_fpem_shared_factorization_g1r3 import load_experiment
from tools.run_predictive_env_iv_patchtst import IndexedDataset

METHODS = (
    "A0", "linear", "head_init", "p01", "p02", "risk_var", "risk_dro",
    "risk_incremental", "shared_grad_norm", "shared_grad_variance",
    "shared_irm", "conditional", "subspace", "ref_global", "ref_prototype",
    "residual_risk", "risk_var_shuffled", "risk_var_random",
    "risk_var_time", "risk_var_uniform", "risk_var_sample",
    "risk_var_hcluster", "risk_var_eiil",
    "map_global", "map_consensus", "map_spectral",
    "pairwise", "shared_private", "shared_private_irm",
    "adversarial_weak", "adversarial_strong",
    "map_projection", "map_shared_irm",
    "map_consensus_shuffled", "map_consensus_random",
    "map_consensus_time", "map_consensus_uniform",
    "map_consensus_hcluster", "map_consensus_eiil",
)


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--base_run_config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--train_stride", type=int, default=16)
    p.add_argument("--val_stride", type=int, default=4)
    p.add_argument("--test_stride", type=int, default=1)
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--allow_legacy_reference_signature", action="store_true")
    return p.parse_args()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


@torch.no_grad()
def collect(model, dataset, args, device, stride):
    wrapped = IndexedDataset(dataset, cycle_len=args.cyclenet_cycle_len)
    loader = DataLoader(Subset(wrapped, range(0, len(dataset), stride)),
                        batch_size=args.batch_size, shuffle=False, num_workers=0)
    parts = {k: [] for k in ("x", "hidden", "mean", "std", "truth", "base", "ids")}
    model.eval()
    for x, y, ids, cycle in loader:
        x, cycle = x.float().to(device), cycle.to(device)
        h, mean, std = model.encode(x, cycle)
        base = model.frozen_h_head(h, cycle).transpose(1, 2) * std + mean
        values = {"x": x, "hidden": h, "mean": mean, "std": std,
                  "truth": y[:, -args.pred_len:], "base": base, "ids": ids}
        for k, value in values.items():
            parts[k].append(value.detach().cpu())
    return {k: torch.cat(v) for k, v in parts.items()}


def prepare(raw, floor):
    y, hist_mean, hist_scale = normalized_pairs(raw["x"], raw["truth"], floor)
    raw.update({"key": history_key(raw["x"], 64).cpu(), "r3_truth": y.cpu(),
                "hist_mean": hist_mean.cpu(), "hist_scale": hist_scale.cpu()})
    raw["target_normalized"] = ((raw["truth"]-raw["mean"])/raw["std"]).transpose(1, 2)
    raw["base_normalized"] = ((raw["base"]-raw["mean"])/raw["std"]).transpose(1, 2)
    return raw


def mse_mae(pred, truth):
    diff = pred - truth
    return {"mse": float(diff.square().mean()), "mae": float(diff.abs().mean()),
            "prediction_std": float(pred.std(unbiased=False))}


def normalized_to_raw(norm, data):
    return norm.transpose(1, 2) * data["std"] + data["mean"]


@torch.no_grad()
def predict_head(net, data, device, batch=64):
    net.eval()
    out = []
    for start in range(0, len(data["hidden"]), batch):
        stop = start + batch
        pred = net(data["hidden"][start:stop].float().to(device),
                   data["q"][start:stop].float().to(device),
                   data["base_normalized"][start:stop].float().to(device))
        out.append(pred.detach().cpu())
    return normalized_to_raw(torch.cat(out), data)


def fit_r3(train, val, args, seed, output):
    saved = output / "r3.pt"
    if saved.exists():
        return RarePatternLearner.from_export(torch.load(saved, map_location="cpu", weights_only=False))
    reference = residual_ridge(train["key"], train["r3_truth"], 100.)
    config = {"head_lr": {"patchtst": 1e-4, "itransformer": 3e-3,
                          "moderntcn": 3e-3}[args.backbone],
              "dictionary_ratio": .1 if args.backbone == "patchtst" else 1.,
              "basis_ratio": .1, "route_ratio": .1 if args.backbone == "patchtst" else 1.,
              "optimizer": "adam", "schedule": "constant", "restore_best": True}
    learner = RarePatternLearner("R3", patterns=8, rank=8, seed=seed,
                                 epochs=10, optimizer_config=config)
    learner.fit(train["key"], train["r3_truth"],
                {"reference_map": reference, "starts": train["ids"],
                 "span": args.seq_len + args.pred_len, "device": "cpu",
                 "validation": (val["key"], val["r3_truth"])})
    torch.save(learner.export(), saved)
    return learner


def stage_cache(cli, output):
    args, model, loaders, device, sha = load_experiment(cli)
    datasets = {name: loader.dataset.dataset for name, loader in loaders.items()}
    raw = {name: collect(model, datasets[name], args, device,
                         {"train": cli.train_stride, "val": cli.val_stride,
                          "test": cli.test_stride}[name])
           for name in ("train", "val", "test")}
    floor = raw["train"]["x"].float().var(1, unbiased=False).add(1e-5).sqrt().median(0).values
    floor = (floor * .25).clamp_min(.01)
    data = {k: prepare(v, floor) for k, v in raw.items()}
    learner = fit_r3(data["train"], data["val"], args, cli.seed, output)
    for split in data.values():
        r3 = learner.query(split["key"])
        split.update({"q": r3["weight"].detach(),
                      "null": r3["null_weight"].detach(),
                      "candidate": r3["candidate"].detach()})
    # Two alternative proxy-group controls are computed from TRAIN only.
    train=data["train"]
    labels=cluster_groups(train,count=train["q"].shape[-1],seed=cli.seed)
    train["hcluster_q"]=torch.nn.functional.one_hot(labels,train["q"].shape[-1]).float()[:,None].expand_as(train["q"])*train["q"].sum(-1,keepdim=True)
    if any(name.endswith("eiil") for name in getattr(cli,"methods","").split(",")):
        eiil=PredictiveConflictEnvironment(len(train["hidden"]),
             env_num=train["q"].shape[-1],steps=20,seed=cli.seed,
             quality_diagnostics=False,random_partition_repeats=1)
        eiil.update(train["base"],train["truth"],torch.arange(len(train["hidden"])),
                    source_split="train")
        train["eiil_q"]=eiil.q[:,None].expand_as(train["q"])*train["q"].sum(-1,keepdim=True)
    global_center, centers, support = prototype_h_centers(data["train"]["hidden"],
                                                           data["train"]["q"])
    write_json(output / "prototype_diagnostics.json",
               {"mass": support["mass"].tolist(), "effective": support["effective"].tolist(),
                "null_train": float(data["train"]["null"].mean()),
                "null_val": float(data["val"]["null"].mean()),
                "null_test": float(data["test"]["null"].mean()),
                "router_frozen_during_shared_training": True,
                "q_source": "same R3 train-only router for constraints and inference"})
    return args, model, data, device, sha, learner, global_center, centers


def method_spec(name):
    rank = 0 if name in ("linear", "head_init") else 16
    reference = "global" if name == "ref_global" else "prototype" if name == "ref_prototype" else "none"
    pretrain = name == "head_init"
    residual = name == "residual_risk"
    return rank, reference, pretrain, residual


def build_head(name, tokens, width, horizon, prototypes=8, pretrained_head=None):
    rank, reference, pretrain, residual = method_spec(name)
    if name.startswith("adversarial_"):
        return PrototypeAdversarialHead(tokens, width, horizon, rank=rank,
                                        prototypes=prototypes)
    if name.startswith("shared_private"):
        return PrototypeSharedPrivateHead(tokens, width, horizon, rank=rank,
                                          prototypes=prototypes)
    return PrototypeGuidedSharedHead(tokens, width, horizon, rank=rank,
        reference_mode=reference, residual_form=residual,
        pretrained_head=pretrained_head if pretrain else None,
        prototypes=prototypes)


def train_method(name, train, val, test, model, args, device, cli, centers):
    rank, reference, pretrain, residual = method_spec(name)
    p, d = train["hidden"].shape[-2:]
    head = build_head(name,p,d,args.pred_len,prototypes=train["q"].shape[-1],
                      pretrained_head=model.h_forecast_head if pretrain else None).to(device)
    head.set_references(centers[0].to(device), centers[1].to(device))
    if pretrain:
        initial = predict_head(head, val, device)
        if not torch.allclose(initial, val["base"], atol=1e-5, rtol=1e-5):
            raise AssertionError("A0 function-preserving initialization failed")
    optimizer = torch.optim.Adam(head.parameters(), lr=3e-4 if pretrain else 1e-3)
    route_mode = (name.split("risk_var_")[1] if name.startswith("risk_var_")
                  and name != "risk_var_sample" else "real")
    sample_level = name == "risk_var_sample"
    if route_mode in ("hcluster","eiil"):
        q_train=train[route_mode+"_q"]
    else:
        q_train = routing_control(train["q"], route_mode, seed=cli.seed,
                                  history_ids=train["ids"])
    generator = torch.Generator().manual_seed(cli.seed)
    map_teacher = None
    if name in ("map_projection", "map_shared_irm"):
        # TRAIN-only prototype mapping. It is fixed during head optimization;
        # no VAL/TEST target enters the teacher or its projection basis.
        hidden_train = train["hidden"].float()
        flat = hidden_train.reshape(-1, hidden_train.shape[-1])
        _, _, right = torch.linalg.svd(flat[:min(2000, len(flat))], full_matrices=False)
        map_basis = right[:min(16, hidden_train.shape[-1])].T.contiguous()
        fitted = fit_prototype_mappings(hidden_train @ map_basis,
                                        train["target_normalized"], train["q"], seed=cli.seed)
        map_teacher = predict_mapping(hidden_train @ map_basis, fitted, "consensus").detach()
    best, history = None, []
    started = time.monotonic()
    for epoch in range(cli.epochs):
        head.train()
        loss_total = 0.
        for ids in torch.randperm(len(train["hidden"]), generator=generator).split(args.batch_size):
            h = train["hidden"][ids].float().to(device)
            q = q_train[ids].float().to(device)
            target = train["target_normalized"][ids].float().to(device)
            base = train["base_normalized"][ids].float().to(device)
            prediction = head(h, q, base)
            forecast = (prediction-target).square().mean()
            penalty = forecast.new_zeros(())
            if name.startswith("risk_var"):
                penalty, _ = prototype_risk_penalty(prediction, target, q,
                    "variance", sample_level=sample_level)
            elif name == "risk_dro":
                penalty, _ = prototype_risk_penalty(prediction, target, q, "dro")
            elif name == "risk_incremental":
                penalty, _ = prototype_risk_penalty(prediction, target, q,
                                                     "incremental", base=base)
            elif name in ("shared_grad_norm", "shared_grad_variance", "shared_irm"):
                variant = {"shared_grad_norm": "norm", "shared_grad_variance": "variance",
                           "shared_irm": "irm"}[name]
                penalty, _ = shared_optimum_penalty(prediction, target, q,
                                                     head.head_weight, variant)
            elif name == "conditional":
                penalty = conditional_agreement(prediction, target, q, h.mean(2))
            elif name == "subspace":
                z = head.forecaster.projection(h)
                q_mass = q.sum((0, 1)).clamp_min(1e-8)
                by_group = torch.einsum("bcm,bcpr->mr", q, z) / (q_mass[:, None]*z.shape[2])
                center = (by_group * q_mass[:, None]).sum(0) / q_mass.sum()
                penalty = ((by_group-center).square().mean(-1)*q_mass/q_mass.sum()).sum()
            elif name == "pairwise":
                penalty = pairwise_prototype_error_loss(prediction, target, q,
                             train["ids"][ids].to(device), args.seq_len + args.pred_len)
            elif name.startswith("shared_private"):
                penalty = head.split_penalty(h)
                if name == "shared_private_irm":
                    extra, _ = shared_optimum_penalty(prediction, target, q,
                                                       head.head_weight, "irm")
                    penalty = penalty + extra
            elif name.startswith("adversarial_"):
                strength = 0.1 if name.endswith("weak") else 1.0
                logits = head.adversarial_logits(h, strength)
                normalized_q = q / q.sum(-1, keepdim=True).clamp_min(1e-8)
                has_route = q.sum(-1) > 1e-8
                ce = -(normalized_q * logits.log_softmax(-1)).sum(-1)
                penalty = ce[has_route].mean() if has_route.any() else forecast.new_zeros(())
            elif name in ("map_projection", "map_shared_irm"):
                teacher = map_teacher[ids].float().to(device)
                penalty = (prediction-teacher).square().mean()
                if name == "map_shared_irm":
                    extra, _ = shared_optimum_penalty(prediction, target, q,
                                                       head.head_weight, "irm")
                    penalty = penalty + extra
            if name == "p02":
                penalty = head.forecaster.orthogonality_loss()
            loss = forecast + 0.01 * penalty
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite {name} objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
            optimizer.step()
            loss_total += float(loss.detach())
        val_metric = mse_mae(predict_head(head, val, device), val["truth"])
        history.append({"epoch": epoch+1, "train_loss": loss_total, **val_metric})
        if best is None or val_metric["mse"] < best["val"]["mse"]:
            best = {"epoch": epoch+1, "val": val_metric,
                    "state": {k: v.detach().cpu().clone() for k,v in head.state_dict().items()}}
        print(name, "epoch", epoch+1, "VAL", round(val_metric["mse"], 6), flush=True)
    head.load_state_dict(best.pop("state"))
    val_prediction=predict_head(head,val,device)
    test_prediction=predict_head(head,test,device)
    val_norm=((val_prediction-val["mean"])/val["std"]).transpose(1,2)
    test_norm=((test_prediction-test["mean"])/test["std"]).transpose(1,2)
    val_group,_,_=prototype_risks(val_norm,val["target_normalized"],val["q"])
    test_group,_,_=prototype_risks(test_norm,test["target_normalized"],test["q"])
    result = {"method": name, "val": best["val"],
              "test": mse_mae(test_prediction, test["truth"]),
              "val_group_risks":val_group.tolist(),
              "test_group_risks":test_group.tolist(),
              "best_epoch": best["epoch"], "trainable_parameters": sum(p.numel() for p in head.parameters()),
              "rank": rank, "reference_mode": reference, "route_control": route_mode,
              "sample_level_risk": sample_level, "elapsed_seconds": time.monotonic()-started,
              "history": history, "objective": "complete-future normalized MSE"}
    return result, head.state_dict()


def mapping_methods(train, val, test, method, seed):
    # In the same compressed H coordinates for every prototype map. The
    # deterministic orthogonal projection is fitted from TRAIN H only.
    h = train["hidden"].float()
    width = min(16, h.shape[-1])
    flat = h.reshape(-1, h.shape[-1])
    _, _, right = torch.linalg.svd(flat[:min(2000, len(flat))], full_matrices=False)
    basis = right[:width].T.contiguous()
    project = lambda data: data["hidden"].float() @ basis
    if method.startswith("map_consensus_"):
        control=method[len("map_consensus_"):]
        if control in ("eiil", "hcluster"):
            fit_q=train[control+"_q"]
        else:
            fit_q=routing_control(train["q"],control,seed=seed,history_ids=train["ids"])
    else:
        fit_q=train["q"]
    fitted = fit_prototype_mappings(project(train), train["target_normalized"],
                                    fit_q, seed=seed)
    kind = "global" if method=="map_global" else "spectral" if method=="map_spectral" else "consensus"
    selected = None
    for shrink in ((0., .25, .5, 1.) if kind == "spectral" else (0.,)):
        val_norm = predict_mapping(project(val), fitted, kind, shrink)
        val_pred = normalized_to_raw(val_norm, val)
        score = mse_mae(val_pred, val["truth"])
        if selected is None or score["mse"] < selected["val"]["mse"]:
            test_norm = predict_mapping(project(test), fitted, kind, shrink)
            selected = {"method": method, "shrink": shrink, "val": score,
                        "test": mse_mae(normalized_to_raw(test_norm, test), test["truth"]),
                        "fit_rows": fitted["fit_rows"], "spectral": fitted["spectral"],
                        "rank": width}
    return selected, {"basis":basis,"fitted":fitted}


def main():
    cli = args_parser()
    out = Path(cli.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(cli.seed)
    args, model, data, device, sha, learner, global_center, centers = stage_cache(cli, out)
    names = cli.methods.split(",")
    for name in names:
        if name not in METHODS:
            raise ValueError(name)
        path = out / name
        if (path / "complete").exists():
            print("SKIP", name, flush=True)
            continue
        path.mkdir(exist_ok=True)
        print("BEGIN", name, flush=True)
        started = time.monotonic()
        if name == "A0":
            result = {"method": name,
                      "val": mse_mae(data["val"]["base"], data["val"]["truth"]),
                      "test": mse_mae(data["test"]["base"], data["test"]["truth"]),
                      "trainable_parameters": 0}
            state = None
        elif name.startswith("map_") and name not in ("map_projection", "map_shared_irm"):
            result, state = mapping_methods(data["train"],data["val"],data["test"],
                                            name,cli.seed)
        else:
            result,state = train_method(name,data["train"],data["val"],data["test"],
                                        model,args,device,cli,(global_center,centers))
        result.update({"backbone": args.backbone, "dataset": args.dataset_name,
                       "pred_len": args.pred_len, "seed": cli.seed,
                       "reference_sha256": sha, "elapsed_total_seconds": time.monotonic()-started,
                       "test_selection_caveat": "TEST ranking is exploratory development only"})
        if state is not None:
            torch.save(state,path/"model.pt")
        write_json(path/"metrics.json",result)
        (path/"complete").write_text("complete\n",encoding="utf-8")
    rows=[]
    for name in names:
        path=out/name/"metrics.json"
        if path.exists():
            row=json.loads(path.read_text())
            rows.append({"method":name,"val_mse":row["val"]["mse"],
                         "val_mae":row["val"]["mae"],"test_mse":row["test"]["mse"],
                         "test_mae":row["test"]["mae"],"reference_sha256":sha})
    with (out/"shared_forecast_results.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]) if rows else ["method"])
        writer.writeheader();writer.writerows(rows)
    write_json(out/"run_config.json",{"cli":vars(cli),"reference_sha256":sha,
        "r3_state":str(out/"r3.pt"),"train_rows":len(data["train"]["hidden"]),
        "val_rows":len(data["val"]["hidden"]),"test_rows":len(data["test"]["hidden"]),
        "route_shape":list(data["train"]["q"].shape),
        "selection":"VAL checkpoint and regularization; exploratory TEST method ranking"})
    print("COMPLETE",out,flush=True)


if __name__=="__main__":
    main()
