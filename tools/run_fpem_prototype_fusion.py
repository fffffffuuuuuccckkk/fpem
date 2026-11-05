"""Staged prototype x fusion pilot using frozen, reusable A0 checkpoints.

All response keys/values are TRAIN-only. TRAIN queries use purged fold-local
key metrics, values and EIIL labels. Epoch/checkpoint selection uses VAL.
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
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.predictive_env.eiil_environment import PredictiveConflictEnvironment
from models.predictive_env.prototype_fusion import PrototypeFusion, PrototypeFusionMemory
from tools.run_fpem_predictive_memory import (
    anchor_records, canonical_environment, collect_predictions_and_hidden, env_labels)
from tools.run_fpem_shared_factorization import (
    load_experiment, ordered_training_loader, prediction, train_loader, write_json)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--base_run_config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--prototype", choices=("P0", "P2", "P3"), required=True)
    p.add_argument("--fusion", choices=("phaseA", "F1", "F2", "F3"), required=True)
    p.add_argument("--mapping", choices=("M0", "M1", "M2"), default="M1")
    p.add_argument("--environment_mode", choices=("predictive", "shuffled", "uniform"), default="predictive")
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--history_size", type=int, default=64)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--temperature", type=float, default=.2)
    p.add_argument("--memory_folds", type=int, default=3)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--lr_fusion", type=float, default=1e-4)
    p.add_argument("--lr_qhat", type=float, default=1e-4)
    p.add_argument("--lambda_qhat", type=float, default=.1)
    p.add_argument("--phase0_cache", default="")
    p.add_argument("--allow_legacy_reference_signature", action="store_true")
    p.add_argument("--max_batches", type=int, default=0)
    return p.parse_args()


def make_bank(cli, args, model, records, q, source_mask=None, forbidden=None, seed_offset=0):
    bank = PrototypeFusionMemory(cli.k, cli.history_size, args.pred_len, model.d_model,
                                 args.env_num, cli.prototype, cli.temperature, cli.topk)
    allowed = torch.ones(len(records["starts"]), dtype=torch.bool) if source_mask is None else source_mask.clone()
    if forbidden is not None:
        lo, hi = forbidden
        starts = records["starts"].long()
        allowed &= (starts + args.seq_len + args.pred_len <= lo) | (starts >= hi)
    bank.fit_key_space(records["x"][allowed], records["response"][allowed], cli.seed+seed_offset)
    report = bank.fit_values(records["x"], records["hidden"], records["response"],
                             q, records["starts"], source_mask=allowed,
                             forbidden_interval=forbidden)
    report["key_supervision"] = "allowed TRAIN anchors only" if cli.prototype == "P2" else "historical TRAIN anchors only"
    return bank, report


def memories_for_pilot(cli, args, model, records, q, device):
    full, full_report = make_bank(cli, args, model, records, q[records["starts"]])
    ntrain = len(q)
    folds, reports = [], []
    starts = records["starts"].long()
    for f in range(cli.memory_folds):
        lo = f*ntrain//cli.memory_folds
        hi = ((f+1)*ntrain-1)//cli.memory_folds + args.seq_len + args.pred_len
        source = (starts + args.seq_len + args.pred_len <= lo) | (starts >= hi)
        selected = torch.where(source)[0]
        if len(selected) < cli.k:
            raise RuntimeError("purged fold has insufficient independent TRAIN anchors")
        manager = PredictiveConflictEnvironment(
            len(selected), env_num=args.env_num, steps=args.eiil_steps,
            lr=args.eiil_lr, balance_weight=args.balance_weight,
            entropy_weight=args.entropy_weight, seed=cli.seed+f+113,
            diagnostic_samples=min(128, len(selected)), random_partition_repeats=10)
        fold_q, stage = manager.update(records["a0_pred"][selected],
                                        records["target"][selected],
                                        torch.arange(len(selected)), source_split="train")
        fold_q, _ = canonical_environment(fold_q, stage)
        fold_q = env_labels(fold_q, cli.environment_mode, cli.seed+f+113)
        anchor_q = torch.full((len(starts), args.env_num), 1/args.env_num)
        anchor_q[selected] = fold_q
        bank, report = make_bank(cli, args, model, records, anchor_q, source,
                                 (lo, hi), seed_offset=f+113)
        assert report["source_anchors"] == len(selected)
        report["environment_source"] = "fold-local EIIL on source TRAIN anchors"
        folds.append(bank.to(device)); reports.append(report)
    return {"full": full.to(device), "folds": folds, "train_count": ntrain}, {
        "full": full_report, "folds": reports}


def forward(model, fusion, memories, x, cycle, sample_ids, args, cli):
    hidden, mean, std = model.encode(x, cycle)
    baseline = prediction(model, hidden, mean, std, cycle, "reference")
    if sample_ids is None:
        selections = [(torch.arange(len(x), device=x.device), memories["full"])]
    else:
        fold = (sample_ids.to(x.device)*cli.memory_folds//memories["train_count"]).clamp_max(cli.memory_folds-1)
        selections = [(torch.where(fold == f)[0], memories["folds"][f]) for f in range(cli.memory_folds)]
    output = torch.zeros_like(baseline)
    diagnostics = {"correction_rms": baseline.new_zeros(()),
                   "retrieval_entropy": baseline.new_zeros(()),
                   "retrieval_distance": baseline.new_zeros(())}
    qhat = fusion.predict_environment(hidden)
    for indices, bank in selections:
        if not len(indices):
            continue
        query = bank.query(x[indices], qhat[indices] if cli.mapping == "M2" else None,
                           use_mapping=cli.mapping == "M2")
        forecast, info = fusion(model, hidden[indices], mean[indices], std[indices],
                                cycle[indices], query)
        output = output.index_copy(0, indices, forecast)
        for name in diagnostics:
            diagnostics[name] = diagnostics[name] + len(indices)/len(x)*info[name]
    return output, baseline, qhat, diagnostics


@torch.no_grad()
def evaluate(model, fusion, memories, loader, args, cli, split):
    fusion.eval(); model.eval()
    mse = mae = a0_mse = a0_mae = count = 0.
    diagnostic = {key: 0. for key in ("correction_rms", "retrieval_entropy", "retrieval_distance")}
    for batch_index, (x, y, ids, cycle) in enumerate(loader):
        if cli.max_batches and batch_index >= cli.max_batches:
            break
        x, cycle = x.float().to(next(model.parameters()).device), cycle.to(next(model.parameters()).device)
        y = y[:, -args.pred_len:].float().to(x.device)
        output, a0, _, diag = forward(model, fusion, memories, x, cycle,
                                      ids if split == "train" else None, args, cli)
        mse += float((output-y).square().sum()); mae += float((output-y).abs().sum())
        a0_mse += float((a0-y).square().sum()); a0_mae += float((a0-y).abs().sum())
        count += y.numel()
        for key in diagnostic:
            diagnostic[key] += float(diag[key])*len(x)
    nsamples = count/(args.pred_len*args.enc_in)
    return {"mse": mse/count, "mae": mae/count, "a0_mse": a0_mse/count,
            "a0_mae": a0_mae/count, **{k: v/nsamples for k,v in diagnostic.items()}}


@torch.no_grad()
def phase_a(records, memories, cli, args):
    """TRAIN-only purged historical retrieval and response usefulness."""
    sample = records["x"]
    starts = records["starts"]
    errors, zeros, weights, entropies = [], [], [], []
    for fold_id, bank in enumerate(memories["folds"]):
        low = fold_id*memories["train_count"]//cli.memory_folds
        high = (fold_id+1)*memories["train_count"]//cli.memory_folds
        selected = (starts >= low) & (starts < high)
        if not bool(selected.any()):
            continue
        query = bank.query(sample[selected].to(bank.keys.device), use_mapping=False)
        target = records["response"][selected].to(bank.keys.device)
        errors.append(float((query["stable_response"]-target).square().mean()))
        zeros.append(float(target.square().mean()))
        weights.append(float(query["weight"].max(-1).values.mean()))
        entropies.append(float(query["entropy"].mean()))
    return {"source_split": "TRAIN purged chronological cross-fit",
            "prototype": cli.prototype, "response_mse": float(np.mean(errors)),
            "zero_response_mse": float(np.mean(zeros)),
            "retrieval_max_weight": float(np.mean(weights)),
            "retrieval_entropy": float(np.mean(entropies)),
            "independent_support": memories["full"].support.cpu().tolist()}


def main():
    cli = parser()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    out = Path(cli.output); out.mkdir(parents=True, exist_ok=True)
    if (out/"run_complete").is_file():
        print("already complete", flush=True); return
    torch.manual_seed(cli.seed); np.random.seed(cli.seed)
    args, model, loaders, device, reference_hash = load_experiment(cli)
    cache_path = Path(cli.phase0_cache) if cli.phase0_cache else None
    cached = torch.load(cache_path, map_location="cpu", weights_only=False) if cache_path and cache_path.is_file() else None
    if cached is not None and cached.get("reference_sha256") != reference_hash:
        raise ValueError("A0 cache/checkpoint mismatch")
    if cached is None:
        records = anchor_records(model, loaders["train"].dataset.dataset, args, device)
        ordered = ordered_training_loader(loaders["train"].dataset.dataset, args)
        a0, target, ids, pooled = collect_predictions_and_hidden(model, ordered, args, device)
        manager = PredictiveConflictEnvironment(
            len(ordered.dataset), env_num=args.env_num, steps=args.eiil_steps,
            lr=args.eiil_lr, balance_weight=args.balance_weight,
            entropy_weight=args.entropy_weight, seed=cli.seed,
            diagnostic_samples=256, random_partition_repeats=10)
        discovered, stage = manager.update(a0, target, ids, source_split="train")
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"records": records, "discovered": discovered, "stage_record": stage,
                        "all_hidden": pooled, "reference_sha256": reference_hash}, cache_path)
    else:
        records, discovered, stage = cached["records"], cached["discovered"], cached["stage_record"]
    q, _ = canonical_environment(discovered, stage)
    q = env_labels(q, cli.environment_mode, cli.seed)
    memories, memory_report = memories_for_pilot(cli, args, model, records, q, device)
    diagnostic = phase_a(records, memories, cli, args)
    write_json(out/"prototype_diagnostics.json", diagnostic)
    write_json(out/"memory_report.json", memory_report)
    write_json(out/"run_config.json", {"cli": vars(cli), "reference_sha256": reference_hash,
               "checkpoint_selection": "VAL MSE", "query": "TRAIN purged; VAL/TEST full TRAIN",
               "backbone_and_head": "frozen A0", "prototype_key_supervision": memory_report["full"]["key_supervision"]})
    if cli.fusion == "phaseA":
        write_json(out/"result.json", {"diagnostics": diagnostic})
        (out/"run_complete").touch(); return
    fusion = PrototypeFusion(model.d_model, model.head_linear.in_features//model.d_model,
                             args.pred_len, cli.history_size, args.env_num,
                             cli.fusion, cli.mapping, cli.rank).to(device)
    train = train_loader(loaders["train"].dataset.dataset, args, cli.seed)
    groups = [{"params": [p for n,p in fusion.named_parameters() if not n.startswith("env_predictor")],
               "lr": cli.lr_fusion},
              {"params": list(fusion.env_predictor.parameters()), "lr": cli.lr_qhat}]
    optimizer = torch.optim.Adam(groups)
    history, best = [], math.inf
    for epoch in range(cli.epochs):
        model.eval(); fusion.train()
        total = 0.; count = 0
        for batch_index, (x,y,ids,cycle) in enumerate(train):
            if cli.max_batches and batch_index >= cli.max_batches:
                break
            x, cycle = x.float().to(device), cycle.to(device)
            y = y[:, -args.pred_len:].float().to(device)
            output, _, qhat, _ = forward(model, fusion, memories, x, cycle, ids, args, cli)
            primary = F.mse_loss(output, y)
            label = q[ids].to(device).detach()
            qhat_loss = -(label*qhat.clamp_min(1e-8).log()).sum(-1).mean()
            loss = primary + cli.lambda_qhat*qhat_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss epoch={epoch+1}")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion.parameters(), 5.)
            optimizer.step()
            total += float(primary.detach()); count += 1
        val = evaluate(model, fusion, memories, loaders["val"], args, cli, "val")
        row = {"epoch": epoch+1, "train_mse": total/max(1,count),
               "val_mse": val["mse"], "val_mae": val["mae"]}
        history.append(row); print(json.dumps(row), flush=True)
        write_json(out/"train_history.json", history)
        if val["mse"] < best:
            best = val["mse"]
            torch.save({"fusion": fusion.state_dict(), "memory_full": memories["full"].state_dict(),
                        "reference_sha256": reference_hash,
                        "best_epoch": epoch+1, "val": val}, out/"best.pt")
    checkpoint = torch.load(out/"best.pt", map_location=device, weights_only=False)
    fusion.load_state_dict(checkpoint["fusion"])
    test = evaluate(model, fusion, memories, loaders["test"], args, cli, "test")
    write_json(out/"result.json", {"best_epoch": checkpoint["best_epoch"],
               "val": checkpoint["val"], "test": test, "diagnostics": diagnostic,
               "reference_sha256": reference_hash, "selection": "VAL only; TEST exploratory"})
    (out/"run_complete").touch()


if __name__ == "__main__":
    main()
