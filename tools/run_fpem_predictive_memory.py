"""Three-backbone TRAIN-only predictive-memory point-forecast pilot.

The probability stage is deliberately absent: it is only justified after
point-prediction and retrieval diagnostics show a benefit.  Checkpoint and
epoch selection use VAL; TEST is evaluated once after selection.
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

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.predictive_env.classification_constraint import (
    EnvironmentClassificationConstraint, soft_cross_entropy)
from models.predictive_env.eiil_environment import PredictiveConflictEnvironment
from models.predictive_env.predictive_memory import PredictiveMemory, history_keys
from models.predictive_env.retrieval_guided_adapter import RetrievalGuidedAdapter
from tools.run_predictive_env_iv_patchtst import IndexedDataset
from tools.run_fpem_shared_factorization import (
    load_experiment, ordered_training_loader, prediction, train_loader, write_json)


def cli_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_run_config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=("phase0", "B0", "B1", "P1", "P2", "P3", "P4"), required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--stage_epochs", type=int, default=2)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--history_size", type=int, default=64)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--memory_folds", type=int, default=3)
    p.add_argument("--temperature", type=float, default=.2)
    p.add_argument("--lr_adapter", type=float, default=1e-4)
    p.add_argument("--lr_head", type=float, default=5e-5)
    p.add_argument("--lambda_inv", type=float, default=.2)
    p.add_argument("--lambda_env", type=float, default=.1)
    p.add_argument("--lambda_qhat", type=float, default=.1)
    p.add_argument("--environment_mode", choices=("predictive", "shuffled", "uniform"), default="predictive")
    p.add_argument("--mapping_shuffle", action="store_true")
    p.add_argument("--input_only", action="store_true")
    p.add_argument("--no_align", action="store_true")
    p.add_argument("--no_variant", action="store_true")
    p.add_argument("--full_residual", action="store_true")
    p.add_argument("--phase0_cache", default="", help="shared TRAIN-only A0/q/anchor cache")
    p.add_argument("--phase0_json", default="", help="reuse the already completed TRAIN diagnostic")
    p.add_argument("--allow_legacy_reference_signature", action="store_true")
    p.add_argument("--max_batches", type=int, default=0, help="smoke only; 0 means full")
    return p.parse_args()


@torch.no_grad()
def collect_predictions_and_hidden(model, loader, args, device, adapter=None,
                                   memories=None, fold_count=3):
    model.eval()
    if adapter is not None:
        adapter.eval()
    preds, ys, ids, pools = [], [], [], []
    for x, y, sample_id, cycle in loader:
        x, cycle = x.float().to(device), cycle.to(device)
        h, mean, std = model.encode(x, cycle)
        if adapter is None or memories is None:
            pred = prediction(model, h, mean, std, cycle, "reference")
        else:
            pred, _, _ = point_forward(model, adapter, memories, x, cycle,
                                        sample_id, args, "P4", fold_count)
        preds.append(pred.cpu())
        ys.append(y[:, -args.pred_len:].float().cpu())
        ids.append(sample_id.cpu())
        pools.append(h.mean((1, 2)).cpu())
    ids = torch.cat(ids)
    if not torch.equal(ids, torch.arange(len(ids))):
        raise RuntimeError("EIIL requires complete chronological TRAIN IDs")
    return torch.cat(preds), torch.cat(ys), ids, torch.cat(pools)


@torch.no_grad()
def anchor_records(model, dataset, args, device):
    # Reuse PTM's seq_len-spaced anchors; support later counts independent
    # complete-window temporal blocks, not the many stride-1 windows.
    starts = torch.arange(0, len(dataset), args.seq_len)
    wrapped = IndexedDataset(dataset, cycle_len=args.cyclenet_cycle_len)
    loader = DataLoader(Subset(wrapped, starts.tolist()), batch_size=args.batch_size,
                        shuffle=False, num_workers=0)
    xs, hs, rs, start, predictions, targets = [], [], [], [], [], []
    for x, y, ids, cycle in loader:
        x, cycle = x.float().to(device), cycle.to(device)
        h, mean, std = model.encode(x, cycle)
        a0 = prediction(model, h, mean, std, cycle, "reference")
        true = y[:, -args.pred_len:].float().to(device)
        scale = x.var(1, keepdim=True, unbiased=False).add(1e-5).sqrt()
        response = ((true - a0) / scale).transpose(1, 2)
        xs.append(x.cpu()); hs.append(h.cpu()); rs.append(response.cpu()); start.append(ids.cpu())
        predictions.append(a0.cpu()); targets.append(true.cpu())
    return {"x": torch.cat(xs), "hidden": torch.cat(hs), "response": torch.cat(rs),
            "starts": torch.cat(start), "a0_pred": torch.cat(predictions),
            "target": torch.cat(targets)}


def env_labels(q, mode, seed):
    q = torch.as_tensor(q, dtype=torch.float32)
    if mode == "predictive":
        return q.clone()
    if mode == "uniform":
        return torch.full_like(q, 1. / q.shape[-1])
    generator = torch.Generator().manual_seed(seed)
    return q[torch.randperm(len(q), generator=generator)]


def canonical_environment(q, record):
    """Fix environment identity from source-only predictive-gradient centroids."""
    q = torch.as_tensor(q, dtype=torch.float32)
    gradients = torch.as_tensor(record["probe_gradients"], dtype=torch.float32)
    order = torch.argsort(gradients[:, 0])
    return q[:, order], order


def create_memories(records, q, keys_bank, args, cli, device):
    n = len(q)
    full = PredictiveMemory(cli.k, cli.history_size, args.pred_len,
                            records["hidden"].shape[-1], args.env_num,
                            cli.temperature, cli.topk)
    full.keys.copy_(keys_bank.keys)
    full_report = full.fit_values(records["x"], records["hidden"],
                                  records["response"], q[records["starts"]],
                                  records["starts"])
    folds, fold_reports = [], []
    for f in range(cli.memory_folds):
        lo = f * n // cli.memory_folds
        # Query starts in [lo, hi_start]; union of complete intervals ends at
        # hi_start + seq_len + pred_len, which also purges boundary overlap.
        hi_start = ((f + 1) * n - 1) // cli.memory_folds
        hi = hi_start + args.seq_len + args.pred_len
        bank = PredictiveMemory(cli.k, cli.history_size, args.pred_len,
                                records["hidden"].shape[-1], args.env_num,
                                cli.temperature, cli.topk)
        bank.keys.copy_(keys_bank.keys)
        source = records["starts"]
        source_end = source + args.seq_len + args.pred_len
        allowed = (source_end <= lo) | (source >= hi)
        # Crucial: these fold-local q values are inferred using only allowed
        # anchor labels, not the full-TRAIN EIIL solution that saw the query.
        selected = torch.where(allowed)[0]
        local_manager = PredictiveConflictEnvironment(
            len(selected), env_num=args.env_num, steps=args.eiil_steps,
            lr=args.eiil_lr, balance_weight=args.balance_weight,
            entropy_weight=args.entropy_weight, seed=cli.seed + f + 113,
            diagnostic_samples=min(128, len(selected)), random_partition_repeats=10)
        fold_q, fold_record = local_manager.update(
            records["a0_pred"][selected], records["target"][selected],
            torch.arange(len(selected)), source_split="train")
        fold_q, _ = canonical_environment(fold_q, fold_record)
        fold_q = env_labels(fold_q, cli.environment_mode, cli.seed + f + 113)
        all_anchor_q = torch.full((len(source), args.env_num), 1. / args.env_num)
        all_anchor_q[selected] = fold_q
        report = bank.fit_values(records["x"], records["hidden"],
                                 records["response"], all_anchor_q,
                                 records["starts"], forbidden_interval=(lo, hi))
        # Validate the exact population used to compute *all* fold values.
        assert int(allowed.sum()) == report["source_anchors"]
        report["environment_source"] = "EIIL on allowed TRAIN anchors only"
        report["environment_gradient_centroids"] = fold_record["probe_gradients"]
        folds.append(bank.to(device)); fold_reports.append(report)
    return {"full": full.to(device), "folds": folds}, {
        "full": full_report, "folds": fold_reports,
        "key_source": "TRAIN history only", "response_source": "TRAIN-only purged folds"}


def point_forward(model, adapter, memories, x, cycle, sample_ids, args, mode,
                  fold_count, mapping_shuffle=False, input_only=False,
                  no_align=False, no_variant=False, full_residual=False):
    hidden, mean, std = model.encode(x, cycle)
    if mode == "B0":
        y = prediction(model, hidden, mean, std, cycle, "reference")
        return y, y, {"h_fused": hidden, "z_inv": hidden, "z_var": torch.zeros_like(hidden)}
    if memories is None or sample_ids is None:
        banks = [(torch.arange(len(x), device=x.device), memories["full"])] if memories else []
    else:
        fold_id = (sample_ids.to(x.device) * fold_count // len(memories["train_q"])).clamp_max(fold_count-1)
        banks = [(torch.where(fold_id == f)[0], memories["folds"][f]) for f in range(fold_count)]
    y_full = x.new_zeros((len(x), args.pred_len, x.shape[-1]))
    y_inv = torch.zeros_like(y_full)
    z_inv_out = torch.zeros_like(hidden)
    z_var_out = torch.zeros_like(hidden)
    qhat_out = x.new_zeros((len(x), adapter.env_predictor[-1].out_features))
    metrics = {k: hidden.new_zeros(()) for k in
               ("gate_mean", "align_effect", "variant_effect", "retrieval_distance", "retrieval_entropy")}
    if mode == "B1":
        factors = adapter(hidden, None, "B1")
        factors["hidden_pooled"] = hidden.mean((1, 2))
        return (prediction(model, factors["h_fused"], mean, std, cycle, "active"),
                prediction(model, factors["z_inv"], mean, std, cycle, "active"), factors)
    for index, bank in banks:
        if len(index) == 0:
            continue
        hh, xx = hidden[index], x[index]
        qhat = adapter.predict_environment(hh)
        query = bank.query(xx, qhat, use_mapping=(mode in ("P3", "P4") and not input_only),
                           shuffle_mapping=mapping_shuffle)
        factors = adapter(hh, query, mode, full_residual, no_align, no_variant)
        y_full = y_full.index_copy(0, index, prediction(
            model, factors["h_fused"], mean[index], std[index], cycle[index], "active"))
        y_inv = y_inv.index_copy(0, index, prediction(
            model, factors["z_inv"], mean[index], std[index], cycle[index], "active"))
        z_inv_out = z_inv_out.index_copy(0, index, factors["z_inv"])
        z_var_out = z_var_out.index_copy(0, index, factors["z_var"])
        qhat_out = qhat_out.index_copy(0, index, factors["qhat"])
        for name, value in (("gate_mean", factors["gate"].mean()),
                            ("align_effect", factors["align_effect"]),
                            ("variant_effect", factors["variant_effect"]),
                            ("retrieval_distance", query["min_distance"].mean()),
                            ("retrieval_entropy", query["entropy"].mean())):
            metrics[name] = metrics[name] + len(index)/len(x)*value
    result = {"z_inv": z_inv_out, "z_var": z_var_out, "qhat": qhat_out,
              "hidden_pooled": hidden.mean((1, 2)), **metrics}
    return y_full, y_inv, result


@torch.no_grad()
def evaluate(model, adapter, memories, loader, args, cli, split):
    model.eval(); adapter.eval()
    totals = {key: 0. for key in ("mse", "mae", "inv_mse", "inv_mae", "a0_mse", "a0_mae",
                                   "gate_mean", "align_effect", "variant_effect",
                                   "retrieval_distance", "retrieval_entropy")}
    nelements = nsamples = 0
    started = time.monotonic()
    for batch_index, (x, y, ids, cycle) in enumerate(loader):
        if cli.max_batches and batch_index >= cli.max_batches:
            break
        x, cycle = x.float().to(next(model.parameters()).device), cycle.to(next(model.parameters()).device)
        y = y[:, -args.pred_len:].float().to(x.device)
        sample_ids = ids if split == "train" else None
        full, inv, info = point_forward(model, adapter, memories, x, cycle, sample_ids,
                                        args, cli.mode, cli.memory_folds,
                                        cli.mapping_shuffle, cli.input_only,
                                        cli.no_align, cli.no_variant, cli.full_residual)
        hidden, mean, std = model.encode(x, cycle)
        a0 = prediction(model, hidden, mean, std, cycle, "reference")
        for name, forecast in (("", full), ("inv_", inv), ("a0_", a0)):
            totals[name + "mse"] += float((forecast-y).square().sum())
            totals[name + "mae"] += float((forecast-y).abs().sum())
        for name in ("gate_mean", "align_effect", "variant_effect", "retrieval_distance", "retrieval_entropy"):
            if name in info:
                totals[name] += float(info[name].detach()) * len(x)
        nelements += y.numel(); nsamples += len(x)
    return {**{k: v/(nsamples if k in ("gate_mean", "align_effect", "variant_effect",
                                      "retrieval_distance", "retrieval_entropy") else nelements)
               for k, v in totals.items()},
            "seconds": time.monotonic()-started,
            "split": split}


def phase0_diagnostics(records, q, memory, all_hidden, outdir, args, cli):
    outdir.mkdir(parents=True, exist_ok=True)
    n = len(q)
    split = max(10, int(.8*n))
    label = q.argmax(-1).numpy()
    train_class = np.unique(label[:split])
    if len(train_class) < 2:
        qhat_acc = float(np.mean(label[split:] == train_class[0]))
    else:
        logreg = LogisticRegression(max_iter=200, random_state=cli.seed)
        logreg.fit(all_hidden[:split].numpy(), label[:split])
        qhat_acc = float((logreg.predict(all_hidden[split:].numpy()) == label[split:]).mean())
    majority = float(np.max(np.bincount(label[split:], minlength=args.env_num))/max(1,n-split))
    keys = history_keys(records["x"], cli.history_size).reshape(-1, cli.history_size).numpy()
    responses = records["response"].reshape(-1, args.pred_len).numpy()
    generator = np.random.default_rng(cli.seed)
    pairs = generator.integers(0, len(keys), size=(min(2000, len(keys)*3), 2))
    dx = np.square(keys[pairs[:, 0]]-keys[pairs[:, 1]]).mean(-1)
    dy = np.square(responses[pairs[:, 0]]-responses[pairs[:, 1]]).mean(-1)
    similarity_corr = float(spearmanr(dx, dy).statistic)
    rows = []
    cluster_label = np.square(keys[:, None] - memory.keys.cpu().numpy()[None]).mean(-1).argmin(-1)
    row_q = np.repeat(q[records["starts"]].numpy(), records["x"].shape[-1], axis=0)
    anchor_count = len(records["starts"])
    for k in range(cli.k):
        support = float(memory.support[k])
        within = float(memory.within[k])
        between = float(memory.between[k])
        mask = cluster_label == k
        null = []
        if mask.sum() >= 8:
            for _ in range(50):
                shuffled = np.repeat(q[records["starts"]].numpy()[
                    generator.permutation(anchor_count)], records["x"].shape[-1], axis=0)[mask]
                rr = responses[mask]
                mass = shuffled.sum(0) + 1e-8
                centers = shuffled.T @ rr / mass[:, None]
                shared = rr.mean(0)
                null.append(float(((mass/mass.sum())[:, None]
                                   * np.square(centers-shared[None])).sum(-1).mean()))
        p_value = (1 + sum(v >= between for v in null))/(1 + len(null)) if null else 1.
        rows.append({"prototype": k, "independent_support": support,
                     "env_effective_support": memory.env_support[k].tolist(),
                     "between": between, "within": within,
                     "between_within_ratio": between/(within+1e-8),
                     "environment_permutation_p": p_value})
    with (outdir / "memory_clusters.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    enough = sum(r["independent_support"] >= 4 and min(r["env_effective_support"]) >= 1
                 for r in rows)
    mapping_significant = sum(r["environment_permutation_p"] < .1 for r in rows)
    evidence = bool(qhat_acc > majority + .02 and similarity_corr > .05 and
                    enough >= 2 and mapping_significant >= 1)
    summary = {"qhat_heldout_hard_accuracy": qhat_acc, "heldout_majority_accuracy": majority,
               "historical_future_distance_spearman": similarity_corr,
               "clusters_with_cross_environment_support": enough,
               "clusters_environment_permutation_p_below_0_1": mapping_significant,
               "evidence_for_conditioned_mapping": evidence,
               "interpretation": "exploratory TRAIN diagnostics, not causal invariance",
               "source_split": "TRAIN only", "rows": rows}
    write_json(outdir / "phase0.json", summary)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].scatter(dx, dy, s=4, alpha=.15)
        axes[0].set(xlabel="historical distance", ylabel="future residual distance")
        axes[1].scatter([r["within"] for r in rows], [r["between"] for r in rows],
                        s=np.maximum(12, [r["independent_support"] for r in rows]))
        axes[1].set(xlabel="within environment", ylabel="between environments")
        fig.tight_layout(); fig.savefig(outdir / "phase0.png", dpi=150); plt.close(fig)
    except ImportError:
        pass
    return summary


def main():
    cli = cli_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    outdir = Path(cli.output); outdir.mkdir(parents=True, exist_ok=True)
    if (outdir / "run_complete").is_file():
        print("already complete, skip", flush=True); return
    torch.manual_seed(cli.seed); np.random.seed(cli.seed)
    args, model, loaders, device, ref_hash = load_experiment(cli)
    ordered = ordered_training_loader(loaders["train"].dataset.dataset, args)
    adapter = RetrievalGuidedAdapter(model.d_model, args.env_num).to(device)
    classifier = EnvironmentClassificationConstraint(model.d_model, args.env_num).to(device)
    manager = PredictiveConflictEnvironment(len(ordered.dataset), env_num=args.env_num,
                                            steps=args.eiil_steps, lr=args.eiil_lr,
                                            balance_weight=args.balance_weight,
                                            entropy_weight=args.entropy_weight, seed=cli.seed,
                                            diagnostic_samples=256, random_partition_repeats=10)
    cache_path = Path(cli.phase0_cache) if cli.phase0_cache else None
    cached = (torch.load(cache_path, map_location="cpu", weights_only=False)
              if cache_path is not None and cache_path.is_file() else None)
    if cached is not None and cached.get("reference_sha256") != ref_hash:
        raise ValueError("phase0 cache was made with a different A0 checkpoint")
    records = (cached["records"] if cached is not None and
               "a0_pred" in cached["records"] else anchor_records(
                   model, loaders["train"].dataset.dataset, args, device))
    key_bank = PredictiveMemory(cli.k, cli.history_size, args.pred_len, model.d_model,
                                args.env_num, cli.temperature, cli.topk)
    key_bank.fit_keys(records["x"], cli.seed)
    commit = __import__("subprocess").run(["git", "rev-parse", "HEAD"],
                    capture_output=True, text=True).stdout.strip()
    write_json(outdir / "run_config.json", {"cli": vars(cli), "base": vars(args),
               "reference_sha256": ref_hash, "git_commit": commit,
               "checkpoint_selection": "VAL MSE", "memory": "TRAIN-only purged cross-fit"})
    # Phase 0 always uses frozen A0 predictions and TRAIN-only environment q.
    if cached is None:
        a0_pred, true, ids, all_hidden = collect_predictions_and_hidden(model, ordered,
                                                                         args, device)
        discovered, stage_record = manager.update(a0_pred, true, ids, source_split="train")
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"records": records, "all_hidden": all_hidden,
                        "discovered": discovered, "stage_record": stage_record,
                        "reference_sha256": ref_hash}, cache_path)
    else:
        all_hidden = cached["all_hidden"]
        discovered, stage_record = cached["discovered"], cached["stage_record"]
        manager.q = torch.as_tensor(discovered, dtype=torch.float32).clone()
        manager.logits = manager.q.clamp_min(1e-8).log()
    canonical_q, order = canonical_environment(discovered, stage_record)
    manager.q = canonical_q.clone()
    manager.logits = manager.q.clamp_min(1e-8).log()
    stage_record["canonical_gradient_scale_order"] = order.tolist()
    q = env_labels(canonical_q, cli.environment_mode, cli.seed)
    memories, reports = create_memories(records, q, key_bank, args, cli, device)
    memories["train_q"] = q
    if cli.phase0_json and Path(cli.phase0_json).is_file():
        phase0 = json.loads(Path(cli.phase0_json).read_text(encoding="utf-8"))
    else:
        phase0 = phase0_diagnostics(records, q, memories["full"], all_hidden,
                                     outdir / "phase0", args, cli)
    write_json(outdir / "memory_stage0.json", reports)
    if cli.mode == "phase0":
        write_json(outdir / "result.json", {"phase0": phase0, "environment": stage_record,
                                            "reference_sha256": ref_hash})
        (outdir / "run_complete").touch(); return
    if cli.mode == "B0":
        val = evaluate(model, adapter, memories, loaders["val"], args, cli, "val")
        test = evaluate(model, adapter, memories, loaders["test"], args, cli, "test")
        write_json(outdir / "result.json", {"val": val, "test": test, "phase0": phase0,
                                            "reference_sha256": ref_hash})
        (outdir / "run_complete").touch(); return
    model.head_linear.requires_grad_(True)
    parameters = list(adapter.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.Adam([{"params": parameters, "lr": cli.lr_adapter},
                                  {"params": model.head_linear.parameters(), "lr": cli.lr_head}])
    train = train_loader(loaders["train"].dataset.dataset, args, cli.seed)
    history = []; best_val = math.inf; stage_records = [stage_record]; began = time.monotonic()
    for epoch in range(cli.epochs):
        if epoch and epoch % cli.stage_epochs == 0:
            if cli.mode == "P4":
                current, true, ids, _ = collect_predictions_and_hidden(
                    model, ordered, args, device, adapter, memories, cli.memory_folds)
                discovered, stage_record = manager.update(current, true, ids, source_split="train")
                canonical_q, order = canonical_environment(discovered, stage_record)
                manager.q = canonical_q.clone()
                manager.logits = manager.q.clamp_min(1e-8).log()
                stage_record["canonical_gradient_scale_order"] = order.tolist()
                q = env_labels(canonical_q, cli.environment_mode, cli.seed + epoch)
                memories, reports = create_memories(records, q, key_bank, args, cli, device)
                memories["train_q"] = q
                stage_records.append(stage_record)
                write_json(outdir / f"memory_stage{epoch//cli.stage_epochs}.json", reports)
        model.eval(); adapter.train(); classifier.train()
        total = inv_total = gate_total = 0.; batches = 0
        for batch_index, (x, y, sample_ids, cycle) in enumerate(train):
            if cli.max_batches and batch_index >= cli.max_batches:
                break
            x, cycle = x.float().to(device), cycle.to(device)
            y = y[:, -args.pred_len:].float().to(device)
            full, inv, info = point_forward(model, adapter, memories, x, cycle,
                                            sample_ids, args, cli.mode, cli.memory_folds,
                                            cli.mapping_shuffle, cli.input_only,
                                            cli.no_align, cli.no_variant, cli.full_residual)
            primary = (full-y).square().mean()
            inv_loss = (inv-y).square().mean()
            label_q = q[sample_ids].to(device).detach()
            env_loss, _ = classifier(info["z_inv"].mean((1, 2)),
                                     info["z_var"].mean((1, 2)), label_q)
            qhat_loss = soft_cross_entropy(adapter.env_predictor(
                info["hidden_pooled"].detach()), label_q)
            loss = primary + cli.lambda_inv*inv_loss + cli.lambda_env*env_loss + cli.lambda_qhat*qhat_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss epoch={epoch+1}")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            total += float(primary.detach()); inv_total += float(inv_loss.detach())
            gate_total += float(info.get("gate_mean", x.new_zeros(())).detach()); batches += 1
        val = evaluate(model, adapter, memories, loaders["val"], args, cli, "val")
        row = {"epoch": epoch+1, "train_mse": total/batches,
               "train_inv_mse": inv_total/batches, "train_gate_mean": gate_total/batches,
               "val_mse": val["mse"], "val_mae": val["mae"],
               "val_inv_mse": val["inv_mse"]}
        history.append(row); print(json.dumps(row), flush=True)
        write_json(outdir / "train_history.json", history)
        if val["mse"] < best_val:
            best_val = val["mse"]
            torch.save({"model": model.state_dict(), "adapter": adapter.state_dict(),
                        "classifier": classifier.state_dict(), "memory": memories["full"].state_dict(),
                        "epoch": epoch+1, "val": val, "reference_sha256": ref_hash},
                       outdir / "best.pt")
    best = torch.load(outdir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"]); adapter.load_state_dict(best["adapter"])
    memories["full"].load_state_dict(best["memory"])
    test = evaluate(model, adapter, memories, loaders["test"], args, cli, "test")
    result = {"mode": cli.mode, "best_epoch": best["epoch"], "val": best["val"],
              "test": test, "phase0": phase0, "environment_stages": stage_records,
              "train_history": history, "reference_sha256": ref_hash,
              "training_seconds": time.monotonic()-began,
              "additional_parameters": sum(p.numel() for p in parameters),
              "test_selection_note": "TEST for exploratory comparison only; epoch from VAL"}
    write_json(outdir / "result.json", result)
    (outdir / "run_complete").touch()


if __name__ == "__main__":
    main()
