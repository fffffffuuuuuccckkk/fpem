"""S4 loss-wise optimizer pilot. Reuses the existing TRAIN-only EIIL protocol."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import resource
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.predictive_env.eiil_environment import PredictiveConflictEnvironment
from models.predictive_env.shared_factorization import EnvironmentPartitionAdapter, make_shared_factor_decomposer
from models.predictive_env.losswise_adam import LosswiseAdam
from tools.run_fpem_shared_factorization import (
    IndexedDataset, build_optimizer, collect_train_predictions, decomposition_latency_ms,
    evaluate, factorize, load_experiment, ordered_training_loader, prediction,
    refit_statistics, train_loader, write_json,
)


MODES = {
    "E0": (0.0, 0.0), "E1": (1.0, 1.0), "E2": (2.0, 1.0),
    "E3": (4.0, 1.0), "E4": (2.0, 0.5), "E5": (4.0, 0.5),
    "GS": (0.0, 0.0), "G1": (0.0, 0.0), "G2": (0.0, 0.0),
}


def args_parse():
    p = argparse.ArgumentParser()
    p.add_argument("--base_run_config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--stage_epochs", type=int, default=0)
    p.add_argument("--lr_head", type=float, required=True)
    p.add_argument("--lr_decomposer", type=float, default=1e-4)
    p.add_argument("--lr_backbone", type=float, default=2e-5)
    p.add_argument("--train_backbone", action="store_true")
    p.add_argument("--environment_mode", default="predictive")
    p.add_argument("--pair_affinity", default="env_affinity")
    p.add_argument("--function_preserving_init", action="store_true")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--eiil_steps", type=int, default=0)
    p.add_argument("--max_batches", type=int, default=0)
    p.add_argument("--prediction_loss", choices=("mse", "mae"), default="mse")
    return p.parse_args()


def norm(grads):
    return math.sqrt(sum(float(g.detach().float().square().sum()) for g in grads if g is not None))


def cosine(gp, ga):
    if gp is None or ga is None:
        return float("nan")
    p, a = gp.detach().float(), ga.detach().float()
    return float((p * a).sum() / (p.norm() * a.norm() + 1e-12))


def surgery(gp, ga, parameter_names, strength=1.0):
    """Remove a fraction of conflicting auxiliary gradient on shared Ws only."""
    if not 0.0 <= strength <= 1.0:
        raise ValueError("surgery strength must lie in [0, 1]")
    projected = list(ga)
    for i, name in enumerate(parameter_names):
        if name != "shared.weight" or gp[i] is None or ga[i] is None:
            continue
        dot = (gp[i] * ga[i]).sum()
        if dot < 0:
            projected[i] = ga[i] - strength * dot / (gp[i].square().sum() + 1e-12) * gp[i]
    return projected


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    cli = args_parse()
    outdir = Path(cli.output)
    outdir.mkdir(parents=True, exist_ok=True)
    args, model, loaders, device, reference_hash = load_experiment(cli)
    ordered = ordered_training_loader(loaders["train"].dataset.dataset, args)
    x, _, _, cycle = next(iter(ordered))
    with torch.no_grad():
        hidden, _, _ = model.encode(x.float().to(device), cycle.to(device))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cli.seed + 811)
        adapter = make_shared_factor_decomposer(
            "supervised_joint_private", hidden.shape[-1], seq_len=args.seq_len,
            affinity=cli.pair_affinity,
            function_preserving_init=cli.function_preserving_init,
        ).to(device)
    if not hasattr(adapter, "shared") or not hasattr(adapter, "private"):
        raise RuntimeError("S4 shared/private parameters not found")
    model.head_linear.requires_grad_(True)
    groups = [{"name": "head", "params": list(model.head_linear.parameters()), "lr": cli.lr_head},
              {"name": "decomposer", "params": list(adapter.parameters()), "lr": cli.lr_decomposer}]
    s4_params = [p for group in groups for p in group["params"]]
    names = ["head." + n for n, _ in model.head_linear.named_parameters()] + [
        n for n, _ in adapter.named_parameters()]
    if len(names) != len(s4_params):
        raise RuntimeError("S4 parameter naming mismatch")
    if cli.mode in ("E0", "GS", "G1", "G2"):
        optimizer = build_optimizer(model, adapter, cli)
        losswise = None
        backbone_optimizer = None
    else:
        losswise = LosswiseAdam(groups)
        optimizer = None
        backbone_optimizer = None
        if cli.train_backbone:
            prefixes = ("patch_embedding", "inverted_embedding", "encoder", "modern_tcn_backbone")
            backbone = [p for name, p in model.named_parameters() if name.startswith(prefixes)]
            for p in backbone:
                p.requires_grad_(True)
            backbone_optimizer = torch.optim.Adam([{"params": backbone, "lr": cli.lr_backbone}])

    n = min(len(ordered.dataset), cli.max_batches * args.batch_size) if cli.max_batches else len(ordered.dataset)
    environment = PredictiveConflictEnvironment(
        n, env_num=args.env_num, steps=cli.eiil_steps or args.eiil_steps,
        lr=args.eiil_lr, balance_weight=args.balance_weight,
        entropy_weight=args.entropy_weight, seed=cli.seed,
        diagnostic_samples=min(128 if cli.max_batches else 512, n),
        random_partition_repeats=10 if cli.max_batches else 100)
    partition = EnvironmentPartitionAdapter(n, args.env_num, cli.environment_mode, cli.seed)
    if cli.max_batches:
        subset = Subset(IndexedDataset(loaders["train"].dataset.dataset,
                                        cycle_len=args.cyclenet_cycle_len), range(n))
        training = DataLoader(subset, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(cli.seed), num_workers=0)
    else:
        training = train_loader(loaders["train"].dataset.dataset, args, cli.seed)

    stage_epochs = cli.stage_epochs or args.stage_epochs
    pred_mult, aux_mult = MODES[cli.mode]
    config = {"cli": vars(cli), "reference_sha256": reference_hash,
              "base_run_config": vars(args), "selection_split": "val",
              "trick_selection_split": "test_exploratory",
              "S4_optimizer": "original_adam" if losswise is None else "two_independent_adam_states",
              "adam_betas": [0.9, 0.999], "adam_eps": 1e-8, "weight_decay": 0,
              "amp": False, "gradient_clipping": False,
              "trainable_parameters": sum(p.numel() for p in s4_params),
              "git_commit": subprocess.run(["git", "rev-parse", "HEAD"],
                                            capture_output=True, text=True).stdout.strip()}
    write_json(outdir / "run_config.json", config)
    history, environment_history, fit_history, gradient_rows, lr_rows = [], [], [], [], []
    best_val = math.inf
    started = time.perf_counter()
    for stage_start in range(0, cli.epochs, stage_epochs):
        stage = stage_start // stage_epochs
        preds, targets, ids = collect_train_predictions(model, adapter, ordered, args, device, cli.max_batches)
        discovered, record = environment.update(preds, targets, ids, source_split="train")
        partition.refresh_after_environment_update(discovered)
        record.update({"stage": stage, "supervision": cli.environment_mode,
                       "assignment_hash": hashlib.sha256(partition.q.numpy().tobytes()).hexdigest()})
        environment_history.append(record)
        fit_started = time.perf_counter()
        fit = refit_statistics(model, adapter, ordered, partition, device, cli.max_batches)
        fit_history.append({"stage": stage, "seconds": time.perf_counter() - fit_started, **fit})
        if stage == 0:
            initial = evaluate(model, adapter, loaders["val"], args, device,
                               max_batches=cli.max_batches)
        for epoch in range(stage_start, min(cli.epochs, stage_start + stage_epochs)):
            model.train() if cli.train_backbone else model.eval()
            model.head_dropout.train()
            adapter.train()
            total, count = 0.0, 0
            grad_acc = {key: 0.0 for key in (
                "pred_grad_norm", "aux_grad_norm", "ws_grad_cosine", "ws_conflict_fraction",
                "pred_update_norm", "aux_update_norm", "s4_update_norm",
                "ws_update_norm", "wv_update_norm", "head_update_norm")}
            batches = 0
            for batch_number, (x, y, sample_ids, cycle) in enumerate(training):
                if cli.max_batches and batch_number >= cli.max_batches:
                    break
                x, y, cycle = x.float().to(device), y[:, -args.pred_len:].float().to(device), cycle.to(device)
                hidden, mean, std = model.encode(x, cycle)
                q = partition.get_assignments(sample_ids).to(device)
                factors = factorize(adapter, hidden, sample_ids, q)
                pred = prediction(model, factors["z_inv_tokens"], mean, std, cycle, head="active")
                primary = ((pred - y).square() if cli.prediction_loss == "mse" else
                           (pred - y).abs()).mean()
                aux = sum(factors["aux_losses"].values(), primary.new_zeros(()))
                loss = primary + aux
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite loss stage={stage} epoch={epoch+1}")
                before = {"ws": adapter.shared.weight.detach().clone(),
                          "wv": adapter.private.weight.detach().clone(),
                          "head": [p.detach().clone() for p in model.head_linear.parameters()]}
                if losswise is None:
                    # Extra gradients are diagnostics only; original Adam still sees exactly Lpred+Laux.
                    ws = adapter.shared.weight
                    gp_ws = torch.autograd.grad(primary, ws, retain_graph=True, allow_unused=True)[0]
                    ga_ws = torch.autograd.grad(aux, ws, retain_graph=True, allow_unused=True)[0]
                    gp = ga = None
                    optimizer.zero_grad(set_to_none=True)
                    if cli.mode in ("GS", "G1", "G2"):
                        gp = torch.autograd.grad(primary, s4_params, retain_graph=True, allow_unused=True)
                        ga = torch.autograd.grad(aux, s4_params, retain_graph=True, allow_unused=True)
                    loss.backward()
                    if cli.mode in ("GS", "G1", "G2"):
                        strength = 0.5 if cli.mode == "G2" else 1.0
                        ga_projected = surgery(gp, ga, names, strength=strength)
                        for p, a, b in zip(s4_params, gp, ga_projected):
                            p.grad = (a if b is None else b if a is None else a + b)
                    optimizer.step()
                    pred_norm = norm([gp_ws]); aux_norm = norm([ga_ws])
                    update_stats = {"prediction_update_norm": float("nan"),
                                    "auxiliary_update_norm": float("nan")}
                else:
                    gp = torch.autograd.grad(primary, s4_params, retain_graph=True, allow_unused=True)
                    ga = torch.autograd.grad(aux, s4_params, retain_graph=cli.train_backbone,
                                             allow_unused=True)
                    ws_index = names.index("shared.weight")
                    gp_ws, ga_ws = gp[ws_index], ga[ws_index]
                    pred_norm, aux_norm = norm(gp), norm(ga)
                    if backbone_optimizer is not None:
                        backbone_optimizer.zero_grad(set_to_none=True)
                        for p in s4_params:
                            p.grad = None
                        loss.backward()
                    update_stats = losswise.step(gp, ga, pred_mult, aux_mult)
                    if backbone_optimizer is not None:
                        backbone_optimizer.step()
                        for p in s4_params:
                            p.grad = None
                ws_cos = cosine(gp_ws, ga_ws)
                ws_update = float((adapter.shared.weight.detach() - before["ws"]).norm())
                wv_update = float((adapter.private.weight.detach() - before["wv"]).norm())
                head_update = math.sqrt(sum(float((p.detach() - old).square().sum())
                                            for p, old in zip(model.head_linear.parameters(), before["head"])))
                stats = {
                    "pred_grad_norm": pred_norm, "aux_grad_norm": aux_norm,
                    "ws_grad_cosine": 0.0 if math.isnan(ws_cos) else ws_cos,
                    "ws_conflict_fraction": float(ws_cos < 0),
                    "pred_update_norm": update_stats["prediction_update_norm"],
                    "aux_update_norm": update_stats["auxiliary_update_norm"],
                    "s4_update_norm": math.sqrt(ws_update**2 + wv_update**2 + head_update**2),
                    "ws_update_norm": ws_update, "wv_update_norm": wv_update,
                    "head_update_norm": head_update}
                for key, value in stats.items():
                    if math.isfinite(value):
                        grad_acc[key] += value
                total += float(primary.detach()) * y.numel()
                count += y.numel()
                batches += 1
            val = evaluate(model, adapter, loaders["val"], args, device, max_batches=cli.max_batches)
            test = evaluate(model, adapter, loaders["test"], args, device, max_batches=cli.max_batches)
            row = {"epoch": epoch+1, "stage": stage, "train_prediction_loss": total/count,
                   "val_mse": val["mse"], "val_mae": val["mae"],
                   "test_mse_diagnostic_only": test["mse"],
                   "test_mae_diagnostic_only": test["mae"]}
            history.append(row)
            gradient_rows.append({"epoch": epoch+1, "stage": stage, **{
                key: value/batches for key, value in grad_acc.items()},
                "effective_lr_ratio": pred_mult/aux_mult if aux_mult else 1.0})
            lr_rows.append({"epoch": epoch+1, "head_prediction_lr": cli.lr_head * (pred_mult or 1),
                            "head_auxiliary_lr": cli.lr_head * (aux_mult or 1),
                            "ws_prediction_lr": cli.lr_decomposer * (pred_mult or 1),
                            "ws_auxiliary_lr": cli.lr_decomposer * (aux_mult or 1),
                            "wv_prediction_lr": cli.lr_decomposer * (pred_mult or 1),
                            "wv_auxiliary_lr": cli.lr_decomposer * (aux_mult or 1),
                            "backbone_lr": cli.lr_backbone if cli.train_backbone else 0})
            print(json.dumps(row), flush=True)
            if val["mse"] < best_val:
                best_val = val["mse"]
                torch.save({"model": model.state_dict(), "decomposer": adapter.state_dict(),
                            "best_epoch": epoch+1, "val": val, "stage": stage,
                            "reference_sha256": reference_hash,
                            "losswise_optimizer": losswise.state_dict() if losswise else None},
                           outdir / "best.pt")
        write_json(outdir / "environment_stages.json", environment_history)
        write_json(outdir / "fit_stages.json", fit_history)
        write_csv(outdir / "gradient_diagnostics.csv", gradient_rows)
        write_csv(outdir / "lr_trace.csv", lr_rows)
        write_json(outdir / "history.json", history)
    best = torch.load(outdir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    adapter.load_state_dict(best["decomposer"])
    test = evaluate(model, adapter, loaders["test"], args, device, max_batches=cli.max_batches)
    result = {"mode": cli.mode, "reference_sha256": reference_hash,
              "best_epoch": best["best_epoch"], "initial_val": initial,
              "val": best["val"], "test": test, "train_history": history,
              "gradient_history": gradient_rows, "lr_history": lr_rows,
              "environment_history": environment_history, "fit_history": fit_history,
              "training_seconds": time.perf_counter()-started,
              "cpu_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
              "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device)
              if device.type == "cuda" else 0,
              "decomposition_milliseconds_per_batch": decomposition_latency_ms(model, adapter, loaders["val"], device)}
    write_json(outdir / "result.json", result)
    (outdir / "run_complete").touch()


if __name__ == "__main__":
    main()
