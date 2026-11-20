"""Low-cost synthetic structure check; q is discovered from predictive conflict.

True groups and true factor directions are used only for offline scoring, never
for fitting unless --oracle_env is explicitly requested as an upper bound.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.predictive_env.eiil_environment import PredictiveConflictEnvironment
from models.predictive_env.shared_factorization import StreamingHiddenStatistics, make_shared_factor_decomposer


def score(operator, directions):
    q, _ = torch.linalg.qr(directions)
    q = q.double()
    projected = operator.double() @ q
    return float(projected.square().sum() / q.square().sum())


def run(noise, group_proportions, predictive_private, oracle_env=False):
    generator = torch.Generator().manual_seed(2021)
    n, d = 600, 16
    groups = torch.multinomial(torch.tensor(group_proportions), n, replacement=True, generator=generator)
    common = torch.eye(d)[:, :2]
    partial = torch.eye(d)[:, 2:5]
    shared_coef = torch.randn(n, 2, generator=generator)
    private_coef = torch.randn(n, generator=generator) * (groups != 2)
    h = shared_coef @ common.T + private_coef[:, None] * partial[:, groups].T
    h += noise * torch.randn(n, d, generator=generator)
    true_y = shared_coef[:, :1] + (0.8 * private_coef[:, None] if predictive_private else 0)
    baseline = 0.7 * shared_coef[:, :1]
    if oracle_env:
        q = torch.nn.functional.one_hot(groups, 3).float()
    else:
        manager = PredictiveConflictEnvironment(n, env_num=3, steps=50, seed=2021,
                                                random_partition_repeats=10,
                                                diagnostic_samples=128)
        q, _ = manager.update(baseline[:, None, :], true_y[:, None, :],
                              torch.arange(n), source_split="train")
        q = torch.as_tensor(q)
    stats = StreamingHiddenStatistics(d, 3)
    stats.add(h[:, None, None, :], q)
    summary = stats.finish()
    records = []
    for method in ("common_subspace", "shared_spectral", "robust_lowrank_sparse",
                   "subspace_recurrence", "subspace_recurrence_poly"):
        model = make_shared_factor_decomposer(method, d)
        model.fit(summary)
        out = model(h[:, None, None, :])
        inv = out["z_inv_tokens"].squeeze(1).squeeze(1)
        var = out["z_var_tokens"].squeeze(1).squeeze(1)
        records.append({
            "method": method,
            "shared_direction_response": score(model.inv_operator, common),
            "partial_direction_response": score(model.var_operator, partial),
            "reconstruction_mae": float((out["reconstruction"] - h[:, None, None, :]).abs().mean()),
            "inv_linear_oracle_mse": float((inv @ torch.linalg.lstsq(inv, true_y).solution
                                            - true_y).square().mean()),
            "global_energy_fraction": float(inv.square().sum() / h.square().sum()),
            "partial_energy_fraction": float(var.square().sum() / h.square().sum()),
        })
    return {"noise": noise, "group_proportions": group_proportions,
            "private_predictive": predictive_private,
            "environment_protocol": "oracle_upper_bound" if oracle_env else "predictive_conflict_DISCOVERED",
            "rows": records}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--oracle_env", action="store_true")
    args = p.parse_args()
    results = [run(noise, sizes, predictive_private, args.oracle_env)
               for noise in (0.05, 0.3)
               for sizes in ((0.6, 0.3, 0.1), (0.34, 0.33, 0.33))
               for predictive_private in (False, True)]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"synthetic_settings": len(results), "method_checks": sum(len(x["rows"]) for x in results),
                      "oracle_env": args.oracle_env}), flush=True)


if __name__ == "__main__":
    main()
