"""Real-checkpoint smoke for all three additive insertion sites."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.predictive_env.exogenous_embedding import ExogenousEmbedding
from tools.run_fpem_eiil3_exog import calendar_batch, check_zero_identity, loader_for
from tools.run_fpem_shared_factorization_g1r3 import load_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case-key", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    assets = {row["case_key"]: row for row in
              (json.loads(line) for line in args.manifest.read_text().splitlines())}
    asset = assets[args.case_key]
    source = argparse.Namespace(base_run_config=asset["base_run_config"],
                                gpu=args.gpu, seed=asset["seed"],
                                allow_legacy_reference_signature=True)
    config, base, loaders, device, sha = load_experiment(source)
    assert sha == asset["A0_sha256"]
    dataset = loaders["train"].dataset.dataset
    adapter = ExogenousEmbedding(base, config.backbone, config.seq_len,
                                 dataset.data_stamp.shape[1], config.d_model).to(device)
    err = check_zero_identity(adapter, loaders["val"].dataset.dataset,
                              config, device)
    x, y, ids, cycle = next(iter(loader_for(dataset, config, 1, 2)))
    x, y, cycle = x.float().to(device), y[:, -config.pred_len:].float().to(device), cycle.to(device)
    mark = calendar_batch(dataset, ids, config.seq_len, device)
    pred = adapter(x, mark, cycle)
    loss = (pred-y).square().mean()
    loss.backward()
    gradient = adapter.proj.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient.abs().max().item() > 0
    assert all(p.grad is None for p in base.parameters())
    with torch.no_grad():
        adapter.proj.weight.add_(1e-3*gradient.sign())
    changed = adapter(x, mark, cycle)
    assert (changed-pred).abs().max().item() > 0
    print(json.dumps({"case": args.case_key, "zero_max_error": err,
                      "gradient_norm": gradient.norm().item(),
                      "changed_prediction_max": (changed-pred).abs().max().item(),
                      "added_params": adapter.proj.weight.numel()}))


if __name__ == "__main__":
    main()
