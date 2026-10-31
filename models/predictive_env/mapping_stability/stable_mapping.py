"""Small, TRAIN-only cross-temporal forecast mappings for frozen backbones."""
from __future__ import annotations

import torch

from models.predictive_env.paired_memory.memory_bank import history_key


def history_features(x: torch.Tensor, size: int = 64) -> torch.Tensor:
    key = history_key(x, size)
    return torch.cat((key, torch.ones_like(key[..., :1])), -1)


def history_scale(x: torch.Tensor, floor: torch.Tensor):
    mean = x.float().mean(1, keepdim=True)
    std = x.float().var(1, keepdim=True, unbiased=False).add(1e-5).sqrt()
    return mean, std.clamp_min(floor.view(1, 1, -1))


def chronological_blocks(starts: torch.Tensor, seq_len: int, pred_len: int,
                         blocks: int = 4):
    """Assign contiguous anchors; purge preceding labels crossing a boundary."""
    if starts.ndim != 1 or not torch.equal(starts, starts.sort().values):
        raise ValueError("TRAIN starts must be chronological")
    n = len(starts)
    if n < blocks * 3:
        raise ValueError("insufficient chronological anchors")
    ids = torch.div(torch.arange(n) * blocks, n, rounding_mode="floor").clamp_max(blocks - 1)
    valid = torch.ones(n, dtype=torch.bool)
    ends = starts + seq_len + pred_len
    for block in range(blocks - 1):
        next_first = starts[(ids == block + 1).nonzero(as_tuple=True)[0][0]]
        valid &= ~((ids == block) & (ends > next_first))
    if any(int((valid & (ids == block)).sum()) < 2 for block in range(blocks)):
        raise ValueError("purged temporal block has fewer than two anchors")
    return ids, valid


def _ridge(features: torch.Tensor, target: torch.Tensor, regularization: float,
           row_weight: torch.Tensor | None = None):
    """Channel-wise ridge, feature [N,C,F], target [N,C,T]."""
    x = features.double().transpose(0, 1)
    y = target.double().transpose(0, 1)
    if row_weight is not None:
        root = row_weight.double().sqrt()[None, :, None]
        x = x * root
        y = y * root
    gram = x.transpose(-1, -2) @ x
    rhs = x.transpose(-1, -2) @ y
    eye = torch.eye(gram.shape[-1], dtype=gram.dtype, device=gram.device)
    return torch.linalg.solve(gram + max(float(regularization), 1e-6) * eye,
                              rhs).float()


def fit_global_ridge(features, residual, regularization, mask=None):
    if mask is not None:
        features, residual = features[mask], residual[mask]
    return _ridge(features, residual, regularization)


def predict_mapping(features: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bcf,cft->bct", features.float(), weight.float())


def fit_shared_mapping(features, residual, block_ids, valid,
                       lambda_shared: float, lambda_private: float,
                       iterations: int = 8):
    """Alternating ridge with centered block offsets (sum_e V_e = 0).

    Block-balanced rows implement the equal-block objective. Only the shared
    coefficient is returned to inference; private offsets are audit-only.
    """
    features, residual, block_ids = features[valid], residual[valid], block_ids[valid]
    count = int(block_ids.max()) + 1
    row_weight = torch.empty(len(block_ids), device=features.device)
    for block in range(count):
        chosen = block_ids == block
        row_weight[chosen] = len(block_ids) / (count * chosen.sum())
    shared = _ridge(features, residual, lambda_shared, row_weight)
    private = shared.new_zeros(count, *shared.shape)
    for _ in range(iterations):
        for block in range(count):
            chosen = block_ids == block
            private[block] = _ridge(
                features[chosen], residual[chosen] -
                predict_mapping(features[chosen], shared), lambda_private)
        private -= private.mean(0, keepdim=True)
        # Variable block sizes prohibit stacking; fill chronological rows.
        remainder = residual.clone()
        for block in range(count):
            chosen = block_ids == block
            remainder[chosen] -= predict_mapping(features[chosen], private[block])
        shared = _ridge(features, remainder, lambda_shared, row_weight)
    return shared, private


def cross_temporal_audit(features, residual, block_ids, valid,
                         lambda_shared: float, lambda_private: float,
                         lambda_global: float):
    """Forward-only TRAIN block transfer, never fitting on evaluation blocks."""
    rows = []
    for heldout in range(2, int(block_ids.max()) + 1):
        source = valid & (block_ids < heldout)
        target = valid & (block_ids == heldout)
        if int(source.sum()) < 4 or int(target.sum()) == 0:
            continue
        shared, _ = fit_shared_mapping(features, residual, block_ids, source,
                                       lambda_shared, lambda_private)
        ordinary = fit_global_ridge(features, residual, lambda_global, source)
        truth = residual[target]
        rows.append({
            "holdout_block": heldout, "train_anchor_count": int(source.sum()),
            "holdout_anchor_count": int(target.sum()),
            "a0_normalized_mse": float(truth.square().mean()),
            "shared_normalized_mse": float((truth - predict_mapping(features[target], shared)).square().mean()),
            "ordinary_normalized_mse": float((truth - predict_mapping(features[target], ordinary)).square().mean()),
        })
    return rows
