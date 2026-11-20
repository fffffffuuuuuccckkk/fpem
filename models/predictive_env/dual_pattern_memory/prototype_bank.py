"""TRAIN-only compact prototypes with complete normalized future mappings."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

from models.predictive_env.dual_consensus_retrieval import entmax15


def normalized_pairs(x: torch.Tensor, y: torch.Tensor, floor: torch.Tensor):
    """Normalize future with past statistics, never with future statistics."""
    mean = x.float().mean(1, keepdim=True)
    std = x.float().var(1, keepdim=True, unbiased=False).add(1e-5).sqrt()
    std = std.clamp_min(floor.reshape(1, 1, -1))
    return ((y.float() - mean) / std).transpose(1, 2), mean, std


class PatternBank:
    """Shared across variables; query complexity depends on M, not TRAIN N."""
    def __init__(self, n_patterns: int, pred_len: int, key_dim: int = 64,
                 temperature: float = .5, ridge: float = 10., seed: int = 2021):
        if n_patterns < 1 or temperature <= 0 or ridge <= 0:
            raise ValueError("invalid pattern-bank hyperparameters")
        self.n_patterns = int(n_patterns)
        self.pred_len = int(pred_len)
        self.key_dim = int(key_dim)
        self.temperature = float(temperature)
        self.ridge = float(ridge)
        self.seed = int(seed)
        self.centers = None
        self.mapping = None
        self.counts = None
        self.block_support = None
        self.loo_mse = None
        self.output_basis = None

    def _flat(self, feature, target, blocks):
        n, c, d = feature.shape
        if d != self.key_dim or target.shape != (n, c, self.pred_len):
            raise ValueError("pattern TRAIN feature/target shapes differ")
        return (feature.reshape(n*c, d).float(),
                target.reshape(n*c, self.pred_len).float(),
                blocks[:, None].expand(n, c).reshape(-1).long())

    def fit(self, feature: torch.Tensor, target: torch.Tensor,
            blocks: torch.Tensor, mapping_feature: torch.Tensor | None = None,
            output_basis: torch.Tensor | None = None,
            preset_centers: torch.Tensor | None = None):
        """Block-balanced MiniBatchKMeans and per-prototype weighted Ridge.

        Mapping can use a larger [u,v] input, while retrieval uses v alone.
        Every future target and prototype mapping uses TRAIN only.
        """
        x, y, group = self._flat(feature.cpu(), target.cpu(), blocks.cpu())
        if output_basis is not None:
            if output_basis.ndim != 2 or output_basis.shape[0] != self.pred_len:
                raise ValueError("shared future basis must be [pred_len, rank]")
            self.output_basis = output_basis.detach().cpu().float()
            y_fit = y @ self.output_basis
        else:
            self.output_basis = None
            y_fit = y
        if mapping_feature is None:
            mapping_feature = feature
        mapping_x = mapping_feature.cpu().reshape(len(x), -1).double()
        if len(x) < self.n_patterns:
            raise ValueError("too few TRAIN anchor-channel pairs")
        block_count = int(group.max()) + 1
        block_mass = torch.bincount(group, minlength=block_count).clamp_min(1)
        balanced = (len(group) / (block_count * block_mass[group])).numpy()
        if preset_centers is None:
            km = MiniBatchKMeans(n_clusters=self.n_patterns, random_state=self.seed,
                                 batch_size=min(256, len(x)), n_init=3,
                                 reassignment_ratio=.01)
            km.fit(x.numpy(), sample_weight=balanced)
            self.centers = torch.from_numpy(km.cluster_centers_.astype(np.float32))
        else:
            if preset_centers.shape != (self.n_patterns, self.key_dim):
                raise ValueError("preset centers must be [n_patterns, key_dim]")
            self.centers = preset_centers.detach().cpu().float().clone()
        weight = self.match_flat(x)
        support = weight > 0
        self.counts = support.sum(0)
        self.block_support = torch.stack([
            support[group == b].sum(0) for b in range(block_count)]).float()
        # Design matrix includes a bias; all targets are complete futures.
        design = torch.cat((mapping_x, mapping_x.new_ones(len(x), 1)), -1)
        outputs = []
        eye = torch.eye(design.shape[-1], dtype=torch.float64)
        eye[-1, -1] = 1e-4  # almost unregularized, still SPD for empty support
        for j in range(self.n_patterns):
            w = weight[:, j].double() * torch.from_numpy(balanced).double()
            w = w / w.sum().clamp_min(1e-9)
            root = w.sqrt()[:, None]
            xw = design * root
            yw = y_fit.double() * root
            chol = torch.linalg.cholesky(xw.T @ xw + self.ridge * eye)
            outputs.append(torch.cholesky_solve(xw.T @ yw, chol).float())
        self.mapping = torch.stack(outputs)
        self.key_dim = feature.shape[-1]
        return {"pairs": len(x), "patterns": self.n_patterns,
                "support": self.counts.tolist(),
                "effective_support": weight.sum(0).tolist(),
                "block_support": self.block_support.tolist(),
                "mapping_shape": list(self.mapping.shape),
                "source": "TRAIN only, time-block-balanced"}

    def match_flat(self, feature: torch.Tensor, null_score: float | None = None):
        if self.centers is None:
            raise RuntimeError("fit bank on TRAIN before query")
        distance = (feature[:, None].float() - self.centers[None].to(feature)).square().mean(-1)
        score = -distance / self.temperature
        if null_score is not None:
            score = torch.cat((score, score.new_full((len(score), 1), float(null_score))), -1)
        return entmax15(score, torch.ones_like(score, dtype=torch.bool))

    def query(self, feature: torch.Tensor,
              mapping_feature: torch.Tensor | None = None,
              null_score: float | None = None):
        """[B,C,D] -> full future [B,C,T], no TRAIN row access."""
        b, c, d = feature.shape
        if d != self.key_dim:
            raise ValueError("query key dimension mismatch")
        if mapping_feature is None:
            mapping_feature = feature
        weights = self.match_flat(feature.reshape(b*c, d), null_score)
        active = weights[:, :self.n_patterns]
        design = torch.cat((mapping_feature.reshape(b*c, -1).float(),
                            feature.new_ones(b*c, 1)), -1)
        if design.shape[-1] != self.mapping.shape[1]:
            raise ValueError("future-map feature dimension mismatch")
        candidates = torch.einsum("bf,mft->bmt", design, self.mapping.to(design))
        if self.output_basis is not None:
            candidates = candidates @ self.output_basis.T.to(candidates)
        prediction = torch.einsum("bm,bmt->bt", active, candidates)
        matched = torch.einsum("bm,md->bd", active,
                               self.centers.to(active))
        null = weights[:, -1] if null_score is not None else active.new_zeros(len(active))
        return {"prediction": prediction.reshape(b,c,self.pred_len),
                "candidate": candidates.reshape(b,c,self.n_patterns,self.pred_len),
                "weight": active.reshape(b,c,self.n_patterns),
                "null_weight": null.reshape(b,c),
                "matched_history": matched.reshape(b,c,self.key_dim),
                "effective_count": (1/active.square().sum(-1).clamp_min(1e-8)).reshape(b,c)}

    def state(self):
        return {"centers": self.centers, "mapping": self.mapping,
                "output_basis": self.output_basis,
                "counts": self.counts, "block_support": self.block_support,
                "n_patterns": self.n_patterns, "pred_len": self.pred_len,
                "key_dim": self.key_dim, "temperature": self.temperature,
                "ridge": self.ridge, "seed": self.seed}

    @classmethod
    def from_state(cls, state):
        bank = cls(state["n_patterns"], state["pred_len"], state["key_dim"],
                   state["temperature"], state["ridge"], state["seed"])
        for name in ("centers", "mapping", "counts", "block_support"):
            setattr(bank, name, state[name])
        bank.output_basis = state.get("output_basis")
        return bank
