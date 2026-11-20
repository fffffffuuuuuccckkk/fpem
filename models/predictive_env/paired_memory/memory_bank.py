"""Real TRAIN history-future pairs; never prototype-averaged futures.

Rows are non-overlapping historical anchors, indexed by original TRAIN sample
start. During TRAIN, a source is eligible only once its full prediction target
has been observed before the query history starts. Validation also uses a
chronological cutoff at its first historical start. Every view returns actual
source rows, including source hidden tokens and their paired true futures.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def history_key(x: torch.Tensor, size: int) -> torch.Tensor:
    """[B,L,C] -> [B,C,R], normalized using history only."""
    series = x.transpose(1, 2).float()
    mean = series.mean(-1, keepdim=True)
    std = series.var(-1, keepdim=True, unbiased=False).add(1e-5).sqrt()
    normal = (series - mean) / std
    return F.interpolate(normal.reshape(-1, 1, series.shape[-1]), size=size,
                         mode="linear", align_corners=False).reshape(
                             x.shape[0], x.shape[2], size)


class PairedMemoryBank(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, history_size: int = 64,
                 density_neighbors: int = 5):
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.history_size = int(history_size)
        self.density_neighbors = int(density_neighbors)
        self.register_buffer("keys", torch.empty(0))
        self.register_buffer("hidden", torch.empty(0))
        self.register_buffer("future", torch.empty(0))
        self.register_buffer("residual", torch.empty(0))
        self.register_buffer("future_raw", torch.empty(0))
        self.register_buffer("residual_raw", torch.empty(0))
        self.register_buffer("historical_std_floor", torch.empty(0))
        self.register_buffer("starts", torch.empty(0, dtype=torch.long))
        self.register_buffer("ends", torch.empty(0, dtype=torch.long))
        self.register_buffer("density", torch.empty(0))
        self.register_buffer("common", torch.empty(0, dtype=torch.bool))
        self.register_buffer("support", torch.empty(0))
        self.register_buffer("typical_future_disagreement", torch.tensor(1.))

    @torch.no_grad()
    def fit(self, records: dict) -> dict:
        """records come exclusively from TRAIN and have channel-matched rows."""
        x = records["x"].float().cpu()
        y = records["target"].float().cpu()
        a0 = records["a0_pred"].float().cpu()
        h = records["hidden"].float().cpu()
        start = records["starts"].long().cpu()
        if x.ndim != 3 or y.shape != (len(x), self.pred_len, x.shape[-1]):
            raise ValueError("paired TRAIN records have inconsistent shapes")
        if h.shape[:2] != (len(x), x.shape[-1]) or not torch.equal(
                start, torch.sort(start).values):
            raise ValueError("hidden layout or chronological TRAIN anchor order is wrong")
        mean = x.mean(1, keepdim=True)
        raw_std = x.var(1, keepdim=True, unbiased=False).add(1e-5).sqrt()
        # This floor is TRAIN-history-only. Near-constant windows otherwise
        # turn ordinary future changes into 1000x normalized inputs.
        floor = (raw_std.median(dim=0).values * .25).clamp_min(.01)
        std = raw_std.clamp_min(floor)
        self.historical_std_floor = floor.squeeze(0)
        self.keys = history_key(x, self.history_size)
        self.hidden = h
        self.future_raw = y.transpose(1,2).contiguous()
        self.residual_raw = (y-a0).transpose(1,2).contiguous()
        self.future = ((y - mean) / std).transpose(1, 2).contiguous()
        self.residual = ((y - a0) / std).transpose(1, 2).contiguous()
        # Robust TRAIN-only scale for all later query-local future MADs.
        future_center = self.future.median(dim=0).values
        self.typical_future_disagreement = (
            self.future - future_center[None]).abs().median().clamp_min(1e-4)
        self.starts = start
        self.ends = start + self.seq_len + self.pred_len
        a, c, _ = self.keys.shape
        if a < 3:
            raise ValueError("at least three TRAIN anchors required")
        density = torch.empty(a, c)
        support = torch.empty(a, c)
        temporal = (start[:, None] - start[None, :]).abs() >= self.seq_len + self.pred_len
        for channel in range(c):
            distance = torch.cdist(self.keys[:, channel], self.keys[:, channel], p=2).square() / self.history_size
            distance.masked_fill_(~temporal, float("inf"))
            k = min(self.density_neighbors, a - 1)
            knn = distance.topk(k, dim=-1, largest=False).values
            finite = torch.isfinite(knn)
            local = (knn.masked_fill(~finite, 0).sum(-1) / finite.sum(-1).clamp_min(1))
            density[:, channel] = 1. / (local + 1e-4)
            radius = torch.median(local[finite.any(-1)]) if bool(finite.any()) else torch.tensor(0.)
            support[:, channel] = ((distance <= radius) & temporal).sum(-1).float()
        self.density = density
        self.support = support
        self.common = density >= density.median(dim=0).values[None]
        common_future = self.future[self.common]
        rare_future = self.future[~self.common]
        common_mean = common_future.mean(0) if len(common_future) else torch.zeros(self.pred_len)
        rare_mean = rare_future.mean(0) if len(rare_future) else torch.zeros(self.pred_len)
        midpoint = torch.median(start)
        first = self.future[start <= midpoint].mean(0)
        second = self.future[start > midpoint].mean(0)
        return {"source": "TRAIN anchors only", "entries": a, "channels": c,
                "common_fraction": float(self.common.float().mean()),
                "density_mean": float(density.mean()),
                "historical_std_floor": self.historical_std_floor.flatten().tolist(),
                "future_input_above_20_fraction": float((self.future.abs()>20).float().mean()),
                "common_future_variance": float(common_future.var(unbiased=False)) if len(common_future) else 0.,
                "rare_future_variance": float(rare_future.var(unbiased=False)) if len(rare_future) else 0.,
                "common_rare_future_mean_gap": float((common_mean-rare_mean).square().mean()) if len(common_future) and len(rare_future) else 0.,
                "first_second_temporal_block_future_mean_gap": float((first-second).square().mean()),
                "low_support_fraction": float((support < 2).float().mean()),
                "first_start": int(start.min()), "last_complete_end": int(self.ends.max())}

    def _select(self, distance, eligible, count, mode, sample_ids):
        b, c, a = distance.shape
        if count == 0:
            empty_index = torch.empty(b, c, 0, dtype=torch.long, device=distance.device)
            empty_valid = torch.empty(b, c, 0, dtype=torch.bool, device=distance.device)
            return empty_index, empty_valid, distance.new_empty(b,c,0)
        # A deterministic tie breaker makes CONCAT slot semantics stable.
        order = distance + torch.arange(a, device=distance.device)[None,None] * 1e-8
        if mode == "random":
            if sample_ids is None:
                sample_ids = torch.arange(b, device=distance.device)
            source = torch.arange(a, device=distance.device)
            pseudo = ((source[None,None] * 1103515245 +
                       sample_ids.to(distance.device)[:,None,None] * 12345 +
                       torch.arange(c, device=distance.device)[None,:,None] * 7919 + 2021) % 2147483647)
            order = pseudo.float()
        ranked = order.masked_fill(~eligible, float("inf"))
        take = min(count, a)
        values, index = ranked.topk(take, dim=-1, largest=False, sorted=True)
        valid = torch.isfinite(values)
        true_distance = distance.gather(-1, index)
        if take < count:
            pad = count - take
            index = F.pad(index, (0,pad))
            valid = F.pad(valid, (0,pad), value=False)
            true_distance = F.pad(true_distance, (0,pad), value=0.)
        return index, valid, true_distance

    @torch.no_grad()
    def retrieve(self, x: torch.Tensor, view: str, common_k: int, rare_k: int,
                 sample_ids: torch.Tensor | None = None, split: str = "test",
                 val_first_start: int | None = None) -> dict:
        if self.keys.numel() == 0:
            raise RuntimeError("fit TRAIN memory before querying")
        if view not in {"all", "common", "rare", "mixed", "random"}:
            raise ValueError(view)
        query = history_key(x, self.history_size)
        key = self.keys.to(query)
        b, c, _ = query.shape
        a = len(self.starts)
        if c != key.shape[1]:
            raise ValueError("query and memory channel counts differ")
        distance = ((query.square().sum(-1)[:, :, None]
                     + key.square().sum(-1).T[None]
                     - 2 * torch.einsum("bcr,acr->bca", query, key))
                    / self.history_size).clamp_min(0.)
        ends = self.ends.to(query.device)
        if split == "train":
            if sample_ids is None:
                raise ValueError("TRAIN retrieval needs original chronological sample IDs")
            eligible_time = ends[None] <= sample_ids.to(query.device)[:,None]
        elif split == "val":
            if val_first_start is None:
                raise ValueError("VAL retrieval needs its first global historical start")
            eligible_time = (ends <= val_first_start)[None].expand(b,-1)
        elif split == "test":
            eligible_time = torch.ones(b,a,dtype=torch.bool,device=query.device)
        else:
            raise ValueError(split)
        eligible = eligible_time[:,None,:].expand(b,c,a)
        common = self.common.to(query.device).T[None].expand(b,-1,-1)
        total = common_k + rare_k
        if view == "mixed":
            parts = [self._select(distance, eligible & common, common_k, "nearest", sample_ids),
                     self._select(distance, eligible & ~common, rare_k, "nearest", sample_ids)]
        elif view == "common":
            parts = [self._select(distance, eligible & common, total, "nearest", sample_ids)]
        elif view == "rare":
            parts = [self._select(distance, eligible & ~common, total, "nearest", sample_ids)]
        else:
            parts = [self._select(distance, eligible, total,
                                  "random" if view == "random" else "nearest", sample_ids)]
        index = torch.cat([part[0] for part in parts], -1)
        valid = torch.cat([part[1] for part in parts], -1)
        selected_dist = torch.cat([part[2] for part in parts], -1)
        channel = torch.arange(c, device=query.device)[None,:,None]
        hidden = self.hidden.to(query.device)[index, channel]
        future = self.future.to(query.device)[index, channel]
        residual = self.residual.to(query.device)[index, channel]
        future_raw = self.future_raw.to(query.device)[index, channel]
        density = self.density.to(query.device)[index, channel]
        support = self.support.to(query.device)[index, channel]
        is_common = self.common.to(query.device)[index, channel] & valid
        hidden = hidden * valid[...,None,None]
        future = future * valid[...,None]
        residual = residual * valid[...,None]
        future_raw = future_raw * valid[...,None]
        return {"hidden": hidden, "future": future, "future_raw": future_raw,
                "residual": residual,
                "indices": index, "valid": valid, "distance": selected_dist,
                "density": density * valid, "support": support * valid,
                "is_common": is_common,
                "source_starts": self.starts.to(query.device)[index],
                "source_ends": ends[index],
                "eligible_count": eligible.sum(-1), "view": view,
                "split": split}
