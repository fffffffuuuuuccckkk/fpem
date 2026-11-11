"""Paired history/future prototypes with a shared frozen-A0 feature adapter.

The bank is fitted without an R3 loss. Inference takes only observed history
keys and the frozen A0 representation; future targets never enter query().
"""
from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


def _weighted_quantile(values: torch.Tensor, weights: torch.Tensor, q: float):
    ordered, positions = values.sort(0)
    mass = weights[:, None].expand_as(values).gather(0, positions)
    cutoff = (mass.cumsum(0) >= q * mass.sum(0).clamp_min(1e-8)).float().argmax(0)
    return ordered.gather(0, cutoff[None]).squeeze(0)


def _sparsemax(logits: torch.Tensor):
    shifted = logits - logits.max(-1, keepdim=True).values
    ordered = shifted.sort(-1, descending=True).values
    cssv = ordered.cumsum(-1) - 1
    k = torch.arange(1, ordered.shape[-1] + 1, device=ordered.device)
    support = ordered > cssv / k
    kmax = support.sum(-1).clamp_min(1)
    tau = cssv.gather(-1, (kmax - 1)[..., None]).squeeze(-1) / kmax
    return (shifted - tau[..., None]).clamp_min(0)


class PairedPrototypeBank(nn.Module):
    """M shared prototypes queried per channel, without a TRAIN-row scan."""

    def __init__(self, patterns: int, tokens: int, width: int, horizon: int,
                 a0_weight: torch.Tensor, a0_bias: torch.Tensor, rank: int = 8,
                 temperature: float = 0.5):
        super().__init__()
        if patterns < 1 or rank < 1 or temperature <= 0:
            raise ValueError("invalid prototype size, rank or temperature")
        if tuple(a0_weight.shape) != (horizon, tokens * width):
            raise ValueError("A0 head shape is incompatible with hidden representation")
        self.patterns, self.tokens = patterns, tokens
        self.width, self.horizon, self.rank = width, horizon, rank
        self.temperature = float(temperature)
        self.centers = nn.Parameter(torch.zeros(patterns, 64))
        self.router = nn.Linear(64, patterns, bias=False)
        nn.init.zeros_(self.router.weight)
        self.adapter_down = nn.Linear(3 * width, rank)
        self.adapter_up = nn.Linear(rank, width, bias=False)
        nn.init.zeros_(self.adapter_up.weight)
        self.register_buffer("a0_weight", a0_weight.detach().float().clone())
        self.register_buffer("a0_bias", a0_bias.detach().float().clone())
        self.register_buffer("hidden_centers", torch.zeros(patterns, tokens, width))
        self.register_buffer("future_centers", torch.zeros(patterns, horizon))
        self.register_buffer("event_support", torch.zeros(patterns))
        self.register_buffer("block_support", torch.zeros(patterns, 4))
        self.register_buffer("medoid_ids", torch.full((patterns,), -1, dtype=torch.long))
        self.register_buffer("stability", torch.zeros(patterns))
        self.fitted = False

    def distance(self, key: torch.Tensor):
        if key.ndim != 3 or key.shape[-1] != 64:
            raise ValueError("history key must be [B,C,64]")
        return (key[:, :, None] - self.centers[None, None]).square().mean(-1)

    def route(self, key: torch.Tensor):
        distance = self.distance(key)
        score = -distance / self.temperature + self.router(key)
        return _sparsemax(score), distance

    def query(self, key: torch.Tensor, hidden: torch.Tensor,
              normalization: dict | None = None):
        if hidden.ndim != 4 or hidden.shape[:2] != key.shape[:2] or hidden.shape[-2:] != (self.tokens, self.width):
            raise ValueError("hidden must be [B,C,P,D] and match history key")
        weight, distance = self.route(key)
        b, c = key.shape[:2]
        h = hidden[:, :, None].expand(b, c, self.patterns, self.tokens, self.width)
        prototype = self.hidden_centers[None, None].expand_as(h)
        features = torch.cat((h, prototype, h - prototype), -1)
        delta = self.adapter_up(F.gelu(self.adapter_down(features)))
        modified = h + delta
        design = modified.permute(0, 1, 2, 4, 3).flatten(-2)
        candidate = F.linear(design, self.a0_weight, self.a0_bias)
        if normalization is not None:
            # The frozen A0 head uses its own RevIN coordinate. Both labels
            # and all downstream corrections use history-only coordinates.
            required = ("model_mean", "model_std", "hist_mean", "hist_scale")
            if any(name not in normalization for name in required):
                raise ValueError("incomplete history/A0 normalization")
            mm = normalization["model_mean"].transpose(1, 2)[:, :, None]
            ms = normalization["model_std"].transpose(1, 2)[:, :, None]
            hm = normalization["hist_mean"].transpose(1, 2)[:, :, None]
            hs = normalization["hist_scale"].transpose(1, 2)[:, :, None]
            candidate = (candidate * ms + mm - hm) / hs.clamp_min(1e-4)
        return {"weight": weight, "distance": distance,
                "candidate": candidate, "adapter_delta": delta,
                "prediction": (weight[..., None] * candidate).sum(2),
                "nearest_distance": distance.min(-1).values}

    predict_candidates = query

    @torch.no_grad()
    def initialize(self, key: torch.Tensor, hidden: torch.Tensor,
                   future: torch.Tensor, starts: torch.Tensor, span: int,
                   strategy: str = "coverage"):
        """Initialize from independent TRAIN events, never from future labels."""
        if future.shape[:2] != key.shape[:2] or future.shape[-1] != self.horizon:
            raise ValueError("TRAIN future shape mismatch")
        if len(starts) != len(key) or span < 1:
            raise ValueError("TRAIN event metadata mismatch")
        chosen = []
        last = -span
        for i, start in enumerate(starts.tolist()):
            if start - last >= span:
                chosen.append(i)
                last = start
        if not chosen:
            raise ValueError("no independent TRAIN events")
        pool = key[chosen].reshape(-1, 64).float()
        if strategy not in {"coverage", "spread"}:
            raise ValueError(strategy)
        if len(pool) < self.patterns:
            raise ValueError("fewer independent event-channel anchors than prototypes")
        if strategy == "spread":
            selected = torch.linspace(0, len(pool) - 1, self.patterns).long().tolist()
        else:
            selected = [0]
            for _ in range(1, self.patterns):
                d = torch.cdist(pool, pool[selected]).square().min(-1).values
                d[selected] = -1
                selected.append(int(d.argmax()))
        self.centers.copy_(pool[selected].to(self.centers))
        self.update_metadata(key, hidden, future, starts, span)
        self.fitted = True
        return {"independent_events": len(chosen), "initial_rows": len(pool),
                "strategy": strategy}

    def fit(self, key: torch.Tensor, hidden: torch.Tensor, future: torch.Tensor,
            starts: torch.Tensor, span: int, *, epochs: int = 10,
            learning_rate: float = 3e-4, strategy: str = "coverage",
            normalization: dict | None = None, lambda_cover: float = 0.2,
            lambda_future: float = 1.0, lambda_div: float = 0.01,
            lambda_adapter: float = 0.01):
        """Fit from TRAIN tensors only; external runner handles time-forward selection.

        The first two epochs train history coverage without future supervision.
        This method is an explicit standalone fit interface. The experiment
        runner additionally performs purged later-TRAIN checkpoint selection.
        """
        if epochs < 3 or len(key) == 0:
            raise ValueError("fit needs at least three epochs and nonempty TRAIN data")
        audit = self.initialize(key, hidden, future, starts, span, strategy)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        trace = []
        for epoch in range(epochs):
            self.train()
            loss, terms = self.losses(
                key, hidden, future, normalization,
                use_future=epoch >= 2, lambda_cover=lambda_cover,
                lambda_div=lambda_div, lambda_adapter=lambda_adapter,
                lambda_future=lambda_future)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite prototype fitting loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.)
            optimizer.step()
            trace.append({name: float(value.detach()) for name, value in terms.items()})
        self.update_metadata(key, hidden, future, starts, span)
        self.eval()
        return {"initialization": audit, "train_trace": trace,
                "selection": "none; use a purged later TRAIN block externally"}

    @torch.no_grad()
    def update_metadata(self, key, hidden, future, starts, span):
        q, _ = self.route(key.to(self.centers.device))
        q = q.detach().cpu().reshape(-1, self.patterns)
        h = hidden.detach().cpu().reshape(-1, self.tokens, self.width)
        y = future.detach().cpu().reshape(-1, self.horizon)
        flat_key = key.detach().cpu().reshape(-1, 64)
        unique = torch.div(starts.cpu().long(), span, rounding_mode="floor")
        event = unique[:, None].expand(-1, key.shape[1]).reshape(-1)
        blocks = torch.div(starts.cpu().long() * 4, max(int(starts.max()) + 1, 1), rounding_mode="floor").clamp(0, 3)
        blocks = blocks[:, None].expand(-1, key.shape[1]).reshape(-1)
        for j in range(self.patterns):
            w = q[:, j]
            if float(w.sum()) <= 1e-8:
                continue
            self.hidden_centers[j].copy_(((w[:, None, None] * h).sum(0) / w.sum()).to(self.hidden_centers))
            self.future_centers[j].copy_(_weighted_quantile(y, w, 0.5).to(self.future_centers))
            active_events = torch.unique(event[w > 1e-4])
            self.event_support[j] = len(active_events)
            for block in range(4):
                self.block_support[j, block] = torch.unique(event[(w > 1e-4) & (blocks == block)]).numel()
            distance = (flat_key - self.centers[j].detach().cpu()).square().mean(-1)
            self.medoid_ids[j] = int(starts[distance.argmin() // key.shape[1]])

    def losses(self, key, hidden, future, normalization=None, *, use_future: bool,
               lambda_cover: float, lambda_div: float, lambda_adapter: float,
               lambda_future: float = 1.0):
        result = self.query(key, hidden, normalization)
        q, distance, candidate = result["weight"], result["distance"], result["candidate"]
        hist = (q * distance).sum(-1).mean()
        nearest = distance.min(-1).values.flatten()
        top = max(1, math.ceil(0.1 * len(nearest)))
        cover = nearest.topk(top).values.mean()
        mse = (candidate - future[:, :, None]).square().mean(-1)
        future_loss = (q * mse).sum(-1).mean() if use_future else hist.new_zeros(())
        if self.patterns > 1:
            dx = torch.pdist(self.centers, p=2).square() / 64
            dy = torch.pdist(self.future_centers.detach().to(dx), p=2).square() / self.horizon
            div = torch.exp(-dx / .1 - dy / .1).mean()
        else:
            div = hist.new_zeros(())
        adapter = result["adapter_delta"].square().mean()
        total = (hist + lambda_cover * cover + lambda_future * future_loss
                 + lambda_div * div + lambda_adapter * adapter)
        return total, {"hist": hist, "cover": cover, "future": future_loss,
                       "div": div, "adapter": adapter}

    def export(self):
        if not self.fitted:
            raise RuntimeError("cannot export an unfitted bank")
        return {"config": {"patterns": self.patterns, "tokens": self.tokens,
                           "width": self.width, "horizon": self.horizon,
                           "rank": self.rank, "temperature": self.temperature},
                "state": {k: v.detach().cpu().clone() for k, v in self.state_dict().items()},
                "fitted": True}

    @classmethod
    def from_export(cls, payload):
        cfg = payload["config"]
        state = payload["state"]
        bank = cls(**cfg, a0_weight=state["a0_weight"], a0_bias=state["a0_bias"])
        bank.load_state_dict(state)
        bank.fitted = bool(payload["fitted"])
        return bank
