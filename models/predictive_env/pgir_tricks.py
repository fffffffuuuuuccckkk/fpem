"""Independent PGIR patch-replacement ablations; original P2 is unchanged."""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from models.predictive_env.pgir import PGIRMask


def local_context_prototype(x, bank, segment_len: int):
    """Match each segment using observed adjacent segments, never its own pixels."""
    b, length, channels = x.shape
    nseg = (length + segment_len - 1) // segment_len
    past = x.transpose(1, 2)
    candidates = F.interpolate(bank.U[:, None].to(x.dtype), size=length,
                               mode="linear", align_corners=False)[:, 0]
    patches = []
    weights = []
    for j in range(nseg):
        left = list(range(max(0, (j-1)*segment_len), j*segment_len))
        right = list(range(min(length, (j+1)*segment_len),
                           min(length, (j+2)*segment_len)))
        context = left + right
        start, end = j*segment_len, min(length, (j+1)*segment_len)
        if not context:
            # With a one-segment history there is no independent matching
            # context, so local replacement must be an identity operation.
            weight = (bank.counts/bank.counts.sum().clamp_min(1)).expand(b,channels,-1)
            patches.append(past[..., start:end])
            weights.append(weight)
            continue
        observed_context = past[..., context]
        candidate_context = candidates[:, context]
        observed_mean = observed_context.mean(-1, keepdim=True)
        observed_std = observed_context.std(-1, unbiased=False, keepdim=True).clamp_min(1e-4)
        candidate_mean = candidate_context.mean(-1, keepdim=True)
        candidate_std = candidate_context.std(-1, unbiased=False, keepdim=True).clamp_min(1e-4)
        observed_shape = (observed_context-observed_mean)/observed_std
        candidate_shape = (candidate_context-candidate_mean)/candidate_std
        distance = (observed_shape[..., None]-candidate_shape.T[None,None]).square().mean(-2)
        weight = torch.softmax(-distance/bank.temperature, dim=-1)
        normalized_patch = (candidates[:,start:end]-candidate_mean)/candidate_std
        patch = torch.einsum("bck,kt->bct",weight,normalized_patch)
        patches.append(patch*observed_std+observed_mean)
        weights.append(weight)
    source = torch.cat(patches, -1).transpose(1, 2)
    return source, torch.stack(weights, 2)


def align_patch_boundaries(source, observed, segment_len: int):
    """Affine endpoint correction using only neighboring historical values."""
    b, length, channels = source.shape
    aligned = source.clone()
    count = (length + segment_len - 1)//segment_len
    for j in range(count):
        start, end = j*segment_len, min(length, (j+1)*segment_len)
        segment = source[:, start:end]
        if start and end < length:
            left = observed[:, start-1] - segment[:, 0]
            right = observed[:, end] - segment[:, -1]
            t = torch.linspace(0, 1, end-start, device=source.device, dtype=source.dtype)
            adjustment = (1-t[None, :, None])*left[:, None] + t[None, :, None]*right[:, None]
        elif start:
            adjustment = (observed[:, start-1]-segment[:, 0])[:, None]
        elif end < length:
            adjustment = (observed[:, end]-segment[:, -1])[:, None]
        else:
            adjustment = torch.zeros_like(segment)
        aligned[:, start:end] = segment + adjustment
    return aligned


class PGIRTrickMask(PGIRMask):
    def __init__(self, segment_len=16, max_replace=2, recent_penalty=0.25,
                 output_bias=0.1, normalize_features=False):
        super().__init__(segment_len)
        self.max_replace = int(max_replace)
        self.recent_penalty = float(recent_penalty)
        self.decision_threshold = 0.0
        self.normalize_features = bool(normalize_features)
        torch.nn.init.constant_(self.net[-1].bias, float(output_bias))
        if self.normalize_features:
            self.register_buffer("feature_mean", torch.zeros(5))
            self.register_buffer("feature_std", torch.ones(5))
            self.register_buffer("feature_stats_fitted", torch.tensor(False))

    @torch.no_grad()
    def fit_feature_normalizer(self, bank, train_dataset, device, batch_size=64):
        """Fit five input statistics on independent TRAIN anchors only."""
        if not self.normalize_features:
            return None
        if int(getattr(train_dataset, "set_type", -1)) != 0:
            raise ValueError("PGIR feature normalization must use TRAIN only")
        starts = np.arange(0, len(train_dataset), int(train_dataset.seq_len))
        data = np.asarray(train_dataset.data_x, dtype=np.float32)
        count = 0
        total = torch.zeros(5, device=device, dtype=torch.float64)
        squares = torch.zeros_like(total)
        for lo in range(0, len(starts), batch_size):
            hist = np.stack([data[int(s):int(s)+train_dataset.seq_len]
                             for s in starts[lo:lo+batch_size]])
            x = torch.from_numpy(hist).to(device)
            query = bank.query(x)
            features, _, _ = self._features(x, query["x_proto"], query, bank)
            values = features.reshape(-1, 5).double()
            total += values.sum(0)
            squares += values.square().sum(0)
            count += len(values)
        if not count:
            raise ValueError("empty TRAIN anchors for feature normalization")
        mean = total/count
        std = (squares/count-mean.square()).clamp_min(0).sqrt().clamp_min(1e-4)
        self.feature_mean.copy_(mean.to(self.feature_mean))
        self.feature_std.copy_(std.to(self.feature_std))
        self.feature_stats_fitted.fill_(True)
        return {"count": count, "mean": mean.cpu().tolist(), "std": std.cpu().tolist()}

    def prepare_features(self, features):
        if not self.normalize_features:
            return features
        if not bool(self.feature_stats_fitted):
            raise RuntimeError("fit TRAIN-only feature normalization before query")
        return (features-self.feature_mean)/self.feature_std

    def forward(self, x, bank, *, local=False, boundary=False, recent=False):
        if not torch.isfinite(self.invariant_threshold):
            raise RuntimeError("fit TRAIN-only threshold before query")
        with torch.no_grad():
            query = bank.query(x)
            features, score, residual_energy = self._features(
                x, query["x_proto"], query, bank)
            invariant = (score >= self.invariant_threshold).to(x.dtype)
        b, length, channels = x.shape
        nseg = invariant.shape[-1]
        net_features = self.prepare_features(features)
        logits = self.net(net_features.reshape(b*channels,nseg,5).transpose(1,2))
        logits = logits[:,0].reshape(b,channels,nseg)
        decision_score = logits
        if recent:
            distance_from_end = nseg - 1 - torch.arange(nseg, device=x.device)
            surcharge = (2-distance_from_end).clamp_min(0).to(x.dtype)*self.recent_penalty
            decision_score = logits + surcharge[None,None]
        eligible = invariant == 0
        candidates = torch.where(eligible & (decision_score < self.decision_threshold),
                                 decision_score, torch.full_like(decision_score, float("inf")))
        selected = candidates.topk(min(self.max_replace,nseg), dim=-1, largest=False).indices
        chosen = torch.zeros_like(logits, dtype=torch.bool)
        chosen.scatter_(-1, selected, True)
        replace = chosen & torch.isfinite(candidates)
        keep_hard = (~replace).to(x.dtype)
        keep_soft = logits.sigmoid()
        mask = keep_hard.detach() - keep_soft.detach() + keep_soft
        # Invariant segments must remain untouched even in backward pass.
        mask = invariant + (1-invariant)*mask
        if local:
            source, local_weight = local_context_prototype(x,bank,self.segment_len)
        else:
            source, local_weight = query["x_proto"], None
        if boundary:
            source = align_patch_boundaries(source,x,self.segment_len)
        time_mask = mask.repeat_interleave(self.segment_len,-1)[...,:length].transpose(1,2)
        selected_x = time_mask*x + (1-time_mask)*source
        info = {"mask":mask,"invariant":invariant,"probability":keep_soft,
                "logits":logits,"decision_score":decision_score,"score":score,
                "features":features,"residual_energy":residual_energy,"query":query,
                "source":source,"local_weight":local_weight,
                "replace_count":replace.sum(-1),
                "replacement_fraction":replace.float().mean(),
                "variation_keep_fraction":((1-invariant)*keep_hard).sum()/(1-invariant).sum().clamp_min(1)}
        return selected_x,info
