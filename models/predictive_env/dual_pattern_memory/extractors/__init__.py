"""TRAIN-only, fixed-capacity DPPM prototype extraction alternatives.

All downstream banks retain the original entmax matcher and weighted Ridge map.
The extractor changes only matching keys and prototype centers.  No test target
is accepted by any query method.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from sklearn.cluster import Birch, MiniBatchKMeans


def _flat(key: torch.Tensor):
    return key.detach().cpu().float().reshape(-1, key.shape[-1]).numpy()


def _event_ids(starts: torch.Tensor, channels: int, span: int):
    # Windows with overlapping labels count once per chronological event.
    event = (starts.long() // max(1, span)).repeat_interleave(channels)
    return event.cpu().numpy()


def _candidate_pool(x, blocks, seed, count=64, birch=False):
    count = min(count, len(x))
    if birch:
        # A bounded streaming microcluster pass, then exactly 64 candidates.
        stream = Birch(threshold=.8, branching_factor=32, n_clusters=None)
        for start in range(0, len(x), 256):
            stream.partial_fit(x[start:start+256])
        micro = np.asarray(stream.subcluster_centers_, dtype=np.float32)
        if len(micro) < count:
            micro = np.concatenate((micro, x[:count-len(micro)]), 0)
        if len(micro) > count:
            km = MiniBatchKMeans(n_clusters=count, random_state=seed,
                                 n_init=3, batch_size=min(256, len(micro)))
            km.fit(micro)
            centers = km.cluster_centers_.astype(np.float32)
        else:
            centers = micro[:count]
    else:
        mass = np.bincount(blocks, minlength=5).clip(min=1)
        weights = len(blocks) / (5 * mass[blocks])
        km = MiniBatchKMeans(n_clusters=count, random_state=seed,
                             batch_size=min(256, len(x)), n_init=3,
                             reassignment_ratio=.01)
        km.fit(x, sample_weight=weights)
        centers = km.cluster_centers_.astype(np.float32)
    distance = ((x[:, None] - centers[None]) ** 2).mean(-1)
    labels = distance.argmin(-1)
    return centers, labels, distance.min(-1)


def _support(labels, events, blocks, count):
    independent = np.zeros(count, dtype=np.float64)
    coverage = np.zeros(count, dtype=np.float64)
    raw = np.bincount(labels, minlength=count)
    for j in range(count):
        take = labels == j
        independent[j] = len(np.unique(events[take]))
        coverage[j] = len(np.unique(blocks[take]))
    return raw, independent, coverage


def _compress(candidates, chosen, count, seed, weights=None):
    chosen = np.asarray(chosen, dtype=np.int64)
    if len(chosen) == 0:
        chosen = np.arange(len(candidates))
    source = candidates[chosen]
    if len(source) < count:
        source = np.concatenate((source, np.repeat(source[-1:], count-len(source), 0)))
    if len(source) == count:
        return torch.from_numpy(source.copy()).float()
    km = MiniBatchKMeans(n_clusters=count, random_state=seed, n_init=3,
                         batch_size=min(256, len(source)))
    w = None if weights is None else np.maximum(np.asarray(weights)[chosen], 1e-6)
    km.fit(source, sample_weight=w)
    return torch.from_numpy(km.cluster_centers_.astype(np.float32))


def _local_event_key(key):
    """Fixed-size, history-only local descriptors; no learned event encoder."""
    x = key.float()
    quarters = x.reshape(*x.shape[:-1], 4, 16)
    mean = quarters.mean(-1)
    slope = quarters[..., -1] - quarters[..., 0]
    volatility = quarters.diff(dim=-1).abs().mean(-1)
    peak = quarters.amax(-1) - quarters.amin(-1)
    descriptors = torch.cat((mean, slope, volatility, peak), -1)
    # Repeat locally relevant descriptors to retain the original 64-D matcher.
    return descriptors.repeat_interleave(4, dim=-1).clamp(-5, 5)


def _ridge(x, y, ridge=1.):
    x = torch.cat((x.double(), torch.ones(len(x), 1, dtype=torch.float64)), -1)
    y = y.double()
    eye = torch.eye(x.shape[1], dtype=torch.float64)
    eye[-1, -1] = 1e-4
    return torch.linalg.solve(x.T @ x / len(x) + ridge * eye,
                              x.T @ y / len(x)).float()


def _signature_predict(x, weight):
    design = torch.cat((x.float(), x.new_ones(len(x), 1)), -1)
    return design @ weight


class PatternExtractor:
    """Uniform extraction/matching adapter; P0 preserves the old path exactly."""

    def __init__(self, mode, ms=16, mr=8, seed=2021, span=816):
        if mode not in {f"P{i}" for i in range(9)}:
            raise ValueError(mode)
        self.mode, self.ms, self.mr = mode, ms, mr
        self.seed, self.span = seed, span
        self.stable_centers = self.rare_centers = None
        self.signature_weight = None
        self.train_key_cache = None
        self.train_mask = None
        self.stats = {}

    def stable_key(self, key):
        if self.mode != "P4":
            return key
        if self.signature_weight is None:
            raise RuntimeError("P4 requires fitted signature predictor")
        raw = key.reshape(-1, 64).cpu().float()
        signature = _signature_predict(raw, self.signature_weight).reshape(*key.shape[:-1], 8)
        # Same observable transform on TRAIN and inference; 64-D output.
        return (key.float() + .25 * signature.repeat_interleave(8, -1)).clamp(-5, 5)

    def rare_key(self, key, stable_query):
        if self.mode == "P0":
            return (key - stable_query["matched_history"]).clamp(-5, 5)
        if self.mode == "P6":
            return _local_event_key(key)
        return self.stable_key(key)

    def train_stable_key(self, key):
        if self.mode == "P4" and self.train_key_cache is not None:
            if key.shape != self.train_key_cache.shape:
                raise ValueError("P4 OOF key is only defined for the fitted TRAIN prefix")
            return self.train_key_cache
        return self.stable_key(key)

    def fit(self, key, future, base, blocks, starts, a0_oof=None):
        begun = time.monotonic()
        n, c, d = key.shape
        if d != 64:
            raise ValueError("DPPM history key must be 64-D")
        blocks_flat = blocks.repeat_interleave(c).cpu().numpy()
        events = _event_ids(starts, c, self.span)
        raw_key = key.cpu().float()
        if self.mode == "P0":
            self.stats = {"mode": "P0", "extraction": "unchanged original MiniBatchKMeans", "elapsed_s": 0.}
            return self
        if self.mode == "P4":
            # Chronological OOF forecast signature; no target enters query.
            yy = future.reshape(n*c, -1).float()
            basis = torch.linalg.svd(yy[blocks.repeat_interleave(c) < 2],
                                     full_matrices=False).Vh[:8].T.float()
            low_y = yy @ basis
            flat = raw_key.reshape(n*c, 64)
            oof = torch.zeros_like(low_y)
            valid = torch.zeros(len(flat), dtype=torch.bool)
            for fold in (1, 2, 3, 4):
                train = blocks_flat < fold
                held = blocks_flat == fold
                if train.sum() < 8 or not held.any():
                    continue
                oof[held] = _signature_predict(flat[held], _ridge(flat[train], low_y[train]))
                valid[held] = True
            if not valid.any():
                raise ValueError("P4 needs at least one chronological OOF signature block")
            self.signature_weight = _ridge(flat, low_y)
            self.train_key_cache = (raw_key + .25 * oof.reshape(n,c,8).repeat_interleave(8,-1)).clamp(-5,5)
            self.train_mask = valid.reshape(n,c).all(-1)
            self.stats["signature_oof_coverage"] = float(valid.float().mean())
            self.stats["signature_train_protocol"] = "chronological OOF; first block excluded"
        stable_key = self.train_stable_key(raw_key)
        if self.train_mask is not None:
            x = _flat(stable_key[self.train_mask])
            blocks_flat = blocks[self.train_mask].repeat_interleave(c).cpu().numpy()
            events = _event_ids(starts[self.train_mask], c, self.span)
        else:
            x = _flat(stable_key)
        candidates, labels, distances = _candidate_pool(x, blocks_flat, self.seed,
                                                        birch=self.mode == "P8")
        raw, independent, coverage = _support(labels, events, blocks_flat, len(candidates))
        frequency = independent / max(independent.sum(), 1)
        valid = (independent >= 2) & (coverage >= 2)
        common_order = np.lexsort((-coverage, -independent))[::-1]
        common_order = np.argsort(-(independent * (1 + coverage / 5)))
        common = common_order[:max(self.ms, min(32, len(candidates)//2))]
        rare_order = np.argsort(independent + .1 * coverage)
        rare = np.asarray([j for j in rare_order if valid[j] and j not in set(common)], dtype=int)
        if len(rare) < self.mr:
            rare = np.asarray([j for j in rare_order if independent[j] >= 2 and j not in set(common)], dtype=int)
        if len(rare) < self.mr:
            rare = rare_order[:self.mr]
        if self.mode == "P2":
            # Low-coverage independent events, clustered in full u (not u-minus-center).
            stable_seed = _compress(candidates, common, self.ms, self.seed, independent)
            uncovered = np.partition(((x[:, None]-stable_seed.numpy()[None])**2).mean(-1),
                                     0, axis=1)[:, 0]
            rare_pool = uncovered >= np.quantile(uncovered, .70)
            valid_event = np.isin(events, np.unique(events[rare_pool]))
            pool = x[rare_pool & valid_event]
            if len(pool) >= self.mr:
                rk = MiniBatchKMeans(n_clusters=self.mr, random_state=self.seed+11,
                                     n_init=3, batch_size=min(256, len(pool)))
                rk.fit(pool)
                self.rare_centers = torch.from_numpy(rk.cluster_centers_.astype(np.float32))
            self.stable_centers = stable_seed
        elif self.mode == "P6":
            self.stable_centers = _compress(candidates, common, self.ms, self.seed, independent)
            event_key = _flat(_local_event_key(raw_key))
            ec, el, _ = _candidate_pool(event_key, blocks_flat, self.seed+11)
            _, ei, ecov = _support(el, events, blocks_flat, len(ec))
            ev = np.where((ei >= 2) & (ecov >= 2))[0]
            ev = ev[np.argsort(ei[ev])] if len(ev) else np.argsort(ei)
            self.rare_centers = _compress(ec, ev[:max(self.mr, len(ev)//2)],
                                          self.mr, self.seed+11, ei)
        else:
            if self.mode in ("P3", "P5", "P7"):
                # P7 rates future-map transfer on chronological held-out blocks.
                # P3 adds at most two supervised center assignment passes.
                future_flat = future.reshape(n*c, -1).float()
                if self.mode == "P5":
                    if a0_oof is None or a0_oof.shape != future.shape:
                        raise ValueError("P5 requires row-aligned chronological A0 OOF [N,C,T]")
                    if not torch.isfinite(a0_oof[blocks >= 2]).all():
                        raise ValueError("P5 A0 OOF missing held-out blocks 2-4")
                base_flat = (a0_oof if self.mode == "P5" else base).reshape(n*c, -1).float()
                rank = min(8, future_flat.shape[0], future_flat.shape[1])
                basis = torch.linalg.svd(future_flat[blocks_flat < 2],
                                         full_matrices=False).Vh[:rank].T.float()
                target_low = future_flat @ basis
                if self.mode == "P3":
                    xx = torch.from_numpy(x)
                    cc = torch.from_numpy(candidates)
                    labels_t = torch.from_numpy(labels)
                    for _ in range(2):
                        maps = []
                        for j in range(len(candidates)):
                            take = labels_t == j
                            if take.sum() < 3:
                                maps.append(_ridge(xx, target_low, 2.))
                            else:
                                maps.append(_ridge(xx[take], target_low[take], 2.))
                        design = torch.cat((xx, torch.ones(len(xx), 1)), -1)
                        predicted = torch.einsum("nf,kfo->nko", design, torch.stack(maps))
                        target_error = (predicted-target_low[:, None]).square().mean(-1)
                        history_error = (xx[:, None]-cc[None]).square().mean(-1)
                        # The future term selects TRAIN members only. Query uses X distances.
                        labels_t = (history_error + .05*target_error).argmin(-1)
                        for j in range(len(candidates)):
                            take = labels_t == j
                            if take.any():
                                cc[j] = xx[take].mean(0)
                    candidates = cc.numpy()
                    labels = labels_t.numpy()
                    raw, independent, coverage = _support(labels, events, blocks_flat, len(candidates))
                    common = np.argsort(-(independent*(1+coverage/5)))[:len(common)]
                    rare = np.asarray([j for j in np.argsort(independent)
                                       if independent[j] >= 2 and coverage[j] >= 2 and j not in set(common)])
                    self.stats["train_x_assignment_agreement"] = float((labels ==
                        ((x[:,None]-candidates[None])**2).mean(-1).argmin(-1)).mean())
                if self.mode in ("P5", "P7"):
                    score_sum = np.zeros(len(candidates))
                    score_count = np.zeros(len(candidates))
                    for fold in (2, 3, 4):
                        tr, te = blocks_flat < fold, blocks_flat == fold
                        if tr.sum() < 65 or not te.any():
                            continue
                        global_map = _ridge(torch.from_numpy(x[tr]), target_low[tr], 1.)
                        for j in range(len(candidates)):
                            member = tr & (labels == j)
                            eval_member = te & (labels == j)
                            if member.sum() < 3 or not eval_member.any():
                                continue
                            local = _ridge(torch.from_numpy(x[member]), target_low[member], 1.)
                            test_key = torch.from_numpy(x[eval_member])
                            local_err = (_signature_predict(test_key, local)-target_low[eval_member]).square().mean(-1)
                            global_err = (_signature_predict(test_key, global_map)-target_low[eval_member]).square().mean(-1)
                            # Also compare to a zero-correction A0 reference.
                            base_err = ((base_flat[eval_member] @ basis)-target_low[eval_member]).square().mean(-1)
                            gain = torch.minimum(global_err, base_err)-local_err
                            score_sum[j] += float(gain.sum())
                            score_count[j] += len(gain)
                    quality = score_sum / np.maximum(score_count, 1)
                    self.stats["candidate_oof_gain"] = quality.tolist()
                    common = np.asarray(sorted(common, key=lambda j: quality[j], reverse=True))
                    if self.mode == "P5":
                        weakness = np.zeros(len(candidates))
                        for j in range(len(candidates)):
                            held = (blocks_flat >= 2) & (labels == j)
                            if held.any():
                                weakness[j] = float((base_flat[held]-future_flat[held]).square().mean())
                        self.stats["candidate_oof_backbone_error"] = weakness.tolist()
                        rare = np.asarray(sorted(rare, key=lambda j: (quality[j] > 0,
                            quality[j], weakness[j]), reverse=True))
                    else:
                        rare = np.asarray(sorted(rare, key=lambda j: quality[j], reverse=True))
            self.stable_centers = _compress(candidates, common[:max(self.ms, 2*self.ms)],
                                            self.ms, self.seed, independent)
            self.rare_centers = _compress(candidates, rare[:max(self.mr, 2*self.mr)],
                                          self.mr, self.seed+11, independent)
        if self.rare_centers is None:
            self.rare_centers = _compress(candidates, rare, self.mr, self.seed+11, independent)
        self.stats.update({"mode": self.mode, "pairs": len(x),
            "candidate_count": len(candidates), "candidate_raw_count": raw.tolist(),
            "candidate_independent_events": independent.tolist(),
            "candidate_block_coverage": coverage.tolist(),
            "candidate_frequency": frequency.tolist(),
            "selected_common": common[:max(self.ms, 2*self.ms)].tolist(),
            "selected_rare": rare[:max(self.mr, 2*self.mr)].tolist(),
            "independent_event_total": int(len(np.unique(events))),
            "build_seconds": time.monotonic()-begun})
        return self

    def export_prototypes(self):
        return self.stable_centers, self.rare_centers, self.stats
