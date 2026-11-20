"""Time-forward, event-balanced evidence for prototype forecasts."""
from __future__ import annotations

import torch


@torch.no_grad()
def prototype_evidence(base, candidate, truth, weight, starts, span,
                       segments: int = 8):
    """Return continuous shared stability and per-segment variation reliability.

    Inputs are all in the same history-normalized coordinate. Call this on a
    purged block not used to fit the bank, never on TEST labels for selection.
    """
    n, c, m, horizon = candidate.shape
    if base.shape != (n, c, horizon) or truth.shape != base.shape or weight.shape != (n, c, m):
        raise ValueError("prototype evidence shape mismatch")
    if len(starts) != n or span < 1:
        raise ValueError("time metadata mismatch")
    # One nonoverlapping event per block; channels still share an event.
    rows, last = [], -span
    for i, start in enumerate(starts.tolist()):
        if start - last >= span:
            rows.append(i)
            last = start
    if len(rows) < 2:
        return torch.zeros(m), torch.zeros(m, segments), {
            "status": "insufficient_independent_events", "event_count": len(rows)}
    base, candidate, truth, weight = (item[rows] for item in (base, candidate, truth, weight))
    improvement = (base[:, :, None] - truth[:, :, None]).square() - (candidate - truth[:, :, None]).square()
    reliability = torch.zeros(m, segments)
    audit = []
    for j in range(m):
        w = weight[:, :, j]
        mass = w.sum()
        effective = float(mass.square() / w.square().sum().clamp_min(1e-8))
        for s in range(segments):
            lo, hi = s * horizon // segments, (s + 1) * horizon // segments
            gains = improvement[:, :, j, lo:hi].mean(-1)
            mean = float((w * gains).sum() / mass.clamp_min(1e-8))
            deviation = float(((w * (gains - mean).square()).sum() / mass.clamp_min(1e-8)).sqrt())
            support = max(0., (effective - 1) / (effective + 4))
            score = max(0., mean) / (abs(mean) + deviation + 1e-8) * support
            reliability[j, s] = score
            audit.append({"prototype": j, "segment": s, "gain": mean,
                          "gain_std": deviation, "effective_channel_support": effective,
                          "reliability": score})
    # Stability is a conservative summary, not a predetermined stable class.
    stability = reliability.mean(-1)
    return stability, reliability, {"status": "evaluated", "event_count": len(rows),
                                    "segments": audit, "source": "purged later TRAIN; no TEST labels"}
