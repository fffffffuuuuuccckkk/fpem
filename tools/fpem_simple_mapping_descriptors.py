"""Small, per-window history-to-future relation descriptors (TRAIN labels only)."""
from __future__ import annotations

import itertools
import numpy as np

NAMES = tuple(f"M{i}" for i in range(1, 9))
DIMS = (4, 4, 4, 3, 3, 3, 3, 4)


def combinations():
    return [group for size in range(1, 9) for group in itertools.combinations(NAMES, size)]


def _segments(a, count=4):
    return np.stack([part.mean(axis=1) for part in np.array_split(a, count, axis=1)], axis=1)


def _slope(a):
    t = np.arange(a.shape[1], dtype=np.float64)
    centered = t - t.mean()
    return (a @ centered) / max(float(centered @ centered), 1e-9)


def _ar1(a):
    left, right = a[:, :-1], a[:, 1:]
    mean_left, mean_right = left.mean(axis=1), right.mean(axis=1)
    centered = left - mean_left[:, None]
    coefficient = np.sum(centered * (right - mean_right[:, None]), axis=1) / (
        np.sum(centered * centered, axis=1) + 1.0)
    coefficient = np.clip(coefficient, -0.98, 0.98)
    bias = mean_right - coefficient * mean_left
    noise = np.sqrt(np.mean((right - coefficient[:, None] * left - bias[:, None]) ** 2, axis=1) + 1e-6)
    return coefficient, bias, noise


def descriptors(x, y, period):
    """x,y are both normalized with *history-only* mean/scale; output M1..M8."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
        raise ValueError("expected matching [sample,time] history/future arrays")
    n, horizon = y.shape
    t_future = np.arange(x.shape[1], x.shape[1] + horizon, dtype=np.float64)
    persistence = np.repeat(x[:, -1:], horizon, axis=1)
    slope = _slope(x)
    trend = x.mean(axis=1, keepdims=True) + slope[:, None] * (
        t_future[None] - (x.shape[1] - 1) / 2)
    seasonal_valid = 2 <= int(period) <= x.shape[1] // 2
    seasonal = x[:, -int(period):][:, np.arange(horizon) % int(period)] if seasonal_valid else persistence
    baseline = seasonal if seasonal_valid else trend
    # The common historical normalization has already removed the original units.
    m1 = _segments(y - persistence)
    m2 = _segments(y - trend)
    m3 = _segments(y - seasonal)
    centered_b = baseline - baseline.mean(axis=1, keepdims=True)
    scale_coef = np.sum(centered_b * (y - y.mean(axis=1, keepdims=True)), axis=1) / (
        np.sum(centered_b ** 2, axis=1) + 1.0)
    scale_coef = np.clip(scale_coef, -5, 5)
    offset = y.mean(axis=1) - scale_coef * baseline.mean(axis=1)
    fit = scale_coef[:, None] * baseline + offset[:, None]
    m4 = np.stack([scale_coef, offset, np.log1p(np.mean((y - fit) ** 2, axis=1))], axis=1)
    errors = np.stack([np.mean((y - prediction) ** 2, axis=1)
                       for prediction in (persistence, trend, seasonal)], axis=1)
    m5 = np.log1p(errors) - np.log1p(errors).mean(axis=1, keepdims=True)
    ax, bx, nx = _ar1(x)
    ay, by, ny = _ar1(y)
    m6 = np.stack([ay - ax, by - bx, np.log((ny + .01) / (nx + .01))], axis=1)
    med_x, med_y = np.median(x, axis=1), np.median(y, axis=1)
    qx = np.quantile(x, [.1, .25, .75, .9], axis=1)
    qy = np.quantile(y, [.1, .25, .75, .9], axis=1)
    iqr_x, iqr_y = qx[2] - qx[1], qy[2] - qy[1]
    m7 = np.stack([med_y - med_x, np.log((iqr_y + .1) / (iqr_x + .1)),
                   ((qy[3] - qy[0]) - (qx[3] - qx[0])) / (iqr_x + .1)], axis=1)
    width = max(2, min(x.shape[1], y.shape[1]) // 4)
    tail, head = x[:, -width:], y[:, :width]
    m8 = np.stack([head.mean(axis=1) - tail.mean(axis=1),
                   width * (_slope(head) - _slope(tail)),
                   np.log((head.std(axis=1) + .1) / (tail.std(axis=1) + .1)),
                   y.mean(axis=1) - x.mean(axis=1)], axis=1)
    output = dict(zip(NAMES, (m1, m2, m3, m4, m5, m6, m7, m8)))
    if not all(np.isfinite(value).all() for value in output.values()):
        raise FloatingPointError("descriptor contains non-finite values")
    return {name: value.astype(np.float32) for name, value in output.items()}, {
        "seasonal_valid": bool(seasonal_valid), "period": int(period),
        "seasonal_fallback": "persistence" if not seasonal_valid else "none"}


def robust_groups(train, groups):
    """Fit robust location/scale on TRAIN A; equalize each included group energy."""
    fitted = {}
    for name in NAMES:
        a = np.asarray(train[name], np.float32)
        center = np.median(a, axis=0)
        scale = np.quantile(a, .75, axis=0) - np.quantile(a, .25, axis=0)
        valid = scale > 1e-5
        fitted[name] = (center, np.where(valid, scale, 1.0), valid)
    return fitted


def combine(values, fitted, names):
    pieces = []
    for name in names:
        center, scale, valid = fitted[name]
        a = np.clip((values[name] - center) / scale, -8, 8)
        a[:, ~valid] = 0
        pieces.append(a / np.sqrt(max(int(valid.sum()), 1)))
    return np.concatenate(pieces, axis=1).astype(np.float32)
