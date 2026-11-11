"""Low-capacity history/future relation and local-operator prototype utilities.

All teacher functions here are TRAIN-only. A student router must be used for
every subsequent prediction or prototype assignment.
"""
from __future__ import annotations

import time
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def softmax_neg(distance, temperature):
    value = -np.asarray(distance, dtype=np.float64) / max(float(temperature), 1e-6)
    value -= value.max(axis=1, keepdims=True)
    weight = np.exp(value)
    return (weight / weight.sum(axis=1, keepdims=True)).astype(np.float32)


def squared_distance(x, center):
    x = np.asarray(x, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32)
    return np.maximum(0, np.mean(x[:, None] ** 2, axis=2) +
                      np.mean(center[None] ** 2, axis=2) -
                      2 * (x @ center.T) / x.shape[1])


def time_statistics(series, period):
    """Fixed four-dimensional state statistics in units of history SD."""
    y = np.asarray(series, dtype=np.float64)
    t = np.arange(y.shape[1], dtype=np.float64)
    centered_t = t - t.mean()
    slope = (y @ centered_t) / max(np.sum(centered_t ** 2), 1e-9)
    centered = y - y.mean(axis=1, keepdims=True)
    harmonic = np.exp(-2j * np.pi * t / max(period, 2))
    amplitude = np.abs(centered @ harmonic / len(t)) * 2
    volatility = np.std(np.diff(y, axis=1), axis=1)
    return np.stack((y.mean(axis=1), slope * period, amplitude, volatility), axis=1).astype(np.float32)


def relation_fingerprint(history, future, period, phase_floor):
    """Eleven TRAIN-supervised relation coordinates; no A0 error is used."""
    x = np.asarray(history, dtype=np.float64)
    y = np.asarray(future, dtype=np.float64)
    sx, sy = time_statistics(x, period), time_statistics(y, period)
    def harmonic(series, offset):
        t = np.arange(series.shape[1], dtype=np.float64) + offset
        return (series - series.mean(axis=1, keepdims=True)) @ (
            np.exp(-2j * np.pi * t / max(period, 2))) / series.shape[1]
    cx, cy = harmonic(x, -x.shape[1]), harmonic(y, 0)
    phase_valid = (2 * np.abs(cx) >= phase_floor) & (2 * np.abs(cy) >= phase_floor)
    delta = np.angle(cy * np.conj(cx))
    phase_sin = np.where(phase_valid, np.sin(delta), 0)
    phase_cos = np.where(phase_valid, np.cos(delta), 0)
    def bands(series):
        power = np.abs(np.fft.rfft(series - series.mean(axis=1, keepdims=True), axis=1)) ** 2
        frequencies = np.fft.rfftfreq(series.shape[1])
        limits = (0, .5 / period, 1 / period, 2 / period, .5 + 1e-8)
        values = []
        for lo, hi in zip(limits[:-1], limits[1:]):
            chosen = (frequencies >= lo) & (frequencies < hi)
            values.append(power[:, chosen].mean(axis=1) if chosen.any() else np.zeros(len(series)))
        values = np.stack(values, axis=1)
        return values / np.maximum(values.sum(axis=1, keepdims=True), 1e-8)
    spectral_ratio = np.clip(np.log((bands(y) + .01) / (bands(x) + .01)), -4, 4)
    result = np.column_stack((sy[:, 0] - sx[:, 0], sy[:, 1] - sx[:, 1],
                              np.clip(np.log((sy[:, 2] + .05) / (sx[:, 2] + .05)), -4, 4),
                              phase_sin, phase_cos, spectral_ratio,
                              np.clip(np.log((sy[:, 3] + .05) / (sx[:, 3] + .05)), -4, 4),
                              phase_valid.astype(float)))
    return result.astype(np.float32), phase_valid


def independent_count(ids, span):
    chosen = 0
    last = -10**20
    for start in np.unique(ids):
        if start >= last:
            chosen += 1
            last = int(start) + span
    return chosen


def local_operators(z, v, ids, span, *, anchor_count=128, neighbor_pool=5000,
                    max_neighbors=12, ridge=1.0):
    """Fit local 4x4 affine maps from history-only neighbor search.

    Neighbors are greedy non-overlapping complete windows. Maps with fewer
    than d_x+2 independent events remain diagnostic only, never 'reliable'.
    """
    z, v, ids = np.asarray(z), np.asarray(v), np.asarray(ids)
    if len(z) < 8:
        return []
    anchors = np.linspace(0, len(z) - 1, min(anchor_count, len(z)), dtype=int)
    finder = NearestNeighbors(n_neighbors=min(neighbor_pool, len(z))).fit(z)
    distances, neighbors = finder.kneighbors(z[anchors])
    output = []
    for anchor, candidate, distance in zip(anchors, neighbors, distances):
        selected = []
        for index in candidate:
            if all(abs(int(ids[index]) - int(ids[other])) >= span for other in selected):
                selected.append(int(index))
            if len(selected) >= max_neighbors:
                break
        if not selected:
            continue
        x, y = z[selected].astype(np.float64), v[selected].astype(np.float64)
        xm, ym = x.mean(axis=0), y.mean(axis=0)
        xc, yc = x - xm, y - ym
        gram = xc.T @ xc + ridge * np.eye(x.shape[1])
        matrix = np.linalg.solve(gram, xc.T @ yc).T
        bias = ym - matrix @ xm
        radius = float(np.quantile(np.linalg.norm(x - xm, axis=1), .9) + .1)
        condition = float(np.linalg.cond(gram))
        support = independent_count(ids[selected], span)
        output.append({"anchor": int(anchor), "matrix": matrix.astype(np.float32),
                       "bias": bias.astype(np.float32), "center": xm.astype(np.float32),
                       "radius": radius, "support": support,
                       "condition": condition,
                       "reliable": support >= x.shape[1] + 2 and condition < 1e6})
    return output


def operator_behaviors(operators, references):
    """Common-reference response; explicitly mark unsupported extrapolation."""
    if not operators:
        return np.empty((0, len(references) * 4)), np.empty((0, len(references)), bool)
    responses, support = [], []
    for operator in operators:
        responses.append(references @ operator["matrix"].T + operator["bias"])
        support.append(np.linalg.norm(references - operator["center"], axis=1) <=
                       1.5 * operator["radius"])
    return np.asarray(responses, np.float32), np.asarray(support, bool)


def train_router(history, teacher, *, seed=2022, epochs=40, num_env=8, lr=0.01):
    """Small 64->32->K student trained with soft-label KL (legacy K=8 default)."""
    teacher = np.asarray(teacher, np.float32)
    if teacher.ndim != 2 or teacher.shape[1] != num_env or len(teacher) != len(history):
        raise ValueError("teacher must have shape [len(history), num_env]")
    torch.manual_seed(seed)
    network = torch.nn.Sequential(torch.nn.Linear(int(np.asarray(history).shape[1]), 32), torch.nn.Tanh(),
                                  torch.nn.Linear(32, num_env))
    x = torch.from_numpy(np.asarray(history, np.float32))
    q = torch.from_numpy(np.asarray(teacher, np.float32))
    optim = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=1e-4)
    t0 = time.perf_counter()
    for _ in range(epochs):
        order = torch.randperm(len(x))
        for batch in order.split(512):
            logits = network(x[batch])
            loss = torch.nn.functional.kl_div(torch.log_softmax(logits, dim=-1),
                                              q[batch], reduction="batchmean")
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
    network.eval()
    return network, time.perf_counter() - t0, sum(p.numel() for p in network.parameters())


@torch.no_grad()
def route(network, history):
    parts = []
    for batch in torch.from_numpy(np.asarray(history, np.float32)).split(1024):
        parts.append(torch.softmax(network(batch), dim=-1).numpy())
    return np.concatenate(parts)


def teacher_quality(teacher, student):
    teacher = np.asarray(teacher)
    student = np.asarray(student)
    return {"top1_agreement": float(np.mean(teacher.argmax(axis=1) == student.argmax(axis=1))),
            "soft_kl": float(np.mean(np.sum(teacher * (np.log(teacher + 1e-8) -
                                                      np.log(student + 1e-8)), axis=1))),
            "teacher_mass": teacher.mean(axis=0).tolist(),
            "student_mass": student.mean(axis=0).tolist()}
