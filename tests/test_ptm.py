"""Focused PTM tests, including TRAIN-only fitting and inference isolation."""
from __future__ import annotations

import copy
import io
import numpy as np
import pytest
import torch

from models.predictive_env.ptm import PrototypeAdapter, PrototypeBank


class TinyTrain:
    def __init__(self, split=0, length=96, channels=3, seq_len=12, pred_len=7):
        rng = np.random.default_rng(24)
        self.set_type = split
        self.seq_len, self.pred_len = seq_len, pred_len
        grid = np.arange(length, dtype=np.float32)[:, None]
        self.data_x = np.sin(grid/5 + np.arange(channels)[None]).astype(np.float32)
        self.data_x += 0.1*rng.normal(size=self.data_x.shape).astype(np.float32)
        self.data_y = self.data_x.copy()
        self._length = length-seq_len-pred_len+1

    def __len__(self):
        return self._length


@pytest.mark.parametrize("shape", [(1, 12, 1), (4, 12, 3), (2, 12, 7)])
def test_query_shapes_and_finite(shape):
    data = TinyTrain(channels=shape[-1])
    bank = PrototypeBank(pred_len=7, k=4, history_size=8)
    fit = bank.fit_train(data)
    assert fit["train_anchor_groups"] == 7
    x = torch.from_numpy(data.data_x[:12]).unsqueeze(0).expand(shape[0], -1, -1)
    out = bank.query(x)
    assert out["x_proto"].shape == x.shape
    assert out["y_proto"].shape == (shape[0], 7, shape[-1])
    assert out["weight"].shape == (shape[0], shape[-1], 4)
    assert torch.isfinite(out["y_proto"]).all()


def test_zero_variance_and_historical_future_scale():
    data = TinyTrain(channels=1)
    bank = PrototypeBank(pred_len=7, k=3, history_size=8)
    bank.fit_train(data)
    out = bank.query(torch.full((1, 12, 1), 17.0))
    assert torch.isfinite(out["x_proto"]).all()
    assert torch.isfinite(out["y_proto"]).all()
    assert torch.allclose(out["y_proto"], torch.full_like(out["y_proto"], 17.0), atol=1e-3)


def test_train_only_updates_and_query_never_reads_future():
    data = TinyTrain()
    bank = PrototypeBank(pred_len=7, k=4, history_size=8)
    with pytest.raises(ValueError):
        bank.fit_train(TinyTrain(split=2))
    bank.fit_train(data)
    before = copy.deepcopy(bank.state_dict())
    x = torch.from_numpy(data.data_x[:12]).unsqueeze(0)
    first = bank.query(x)["y_proto"]
    data.data_y[:] = 9999.0
    second = bank.query(x)["y_proto"]
    assert torch.equal(first, second)
    assert all(torch.equal(before[key], bank.state_dict()[key]) for key in before)
    with pytest.raises(ValueError):
        bank.update_train_epoch(TinyTrain(split=1))


def test_epoch_ema_and_state_roundtrip():
    data = TinyTrain()
    bank = PrototypeBank(pred_len=7, k=4, history_size=8, beta=0.5)
    bank.fit_train(data)
    old = bank.V.clone()
    data.data_y += 1.0
    bank.update_train_epoch(data)
    assert not torch.equal(old, bank.V)
    buffer = io.BytesIO()
    torch.save(bank.state_dict(), buffer)
    buffer.seek(0)
    other = PrototypeBank(pred_len=7, k=4, history_size=8)
    other.load_state_dict(torch.load(buffer, weights_only=True))
    x = torch.from_numpy(data.data_x[:12]).unsqueeze(0)
    assert torch.equal(bank.query(x)["y_proto"], other.query(x)["y_proto"])


def test_input_gate_identity_then_gradient():
    data = TinyTrain()
    bank = PrototypeBank(pred_len=7, k=4, history_size=8)
    bank.fit_train(data)
    x = torch.from_numpy(data.data_x[:12]).unsqueeze(0)
    query = bank.query(x)
    adapter = PrototypeAdapter("input")
    identity, _ = adapter.enhance_input(x, query, schedule=0.0)
    assert torch.equal(identity, x)
    changed, _ = adapter.enhance_input(x, query, schedule=1.0)
    assert not torch.equal(changed, x)
    changed.square().mean().backward()
    assert adapter.gate[-1].weight.grad is not None
    assert adapter.gate[-1].weight.grad.abs().sum() > 0


def test_shape_calibration_initial_identity_and_mean_conservation():
    data = TinyTrain()
    bank = PrototypeBank(pred_len=7, k=4, history_size=8)
    bank.fit_train(data)
    query = bank.query(torch.from_numpy(data.data_x[:12]).unsqueeze(0))
    adapter = PrototypeAdapter("shape")
    y0 = torch.randn(1, 7, 3)
    original, _ = adapter.calibrate_shape(y0, query)
    assert torch.equal(original, y0)
    with torch.no_grad():
        adapter.gate[-1].bias.fill_(0.5)
    calibrated, gate = adapter.calibrate_shape(y0, query)
    assert torch.allclose(calibrated.mean(1), y0.mean(1), atol=1e-6)
    assert not torch.equal(calibrated, y0)
    assert gate.abs().max() <= 0.2


def test_nonoverlap_anchor_count_is_stride_invariant():
    data = TinyTrain(length=120, seq_len=12, pred_len=7)
    bank = PrototypeBank(pred_len=7, k=4, history_size=8)
    fitted = bank.fit_train(data)
    assert fitted["train_anchor_groups"] == len(range(0, len(data), data.seq_len))
    assert sum(fitted["initial_cluster_counts"]) == fitted["train_anchor_groups"]*3
