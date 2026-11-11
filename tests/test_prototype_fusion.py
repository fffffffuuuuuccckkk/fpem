import pytest
import torch
from torch import nn

from models.predictive_env.prototype_fusion import PrototypeFusion, PrototypeFusionMemory


class FrozenHead(nn.Module):
    def __init__(self, channels=2, patches=3, width=8, horizon=7):
        super().__init__()
        self.head_flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(patches*width, horizon)

    def frozen_h_head(self, hidden, cycle=None):
        return self.linear(self.head_flatten(hidden.permute(0, 1, 3, 2)))


@pytest.mark.parametrize("mode", ["P0", "P2", "P3"])
def test_memory_no_query_label_and_complete_interval_purge(mode):
    torch.manual_seed(7)
    n, length, channels, horizon, width = 48, 12, 2, 7, 8
    x = torch.randn(n, length, channels)
    h = torch.randn(n, channels, 3, width)
    response = torch.randn(n, channels, horizon)
    q = torch.softmax(torch.randn(n, 3), -1)
    starts = torch.arange(n)*length
    bank = PrototypeFusionMemory(8, 12, horizon, width, 3, mode)
    allowed = ((starts+length+horizon <= 150) | (starts >= 320))
    bank.fit_key_space(x[allowed], response[allowed])
    report = bank.fit_values(x, h, response, q, starts, allowed, (150, 320))
    assert report["source_anchors"] == int(allowed.sum())
    query_before = bank.query(x[:2])["stable_response"]
    response[~allowed] += 1000
    bank2 = PrototypeFusionMemory(8, 12, horizon, width, 3, mode)
    bank2.fit_key_space(x[allowed], response[allowed])
    bank2.fit_values(x, h, response, q, starts, allowed, (150, 320))
    assert torch.allclose(query_before, bank2.query(x[:2])["stable_response"], atol=1e-5)


@pytest.mark.parametrize("fusion", ["F1", "F2", "F3"])
@pytest.mark.parametrize("mapping", ["M0", "M1", "M2"])
def test_zero_start_a0_and_gradient(fusion, mapping):
    torch.manual_seed(3)
    b, c, p, d, horizon, r, k = 2, 2, 3, 8, 7, 12, 8
    model = FrozenHead(c, p, d, horizon)
    module = PrototypeFusion(d, p, horizon, r, 3, fusion, mapping, rank=2)
    hidden = torch.randn(b, c, p, d)
    query = {"stable_response": torch.randn(b,c,horizon),
             "env_response": torch.randn(b,c,horizon),
             "c_inv": torch.rand(b,c), "proto_hidden": torch.randn(b,c,d),
             "keys": torch.randn(k,r), "hidden_values": torch.randn(k,d),
             "entropy": torch.rand(b,c), "min_distance": torch.rand(b,c)}
    mean = torch.zeros(b,1,c); std = torch.ones(b,1,c)
    output, _ = module(model, hidden, mean, std, None, query)
    expected = model.frozen_h_head(hidden).permute(0,2,1)
    assert torch.allclose(output, expected, atol=1e-7)
    loss = output.square().mean(); loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for p in module.parameters())


def test_m2_requires_only_history_predicted_q():
    module = PrototypeFusion(8, 3, 7, 12, 3, "F1", "M2")
    hidden = torch.randn(4, 2, 3, 8)
    assert module.predict_environment(hidden).shape == (4,3)
