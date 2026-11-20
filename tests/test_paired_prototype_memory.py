import torch
import torch.nn.functional as F
import pytest

from models.predictive_env.paired_prototype_memory import PairedPrototypeBank
from models.predictive_env.paired_prototype_memory.fusion import reliable_variation, stable_consensus
from models.predictive_env.paired_prototype_memory.stability import prototype_evidence
from models.predictive_env.paired_prototype_memory.uncertainty import fit_residual_memory, memory_features
from models.predictive_env.paired_prototype_memory.visualize import (
    plot_bank, plot_stability, plot_memory, plot_interval,
)


def example():
    torch.manual_seed(5)
    n, c, p, d, t = 7, 3, 2, 5, 9
    key, hidden = torch.randn(n, c, 64), torch.randn(n, c, p, d)
    future = torch.randn(n, c, t)
    starts = torch.arange(n) * 12
    linear = torch.nn.Linear(p * d, t)
    bank = PairedPrototypeBank(4, p, d, t, linear.weight, linear.bias, rank=3)
    bank.initialize(key, hidden, future, starts, 12)
    return bank, key, hidden, future, starts, linear


def test_shapes_zero_adapter_and_gradients():
    bank, key, hidden, future, starts, linear = example()
    result = bank.query(key, hidden)
    assert result["candidate"].shape == (7, 3, 4, 9)
    assert result["weight"].shape == (7, 3, 4)
    base = F.linear(hidden.permute(0, 1, 3, 2).flatten(-2), linear.weight, linear.bias)
    assert torch.allclose(result["candidate"], base[:, :, None].expand_as(result["candidate"]), atol=1e-6)
    loss, terms = bank.losses(key, hidden, future, use_future=True,
                              lambda_cover=.2, lambda_div=.01, lambda_adapter=.01)
    loss.backward()
    assert torch.isfinite(loss)
    assert bank.adapter_up.weight.grad is not None
    assert bank.adapter_up.weight.grad.abs().sum() > 0
    assert bank.a0_weight.grad is None


def test_fit_interface_and_frozen_query():
    bank, key, hidden, future, starts, _ = example()
    report = bank.fit(key, hidden, future, starts, 12, epochs=3)
    assert len(report["train_trace"]) == 3
    bank.requires_grad_(False)
    with torch.no_grad():
        result = bank.query(key, hidden)
    assert torch.isfinite(result["candidate"]).all()


def test_query_never_uses_future_and_restores_exactly():
    bank, key, hidden, future, starts, _ = example()
    before = bank.query(key, hidden)
    restored = PairedPrototypeBank.from_export(bank.export())
    after = restored.query(key, hidden)
    for field in ("candidate", "weight", "nearest_distance"):
        assert torch.allclose(before[field], after[field], atol=1e-7)
    assert "future" not in bank.query.__code__.co_varnames[:4]


def test_null_fallback_evidence_and_memory():
    bank, key, hidden, future, starts, _ = example()
    query = bank.query(key, hidden)
    stable = query["prediction"]
    zero_reliability = torch.zeros(4, 3)
    output, correction = reliable_variation(stable, query["candidate"], query["weight"], zero_reliability)
    assert torch.equal(output, stable) and torch.count_nonzero(correction) == 0
    consensus, mass = stable_consensus(query["candidate"], query["weight"],
                                       torch.zeros(4), stable)
    assert torch.equal(consensus, stable) and torch.count_nonzero(mass) == 0
    stability, reliability, audit = prototype_evidence(stable, query["candidate"],
        future, query["weight"], starts, 12, segments=3)
    assert stability.shape == (4,) and reliability.shape == (4, 3)
    assert audit["status"] == "evaluated"
    memory = fit_residual_memory(future - stable, query["weight"], starts, 12)
    features = memory_features(memory, query["weight"], query["distance"],
        torch.ones(7, 1, 3), stability, reliability)
    assert features.shape == (7, 3, 9, 14)
    assert torch.isfinite(features).all()


@pytest.mark.parametrize("tokens,width,channels,horizon", [(12, 128, 7, 720),
                                                        (1, 64, 7, 336),
                                                        (6, 64, 1, 96)])
def test_backbone_agnostic_hidden_shapes(tokens, width, channels, horizon):
    torch.manual_seed(17)
    key = torch.randn(2, channels, 64)
    hidden = torch.randn(2, channels, tokens, width)
    head = torch.nn.Linear(tokens * width, horizon)
    bank = PairedPrototypeBank(4, tokens, width, horizon, head.weight, head.bias)
    result = bank.query(key, hidden)
    assert result["candidate"].shape == (2, channels, 4, horizon)
    assert torch.isfinite(result["candidate"]).all()


def test_quantile_memory_baseline_uses_fixed_point_and_finite_scores():
    from tools.run_fpem_paired_prototype import quantile_memory_baseline
    bank, key, hidden, future, starts, _ = example()
    query = bank.query(key, hidden)
    point = query["prediction"].detach()
    memory = fit_residual_memory(future - point, query["weight"].detach(), starts, 12)
    block = {"scale": torch.ones(7, 1, 3),
             "hist_mean": torch.zeros(7, 1, 3),
             "truth": future.transpose(1, 2)}
    metrics = quantile_memory_baseline(memory, query, point, block,
                                       global_only=True, samples=8, batch=3)
    assert "CRPS" in metrics and torch.isfinite(torch.tensor(metrics["CRPS"]))


def test_visual_diagnostics_render(tmp_path):
    bank, key, hidden, future, starts, _ = example()
    query = bank.query(key, hidden)
    stability, reliability, evidence = prototype_evidence(
        query["prediction"], query["candidate"], future,
        query["weight"], starts, 12, segments=3)
    memory = fit_residual_memory(future - query["prediction"],
                                 query["weight"].detach(), starts, 12)
    plot_bank(bank, {"key": key}, tmp_path)
    plot_stability(stability, reliability, tmp_path, evidence)
    plot_memory(memory, tmp_path)
    plot_interval(future[0, 0], query["prediction"][0, 0],
                  torch.randn(10, future.shape[-1]), tmp_path / "interval.png")
    assert (tmp_path / "prototype_similarity.png").exists()
    assert (tmp_path / "stability_and_reliability.png").exists()
    assert (tmp_path / "residual_memory_00.png").exists()
    assert (tmp_path / "interval.png").exists()
