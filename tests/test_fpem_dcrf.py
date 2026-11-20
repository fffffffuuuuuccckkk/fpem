"""Numerical, support, NULL, and time-causal checks for DCRF."""
import torch

from models.predictive_env.dual_consensus_retrieval import (
    entmax15, selection_from_memory, two_stage_weights,
    weighted_ridge_evidence, weighted_median)


def test_entmax_sparse_normalized_and_no_memory():
    score = torch.tensor([[0., -.1, -10.], [1., 2., 3.]])
    valid = torch.tensor([[True, True, True], [False, False, False]])
    weight = entmax15(score, valid)
    assert torch.allclose(weight[0].sum(), torch.tensor(1.), atol=1e-5)
    assert weight[0, 2] == 0
    assert weight[1].abs().sum() == 0
    assert torch.isfinite(weight).all()


def test_null_can_reject_and_same_source_can_enter_both():
    raw = torch.tensor([.5, .5, 0.])
    key = torch.zeros(2)
    memory = torch.zeros(3, 2)
    common, variation, stable, variant, null = two_stage_weights(
        raw, key, memory, memory, memory + 20,
        torch.zeros(3), torch.zeros(3), 1., 1., 0., 0., 0.)
    assert null == 1
    assert variant.sum() == 0
    assert bool((stable[:2] > 0).all())
    assert torch.allclose(common, torch.zeros_like(common))
    assert torch.allclose(variation, torch.zeros_like(variation))


def test_weighted_ridge_loo_matches_explicit_fit():
    torch.manual_seed(9)
    x = torch.randn(5, 4)
    y = torch.randn(5, 3)
    q = torch.randn(4)
    w = torch.tensor([.1, .2, .3, .15, .25])
    lam = 1.5
    got = weighted_ridge_evidence(q, x, y, w, lam)
    explicit = []
    for i in range(5):
        keep = torch.arange(5) != i
        wx = x[keep].double() * w[keep, None].double().sqrt()
        wy = y[keep].double() * w[keep, None].double().sqrt()
        # The exact LOO formula keeps the original objective weights.
        coefficient = torch.linalg.solve(wx.T @ wx + lam * torch.eye(4), wx.T @ wy)
        explicit.append((q.double() @ coefficient).float())
    expected = weighted_median(torch.stack(explicit), w)
    assert torch.allclose(got.common, expected, atol=1e-5)
    assert got.supported and torch.isfinite(got.gain).all()


def test_adaptive_retrieval_only_eligible_and_no_fixed_k():
    memory = torch.tensor([[0.], [.02], [.04], [5.]])
    query = torch.tensor([0.])
    eligible = torch.tensor([True, True, False, False])
    weight, _ = selection_from_memory(query, memory, eligible, 1.)
    assert weight[:2].sum() > .999
    assert weight[2:].sum() == 0
    assert (weight[:2] > 0).sum() == 2
    empty, _ = selection_from_memory(query, memory, torch.zeros_like(eligible), 1.)
    assert empty.sum() == 0
