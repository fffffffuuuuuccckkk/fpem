"""Small numerical and causality checks for the SMRVA point pilot."""
import torch

from models.predictive_env.paired_memory import PairedMemoryBank
from models.predictive_env.mapping_stability import (
    chronological_blocks, fit_shared_mapping, history_features,
    local_variation_mapping, predict_mapping, variation_activity,
    variation_description,
)


def test_chronological_purge_and_zero_sum():
    starts = torch.arange(32) * 96
    blocks, valid = chronological_blocks(starts, 96, 192, 4)
    for block in range(3):
        boundary = starts[(blocks == block + 1)][0]
        assert bool(((starts[valid & (blocks == block)] + 96 + 192) <= boundary).all())
    torch.manual_seed(7)
    x = torch.randn(32, 2, 5)
    y = torch.randn(32, 2, 3)
    shared, private = fit_shared_mapping(x, y, blocks, valid, 1., 10.)
    assert shared.shape == (2, 5, 3)
    assert torch.allclose(private.sum(0), torch.zeros_like(shared), atol=1e-5)
    assert torch.isfinite(predict_mapping(x, shared)).all()


def test_analytic_loo_agrees_with_explicit_refit():
    torch.manual_seed(11)
    query = torch.randn(1, 1, 4)
    source = torch.randn(1, 1, 5, 4)
    target = torch.randn(1, 1, 5, 3)
    valid = torch.ones(1, 1, 5, dtype=torch.bool)
    lam = 2.5
    result = local_variation_mapping(query, source, target, valid, lam)
    for index in range(5):
        keep = torch.arange(5) != index
        x = source[0, 0, keep]
        y = target[0, 0, keep]
        estimate = source[0, 0, index] @ x.T @ torch.linalg.solve(
            x @ x.T + lam * torch.eye(4), y)
        expected = target[0, 0, index] - estimate
        assert torch.allclose(result["loo_residual"][0, 0, index], expected, atol=1e-5)
    assert result["candidate"].shape == (1, 1, 3)


def test_no_valid_neighbours_and_constant_history_are_finite():
    query = torch.zeros(2, 3, 4)
    source = torch.zeros(2, 3, 5, 4)
    target = torch.ones(2, 3, 5, 7)
    valid = torch.zeros(2, 3, 5, dtype=torch.bool)
    result = local_variation_mapping(query, source, target, valid, 100.)
    assert result["candidate"].abs().sum() == 0
    assert result["map_reliability"].abs().sum() == 0
    history = torch.ones(2, 96, 3)
    floor = torch.full((1, 3), .01)
    assert history_features(history).shape == (2, 3, 65)
    assert torch.isfinite(variation_description(history, floor)).all()
    assert torch.isfinite(variation_activity(history, floor)).all()


def test_train_retrieval_never_uses_unobserved_future():
    torch.manual_seed(5)
    x = torch.randn(12, 96, 2)
    records = {"x": x, "target": torch.randn(12, 24, 2),
               "a0_pred": torch.zeros(12, 24, 2),
               "hidden": torch.zeros(12, 2, 1, 8),
               "starts": torch.arange(12) * 96}
    bank = PairedMemoryBank(96, 24)
    bank.fit(records)
    chosen = bank.retrieve(x[4:6], view="all", common_k=5, rare_k=0,
                           sample_ids=records["starts"][4:6], split="train")
    assert bool((chosen["source_ends"][chosen["valid"]] <=
                 records["starts"][4:6, None, None].expand_as(
                     chosen["source_ends"])[chosen["valid"]]).all())
