import torch

from models.predictive_env.consensus_adaptation import local_operator_adapt
from models.predictive_env.consensus_adaptation.local_head_adapter import unified_operator_adapt
from models.predictive_env.future_consensus import future_consensus, future_difference_features
from models.predictive_env.learned_memory import LearnabilityMemory, LinearHistoryDecoder
from models.predictive_env.residual_difference import (
    ResidualDifferenceCondition, residual_difference_features)
from tools.run_fpem_residual_difference import add_features, add_consensus_features
from tools.run_fpem_learnability_pilot import new_mode_retrieval


def test_linear_decoder_detached_hidden_training_contract():
    hidden = torch.randn(2, 3, 4, 5, requires_grad=True)
    decoder = LinearHistoryDecoder(4, 5, 8)
    value = decoder(hidden.detach()).square().mean()
    value.backward()
    assert hidden.grad is None
    assert decoder.linear.weight.grad is not None
    assert decoder(hidden.detach()).shape == (2, 3, 8)


def test_dual_ridge_zero_and_stable_protection():
    query = torch.ones(1, 1, 1, 2)
    sources = torch.tensor([[[[[1., 0.]], [[1., 0.]], [[1., 0.]]]]])
    residual = torch.tensor([[[[0.], [2.], [2.]]]])
    valid = torch.ones(1, 1, 3, dtype=torch.bool)
    stable = torch.ones(1, 1, 3)
    under = torch.ones(1, 1, 3)
    plain, _ = local_operator_adapt(query, sources, residual, valid, stable,
                                    under, 1, lambda_stable=0., lambda_ridge=.1)
    protected, diag = local_operator_adapt(query, sources, residual, valid,
                                            stable, under, 1,
                                            lambda_stable=10., lambda_ridge=.1)
    assert plain.item() > protected.item() > 0
    assert diag["stable_prediction_shift_mse"] > 0
    absent, _ = local_operator_adapt(query, sources, residual,
                                     torch.zeros_like(valid), stable, under, 1)
    assert torch.equal(absent, torch.zeros_like(absent))


def test_causal_memory_and_soft_group_labels():
    n, length, horizon = 20, 8, 4
    starts = torch.arange(n) * length
    x = torch.randn(n, length, 1)
    y = torch.randn(n, horizon, 1)
    hidden = torch.randn(n, 1, 2, 3)
    bank = LearnabilityMemory(length, horizon, history_size=8)
    bank.fit({"x": x, "target": y, "a0_pred": torch.zeros_like(y),
              "hidden": hidden, "starts": starts})
    fitted = {"stable_probability_joint": torch.linspace(.01, .99, n)}
    density = bank.density_common.clone()
    bank.set_group(fitted, "joint")
    train = bank.retrieve(x[5:6], "mixed", 2, 2, sample_ids=starts[5:6], split="train")
    assert torch.all(train["source_ends"][train["valid"]] <= starts[5])
    assert torch.allclose(train["stable_probability"] +
                          train["underlearned_probability"],
                          train["valid"].float())
    random = bank.retrieve(x[15:16], "random_group", 2, 2,
                           sample_ids=starts[15:16], split="train")
    assert torch.all(random["source_ends"][random["valid"]] <= starts[15])
    bank.set_group(fitted, "density")
    assert torch.equal(bank.common, density)


def test_residual_statistics_and_identity_condition():
    residual = torch.tensor([[[[1., 2., 3.], [3., 4., 5.], [0., 0., 0.]]]])
    valid = torch.tensor([[[True, True, False]]])
    features = residual_difference_features(residual, valid)
    assert features.shape == (1, 1, 3, 7)
    assert torch.allclose(features[0, 0, :, 0], torch.tensor([1., 2., 3.]))
    condition = torch.randn(1, 1, 1, 6)
    film = ResidualDifferenceCondition(7, 6, patch_len=4, mode="film")
    assert torch.allclose(film(condition, features), condition)
    scale = ResidualDifferenceCondition(7, 6, patch_len=4, mode="output_scale")
    assert torch.allclose(scale.scale_correction(features), torch.ones(1, 1, 3))
    loss = film(condition, features).square().mean()
    loss.backward()
    assert film.net[-1].weight.grad is not None


def test_random_residual_control_respects_train_chronology():
    horizon = 4
    record = {
        "ids": torch.tensor([15, 40]),
        "target": torch.zeros(2, horizon, 1),
        "source_ids": torch.tensor([[[1, 3]], [[1, 3]]]),
        "valid": torch.ones(2, 1, 2, dtype=torch.bool),
        "stable": torch.tensor([[[True, False]], [[True, False]]]),
        "a0_residual": torch.zeros(2, 1, 2, horizon),
    }
    residual = {0: torch.ones(1, horizon), 5: 5 * torch.ones(1, horizon),
                30: 30 * torch.ones(1, horizon)}
    features, _ = add_features(record, residual, "U8", train_query=True,
                               seq_len=4, pred_len=4)
    assert features[0, 0, :, 0].max() <= 5
    assert features[1, 0, :, 0].max() <= 30


def test_unified_topk_reads_reconstruction_label_after_causal_retrieval():
    n, length, horizon = 20, 8, 4
    starts = torch.arange(n) * length
    x = torch.randn(n, length, 1)
    bank = LearnabilityMemory(length, horizon, history_size=8)
    bank.fit({"x": x, "target": torch.randn(n, horizon, 1),
              "a0_pred": torch.zeros(n, horizon, 1),
              "hidden": torch.randn(n, 1, 2, 3), "starts": starts})
    posterior = torch.linspace(.05, .95, n)
    bank.set_group({"stable_probability_rec": posterior}, "reconstruction")
    retrieved = bank.retrieve(x[12:13], view="all", common_k=5, rare_k=0,
                              sample_ids=starts[12:13], split="train")
    assert retrieved["indices"].shape == (1, 1, 5)
    assert torch.all(retrieved["source_ends"][retrieved["valid"]] <= starts[12])
    assert torch.allclose(retrieved["p_well"] + retrieved["p_poor"],
                          retrieved["valid"].float())
    # The selected indices are distance/causality based, not class-quota based.
    assert torch.allclose(retrieved["p_well"],
        posterior[retrieved["indices"]] * retrieved["valid"])
    mismatched = new_mode_retrieval(bank, x[12:13], starts[12:13], "B6",
                                    "train", None, 5)
    assert torch.all(mismatched["label_source_ends"][mismatched["valid"]] <= starts[12])
    assert not torch.equal(mismatched["future"], retrieved["future"])


def test_future_consensus_reliability_and_zero_disagreement():
    future = torch.zeros(1, 1, 5, 7)
    valid = torch.tensor([[[True, False, False, False, False]]])
    single = future_consensus(future, valid, 1.)
    assert torch.equal(single["weight"], torch.zeros_like(single["weight"]))
    valid[:] = True
    many = future_consensus(future, valid, 1.)
    assert torch.all(many["weight"] > 0)
    features = future_difference_features(future, valid, 1.)
    assert features.shape == (1, 1, 7, 7)
    assert torch.isfinite(features).all()


def test_unified_operator_empty_group_and_robust_solver():
    query = torch.randn(2, 1, 2, 3)
    source = torch.randn(2, 1, 5, 2, 3)
    target = torch.randn(2, 1, 5, 7)
    valid = torch.ones(2, 1, 5, dtype=torch.bool)
    posterior = torch.ones(2, 1, 5)  # Poor class is empty.
    consensus = torch.rand(2, 1, 7)
    for mode in ("legacy_protect", "both_square", "well_square_poor_huber",
                 "all_huber"):
        correction, diagnostics = unified_operator_adapt(
            query, source, target, valid, posterior, consensus,
            lambda_ridge=100., mode=mode, use_loss_weight=True,
            use_output_weight=True)
        assert correction.shape == (2, 1, 7)
        assert torch.isfinite(correction).all()
        assert diagnostics["poor_count"] == 0
    zero, _ = unified_operator_adapt(query, source, target,
        torch.zeros_like(valid), posterior, consensus, mode="all_huber")
    assert torch.equal(zero, torch.zeros_like(zero))


def test_consensus_feature_controls_never_read_query_future():
    record = {
        "ids": torch.tensor([100]), "target": torch.zeros(1, 4, 1),
        "source_ids": torch.tensor([[[4, 12, 20]]]),
        "valid": torch.ones(1, 1, 3, dtype=torch.bool),
        "stable": torch.tensor([[[True, False, True]]]),
        "a0_residual": torch.zeros(1, 1, 3, 4),
        "source_future": torch.randn(1, 1, 3, 4),
        "random_source_future": torch.randn(1, 1, 3, 4),
        "shuffled_source_future": torch.randn(1, 1, 3, 4),
        "random_valid": torch.ones(1, 1, 3, dtype=torch.bool),
        "p_well": torch.tensor([[[.8, .2, .9]]]),
        "source_typical_scale": torch.ones(1),
    }
    residual_map = {4: torch.ones(1, 4), 12: torch.ones(1, 4),
                    20: torch.ones(1, 4)}
    first, _ = add_consensus_features(record, residual_map, "C5")
    record["target"] = torch.full_like(record["target"], 999.)
    second, _ = add_consensus_features(record, residual_map, "C5")
    assert torch.equal(first, second)
    assert first.shape == (1, 1, 4, 14)
    shuffled, _ = add_consensus_features(record, residual_map, "C6")
    assert not torch.equal(first[..., 7:], shuffled[..., 7:])
