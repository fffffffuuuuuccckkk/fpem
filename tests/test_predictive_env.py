from types import SimpleNamespace

import pytest
import torch

from models.PatchTST_PredictiveEnvIV import ABLATIONS, Model
from models.predictive_env import (
    PredictiveConflictEnvironment,
    environment_risk_consistency,
    gradient_reverse,
    ComplementaryGateDecomposer,
    DirectGatedVariantFusion,
    FilmVariantFusion,
    HorizonFutureVariant,
    YFilmDecayFusion,
    SignedGateDecomposer,
    signed_gate_diagnostics,
    variant_gain_loss,
)
from models.predictive_env.future_var_consistency import future_variant_objective
from models.predictive_env.conditional_variant_predictor import (
    conditional_gain_ranking_loss,
    reliability_weighted_utility_loss,
)
from tools.run_predictive_env_iv_patchtst import (
    FutureTeacherIndexedDataset,
    analytic_patch_reliability_target,
    build_training_optimizer,
    expand_patch_reliability,
    future_patch_centers,
    future_patch_mean,
    gradient_relative_maturity,
    horizon_anchor_positions,
    horizon_effect_objective,
    invariant_anchored_variant_loss,
    make_train_loader,
    patch_correction_usefulness_target,
    reliability_pairwise_ranking_loss,
    reliability_target_stability,
    update_gamma_beta_learning_rate,
)


def cfg(
    experiment,
    constraint="contrastive",
    env_num=3,
    decomposition="projection",
):
    return SimpleNamespace(
        seq_len=32,
        pred_len=8,
        d_model=16,
        n_heads=2,
        e_layers=1,
        d_ff=32,
        dropout=0.0,
        factor=1,
        activation="gelu",
        patch_len=8,
        predictive_env_stride=4,
        predictive_env_ablation=experiment,
        predictive_env_bottleneck=8,
        predictive_env_num=env_num,
        predictive_env_grl_weight=1.0,
        representation_constraint=constraint,
        decomposition_type=decomposition,
    )


def test_all_ablations_forward_backward():
    for constraint in ("contrastive", "classification"):
        for decomposition in (
            "projection",
            "signed_gate",
            "complementary_gate",
        ):
            for experiment in ABLATIONS:
                model = Model(cfg(experiment, constraint, 3, decomposition))
                output = model.forward_components(torch.randn(4, 32, 3))
                assert output["prediction"].shape == (4, 8, 3)
                assert output["decomposition_gate"].shape == (4, 3, 8, 16)
                output["prediction"].square().mean().backward()


def test_zero_initialized_adaptations():
    diagnostics = Model(cfg("A4")).identity_diagnostics(torch.randn(3, 32, 3))
    assert max(diagnostics.values()) < 1e-8


def test_mapping_ablation_is_removed():
    with pytest.raises(ValueError, match="Mapping ablations"):
        Model(cfg("A5"))


def test_eiil_train_only_and_balanced():
    n = 30
    prediction = torch.randn(n, 8, 2)
    target = prediction.clone()
    target[:15] += 1
    target[15:] -= 1
    environment = PredictiveConflictEnvironment(n, 3, steps=5)
    assignment, report = environment.update(
        prediction, target, torch.arange(n), source_split="train"
    )
    assert assignment.shape == (n, 3)
    assert abs(assignment.mean(0).sum() - 1) < 1e-6
    assert report["source_split"] == "TRAIN only"
    assert report["D_within_grad"] >= 0
    assert report["D_between_grad"] >= 0
    assert report["gradient_separation"] >= 0
    assert report["random_partition_repeats"] == 100
    assert torch.isfinite(torch.tensor(report["random_partition_z_score"]))
    assert report["min_environment_mass"] > 0
    assert sorted(report["hungarian_permutation"]) == [0, 1, 2]
    assert report["alignment_overlap_after"] >= report["alignment_overlap_before"]
    with pytest.raises(ValueError):
        environment.update(prediction, target, torch.arange(n), source_split="test")


def test_generic_losses_are_finite():
    assignment = torch.softmax(torch.randn(8, 3), -1)
    loss, risks = environment_risk_consistency(torch.rand(8), assignment)
    assert torch.isfinite(loss + risks.mean())


def test_hungarian_alignment_restores_environment_columns():
    environment = PredictiveConflictEnvironment(6, 3)
    previous = torch.tensor(
        [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]],
        dtype=torch.float32,
    )
    current = previous[:, [2, 0, 1]]
    gradients = torch.tensor([[20.0, 21.0], [0.0, 1.0], [10.0, 11.0]])
    aligned, _, permutation, before, after, _ = environment._align_to_previous(
        previous, current, gradients
    )
    assert permutation.tolist() == [1, 2, 0]
    assert torch.equal(aligned, previous)
    assert after > before


def test_gradient_aware_matching_activates_after_first_update():
    n = 18
    prediction = torch.randn(n, 4, 2)
    target = torch.randn(n, 4, 2)
    environment = PredictiveConflictEnvironment(
        n, 3, steps=2, matching="gradient_aware"
    )
    _, first = environment.update(
        prediction, target, torch.arange(n), source_split="train"
    )
    _, second = environment.update(
        prediction * 0.9, target, torch.arange(n), source_split="train"
    )
    assert first["environment_matching_used"] == "overlap_first_update"
    assert second["environment_matching_used"] == "gradient_aware"


def test_soft_classification_detaches_q_and_grl_reverses_gradient():
    model = Model(cfg("A2", "classification", 3))
    output = model.forward_components(torch.randn(5, 32, 2))
    q = torch.softmax(torch.randn(5, 3), -1).requires_grad_(True)
    loss, diagnostics = model.environment_classification_loss(
        output["z_inv"], output["z_var"], q
    )
    loss.backward()
    assert q.grad is None
    assert torch.isfinite(diagnostics["inv_env_soft_ce"])
    assert torch.isfinite(diagnostics["var_env_soft_ce"])
    value = torch.ones(3, requires_grad=True)
    gradient_reverse(value, 2.0).sum().backward()
    assert torch.equal(value.grad, torch.full_like(value, -2.0))


def test_signed_gate_is_exact_partition_of_hidden_features():
    decomposer = SignedGateDecomposer(12, 6)
    hidden = torch.randn(3, 2, 5, 12)
    z_inv, z_var, gate = decomposer(hidden)
    assert gate.shape == hidden.shape
    assert torch.allclose(z_inv, hidden * torch.relu(gate))
    assert torch.allclose(z_var, hidden * torch.relu(-gate))
    assert not torch.any((torch.relu(gate) > 0) & (torch.relu(-gate) > 0))
    diagnostics = signed_gate_diagnostics(gate)
    assert 0 <= diagnostics["gate/positive_ratio"] <= 1
    assert 0 <= diagnostics["gate/negative_ratio"] <= 1
    assert 0 <= diagnostics["gate/near_zero_ratio"] <= 1


def test_complementary_gate_conserves_hidden_information():
    decomposer = ComplementaryGateDecomposer(12, 6)
    hidden = torch.randn(3, 2, 5, 12)
    z_inv, z_var, gate = decomposer(hidden)
    assert gate.shape == hidden.shape
    assert torch.allclose(z_inv + z_var, hidden, atol=1e-7, rtol=1e-6)
    assert float(((z_inv + z_var) - hidden).abs().mean()) < 1e-7
    assert torch.all((1.0 + gate) / 2.0 >= 0)
    assert torch.all((1.0 - gate) / 2.0 >= 0)


def test_direct_variant_fusion_matches_requested_formula():
    fusion = DirectGatedVariantFusion(12)
    z_inv = torch.randn(3, 2, 5, 12)
    z_var = torch.randn(3, 2, 5, 12)
    final, gate, applied = fusion(z_inv, z_var)
    assert gate.shape == (3, 2, 5, 1)
    assert torch.all((gate >= 0) & (gate <= 1))
    assert torch.allclose(applied, gate * z_var)
    assert torch.allclose(final, z_inv + gate * z_var)


def test_featurewise_direct_variant_fusion_has_one_gate_per_feature():
    fusion = DirectGatedVariantFusion(12, gate_type="feature")
    z_inv = torch.randn(3, 2, 5, 12)
    z_var = torch.randn(3, 2, 5, 12)
    final, gate, applied = fusion(z_inv, z_var)
    assert gate.shape == z_var.shape
    assert torch.all((gate >= 0) & (gate <= 1))
    assert torch.allclose(applied, gate * z_var)
    assert torch.allclose(final, z_inv + gate * z_var)


def test_film_variant_fusion_is_featurewise_and_identity_initialized():
    fusion = FilmVariantFusion(12, bottleneck=6, gamma_scale=0.1, beta_scale=0.05)
    z_inv = torch.randn(3, 2, 5, 12)
    z_var = torch.randn(3, 2, 5, 12)
    final, strength, applied, gamma, beta = fusion(z_inv, z_var)
    assert gamma.shape == z_inv.shape
    assert beta.shape == z_inv.shape
    assert strength.shape == z_inv.shape
    assert torch.count_nonzero(gamma) == 0
    assert torch.count_nonzero(beta) == 0
    assert torch.equal(final, z_inv)
    assert torch.count_nonzero(applied) == 0


def test_film_model_uses_shared_head_without_direct_zvar_addition():
    configuration = cfg("A2", "classification", 3, "complementary_gate")
    configuration.variant_fusion_mode = "film"
    model = Model(configuration)
    output = model.forward_components(torch.randn(4, 32, 3))
    assert output["film_gamma"].shape == output["z_inv_tokens"].shape
    assert output["film_beta"].shape == output["z_inv_tokens"].shape
    assert torch.equal(output["final_tokens"], output["z_inv_tokens"])
    assert torch.equal(
        output["prediction"], output["invariant_prediction"]
    )


def test_film_decay_reg_is_the_same_zero_initialized_z_space_path():
    configuration = cfg("A2", "classification", 3, "complementary_gate")
    configuration.variant_fusion_mode = "film_decay_reg"
    model = Model(configuration)
    output = model.forward_components(torch.randn(4, 32, 3))
    assert output["film_gamma"].shape == output["z_inv_tokens"].shape
    assert output["y_film_gamma"] is None
    assert torch.equal(output["prediction"], output["invariant_prediction"])


def test_film_decay_reg_without_decay_loss_reproduces_film_forward():
    configuration = cfg("A2", "classification", 3, "complementary_gate")
    configuration.variant_fusion_mode = "film"
    torch.manual_seed(2021)
    baseline = Model(configuration).eval()
    configuration.variant_fusion_mode = "film_decay_reg"
    torch.manual_seed(2021)
    decay = Model(configuration).eval()
    decay.load_state_dict(baseline.state_dict(), strict=False)
    value = torch.randn(4, 32, 3)
    baseline_output = baseline.forward_components(value)
    decay_output = decay.forward_components(value)
    assert torch.equal(
        baseline_output["prediction"], decay_output["prediction"]
    )


def test_y_film_decay_is_output_space_identity_with_monotonic_decay():
    fusion = YFilmDecayFusion(12, 8, bottleneck=6, decay_bias=-4.0)
    invariant = torch.randn(3, 2, 8)
    z_var = torch.randn(3, 2, 5, 12)
    full, strength, delta, gamma, beta, rho, decay = fusion(invariant, z_var)
    assert full.shape == invariant.shape
    assert strength.shape == delta.shape == gamma.shape == beta.shape == invariant.shape
    assert rho.shape == (3, 2, 1)
    assert decay.shape == invariant.shape
    assert torch.equal(full, invariant)
    assert torch.count_nonzero(delta) == 0
    assert torch.all(decay[..., 1:] <= decay[..., :-1])


def test_y_film_decay_model_keeps_latent_tokens_unmodified_for_backbones():
    for backbone in ("patchtst", "cyclenet"):
        configuration = cfg("A2", "classification", 3, "complementary_gate")
        configuration.variant_fusion_mode = "y_film_decay"
        configuration.predictive_env_backbone = backbone
        configuration.enc_in = 3
        configuration.cyclenet_cycle_len = 24
        model = Model(configuration)
        output = model.forward_components(torch.randn(4, 32, 3), torch.arange(4))
        assert torch.equal(output["final_tokens"], output["z_inv_tokens"])
        assert torch.equal(output["prediction"], output["invariant_prediction"])
        assert output["y_film_gamma"].shape == (4, 3, 8)
        assert output["decay_rho"].shape == (4, 3, 1)


def test_horizon_future_variant_is_continuous_and_identity_initialized():
    fusion = HorizonFutureVariant(
        12, 8, horizon_dim=5, bottleneck=6,
        gamma_scale=0.1, beta_scale=0.05,
        future_patch_len=3,
    )
    invariant = torch.randn(3, 2, 8)
    z_inv = torch.randn(3, 2, 5, 12)
    z_var = torch.randn(3, 2, 5, 12)
    output = fusion(invariant, z_inv, z_var)
    current = z_var.mean(dim=2).unsqueeze(2).expand(-1, -1, 3, -1)
    assert output["future_zvar"].shape == (3, 2, 3, 12)
    assert output["reliability"].shape == (3, 3)
    assert torch.equal(output["future_zvar"], current)
    assert torch.equal(output["prediction"], invariant)
    assert torch.count_nonzero(output["variation"]) == 0
    coordinate = fusion.horizon_coordinates(invariant.device, invariant.dtype)
    assert torch.allclose(coordinate[:, 0], torch.tensor([0.25, 0.625, 0.9375]))
    output["prediction"].square().mean().backward()
    assert fusion.modulation_generator[-1].weight.grad is not None


def test_environment_disagreement_estimates_are_isolated_and_patchwise():
    fusion = HorizonFutureVariant(
        6, 7, horizon_dim=4, bottleneck=5,
        future_patch_len=3, environment_count=2,
    )
    invariant = torch.randn(2, 3, 7)
    z_inv = torch.randn(2, 3, 4, 6, requires_grad=True)
    z_var = torch.randn(2, 3, 4, 6, requires_grad=True)
    output = fusion(invariant, z_inv, z_var)
    assert output["environment_corrections"].shape == (2, 2, 3, 3, 3)
    assert output["environment_disagreement"].shape == (2, 3)
    assert torch.count_nonzero(output["environment_corrections"]) == 0
    output["environment_corrections"].square().mean().backward()
    assert z_var.grad is None
    assert z_inv.grad is None
    for head in fusion.environment_correction_heads:
        assert head[-1].weight.grad is not None


def test_future_patch_reliability_only_gates_evaluation_prediction():
    fusion = HorizonFutureVariant(4, 5, future_patch_len=3)
    with torch.no_grad():
        fusion.modulation_generator[-1].bias[:3].fill_(0.2)
        fusion.modulation_generator[-1].bias[3:].fill_(0.1)
    invariant = torch.randn(2, 1, 5, requires_grad=True)
    output = fusion(
        invariant,
        torch.randn(2, 1, 2, 4),
        torch.randn(2, 1, 2, 4),
    )
    expected = (1.0 + output["gamma"]) * invariant + output["beta"]
    assert torch.allclose(output["prediction"], expected)
    detached_modulation = (
        (1.0 + output["gamma"]) * invariant.detach() + output["beta"]
    )
    detached_modulation.square().mean().backward()
    assert invariant.grad is None
    assert fusion.modulation_generator[-1].weight.grad is not None
    fusion.eval()
    evaluated = fusion(
        invariant.detach(),
        torch.randn(2, 1, 2, 4),
        torch.randn(2, 1, 2, 4),
    )
    raw_delta = evaluated["gamma"] * invariant.detach() + evaluated["beta"]
    expanded_r = expand_patch_reliability(
        evaluated["reliability"], 5, 3
    ).permute(0, 2, 1)
    assert torch.allclose(
        evaluated["prediction"], invariant.detach() + expanded_r * raw_delta
    )

    fusion_without_gate = HorizonFutureVariant(
        4, 5, future_patch_len=3, use_reliability_gate=False
    )
    fusion_without_gate.load_state_dict(fusion.state_dict())
    fusion_without_gate.eval()
    ungated = fusion_without_gate(
        invariant.detach(),
        torch.randn(2, 1, 2, 4),
        torch.randn(2, 1, 2, 4),
    )
    ungated_delta = ungated["gamma"] * invariant.detach() + ungated["beta"]
    assert torch.allclose(
        ungated["prediction"], invariant.detach() + ungated_delta
    )


def test_gradient_relative_maturity_is_current_batch_and_detached():
    invariant_parameter = torch.nn.Parameter(torch.tensor(1.0))
    variant_parameter = torch.nn.Parameter(torch.tensor(1.0))
    invariant_loss = (10.0 * invariant_parameter).square()
    full_loss = variant_parameter.square()
    early, g_inv_early, g_var = gradient_relative_maturity(
        invariant_loss, full_loss, [invariant_parameter], [variant_parameter]
    )
    mature, g_inv_mature, _ = gradient_relative_maturity(
        (0.1 * invariant_parameter).square(),
        variant_parameter.square(),
        [invariant_parameter],
        [variant_parameter],
    )
    assert g_inv_early > g_var > g_inv_mature
    assert early < mature
    assert not early.requires_grad and not mature.requires_grad


def test_gradient_relative_maturity_uses_l2_parameter_norm():
    invariant_parameter = torch.nn.Parameter(torch.ones(2))
    variant_parameter = torch.nn.Parameter(torch.ones(200))
    invariant_loss = invariant_parameter.sum()
    variant_loss = variant_parameter.sum()
    maturity, g_inv, g_var = gradient_relative_maturity(
        invariant_loss,
        variant_loss,
        [invariant_parameter],
        [variant_parameter],
    )
    assert torch.allclose(g_inv, torch.tensor(2.0).sqrt())
    assert torch.allclose(g_var, torch.tensor(200.0).sqrt())
    expected = g_var / (g_inv + g_var + 1e-8)
    assert torch.allclose(maturity, expected)


def test_differential_optimizer_and_adaptive_gamma_beta_lr():
    config = cfg("A2", "classification", 3, "complementary_gate")
    config.variant_fusion_mode = "horizon_future_var"
    config.future_patch_len = 4
    model = Model(config)
    args = SimpleNamespace(
        differential_lr=True,
        lr=1e-4,
        lr_backbone=2e-5,
        lr_inv_head=5e-5,
        lr_decomposer=1e-4,
        lr_env_head=1e-4,
        lr_variant=1e-4,
        lr_gamma_beta_base=1e-4,
        lr_reliability=3e-4,
        adaptive_variant_lr=False,
        fixed_gamma_beta_lr=False,
    )
    optimizer = build_training_optimizer(model, args)
    groups = {group["name"]: group for group in optimizer.param_groups}
    assert groups["backbone"]["lr"] == pytest.approx(2e-5)
    assert groups["inv_head"]["lr"] == pytest.approx(5e-5)
    assert groups["reliability"]["lr"] == pytest.approx(3e-4)
    assert groups["gamma_beta"]["lr"] == pytest.approx(2e-5)
    update_gamma_beta_learning_rate(optimizer, torch.tensor(0.25), args)
    assert groups["gamma_beta"]["lr"] == pytest.approx(4e-5)
    update_gamma_beta_learning_rate(optimizer, torch.tensor(1.0), args)
    assert groups["gamma_beta"]["lr"] == pytest.approx(1e-4)
    args.adaptive_variant_lr = True
    update_gamma_beta_learning_rate(optimizer, torch.tensor(0.25), args)
    assert groups["variant"]["lr"] == pytest.approx(4e-5)
    update_gamma_beta_learning_rate(optimizer, torch.tensor(1.0), args)
    assert groups["variant"]["lr"] == pytest.approx(1e-4)
    args.fixed_gamma_beta_lr = True
    fixed_optimizer = build_training_optimizer(model, args)
    fixed_groups = {
        group["name"]: group for group in fixed_optimizer.param_groups
    }
    assert fixed_groups["gamma_beta"]["lr"] == pytest.approx(1e-4)


def test_analytic_patch_reliability_and_head_gradient_isolation():
    invariant = torch.zeros(2, 5, 1)
    raw = torch.full_like(invariant, 2.0)
    target = torch.ones_like(invariant)
    analytic = analytic_patch_reliability_target(
        target, invariant, raw, future_patch_len=3
    )
    assert analytic.shape == (2, 2)
    assert torch.allclose(analytic, torch.full_like(analytic, 0.5))

    fusion = HorizonFutureVariant(4, 5, future_patch_len=3)
    z_inv = torch.randn(2, 1, 2, 4, requires_grad=True)
    z_var = torch.randn(2, 1, 2, 4, requires_grad=True)
    output = fusion(torch.zeros(2, 1, 5), z_inv, z_var)
    reliability_loss = torch.nn.functional.mse_loss(
        output["reliability"], torch.zeros_like(output["reliability"])
    )
    reliability_loss.backward()
    assert z_inv.grad is None and z_var.grad is None
    assert fusion.modulation_generator[-1].weight.grad is None
    assert fusion.reliability_net[-1].weight.grad is not None


def test_reliability_target_stability_uses_fixed_sample_correspondence():
    sample_id = torch.tensor([0, 1, 2])
    target = torch.tensor([[0.0, 0.25], [0.5, 1.0], [0.2, 0.8]])
    previous = (sample_id.clone(), target.clone())
    result = reliability_target_stability(
        (sample_id, target), previous_epoch=previous, previous_stage=previous
    )
    assert result["r_target_epoch_correlation"] == pytest.approx(1.0)
    assert result["r_target_epoch_rank_stability"] == pytest.approx(1.0)
    assert result["r_target_stage_correlation"] == pytest.approx(1.0)
    assert result["r_target_clip_zero_ratio"] == pytest.approx(1.0 / 6.0)
    assert result["r_target_clip_one_ratio"] == pytest.approx(1.0 / 6.0)
    assert result["r_target_fixed_sample_count"] == 3


def test_reliability_ranking_and_usefulness_targets():
    target = torch.ones(2, 4, 1)
    invariant = torch.zeros_like(target)
    useful_raw = target.clone()
    usefulness = patch_correction_usefulness_target(
        target, invariant, useful_raw, future_patch_len=2
    )
    assert torch.equal(usefulness, torch.ones_like(usefulness))
    analytic = torch.tensor([[0.0, 1.0]])
    aligned = reliability_pairwise_ranking_loss(
        torch.tensor([[0.1, 0.9]]), analytic
    )
    reversed_prediction = reliability_pairwise_ranking_loss(
        torch.tensor([[0.9, 0.1]]), analytic
    )
    assert aligned < reversed_prediction


def test_invariant_anchored_variant_loss_detaches_anchor_and_normalizes():
    invariant = torch.tensor([[1.0, 2.0]], requires_grad=True)
    delta = torch.tensor([[0.2, -0.1]], requires_grad=True)
    target = torch.tensor([[1.1, 1.9]])
    full = invariant.detach() + delta
    losses = invariant_anchored_variant_loss(
        invariant, full, target, lambda_maturity=1.0
    )
    losses["loss"].backward()
    expected_inv_grad = 2.0 * (invariant.detach() - target) / invariant.numel()
    assert torch.allclose(invariant.grad, expected_inv_grad)
    assert delta.grad is not None
    assert not losses["invariant_error"].requires_grad
    near = invariant_anchored_variant_loss(
        invariant.detach(), invariant.detach() + 0.1,
        invariant.detach() + 0.01, 1.0,
    )
    far = invariant_anchored_variant_loss(
        invariant.detach(), invariant.detach() + 0.1,
        invariant.detach() + 1.0, 1.0,
    )
    assert near["anchor_loss"] > far["anchor_loss"]


def test_invariant_anchored_variant_loss_can_disable_anchor_only():
    invariant = torch.tensor([[1.0, 2.0]], requires_grad=True)
    delta = torch.tensor([[0.2, -0.1]], requires_grad=True)
    target = torch.tensor([[1.1, 1.9]])
    full = invariant.detach() + delta
    losses = invariant_anchored_variant_loss(
        invariant,
        full,
        target,
        lambda_maturity=0.5,
        lambda_anchor=0.0,
    )
    expected = losses["invariant_loss"] + 0.5 * losses["full_loss"]
    assert torch.allclose(losses["loss"], expected)
    assert losses["anchor_loss"] > 0


def test_horizon_future_model_uses_y_space_and_shared_teacher():
    configuration = cfg("A2", "classification", 3, "complementary_gate")
    configuration.variant_fusion_mode = "horizon_future_var"
    configuration.future_patch_len = 3
    model = Model(configuration)
    output = model.forward_components(torch.randn(4, 32, 3))
    assert torch.equal(output["final_tokens"], output["z_inv_tokens"])
    assert torch.equal(output["prediction"], output["invariant_prediction"])
    assert output["horizon_future_zvar"].shape == (4, 3, 3, 16)
    windows = torch.randn(4, 3, 32, 3)
    target = model.horizon_future_variant_targets(windows)
    assert target.shape == (4, 3, 3, 16)
    assert not target.requires_grad


def test_horizon_future_model_exposes_environment_disagreement_feature():
    configuration = cfg("A2", "classification", 2, "complementary_gate")
    configuration.variant_fusion_mode = "horizon_future_var"
    configuration.future_patch_len = 3
    configuration.reliability_environment_disagreement = True
    model = Model(configuration)
    output = model.forward_components(torch.randn(4, 32, 3))
    assert output["horizon_environment_corrections"].shape == (4, 2, 3, 3, 3)
    assert output["horizon_environment_disagreement"].shape == (4, 3)


def test_future_patch_centers_reduce_and_mask_partial_last_patch():
    assert future_patch_centers(8, 3).tolist() == [2, 5, 7]
    values = torch.arange(8.0).view(1, 1, 8)
    reduced = future_patch_mean(values, 3)
    assert torch.allclose(reduced, torch.tensor([[[1.0, 4.0, 6.5]]]))
    fusion = HorizonFutureVariant(4, 8, future_patch_len=3)
    output = fusion(
        torch.randn(2, 1, 8),
        torch.randn(2, 1, 2, 4),
        torch.randn(2, 1, 2, 4),
    )
    assert output["prediction"].shape == (2, 1, 8)
    assert output["gamma"].shape == (2, 1, 8)


def test_horizon_decay_objective_uses_relative_bins_and_normalized_effect():
    invariant = torch.zeros(2, 8, 3)
    effect = torch.tensor([4.0, 4.0, 3.0, 3.0, 2.0, 2.0, 1.0, 1.0])
    full = effect.view(1, 8, 1).expand_as(invariant)
    metrics = horizon_effect_objective(full, invariant, bins=4)
    assert torch.allclose(metrics["var_effect_bins"], torch.tensor([4.0, 3.0, 2.0, 1.0]))
    assert metrics["decay_mono_loss"] == 0
    assert torch.allclose(metrics["var_effect_far_near_ratio"], torch.tensor(0.25))


def test_featurewise_gate_does_not_shift_following_initialization_rng():
    torch.manual_seed(2021)
    _ = DirectGatedVariantFusion(12, gate_type="token")
    token_following = torch.nn.Linear(12, 7).weight.detach().clone()
    torch.manual_seed(2021)
    _ = DirectGatedVariantFusion(12, gate_type="feature")
    feature_following = torch.nn.Linear(12, 7).weight.detach().clone()
    assert torch.equal(token_following, feature_following)


def test_featurewise_fusion_gate_shape_for_all_backbones():
    for backbone in ("patchtst", "cyclenet", "itransformer"):
        config = cfg("A2", "classification", 3, "complementary_gate")
        config.variant_fusion_mode = "direct_gated"
        config.variant_fusion_gate_type = "feature"
        config.predictive_env_backbone = backbone
        config.enc_in = 3
        config.embed = "timeF"
        config.freq = "h"
        config.cyclenet_cycle_len = 24
        model = Model(config)
        output = model.forward_components(
            torch.randn(4, 32, 3), torch.arange(4)
        )
        assert output["feature_gate"].shape == output["z_var_tokens"].shape


def test_variant_gain_loss_is_samplewise_and_detaches_invariant_target():
    full = torch.tensor([0.7, 0.2, 0.4], requires_grad=True)
    invariant = torch.tensor([0.5, 0.3, 0.4], requires_grad=True)
    temperature = 0.05
    loss = variant_gain_loss(full, invariant, temperature)
    expected = temperature * torch.nn.functional.softplus(
        (full - invariant.detach()) / temperature
    ).mean()
    assert torch.allclose(loss, expected)
    loss.backward()
    assert full.grad is not None and torch.all(full.grad > 0)
    assert invariant.grad is None


def test_variant_gain_loss_rejects_batch_averages():
    with pytest.raises(ValueError, match="sample-wise"):
        variant_gain_loss(torch.tensor(0.2), torch.tensor(0.3), 0.01)


def test_future_variant_teacher_uses_shared_stop_gradient_path():
    model = Model(cfg("A2", "classification", 3, "complementary_gate"))
    future = torch.randn(4, 32, 3)
    target = model.future_variant_target(future)
    prediction = model.future_var_predictor(
        model.forward_components(torch.randn(4, 32, 3))["z_var"]
    )
    metrics = future_variant_objective(prediction, target)
    assert target.shape == (4, 16)
    assert not target.requires_grad
    assert prediction.requires_grad
    assert torch.isfinite(metrics["loss"])
    metrics["loss"].backward()
    assert any(
        parameter.grad is not None
        for parameter in model.future_var_predictor.parameters()
    )


def test_future_teacher_window_is_earliest_seq_len_and_train_bounded():
    class FakeDataset:
        def __init__(self):
            self.data_x = torch.arange(30).view(30, 1)

        def __len__(self):
            # Emulate seq_len=8, pred_len=3 base samples.
            return 20

        def __getitem__(self, index):
            x = self.data_x[index : index + 8]
            y = self.data_x[index + 8 : index + 11]
            return x, y, None, None

    wrapped = FutureTeacherIndexedDataset(FakeDataset(), seq_len=8)
    diagnostic = FutureTeacherIndexedDataset(
        FakeDataset(), seq_len=8, valid_only=True
    )
    assert len(wrapped) == 20  # Forecasting samples are unchanged when lambda=0.
    assert len(diagnostic) == 15  # 30 - 2*8 + 1 valid TRAIN teachers.
    _, _, _, first_future, first_valid, _, _ = wrapped[0]
    _, _, _, last_future, last_valid, _, _ = diagnostic[len(diagnostic) - 1]
    assert torch.equal(first_future, torch.arange(8, 16).view(8, 1))
    assert torch.equal(last_future, torch.arange(22, 30).view(8, 1))
    assert first_valid and last_valid
    _, _, _, invalid_future, invalid, _, _ = wrapped[len(wrapped) - 1]
    assert not invalid
    assert torch.count_nonzero(invalid_future) == 0
    anchors = horizon_anchor_positions(8, 4)
    windows, valid_mask, cycles = wrapped.horizon_teacher_batch(
        torch.tensor([0, 14, 19]), anchors
    )
    assert anchors.tolist() == [1, 3, 6, 8]
    assert valid_mask.tolist() == [True, True, False]
    assert windows.shape == (2, 4, 8, 1)
    assert torch.equal(windows[0, 0], torch.arange(1, 9).view(8, 1))
    assert torch.equal(windows[0, -1], torch.arange(8, 16).view(8, 1))
    assert cycles.shape == (2, 4)


def test_train_shuffle_rng_is_independent_of_model_initialization():
    class FakeDataset:
        def __init__(self):
            self.data_x = torch.arange(80).view(80, 1)

        def __len__(self):
            return 64

        def __getitem__(self, index):
            x = self.data_x[index : index + 8]
            y = self.data_x[index + 8 : index + 11]
            return x, y, None, None

    args = SimpleNamespace(
        seq_len=8, batch_size=7, num_workers=0, seed=2021,
        cyclenet_cycle_len=1,
    )
    first_loader = make_train_loader(FakeDataset(), args)
    # Consume substantial global RNG after loader creation, as adding a head does.
    _ = torch.nn.Sequential(torch.nn.Linear(31, 127), torch.nn.Linear(127, 19))
    first_order = torch.cat([batch[2] for batch in first_loader])
    _ = torch.randn(10000)
    second_loader = make_train_loader(FakeDataset(), args)
    second_order = torch.cat([batch[2] for batch in second_loader])
    assert torch.equal(first_order, second_order)
    assert torch.equal(torch.sort(first_order).values, torch.arange(64))


def test_conditional_variant_predictor_detaches_invariant_condition():
    model = Model(cfg("A2", "classification", 3, "complementary_gate"))
    z_inv = torch.randn(4, 3, 8, 16, requires_grad=True)
    z_var = torch.randn(4, 3, 8, 16, requires_grad=True)
    prediction = model.variant_conditional_predictor(z_inv, z_var)
    assert prediction.shape == (4, 8, 3)
    prediction.square().mean().backward()
    assert z_inv.grad is None
    assert z_var.grad is not None and float(z_var.grad.abs().sum()) > 0


def test_utility_loss_is_reliability_weighted_and_only_penalizes_harm():
    reliability = torch.tensor([0.5, 0.5], requires_grad=True)
    full_loss = torch.tensor([2.0, 1.0])
    invariant_loss = torch.tensor([1.0, 2.0])
    loss = reliability_weighted_utility_loss(
        reliability, full_loss, invariant_loss
    )
    assert torch.allclose(loss, torch.tensor(0.25))
    loss.backward()
    assert torch.allclose(reliability.grad, torch.tensor([0.5, 0.0]))


def test_conditional_negative_is_deterministic_derangement():
    model = Model(cfg("A2", "classification", 3, "complementary_gate"))
    z_inv = torch.randn(5, 3, 8, 16)
    z_var = torch.randn(5, 3, 8, 16)
    paired, shuffled, permutation = (
        model.variant_conditional_predictor.paired_and_shuffled(z_inv, z_var)
    )
    expected = torch.roll(torch.arange(5), shifts=1)
    assert torch.equal(permutation.cpu(), expected)
    assert torch.all(permutation != torch.arange(5))
    assert paired.shape == shuffled.shape == (5, 8, 3)


def test_conditional_ranking_detaches_shuffled_loss():
    paired_loss = torch.tensor([2.0, 0.5], requires_grad=True)
    shuffled_loss = torch.tensor([1.0, 1.0], requires_grad=True)
    loss = conditional_gain_ranking_loss(paired_loss, shuffled_loss, margin=0.0)
    assert torch.allclose(loss, torch.tensor(0.5))
    loss.backward()
    assert torch.allclose(paired_loss.grad, torch.tensor([0.5, 0.0]))
    assert shuffled_loss.grad is None


def test_frozen_h_head_is_separate_from_trainable_z_forecast_head():
    model = Model(cfg("A2", "classification", 3, "complementary_gate"))
    model.initialize_h_forecast_head()
    model.eval()
    x = torch.randn(3, 32, 2)
    before = model.forward_components(x)["h_prediction"].detach()
    assert all(not parameter.requires_grad for parameter in model.h_forecast_head.parameters())
    with torch.no_grad():
        model.head_linear.weight.add_(1.0)
        model.head_linear.bias.add_(1.0)
    after = model.forward_components(x)["h_prediction"].detach()
    assert torch.equal(before, after)


def test_detached_environment_decomposition_cannot_update_hidden():
    model = Model(cfg("A2", "classification", 3, "complementary_gate"))
    hidden = torch.randn(4, 2, 8, 16, requires_grad=True)
    z_inv, z_var, _ = model.detached_decomposition(hidden)
    (z_inv.square().mean() + z_var.square().mean()).backward()
    assert hidden.grad is None
    assert any(parameter.grad is not None for parameter in model.decomposer.parameters())


def test_all_detached_environment_losses_leave_encoder_without_gradient():
    configuration = cfg("A2", "classification", 3, "complementary_gate")
    configuration.variant_fusion_mode = "direct_gated"
    model = Model(configuration)
    output = model.forward_components(torch.randn(4, 32, 2))
    environment_output = model.detached_environment_components(output)
    q = torch.softmax(torch.randn(4, 3), dim=-1)
    domain_loss, _ = model.environment_classification_loss(
        environment_output["z_inv"], environment_output["z_var"], q
    )
    loss = environment_output["prediction"].square().mean() + domain_loss
    loss.backward()
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    assert all(
        parameter.grad is None for parameter in model.patch_embedding.parameters()
    )
    assert any(parameter.grad is not None for parameter in model.decomposer.parameters())


def test_rms_fusion_calibration_preserves_each_sample_variable_scale():
    configuration = cfg("A2", "classification", 3, "complementary_gate")
    configuration.variant_fusion_mode = "direct_gated"
    configuration.fusion_scale_calibration = "rms"
    model = Model(configuration)
    output = model.forward_components(torch.randn(4, 32, 3))
    hidden_rms = output["hidden_tokens"].square().mean((-2, -1)).sqrt()
    fused_rms = output["final_tokens"].square().mean((-2, -1)).sqrt()
    assert torch.allclose(hidden_rms, fused_rms, atol=1e-6, rtol=1e-5)
