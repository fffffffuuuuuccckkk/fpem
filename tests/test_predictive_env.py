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
    SignedGateDecomposer,
    signed_gate_diagnostics,
)
from models.predictive_env.future_var_consistency import future_variant_objective
from models.predictive_env.conditional_variant_predictor import (
    conditional_gain_ranking_loss,
    reliability_weighted_utility_loss,
)
from tools.run_predictive_env_iv_patchtst import (
    FutureTeacherIndexedDataset,
    make_train_loader,
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
    _, _, _, first_future, first_valid = wrapped[0]
    _, _, _, last_future, last_valid = diagnostic[len(diagnostic) - 1]
    assert torch.equal(first_future, torch.arange(8, 16).view(8, 1))
    assert torch.equal(last_future, torch.arange(22, 30).view(8, 1))
    assert first_valid and last_valid
    _, _, _, invalid_future, invalid = wrapped[len(wrapped) - 1]
    assert not invalid
    assert torch.count_nonzero(invalid_future) == 0


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

    args = SimpleNamespace(seq_len=8, batch_size=7, num_workers=0, seed=2021)
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
