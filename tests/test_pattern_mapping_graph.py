"""Unit tests for the research-critical FPem-PMG behaviors."""

import inspect
from types import SimpleNamespace

import pytest
import torch

from models.fpem import (
    combine_invariant_evidence,
    EnvironmentFusion,
    PatternMappingFPem,
    PatternMappingGraphBuilder,
    PatternMappingRetriever,
    StableMappingGraph,
    StablePatternGraph,
    TypedVariationFusion,
)
from models.fpem.pattern_mapping_graph import InvariantPatternEncoder, VariantMappingCorrection
from models.fpem.future_mapping import (
    AdaptiveVariantFutureMapper, FutureForecastCorrection,
    FutureMappingMemoryBuilder, HistoryMappingContextEncoder, StableFutureMappingMemory,
)


def test_mapping_mixture_is_soft_train_only_and_normalized():
    builder = PatternMappingGraphBuilder(2, seq_len=2)
    mapping = {
        "src_index": torch.arange(6),
        "coverage": torch.tensor([.95, .8, .65, .3, .2, .1]),
        "counts": torch.tensor([100., 80., 60., 20., 15., 10.]),
        "sample_entropy": torch.tensor([.95, .9, .8, .3, .2, .1]),
        "stability": torch.tensor([.9, .85, .8, .75, .7, .65]),
        "predictive_gain": torch.zeros(6),
        "delta_means": torch.randn(6, 2),
    }
    builder._attach_mapping_mixture(mapping)
    assert torch.allclose(mapping["p_inv"] + mapping["p_var"], torch.ones(6))
    assert torch.isfinite(mapping["p_inv"]).all()
    assert mapping["p_inv"][:3].mean() > mapping["p_inv"][3:].mean()


def test_variant_mapping_query_is_sample_conditioned_and_temporally_aligned():
    _, mapping = _graphs(include_ab=True)
    mapping.p_inv = torch.tensor([1.0, 1.0, 0.0])
    mapping.p_var = 1.0 - mapping.p_inv
    mapping.local_delta_means = mapping.delta_means.clone()
    means = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    query = torch.tensor([
        [[[1.0, 0.0], [0.0, 1.0]]],
        [[[1.0, 0.0], [1.0, 0.0]]],
    ])
    responsibility = torch.tensor([
        [[[1.0, 0.0], [0.0, 1.0]]],
        [[[1.0, 0.0], [1.0, 0.0]]],
    ])
    result = mapping.query(query, responsibility, means)
    assert result["variant_activation"][0, 0, 1] > result["variant_activation"][1, 0, 1]
    assert torch.equal(result["variant_delta"][:, :, 0], torch.zeros_like(result["variant_delta"][:, :, 0]))
    assert result["variant_delta"].shape == query.shape


def test_variant_mapping_correction_zero_init_is_identity_with_conservative_gate():
    module = VariantMappingCorrection(4, 2)
    z_typed = torch.randn(2, 1, 3, 4)
    result = module(z_typed, torch.randn(2, 1, 3, 2), torch.ones(2, 1, 3))
    assert torch.equal(result["variant_mapping_correction"], torch.zeros_like(z_typed))
    assert torch.allclose(
        result["variant_mapping_gate"],
        torch.full_like(result["variant_mapping_gate"], torch.sigmoid(torch.tensor(-2.0))),
    )


def test_future_memory_is_train_block_aggregated_and_leave_block_out():
    history = torch.randn(8, 2, 3, 5)
    beta = torch.softmax(torch.randn(8, 2, 2, 4), -1)
    statistics = FutureMappingMemoryBuilder(seq_len=2).build(history, beta)
    assert torch.all((statistics["p_inv"] >= 0) & (statistics["p_inv"] <= 1))
    assert statistics["history_keys"].shape[1:] == (3, 5)
    memory = StableFutureMappingMemory()
    memory.load_statistics(statistics)
    encoder = HistoryMappingContextEncoder(5, 8)
    result = memory.query(history[:2], encoder, leave_nearest_out=True)
    assert result["stable_future_logits"].shape == (2, 2, 2, 4)
    assert torch.all(result["excluded_memory_index"] >= 0)
    assert torch.isfinite(result["stable_future_logits"]).all()


def test_adaptive_future_mapper_is_sample_conditioned_and_modes_are_exact():
    mapper = AdaptiveVariantFutureMapper(8, future_patterns=4, future_patches=2)
    sequence = torch.randn(2, 1, 3, 8)
    context = torch.randn(2, 1, 8)
    stable = torch.randn(2, 1, 2, 4)
    summary = torch.randn(2, 1, 6)
    off = mapper(sequence, context, stable, summary, "off")
    unit = mapper(sequence, context, stable, summary, "unit")
    assert torch.equal(off["final_future_logits"], stable)
    assert torch.equal(unit["future_variant_gate"], torch.ones_like(unit["future_variant_gate"]))
    assert not torch.allclose(unit["variant_future_logits"][0], unit["variant_future_logits"][1])


def test_future_forecast_correction_is_identity_at_initialization():
    module = FutureForecastCorrection(8)
    base = torch.randn(2, 6, 3)
    context = torch.randn_like(base)
    result = module(base, context, torch.rand(2, 3), torch.rand(2, 3, 1, 1))
    assert torch.equal(result["future_forecast"], base)
    assert torch.equal(result["future_forecast_correction"], torch.zeros_like(base))


def _graphs(include_ab=False, abnormal_ab_delta=False):
    pattern = StablePatternGraph(2)
    pattern.load_statistics(
        {
            "means": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "variances": torch.full((2, 2), 0.01),
            "counts": torch.tensor([100.0, 100.0]),
            "window_support": torch.tensor([50.0, 50.0]),
            "predictive_support": torch.ones(2),
            "stability": torch.ones(2),
            "distance_cdf": torch.tensor([0.0, 0.05, 0.1, 0.2, 0.4]),
        }
    )
    mapping = StableMappingGraph(2)
    src = [0, 1]
    dst = [0, 1]
    delta_means = [[0.0, 0.0], [0.0, 0.0]]
    if include_ab:
        src.append(0)
        dst.append(1)
        delta_means.append([0.0, 0.0] if abnormal_ab_delta else [-1.0, 1.0])
    edge_count = len(src)
    mapping.load_statistics(
        {
            "src_index": torch.tensor(src, dtype=torch.long),
            "dst_index": torch.tensor(dst, dtype=torch.long),
            "counts": torch.full((edge_count,), 100.0),
            "window_support": torch.full((edge_count,), 50.0),
            "delta_means": torch.tensor(delta_means),
            "delta_variances": torch.full((edge_count, 2), 0.01),
            "stability": torch.ones(edge_count),
            "distance_cdf": torch.tensor([0.0, 0.05, 0.1, 0.2, 0.4]),
        }
    )
    return pattern, mapping


def test_high_support_pattern_and_mapping_have_high_invariant_evidence():
    pattern, mapping = _graphs()
    result = PatternMappingRetriever(pattern, mapping)(
        torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]]), True, True
    )
    assert result["c_inv"][0, 0, 1] > 0.9
    assert 1.0 - result["c_inv"][0, 0, 1] < 0.1


def test_cinv_b0_b1_b2_formulas_and_activation_regression():
    c_pat = torch.tensor([0.2, 0.5, 1.0])
    c_map = torch.tensor([0.0, 0.4, 1.0])
    product = combine_invariant_evidence(c_pat, c_map, "product")
    soft_mapping = combine_invariant_evidence(c_pat, c_map, "soft_mapping")
    pattern_only = combine_invariant_evidence(c_pat, c_map, "pattern_only")
    assert torch.equal(product, c_pat * c_map)
    assert torch.equal(1.0 - product, 1.0 - c_pat * c_map)
    assert torch.equal(soft_mapping, c_pat * (1.0 + c_map) * 0.5)
    assert soft_mapping[0] == 0.5 * c_pat[0]
    assert soft_mapping[-1] == c_pat[-1]
    assert torch.equal(pattern_only, c_pat)


def test_pattern_only_cinv_still_queries_mapping_graph():
    pattern, mapping = _graphs(include_ab=False)
    result = PatternMappingRetriever(pattern, mapping)(
        torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]), True, True, "pattern_only"
    )
    assert torch.equal(result["c_inv"], result["c_pat"])
    assert result["mapping"]["novelty"][0, 0, 1] > 0.9
    assert result["c_map"][0, 0, 1] < 0.1
    assert result["mapping"]["context"].shape[-1] == 2


def test_pattern_query_exposes_shape_recurrence_and_predictive_confidence():
    pattern, _ = _graphs()
    query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    result = pattern.query(query)
    responsibility = result["responsibility"]
    recurrence = torch.log1p(pattern.window_support) / torch.log1p(pattern.window_support.max()).clamp_min(1.0)
    assert torch.equal(result["shape_compatibility"], 1.0 - result["novelty"])
    assert torch.allclose(result["recurrence_confidence"], responsibility @ recurrence)
    assert torch.allclose(result["predictive_confidence"], responsibility @ pattern.predictive_support)


def test_unseen_pattern_has_high_pattern_novelty():
    pattern, _ = _graphs()
    result = pattern.query(torch.tensor([[[[-1.0, 0.0]]]]))
    assert result["novelty"].item() > 0.9


def test_seen_patterns_but_unseen_mapping_activates_variant():
    pattern, mapping = _graphs(include_ab=False)
    result = PatternMappingRetriever(pattern, mapping)(
        torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]), True, True
    )
    assert result["pattern"]["novelty"].max() < 0.1
    assert result["mapping"]["novelty"][0, 0, 1] > 0.9
    assert 1.0 - result["c_inv"][0, 0, 1] > 0.9


def test_seen_mapping_pair_with_abnormal_delta_is_novel():
    pattern, mapping = _graphs(include_ab=True, abnormal_ab_delta=True)
    result = PatternMappingRetriever(pattern, mapping)(
        torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]), True, True
    )
    assert result["mapping"]["novelty"][0, 0, 1] > 0.9


def test_variant_encoder_is_not_subtraction():
    module = PatternMappingFPem(4, pattern_dim=2, env_dim=2, use_pattern=False)
    hidden = torch.randn(2, 1, 3, 4)
    result = module.decompose(hidden)
    assert not torch.allclose(result["z_var_raw"], hidden - result["z_inv"])


def test_invariant_encoder_falls_back_to_graph_context_without_copying_hidden():
    encoder = InvariantPatternEncoder(hidden_dim=4, pattern_dim=2, zinv_mode="interpolate")
    hidden = torch.randn(2, 1, 3, 4)
    pattern_context = torch.randn(2, 1, 3, 2)
    mapping_context = torch.randn(2, 1, 3, 2)
    low_result = encoder(
        hidden, pattern_context, mapping_context, torch.zeros(2, 1, 3)
    )
    high_result = encoder(
        hidden, pattern_context, mapping_context, torch.ones(2, 1, 3)
    )
    low_evidence = low_result["z_inv"]
    graph_context = low_result["graph_context"]
    high_evidence = high_result["z_inv"]
    assert torch.allclose(low_evidence, graph_context)
    assert not torch.allclose(low_evidence, hidden)
    assert not torch.allclose(high_evidence, hidden)


def test_zinv_modes_reproduce_interpolation_hidden_only_and_zero_init_correction():
    hidden = torch.randn(2, 3, 4, 6)
    pattern_context = torch.randn(2, 3, 4, 2)
    mapping_context = torch.randn(2, 3, 4, 2)
    stable_delta = torch.randn(2, 3, 4, 2)
    c_pat = torch.rand(2, 3, 4)

    c0 = InvariantPatternEncoder(6, 2, "interpolate")(
        hidden, pattern_context, mapping_context, c_pat, stable_delta
    )
    expected_c0 = c_pat.unsqueeze(-1) * c0["invariant_base"] + (
        1.0 - c_pat.unsqueeze(-1)
    ) * c0["graph_context"]
    assert torch.allclose(c0["z_inv"], expected_c0)

    c1 = InvariantPatternEncoder(6, 2, "hidden_only")(
        hidden, pattern_context, mapping_context, c_pat, stable_delta
    )
    assert torch.equal(c1["z_inv"], c1["invariant_base"])

    c2_encoder = InvariantPatternEncoder(6, 2, "stable_relation_correction")
    c2 = c2_encoder(hidden, pattern_context, mapping_context, c_pat, stable_delta)
    assert torch.equal(c2["z_inv"], c2["invariant_base"])
    assert torch.count_nonzero(c2["graph_correction"]) == 0
    expected_gate = c_pat.unsqueeze(-1) * torch.sigmoid(torch.tensor(-2.0))
    assert torch.allclose(c2["graph_gate"], expected_gate, atol=1e-7)
    assert torch.count_nonzero(c2_encoder.graph_correction_encoder[-1].weight) == 0


def test_stable_mapping_delta_preserves_batch_channel_temporal_correspondence():
    pattern, mapping = _graphs(include_ab=True)
    query = torch.tensor(
        [
            [[[1.0, 0.0], [0.0, 1.0]]],
            [[[1.0, 0.0], [1.0, 0.0]]],
        ]
    )
    pattern_result = pattern.query(query)
    result = mapping.query(query, pattern_result["responsibility"], pattern.means, use_delta=True)
    assert result["stable_delta"].shape == query.shape
    assert torch.equal(result["stable_delta"][:, :, 0], torch.zeros_like(query[:, :, 0]))
    assert torch.allclose(result["stable_delta"][0, 0, 1], torch.tensor([-1.0, 1.0]), atol=1e-3)
    assert torch.allclose(result["stable_delta"][1, 0, 1], torch.tensor([0.0, 0.0]), atol=1e-3)


def test_zero_reliability_is_exact_invariant_fallback():
    fusion = EnvironmentFusion(hidden_dim=4, env_dim=2)
    z_inv = torch.randn(2, 1, 3, 4)
    environment = torch.randn(2, 2)
    fused, _, _ = fusion(z_inv, environment, torch.zeros(2))
    assert torch.allclose(fused, z_inv, atol=1e-7)


def test_checkpoint_roundtrip_preserves_dynamic_graph_statistics():
    pattern, mapping = _graphs(include_ab=True)
    module = PatternMappingFPem(4, pattern_dim=2, env_dim=2)
    module.pattern_graph.load_statistics(pattern.export())
    module.mapping_graph.load_statistics(mapping.export())
    state = module.state_dict()
    restored = PatternMappingFPem(4, pattern_dim=2, env_dim=2)
    restored.load_state_dict(state)
    assert torch.equal(restored.pattern_graph.means, module.pattern_graph.means)
    assert torch.equal(restored.mapping_graph.src_index, module.mapping_graph.src_index)
    assert torch.equal(restored.mapping_graph.delta_variances, module.mapping_graph.delta_variances)


def test_graph_builder_api_has_no_validation_or_test_loader():
    parameters = inspect.signature(PatternMappingGraphBuilder.build_from_embeddings).parameters
    assert "val_loader" not in parameters
    assert "test_loader" not in parameters
    embeddings = torch.randn(6, 1, 3, 2)
    artifact = PatternMappingGraphBuilder(2, seq_len=3).build_from_embeddings(
        embeddings, torch.randn(6, 1, 4)
    )
    assert artifact["metadata"]["source_split"] == "train"
    assert artifact["metadata"]["predictive_target_representation"] == "normalized_full_horizon"


def test_sliding_windows_count_support_once_per_non_overlapping_anchor():
    embeddings = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]]).repeat(8, 1, 1, 1)
    futures = torch.ones(8, 1, 3)
    artifact = PatternMappingGraphBuilder(2, seq_len=4).build_from_embeddings(embeddings, futures)
    assert artifact["metadata"]["number_of_anchor_groups"] == 2
    assert artifact["pattern_graph"]["window_support"].max().item() == 2
    if artifact["mapping_graph"]["window_support"].numel():
        assert artifact["mapping_graph"]["window_support"].max().item() == 2


def test_mapping_builder_has_no_fixed_top2():
    source = inspect.getsource(PatternMappingGraphBuilder._mapping_statistics)
    assert "topk" not in source
    assert "_posterior_mass_prune" in source


def test_retrospective_api_cannot_receive_future_target():
    from models.PatchTST_FPEM import Model

    parameters = inspect.signature(Model._retrospective_gain).parameters
    assert list(parameters) == ["self", "x_norm"]


def test_patchtst_pmg_forward_loss_and_diagnostics_smoke():
    from models.PatchTST_FPEM import Model, _PatchTSTEncoder

    config = SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=32,
        pred_len=8,
        enc_in=3,
        d_model=8,
        patch_len=16,
        factor=1,
        dropout=0.0,
        n_heads=2,
        d_ff=16,
        activation="gelu",
        e_layers=1,
        fpem_pmg_enabled=1,
        fpem_pmg_pattern_dim=4,
        fpem_pmg_env_dim=3,
    )
    model = Model(config)
    assert model.use_retrospective is False
    assert model.representation_space == "embedding"
    assert tuple(model.pmg.reliability_head.linear.weight.shape) == (1, 3)
    x = torch.randn(2, 32, 3)
    marks = torch.zeros(2, 32, 4)
    decoder = torch.zeros(2, 24, 3)
    assert model.fpem_pmg_warmup_forecast(x).shape == (2, 8, 3)
    model.fpem_pmg_warmup_forecast(x).sum().backward()
    assert any(parameter.grad is not None for parameter in model.pmg.projector.parameters())
    prediction = model(x, marks, decoder, marks[:, :24])
    assert prediction.shape == (2, 8, 3)
    extra_loss, logs = model.fpem_extra_loss(torch.randn(2, 8, 3))
    assert torch.isfinite(extra_loss)
    assert "loss/inv" in logs
    diagnostics = model.forward_with_diagnostics(x, marks, decoder, marks[:, :24])
    assert set(diagnostics) == {
        "prediction",
        "prediction_inv",
        "pattern_novelty",
        "mapping_novelty",
        "variation_activation",
        "environment",
        "environment_reliability",
    }
    model.freeze_fpem_pmg_graph_space()
    model.train()
    assert not any(parameter.requires_grad for parameter in model.encoder_backbone.patch_embedding.parameters())
    assert any(parameter.requires_grad for parameter in model.encoder_backbone.encoder.parameters())
    assert not any(parameter.requires_grad for parameter in model.pmg.projector.parameters())
    assert model.encoder_backbone.encoder.training is True
    assert model.encoder_backbone.patch_embedding.training is False
    assert model.pmg.projector.training is False
    assert model.pmg.invariant_encoder.training is True

    encoder = _PatchTSTEncoder(config, patch_len=16, stride=8)
    encoder.eval()
    with torch.no_grad():
        direct_embedding, n_vars = encoder.patch_embedding(x.permute(0, 2, 1))
        features = encoder.forward_features(x)
    assert n_vars == 3
    assert features["embedding"].shape == features["hidden"].shape == (2, 3, 4, 8)
    assert torch.equal(features["embedding"], direct_embedding.reshape(2, 3, 4, 8))
    assert not torch.allclose(features["embedding"], features["hidden"])
    expected_delta = features["embedding"][:, :, 1:, :] - features["embedding"][:, :, :-1, :]
    assert expected_delta.shape[2] == features["embedding"].shape[2] - 1


def test_raw_patch_extraction_matches_patchtst_temporal_anchors():
    from models.PatchTST_FPEM import _PatchTSTEncoder, extract_raw_patches

    config = SimpleNamespace(
        d_model=8, dropout=0.0, factor=1, n_heads=2, d_ff=16,
        activation="gelu", e_layers=1,
    )
    x = torch.arange(2 * 32 * 3, dtype=torch.float32).reshape(2, 32, 3)
    encoder = _PatchTSTEncoder(config, patch_len=16, stride=8)
    features = encoder.forward_features(x)
    raw = extract_raw_patches(x, patch_len=16, stride=8, padding=8)
    manual = torch.nn.functional.pad(x.permute(0, 2, 1), (0, 8), mode="replicate").unfold(-1, 16, 8)
    assert torch.equal(raw["patches"], manual)
    assert raw["patches"].shape[:3] == features["embedding"].shape[:3]
    assert raw["patches"].shape[:3] == features["hidden"].shape[:3]


def test_raw_patch_shape_normalization_is_finite_for_constant_patch():
    from models.PatchTST_FPEM import extract_raw_patches

    raw = extract_raw_patches(torch.ones(2, 32, 3), 16, 8, 8)
    assert torch.isfinite(raw["normalized"]).all()
    assert torch.count_nonzero(raw["normalized"]) == 0
    assert torch.count_nonzero(raw["std"]) == 0


def test_raw_graph_is_train_only_interpretable_and_deterministic():
    raw_a = torch.tensor([-1.3416, -0.4472, 0.4472, 1.3416])
    raw_b = raw_a.flip(0)
    one_window = torch.stack(
        [torch.stack([raw_a, raw_b, raw_a]), torch.stack([raw_b, raw_a, raw_b])]
    )
    raw_windows = one_window.unsqueeze(0).repeat(12, 1, 1, 1)
    futures = torch.stack(
        [torch.linspace(-1, 1, 6), torch.linspace(1, -1, 6)]
    ).unsqueeze(0).repeat(12, 1, 1)
    builder = PatternMappingGraphBuilder(
        4, seq_len=3, representation_space="raw", patch_len=4, stride=2
    )
    raw_mean = torch.arange(12).float()[:, None, None].expand(12, 2, 3)
    raw_std = torch.ones(12, 2, 3)
    first = builder.build_from_embeddings(
        raw_windows, futures, raw_patch_mean=raw_mean, raw_patch_std=raw_std
    )
    second = builder.build_from_embeddings(
        raw_windows, futures, raw_patch_mean=raw_mean, raw_patch_std=raw_std
    )
    assert first["metadata"]["source_split"] == "train"
    assert first["metadata"]["representation_space"] == "raw"
    assert first["metadata"]["projector_used"] is False
    assert first["pattern_graph"]["means"].shape == second["pattern_graph"]["means"].shape
    assert torch.equal(first["pattern_graph"]["window_support"], second["pattern_graph"]["window_support"])
    assert torch.allclose(first["pattern_graph"]["stability"], second["pattern_graph"]["stability"])
    assert first["pattern_graph"]["prototype_medoid"].shape == first["pattern_graph"]["means"].shape
    assert first["pattern_graph"]["medoid_window_index"].min() >= 0
    assert first["pattern_graph"]["medoid_window_index"].max() < raw_windows.shape[0]
    assert first["pattern_graph"]["medoid_variable_index"].max() < raw_windows.shape[1]
    assert first["pattern_graph"]["medoid_patch_index"].max() < raw_windows.shape[2]
    assert first["mapping_graph"]["delta_means"].shape[-1] == 4
    assert first["pattern_graph"]["future_ratio"].shape == first["pattern_graph"]["stability"].shape
    assert first["pattern_graph"]["future_prototypes"].shape[-1] == futures.shape[-1]
    assert first["mapping_graph"]["future_prototypes"].shape[-1] == futures.shape[-1]
    assert first["mapping_graph"]["predictive_gain"].shape == first["mapping_graph"]["stability"].shape
    assert first["pattern_graph"]["stable_level_center"].shape[1] == 2
    assert first["pattern_graph"]["stable_log_scale_center"].shape[1] == 2
    assert torch.isfinite(first["pattern_graph"]["stable_level_mad"]).all()
    assert first["pattern_graph"]["shift_score_cdf"].numel() > 0
    assert first["pattern_graph"]["scale_score_cdf"].numel() > 0


def test_predictive_stability_modes_use_only_supplied_train_statistics():
    raw = torch.randn(12, 2, 3, 4)
    raw = (raw - raw.mean(-1, keepdim=True)) / (raw.std(-1, keepdim=True, unbiased=False) + 1e-5)
    futures = torch.randn(12, 2, 6)
    futures[:, 1] = 4.0 * futures[:, 1] + 10.0
    history_last = torch.randn(12, 2)
    history_std = torch.rand(12, 2) + 0.5
    raw_mean = torch.randn(12, 2, 3)
    raw_std = torch.rand(12, 2, 3) + 0.2
    artifacts = {}
    for mode in ("per_future_znorm", "train_global", "relative_future"):
        artifacts[mode] = PatternMappingGraphBuilder(
            4, seq_len=3, representation_space="raw", patch_len=4,
            stride=2, predictive_stability_mode=mode,
        ).build_from_embeddings(
            raw, futures, history_last=history_last, history_std=history_std,
            raw_patch_mean=raw_mean, raw_patch_std=raw_std,
        )
        assert artifacts[mode]["metadata"]["predictive_stability_mode"] == mode
        assert torch.isfinite(artifacts[mode]["pattern_graph"]["future_ratio"]).all()
    expected_mean = futures.mean(dim=(0, 2))
    assert torch.allclose(
        artifacts["train_global"]["metadata"]["future_train_mean"], expected_mean
    )
    assert "future_train_mean" not in artifacts["per_future_znorm"]["metadata"]


def test_mapping_use_ablation_keeps_pattern_gate_fixed():
    pattern, mapping = _graphs(include_ab=True)
    full = PatternMappingFPem(4, pattern_dim=2, env_dim=2, mapping_use_mode="full")
    none = PatternMappingFPem(4, pattern_dim=2, env_dim=2, mapping_use_mode="none")
    none.load_state_dict(full.state_dict())
    for module in (full, none):
        module.pattern_graph.load_statistics(pattern.export())
        module.mapping_graph.load_statistics(mapping.export())
    hidden = torch.randn(1, 1, 2, 4)
    graph_query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    full_result = full.decompose(hidden, graph_query=graph_query)
    none_result = none.decompose(hidden, graph_query=graph_query)
    assert torch.equal(full_result["c_inv"], none_result["c_inv"])
    assert torch.equal(full_result["variation_activation"], none_result["variation_activation"])
    assert not torch.allclose(full_result["consensus_context"], none_result["consensus_context"])


def test_raw_mode_bypasses_projector_and_keeps_backbone_trainable():
    from models.PatchTST_FPEM import Model, extract_raw_patches

    config = SimpleNamespace(
        task_name="long_term_forecast", seq_len=32, pred_len=8, enc_in=3,
        d_model=8, patch_len=16, factor=1, dropout=0.0, n_heads=2,
        d_ff=16, activation="gelu", e_layers=1, fpem_pmg_enabled=1,
        fpem_pmg_pattern_dim=4, fpem_pmg_env_dim=3,
        fpem_pmg_representation_space="raw",
    )
    model = Model(config)
    x = torch.randn(2, 32, 3)
    query = model.fpem_pmg_pattern_representation(x)
    expected = extract_raw_patches(x, 16, 8, 8)["normalized"]
    assert torch.equal(query, expected)
    model.fpem_pmg_warmup_forecast(x).sum().backward()
    assert all(parameter.grad is None for parameter in model.pmg.projector.parameters())
    model.freeze_fpem_pmg_graph_space()
    model.train()
    assert all(parameter.requires_grad for parameter in model.encoder_backbone.parameters())
    assert not any(parameter.requires_grad for parameter in model.pmg.projector.parameters())
    assert model.encoder_backbone.training is True


def test_raw_freeze_patch_control_only_freezes_patch_embedding():
    from models.PatchTST_FPEM import Model

    config = SimpleNamespace(
        task_name="long_term_forecast", seq_len=32, pred_len=8, enc_in=3,
        d_model=8, patch_len=16, factor=1, dropout=0.0, n_heads=2,
        d_ff=16, activation="gelu", e_layers=1, fpem_pmg_enabled=1,
        fpem_pmg_pattern_dim=4, fpem_pmg_env_dim=3,
        fpem_pmg_representation_space="raw",
        fpem_pmg_freeze_patch_embedding_stage2="true",
    )
    model = Model(config)
    model.freeze_fpem_pmg_graph_space()
    model.train()
    assert not any(parameter.requires_grad for parameter in model.encoder_backbone.patch_embedding.parameters())
    assert any(parameter.requires_grad for parameter in model.encoder_backbone.encoder.parameters())
    assert model.encoder_backbone.patch_embedding.training is False
    assert model.encoder_backbone.encoder.training is True


def test_raw_deviation_variant_input_is_linear_and_keeps_activation_definition():
    module = PatternMappingFPem(
        6, pattern_dim=4, env_dim=3, use_pattern=False,
        variant_input_mode="raw_deviation",
    )
    assert isinstance(module.variant_encoder.raw_deviation_projection, torch.nn.Linear)
    assert module.variant_encoder.raw_deviation_projection.in_features == 4
    assert module.variant_encoder.raw_deviation_projection.out_features == 6
    assert module.variant_encoder.net[1].in_features == 18
    hidden = torch.randn(2, 1, 3, 6)
    graph_query = torch.randn(2, 1, 3, 4)
    result = module.decompose(hidden, graph_query=graph_query)
    assert result["raw_deviation_projected"].shape == hidden.shape
    assert result["z_var"].shape == hidden.shape
    assert torch.equal(result["variation_activation"], 1.0 - result["c_pat"])
    assert result["z_var_raw"].shape[-1] == 6
    assert result["pattern_deviation"].shape[-1] == 4


def test_channel_conditioned_stable_geometry_and_empirical_novelty():
    pattern, _ = _graphs()
    statistics = pattern.export()
    statistics.update({
        "stable_level_center": torch.tensor([[10.0], [20.0]]),
        "stable_level_mad": torch.tensor([[2.0], [4.0]]),
        "stable_log_scale_center": torch.log(torch.tensor([[2.0], [8.0]])),
        "stable_log_scale_mad": torch.tensor([[0.2], [0.4]]),
        "geometry_support": torch.tensor([[20.0], [30.0]]),
        "shift_score_cdf": torch.tensor([0.0, 0.5, 1.0, 2.0, 4.0]),
        "scale_score_cdf": torch.tensor([0.0, 0.5, 1.0, 2.0, 4.0]),
    })
    pattern.load_statistics(statistics)
    responsibility = torch.tensor([[[[0.25, 0.75]]]])
    current_level = torch.tensor([[[19.0]]])
    current_scale = torch.tensor([[[4.0]]])
    result = pattern.geometry(responsibility, current_level, current_scale)
    assert torch.allclose(result["stable_level"], torch.tensor([[[17.5]]]))
    expected_scale = torch.exp(0.25 * torch.log(torch.tensor(2.0)) + 0.75 * torch.log(torch.tensor(8.0)))
    assert torch.allclose(result["stable_scale"], expected_scale.view(1, 1, 1))
    assert torch.allclose(
        result["shift_signed"], (current_level - result["stable_level"]) / result["stable_scale"]
    )
    assert torch.allclose(
        result["scale_signed"], torch.log(current_scale / result["stable_scale"])
    )
    assert (result["u_shift"] >= 0).all() and (result["u_shift"] <= 1).all()
    assert (result["u_scale"] >= 0).all() and (result["u_scale"] <= 1).all()


def test_factorized_full_is_zero_init_residual_with_soft_or_activation():
    pattern, mapping = _graphs(include_ab=True)
    statistics = pattern.export()
    statistics.update({
        "stable_level_center": torch.tensor([[0.0], [1.0]]),
        "stable_level_mad": torch.ones(2, 1),
        "stable_log_scale_center": torch.zeros(2, 1),
        "stable_log_scale_mad": torch.ones(2, 1),
        "geometry_support": torch.full((2, 1), 10.0),
        "shift_score_cdf": torch.tensor([0.0, 0.1, 0.2, 0.5, 1.0]),
        "scale_score_cdf": torch.tensor([0.0, 0.1, 0.2, 0.5, 1.0]),
    })
    pattern.load_statistics(statistics)
    module = PatternMappingFPem(
        6, pattern_dim=2, env_dim=3, variant_input_mode="factorized_full"
    )
    module.load_graph_statistics({"pattern_graph": pattern.export(), "mapping_graph": mapping.export()})
    hidden = torch.randn(2, 1, 2, 6)
    query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]).repeat(2, 1, 1, 1)
    current_level = torch.tensor([[[0.0, 3.0]], [[-2.0, 1.0]]])
    current_scale = torch.tensor([[[1.0, 2.0]], [[0.5, 1.0]]])
    result = module.decompose(
        hidden, graph_query=query,
        raw_patch_mean=current_level, raw_patch_std=current_scale,
    )
    assert torch.count_nonzero(result["raw_variation_correction"]) == 0
    assert torch.allclose(result["z_var_raw"], result["variant_latent"])
    assert torch.allclose(
        result["raw_variation_gate"],
        torch.full_like(result["raw_variation_gate"], torch.sigmoid(torch.tensor(-2.0))),
    )
    expected_geo = 1.0 - (1.0 - result["raw_variation_u_shift"]) * (
        1.0 - result["raw_variation_u_scale"]
    )
    expected_full = 1.0 - (1.0 - result["raw_variation_a_shape"]) * (
        1.0 - result["raw_variation_u_shift"]
    ) * (1.0 - result["raw_variation_u_scale"])
    assert torch.allclose(result["raw_variation_a_geo"], expected_geo)
    assert torch.allclose(result["variation_activation"], expected_full)


def test_factorized_mode_rejects_old_graph_without_geometry_but_latent_does_not():
    pattern, mapping = _graphs()
    hidden = torch.randn(1, 1, 2, 4)
    query = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    level = torch.zeros(1, 1, 2)
    scale = torch.ones(1, 1, 2)
    latent = PatternMappingFPem(4, pattern_dim=2, env_dim=2, variant_input_mode="latent_only")
    factorized = PatternMappingFPem(4, pattern_dim=2, env_dim=2, variant_input_mode="factorized_shape")
    artifact = {"pattern_graph": pattern.export(), "mapping_graph": mapping.export()}
    latent.load_graph_statistics(artifact)
    factorized.load_graph_statistics(artifact)
    latent.decompose(hidden, graph_query=query, raw_patch_mean=level, raw_patch_std=scale)
    with pytest.raises(RuntimeError, match="requires rebuilt raw graph"):
        factorized.decompose(hidden, graph_query=query, raw_patch_mean=level, raw_patch_std=scale)


def test_typed_fusion_zero_init_is_identity_and_raw_branches_ignore_zinv_content():
    fusion = TypedVariationFusion(6, 4, "full")
    z_inv = torch.randn(2, 1, 3, 6)
    d_shape = torch.randn(2, 1, 3, 4)
    d_scale = torch.randn(2, 1, 3)
    d_shift = torch.randn(2, 1, 3)
    evidence = torch.rand(2, 1, 3)
    first = fusion(z_inv, d_shape, d_scale, d_shift, evidence, evidence, evidence)
    second = fusion(z_inv + 100.0, d_shape, d_scale, d_shift, evidence, evidence, evidence)
    assert torch.equal(first["shape_delta"], torch.zeros_like(first["shape_delta"]))
    assert torch.equal(first["scale_raw"], torch.zeros_like(first["scale_raw"]))
    assert torch.equal(first["shift_raw"], torch.zeros_like(first["shift_raw"]))
    assert torch.equal(first["z_typed"], z_inv)
    assert torch.equal(first["scale_factor"], torch.ones_like(first["scale_factor"]))
    for key in ("shape_gate", "scale_gate", "shift_gate", "shape_delta", "scale_raw", "shift_raw"):
        assert torch.equal(first[key], second[key])


def test_typed_fusion_uses_shape_then_multiplicative_scale_then_additive_shift():
    fusion = TypedVariationFusion(3, 2, "full")
    with torch.no_grad():
        fusion.shape_encoder[-1].bias.fill_(0.5)
        fusion.scale_encoder[-1].bias.fill_(0.2)
        fusion.shift_encoder[-1].bias.fill_(0.3)
        for gate in (fusion.shape_gate, fusion.scale_gate, fusion.shift_gate):
            gate.bias.zero_()
    z_inv = torch.ones(1, 1, 1, 3)
    d_shape = torch.tensor([[[[1.0, -1.0]]]])
    scalar = torch.ones(1, 1, 1)
    result = fusion(z_inv, d_shape, scalar, scalar, scalar, scalar, scalar)
    expected_shape = z_inv + 0.5 * 0.5
    expected_factor = torch.exp(torch.full_like(z_inv, 0.5 * torch.tanh(torch.tensor(0.2))))
    expected_shift = torch.full_like(z_inv, 0.5 * 0.3)
    assert torch.allclose(result["z_after_shape"], expected_shape)
    assert torch.allclose(result["z_after_scale"], expected_factor * expected_shape)
    assert torch.allclose(result["z_typed"], expected_factor * expected_shape + expected_shift)
    assert torch.allclose(result["z_scale_only"], expected_factor * z_inv)
    assert torch.allclose(result["z_shift_only"], z_inv + expected_shift)


def test_typed_component_masks_are_exact_identities():
    z_inv = torch.randn(1, 1, 2, 4)
    d_shape = torch.randn(1, 1, 2, 3)
    scalar = torch.rand(1, 1, 2)
    shape = TypedVariationFusion(4, 3, "shape")(
        z_inv, d_shape, scalar, scalar, scalar, scalar, scalar
    )
    geometry = TypedVariationFusion(4, 3, "geometry")(
        z_inv, d_shape, scalar, scalar, scalar, scalar, scalar
    )
    assert torch.equal(shape["scale_factor"], torch.ones_like(shape["scale_factor"]))
    assert torch.equal(shape["shift_bias"], torch.zeros_like(shape["shift_bias"]))
    assert torch.equal(geometry["shape_correction"], torch.zeros_like(geometry["shape_correction"]))


def test_typed_raw_requires_raw_representation_space():
    from models.PatchTST_FPEM import Model

    config = SimpleNamespace(
        task_name="long_term_forecast", seq_len=32, pred_len=8, enc_in=3,
        d_model=8, patch_len=16, factor=1, dropout=0.0, n_heads=2,
        d_ff=16, activation="gelu", e_layers=1, fpem_pmg_enabled=1,
        fpem_pmg_pattern_dim=4, fpem_pmg_env_dim=3,
        fpem_pmg_representation_space="embedding", fpem_pmg_fusion_mode="typed_raw",
    )
    with pytest.raises(ValueError, match="typed raw fusion requires raw representation space"):
        Model(config)


def test_legacy_checkpoint_without_typed_module_still_loads_strictly():
    source = PatternMappingFPem(4, pattern_dim=2, env_dim=2, fusion_mode="legacy_environment")
    old_state = {
        key: value for key, value in source.state_dict().items()
        if "typed_fusion." not in key
    }
    restored = PatternMappingFPem(4, pattern_dim=2, env_dim=2, fusion_mode="legacy_environment")
    restored.load_state_dict(old_state, strict=True)


def test_hidden_representation_mode_preserves_full_backbone_freeze():
    from models.PatchTST_FPEM import Model

    config = SimpleNamespace(
        task_name="long_term_forecast", seq_len=32, pred_len=8, enc_in=3,
        d_model=8, patch_len=16, factor=1, dropout=0.0, n_heads=2,
        d_ff=16, activation="gelu", e_layers=1, fpem_pmg_enabled=1,
        fpem_pmg_pattern_dim=4, fpem_pmg_env_dim=3,
        fpem_pmg_representation_space="hidden",
    )
    model = Model(config)
    x = torch.randn(2, 32, 3)
    features = model.encoder_backbone.forward_features(x)
    hidden_result = model.pmg.decompose(features["hidden"], features["hidden"])
    legacy_result = model.pmg.decompose(features["hidden"])
    assert torch.allclose(hidden_result["query"], legacy_result["query"])
    model.freeze_fpem_pmg_graph_space()
    model.train()
    assert not any(parameter.requires_grad for parameter in model.encoder_backbone.parameters())
    assert model.encoder_backbone.training is False


def test_matching_a0_checkpoint_initializes_pmg_backbone_and_head(tmp_path):
    from models.PatchTST_FPEM import Model

    config = SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=32,
        pred_len=8,
        enc_in=3,
        d_model=8,
        patch_len=16,
        factor=1,
        dropout=0.0,
        n_heads=2,
        d_ff=16,
        activation="gelu",
        e_layers=1,
        fpem_pmg_enabled=1,
        fpem_pmg_pattern_dim=4,
        fpem_pmg_env_dim=3,
    )
    source = Model(config)
    a0_state = dict(source.encoder_backbone.state_dict())
    a0_state.update({"head." + key: value for key, value in source.head_shared.state_dict().items()})
    checkpoint = tmp_path / "a0.pth"
    torch.save(a0_state, checkpoint)
    restored = Model(config)
    loaded = restored.load_fpem_pmg_a0_checkpoint(str(checkpoint))
    assert loaded == len(source.encoder_backbone.state_dict()) + len(source.head_shared.state_dict())
    assert restored._stage0_loaded_a0 is True
    assert torch.equal(
        restored.encoder_backbone.patch_embedding.value_embedding.weight,
        source.encoder_backbone.patch_embedding.value_embedding.weight,
    )
    assert torch.equal(restored.head_shared.linear.weight, source.head_shared.linear.weight)


def test_legacy_fpem_path_still_runs_when_pmg_is_disabled():
    from models.PatchTST_FPEM import Model

    config = SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=32,
        pred_len=8,
        enc_in=3,
        d_model=8,
        patch_len=16,
        factor=1,
        dropout=0.0,
        n_heads=2,
        d_ff=16,
        activation="gelu",
        e_layers=1,
        fpem_pmg_enabled=0,
    )
    model = Model(config)
    x = torch.randn(2, 32, 3)
    marks = torch.zeros(2, 32, 4)
    decoder = torch.zeros(2, 24, 3)
    prediction = model(x, marks, decoder, marks[:, :24])
    assert prediction.shape == (2, 8, 3)
    extra_loss, logs = model.fpem_extra_loss(torch.randn(2, 8, 3))
    assert torch.isfinite(extra_loss)
    assert "fpem_ts/loss_inv" in logs
