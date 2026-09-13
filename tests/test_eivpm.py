from types import SimpleNamespace
import torch

from models.PatchTST_EIVPM import Model, ABLATIONS
from models.eivpm.pattern_bank import EnvironmentPatternBank
from models.eivpm.mapping_bank import EnvironmentMappingBank


def config(ablation):
    return SimpleNamespace(task_name="long_term_forecast", seq_len=32, pred_len=8, enc_in=3,
        d_model=16, n_heads=2, e_layers=1, d_ff=32, dropout=0., factor=1,
        activation="gelu", patch_len=8, eivpm_stride=4, eivpm_ablation=ablation,
        eivpm_pattern_count=4, eivpm_adapter_dim=8, eivpm_mapping_rank=2)


def fitted(model):
    model.pattern_bank.prototypes.normal_()
    model.pattern_bank.p_inv.copy_(torch.tensor([.9, .7, .2, .1]))
    model.pattern_bank.fitted.fill_(True)
    model.mapping_bank.p_inv.copy_(torch.linspace(.1, .9, 16))
    model.mapping_bank.fitted.fill_(True)


def test_all_ablation_forward_backward():
    for name in ABLATIONS:
        model = Model(config(name)); fitted(model)
        x = torch.randn(2, 32, 3)
        y = model(x)
        assert y.shape == (2, 8, 3)
        y.square().mean().backward()


def test_zero_initialized_plugin_is_identity():
    model = Model(config("E9")); fitted(model)
    diff = model.initial_identity_differences(torch.randn(2, 32, 3))
    assert diff["initial_pattern_abs_diff"] < 1e-8
    assert diff["initial_prediction_abs_diff"] < 1e-8


def test_soft_mapping_has_correct_temporal_correspondence():
    bank = EnvironmentMappingBank(3, 5)
    resp = torch.zeros(1, 2, 4, 3)
    resp[:, :, 0, 0] = 1; resp[:, :, 1, 1] = 1; resp[:, :, 2, 2] = 1; resp[:, :, 3, 0] = 1
    bank.fit([resp], [torch.tensor([0])])
    transition = bank.env_transition.reshape(1, 3, 3)[0]
    assert transition[0, 1] > 0 and transition[1, 2] > 0 and transition[2, 0] > 0
    assert transition[0, 2] == 0


def test_train_only_environment_guard(tmp_path):
    from models.eivpm.environment_decomposer import FoilEnvironmentProvider
    x = torch.randn(20, 12, 3).numpy(); y = torch.randn(20, 4, 3).numpy()
    labels = FoilEnvironmentProvider(2).load_or_create(str(tmp_path / "env.npz"), x, y)
    assert labels.soft.shape == (20, 2)
    assert len(labels.labels) == 20


def test_pattern_fit_uses_spherical_prototypes_and_non_degenerate_split():
    torch.manual_seed(4)
    bank = EnvironmentPatternBank(6, pattern_count=4, scales=(1, 2), temperature=.08)
    batches, environments = [], []
    for _ in range(3):
        hidden = torch.randn(12, 2, 5, 6)
        hidden[:6, :, :, 0] += 4.0
        hidden[6:, :, :, 1] += 4.0
        batches.append(hidden)
        environments.append(torch.tensor([0] * 6 + [1] * 6))
    bank.fit(batches, environments, max_segments=10000, iterations=8)
    assert torch.allclose(bank.prototypes.norm(dim=-1), torch.ones(4), atol=1e-5)
    assert not torch.allclose(bank.p_inv, torch.full_like(bank.p_inv, .5), atol=1e-6)


def test_pattern_occurrence_is_environment_mean_of_sample_presence():
    torch.manual_seed(7)
    bank = EnvironmentPatternBank(4, pattern_count=3, scales=(1,), temperature=.1)
    hidden = torch.randn(5, 2, 4, 4)
    env = torch.tensor([0, 0, 1, 1, 1])
    bank.fit([hidden], [env], max_segments=1000, iterations=5)
    presence = bank.sample_presence(hidden)
    expected = torch.stack([presence[env == e].mean(0) for e in range(2)], dim=1)
    assert torch.allclose(bank.env_occurrence.cpu(), expected, atol=1e-6)
