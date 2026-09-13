from types import SimpleNamespace
import torch

from models.PatchTST_DomainIV import Model, EXPERIMENTS
from models.domain_iv.losses import domain_constraint_loss, linear_hsic


def config(experiment):
    return SimpleNamespace(pred_len=8, seq_len=32, enc_in=3, d_model=16, n_heads=2,
        e_layers=1, d_ff=32, dropout=0., factor=1, activation="gelu", patch_len=8,
        domain_iv_stride=4, domain_iv_experiment=experiment, domain_iv_bottleneck=8,
        domain_iv_rank=2, domain_iv_env_num=3)


def test_all_domain_iv_ablations_forward_backward():
    for name in EXPERIMENTS:
        model = Model(config(name)); x = torch.randn(4, 32, 3)
        result = model.forward_components(x)
        assert result["prediction"].shape == (4, 8, 3)
        loss = result["prediction"].square().mean() + result["invariant_prediction"].square().mean()
        if name != "D0":
            q = torch.softmax(torch.randn(4, 3), -1)
            domain, _ = domain_constraint_loss(model.domain_constraint, result["z_inv"], result["z_var"], q, result["q_logits"])
            loss = loss + domain
        loss.backward()


def test_dynamic_mapping_zero_initialization():
    for name in ("D2", "D3", "D5", "D6"):
        model = Model(config(name))
        assert model.initial_mapping_difference(torch.randn(3, 32, 3)) < 1e-8


def test_environment_similarity_is_permutation_invariant():
    q = torch.softmax(torch.randn(7, 4), -1)
    permutation = torch.tensor([2, 0, 3, 1])
    assert torch.allclose(q @ q.t(), q[:, permutation] @ q[:, permutation].t(), atol=1e-7)


def test_hsic_detects_dependence():
    x = torch.randn(64, 5)
    dependent = linear_hsic(x, x)
    independent = linear_hsic(x, torch.randn(64, 5))
    assert dependent > independent
