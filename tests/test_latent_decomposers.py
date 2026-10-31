import io

import pytest
import torch

from models.predictive_env.latent_decomposers import (
    EnvGuidedSubspaceDecomposer, LowRankSubspaceDecomposer,
    ReferenceGuidedDecomposer,
)
from models.predictive_env.latent_disentanglement_losses import orthogonality_loss
from models.predictive_env.signed_gate_decomposition import ComplementaryGateDecomposer


@pytest.mark.parametrize("shape", [(2, 1, 1, 16), (3, 5, 7, 32), (2, 2, 3, 8)])
@pytest.mark.parametrize("kind", ["low", "environment", "reference"])
def test_shapes_reconstruction_gradients_and_checkpoint(shape, kind):
    d = shape[-1]
    if kind == "low":
        model = LowRankSubspaceDecomposer(d, min(4, d))
    elif kind == "environment":
        model = EnvGuidedSubspaceDecomposer(d, min(4, d))
        model.update_environment_statistics(torch.randn(10, d),
                                            torch.softmax(torch.randn(10, 3), -1))
    else:
        model = ReferenceGuidedDecomposer(d, 3)
        model.update_references(torch.randn(10, d),
                                torch.softmax(torch.randn(10, 3), -1))
    hidden = torch.randn(shape, requires_grad=True)
    z_inv, z_var, gate = model(hidden)
    assert z_inv.shape == z_var.shape == gate.shape == hidden.shape
    assert torch.allclose(z_inv + z_var, hidden, atol=1e-6)
    loss = z_inv.square().mean() + z_var.square().mean()
    if kind == "environment":
        loss = loss + 0.05 * model.environment_subspace_loss()
    if kind == "reference":
        q = torch.softmax(torch.randn(shape[0], 3), -1)
        loss = loss + model.q_supervision_loss(hidden, q)
    loss.backward()
    assert torch.isfinite(hidden.grad).all()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())
    if kind != "reference":
        basis = model.orthogonal_basis()
        assert torch.allclose(basis.T @ basis, torch.eye(basis.shape[1]), atol=1e-5)
        assert z_var.detach().square().mean().sqrt() < hidden.detach().square().mean().sqrt() * 0.1
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    state = torch.load(stream, weights_only=True)
    model.load_state_dict(state)


def test_environment_statistics_change_with_shuffled_q():
    model = EnvGuidedSubspaceDecomposer(8, 2)
    h = torch.randn(40, 8)
    q = torch.softmax(torch.randn(40, 3), -1)
    model.update_environment_statistics(h, q)
    correct = model.between_scatter.clone()
    model.update_environment_statistics(h, q.roll(7, 0))
    assert not torch.allclose(correct, model.between_scatter)


def test_reference_inference_depends_only_on_history():
    model = ReferenceGuidedDecomposer(8, 3)
    h = torch.randn(12, 8)
    q = torch.softmax(torch.randn(12, 3), -1)
    model.update_references(h, q)
    hidden = torch.randn(2, 4, 3, 8)
    with torch.no_grad():
        first = model(hidden)[0]
        second = model(hidden)[0]
    assert torch.equal(first, second)


def test_legacy_complementary_gate_unchanged():
    model = ComplementaryGateDecomposer(8)
    h = torch.randn(2, 3, 4, 8)
    inv, var, gate = model(h)
    assert torch.allclose(inv, h * ((1 + gate) / 2))
    assert torch.allclose(var, h * ((1 - gate) / 2))
    assert torch.allclose(inv + var, h)


def test_function_preserving_scale_and_orth_loss():
    h = torch.randn(2, 3, 4, 8)
    head = torch.nn.Linear(8 * 4, 6)
    reference = head(h.transpose(-1, -2).flatten(-2))
    with torch.no_grad():
        head.weight.mul_(2)
    prediction = head((h * 0.5).transpose(-1, -2).flatten(-2))
    assert torch.allclose(prediction, reference, atol=1e-6)
    assert torch.isfinite(orthogonality_loss(h, h, "cosine"))
    assert torch.isfinite(orthogonality_loss(h, h, "cross_cov"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_single_batch():
    model = LowRankSubspaceDecomposer(32, 8).cuda()
    h = torch.randn(2, 5, 4, 32, device="cuda", requires_grad=True)
    inv, var, _ = model(h)
    (inv.square().mean() + var.square().mean()).backward()
    assert torch.isfinite(h.grad).all()
