import pytest
import torch

from models.predictive_env.group_consensus import GroupConsensusSubspace, ConsensusAwareFusion


@pytest.mark.parametrize("mode", ["random", "static", "ema", "robust"])
@pytest.mark.parametrize("shape", [(4, 1, 1, 16), (3, 2, 5, 16)])
def test_projection_reconstruction_orthogonality(mode, shape):
    torch.manual_seed(1)
    block = GroupConsensusSubspace(16, 4, mode, refresh_steps=1)
    hidden = torch.randn(shape, requires_grad=True)
    if mode in ("ema", "robust"):
        block.initialize_from_training([hidden.detach(), hidden.detach()*1.1])
        block.train(); block.update(hidden.detach())
    inv, dyn, shared, continuity, diag = block(hidden)
    assert torch.allclose(inv+dyn, hidden, atol=1e-5)
    assert float(diag["orthogonality"]) < 1e-4
    assert float(diag["basis_orthogonality"]) < 1e-4
    assert shared.min() >= 0 and shared.max() <= 1
    assert continuity.shape == hidden.shape[:-1]
    if shape[2] == 1:
        assert torch.count_nonzero(continuity) == 0
    (inv.square().mean()+dyn.square().mean()).backward()
    assert torch.isfinite(hidden.grad).all()
    if mode == "static":
        assert block.raw_basis.grad is not None
        assert torch.isfinite(block.raw_basis.grad).all()


def test_memory_is_train_only_and_checkpointable():
    block = GroupConsensusSubspace(8, 2, "robust", refresh_steps=1)
    batch = torch.randn(4, 2, 3, 8)
    block.initialize_from_training([batch])
    first = block.basis().clone()
    block.train(); block.update(batch+0.2)
    assert int(block.updates) == 1
    block.eval()
    with pytest.raises(RuntimeError):
        block.update(batch)
    clone = GroupConsensusSubspace(8, 2, "robust", refresh_steps=1)
    clone.load_state_dict(block.state_dict())
    assert torch.allclose(clone.basis(), block.basis())
    assert not torch.allclose(block.second_moment, torch.zeros_like(block.second_moment))
    assert first.shape == block.basis().shape


@pytest.mark.parametrize("mode", ["fixed", "ordinary", "consensus"])
def test_fusion_gate_shape_and_gradient(mode):
    gate = ConsensusAwareFusion(8, 12, mode)
    inv = torch.randn(2, 3, 4, 8, requires_grad=True)
    dyn = torch.randn_like(inv)
    score = torch.rand(2, 3, 4)
    result = gate(inv, dyn, score, score, 12)
    assert result.shape == (2, 12, 3)
    assert result.min() >= 0 and result.max() <= 1
    if mode != "fixed":
        result.mean().backward()
        assert gate.net[-1].weight.grad is not None
