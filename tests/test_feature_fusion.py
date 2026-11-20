import io

import pytest
import torch

from models.predictive_env.feature_fusion import FeatureFusion


@pytest.mark.parametrize("mode", ["F1", "F2"])
@pytest.mark.parametrize("shape", [(2, 1, 1, 16), (3, 7, 12, 32)])
def test_initial_identity_and_checkpoint(mode, shape):
    torch.manual_seed(4)
    fusion = FeatureFusion(shape[-1], mode)
    inv = torch.randn(shape)
    var = torch.randn(shape)
    output = fusion(inv, var)
    assert output.shape == inv.shape
    assert torch.allclose(output, inv, atol=1e-7, rtol=0)
    buffer = io.BytesIO()
    torch.save(fusion.state_dict(), buffer)
    buffer.seek(0)
    restored = FeatureFusion(shape[-1], mode)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    assert torch.equal(restored(inv, var), output)


@pytest.mark.parametrize("mode", ["F1", "F2"])
def test_variant_path_learns(mode):
    torch.manual_seed(5)
    fusion = FeatureFusion(8, mode)
    inv = torch.randn(4, 2, 3, 8)
    var = torch.randn_like(inv)
    target = inv + 0.25 * var
    optimizer = torch.optim.Adam(fusion.parameters(), lr=0.05)
    initial = (fusion(inv, var)-target).square().mean().item()
    for _ in range(12):
        optimizer.zero_grad()
        loss = (fusion(inv, var)-target).square().mean()
        loss.backward()
        optimizer.step()
    assert (fusion(inv, var)-target).square().mean().item() < initial


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        FeatureFusion(8, "F1")(torch.zeros(2, 1, 3, 8), torch.zeros(2, 1, 4, 8))
