import importlib.util
from pathlib import Path
import unittest

import torch


spec = importlib.util.spec_from_file_location(
    "exogenous_token_fusion",
    Path(__file__).resolve().parents[1] / "models" / "predictive_env"
    / "exogenous_token_fusion.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ExogenousTokenFusionTest(unittest.TestCase):
    def test_zero_gate_and_aligned_positions(self):
        torch.manual_seed(4)
        fusion = module.PatchExogenousTokenFusion(8, 2).eval()
        target = torch.randn(2, 3, 4, 8)
        exog = torch.randn(2, 5, 4, 8)
        self.assertTrue(torch.equal(fusion(target, exog), target))
        fusion.gate.data.fill_(1)
        original = fusion(target, exog)
        changed = exog.clone()
        changed[:, :, 1, :] += 10
        modified = fusion(target, changed)
        self.assertTrue(torch.equal(original[:, :, 0], modified[:, :, 0]))
        self.assertFalse(torch.allclose(original[:, :, 1], modified[:, :, 1]))

    def test_gate_receives_gradient_at_initialization(self):
        fusion = module.PatchExogenousTokenFusion(8, 2)
        target = torch.randn(2, 3, 4, 8)
        exog = torch.randn(2, 2, 4, 8)
        fusion(target, exog).square().sum().backward()
        self.assertGreater(fusion.gate.grad.abs().item(), 0)

    def test_rejects_mismatched_patches(self):
        fusion = module.PatchExogenousTokenFusion(8, 2)
        with self.assertRaises(ValueError):
            fusion(torch.randn(2, 3, 4, 8), torch.randn(2, 2, 5, 8))

    def test_global_fusion_shape_and_gradients(self):
        torch.manual_seed(8)
        fusion = module.GlobalExogenousTokenFusion(16, 8, 2, gate_init=0.1)
        target = torch.randn(2, 3, 5, 8, requires_grad=True)
        history = torch.randn(2, 16, 3, requires_grad=True)
        exog = torch.randn(2, 16, 4, requires_grad=True)
        output = fusion(target, history, exog)
        self.assertEqual(output.shape, target.shape)
        output.square().mean().backward()
        self.assertGreater(exog.grad.abs().sum().item(), 0)
        self.assertGreater(fusion.global_embedding.weight.grad.abs().sum().item(), 0)
        self.assertGreater(fusion.patch_exog_attention.in_proj_weight.grad.abs().sum().item(), 0)

    def test_global_fusion_has_full_history_and_no_cross_target_mixing(self):
        torch.manual_seed(9)
        fusion = module.GlobalExogenousTokenFusion(16, 8, 2, gate_init=0.5).eval()
        target = torch.randn(2, 3, 5, 8)
        history = torch.randn(2, 16, 3)
        exog = torch.randn(2, 16, 4)
        original = fusion(target, history, exog)

        changed_exog = exog.clone()
        changed_exog[:, 0, 0] += 10
        exog_output = fusion(target, history, changed_exog)
        # A change at one history position is visible to every target patch.
        per_patch = (exog_output - original).abs().sum(dim=(0, 1, 3))
        self.assertTrue(torch.all(per_patch > 0))

        changed_history = history.clone()
        changed_history[:, :, 0] += 10
        history_output = fusion(target, changed_history, exog)
        self.assertFalse(torch.allclose(history_output[:, 0], original[:, 0]))
        self.assertTrue(torch.equal(history_output[:, 1:], original[:, 1:]))

    def test_global_zero_gate_is_exact_identity(self):
        fusion = module.GlobalExogenousTokenFusion(16, 8, 2)
        target = torch.randn(2, 3, 5, 8)
        history = torch.randn(2, 16, 3)
        exog = torch.randn(2, 16, 4)
        self.assertTrue(torch.equal(fusion(target, history, exog), target))


if __name__ == "__main__":
    unittest.main()
