import unittest

import numpy as np

from tools.run_fpem_eiil3_r3_input_ablation import variant_design, direction_diagnostics


class InputAblationTest(unittest.TestCase):
    def test_design_and_linear_equivalence(self):
        rng = np.random.default_rng(5)
        f03 = rng.normal(size=(4, 3)).astype(np.float32)
        a0 = rng.normal(size=(4, 3)).astype(np.float32)
        key = rng.normal(size=(4, 64)).astype(np.float32)
        b = variant_design("B", f03, a0, key)
        c = variant_design("C", f03, a0, key)
        d = variant_design("D", f03, a0, key)
        self.assertEqual(b.shape, (4, 3 + 64 + 11))
        self.assertEqual(c.shape, (4, 6 + 64 + 11))
        self.assertEqual(d.shape, c.shape)
        np.testing.assert_allclose(c[:, 3:6], c[:, :3] - d[:, 3:6], atol=1e-6)
        # A linear expert on [F03,A0,key] can be reparameterized on
        # [F03,F03-A0,key], but Adam/weight decay need not follow equal paths.
        wf = rng.normal(size=(3, 3)).astype(np.float32)
        wa = rng.normal(size=(3, 3)).astype(np.float32)
        tail = rng.normal(size=(3, b.shape[1] - 3)).astype(np.float32)
        yc = c[:, :3] @ wf.T + c[:, 3:6] @ wa.T + c[:, 6:] @ tail.T
        yd = d[:, :3] @ (wf + wa).T + d[:, 3:6] @ (-wa).T + d[:, 6:] @ tail.T
        np.testing.assert_allclose(yc, yd, atol=1e-5)

    def test_direction_statistics(self):
        y = np.array([[2., 3.]], dtype=np.float32)
        a0 = np.array([[0., 0.]], dtype=np.float32)
        f03 = np.array([[1., 0.]], dtype=np.float32)
        r3 = np.array([[0., 1.]], dtype=np.float32)
        stats = direction_diagnostics(y, a0, f03, r3)
        self.assertAlmostEqual(stats["cosine_f03_r3"], 0.)
        self.assertAlmostEqual(stats["naive_sum_mse"], 2.5)


if __name__ == "__main__":
    unittest.main()
