import unittest

import torch

from tools.fpem_paper_final_counterfactual import condition_record, wrong_index


class TestPaperFinalCounterfactual(unittest.TestCase):
    def setUp(self):
        n, c, d = 20, 3, 14
        self.record = {
            "ids": torch.arange(n),
            "features": torch.arange(n * c * d).reshape(n, c, d).float(),
            "scale": torch.linspace(.1, 2., n).view(n, 1, 1).expand(n, c, 1),
            "truth": torch.randn(n, 24, c),
            "y_inv": torch.randn(n, 24, c),
            "future": torch.randn(n, c, 64),
        }

    def test_derangements_and_no_target_change(self):
        for mode in ("far_time", "scale_stratum"):
            perm = wrong_index(self.record, mode)
            self.assertTrue(torch.all(perm != torch.arange(len(perm))))
            self.assertEqual(len(torch.unique(perm)), len(perm))
        for mode in ("zero", "far_time", "scale_stratum", "variable"):
            changed = condition_record(self.record, mode)
            for key in ("truth", "y_inv", "future", "scale", "ids"):
                self.assertIs(changed[key], self.record[key])
            self.assertEqual(changed["features"].shape,
                             self.record["features"].shape)
        self.assertTrue(torch.equal(condition_record(self.record, "zero")["features"],
                                    torch.zeros_like(self.record["features"])))


if __name__ == "__main__":
    unittest.main()
