import unittest

import torch

from tools.run_fpem_s4_losswise import surgery


class PredictionPrioritySurgeryTest(unittest.TestCase):
    def test_full_and_half_conflict_removal(self):
        gp = [torch.tensor([1.0, 0.0])]
        ga = [torch.tensor([-2.0, 3.0])]
        full = surgery(gp, ga, ["shared.weight"], strength=1.0)[0]
        half = surgery(gp, ga, ["shared.weight"], strength=0.5)[0]
        self.assertTrue(torch.allclose(full, torch.tensor([0.0, 3.0])))
        self.assertTrue(torch.allclose(half, torch.tensor([-1.0, 3.0])))
        self.assertTrue(torch.allclose(ga[0], torch.tensor([-2.0, 3.0])))

    def test_nonconflict_and_nonshared_unchanged(self):
        gp = [torch.tensor([1.0]), torch.tensor([1.0])]
        ga = [torch.tensor([2.0]), torch.tensor([-2.0])]
        out = surgery(gp, ga, ["shared.weight", "private.weight"], strength=1.0)
        self.assertTrue(torch.equal(out[0], ga[0]))
        self.assertTrue(torch.equal(out[1], ga[1]))

    def test_invalid_strength(self):
        with self.assertRaises(ValueError):
            surgery([], [], [], strength=1.1)


if __name__ == "__main__":
    unittest.main()
