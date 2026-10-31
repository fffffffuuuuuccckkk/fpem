import math
import unittest

import torch

from models.predictive_env.losswise_adam import LosswiseAdam


class LosswiseAdamTest(unittest.TestCase):
    def test_independent_equal_gradients(self):
        p = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
        optimizer = LosswiseAdam([{"name": "shared", "params": [p], "lr": 0.01}])
        optimizer.step([torch.tensor([2.0, -3.0])], [torch.tensor([2.0, -3.0])])
        self.assertTrue(torch.allclose(p, torch.tensor([0.98, -0.98]), atol=1e-6))
        self.assertEqual(optimizer.state[p]["prediction"]["step"], 1)
        self.assertEqual(optimizer.state[p]["auxiliary"]["step"], 1)

    def test_missing_gradients_do_not_update_state(self):
        p = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = LosswiseAdam([{"name": "shared", "params": [p], "lr": 0.01}])
        optimizer.step([None], [torch.tensor([1.0])])
        self.assertAlmostEqual(float(p), 0.99, places=6)
        self.assertNotIn("prediction", optimizer.state[p])
        optimizer.step([torch.tensor([1.0])], [None])
        self.assertEqual(optimizer.state[p]["prediction"]["step"], 1)
        self.assertEqual(optimizer.state[p]["auxiliary"]["step"], 1)

    def test_weight_decay_once(self):
        p = torch.nn.Parameter(torch.tensor([2.0]))
        optimizer = LosswiseAdam([{"name": "shared", "params": [p], "lr": 0.01}], weight_decay=0.1)
        optimizer.step([torch.tensor([1.0])], [torch.tensor([1.0])])
        self.assertAlmostEqual(float(p), 1.978, places=6)

    def test_state_isolation(self):
        p = torch.nn.Parameter(torch.tensor([0.0]))
        optimizer = LosswiseAdam([{"name": "shared", "params": [p], "lr": 0.01}])
        optimizer.step([torch.tensor([2.0])], [torch.tensor([-5.0])])
        self.assertAlmostEqual(float(optimizer.state[p]["prediction"]["m"]), 0.2, places=6)
        self.assertAlmostEqual(float(optimizer.state[p]["auxiliary"]["m"]), -0.5, places=6)
        self.assertAlmostEqual(float(p), 0.0, places=6)

    def test_fp32_and_cuda_autocast_finite(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            p = torch.nn.Parameter(torch.ones(4, device=device))
            optimizer = LosswiseAdam([{"name": "shared", "params": [p], "lr": 1e-4}])
            optimizer.step([torch.ones_like(p)], [-torch.ones_like(p)])
            self.assertTrue(torch.isfinite(p).all())
            self.assertTrue(math.isfinite(float(p.mean())))


if __name__ == "__main__":
    unittest.main()
