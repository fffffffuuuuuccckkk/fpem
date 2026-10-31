"""CPU contract tests for all new DPPM rare learners."""
import unittest

import torch

from models.predictive_env.dual_pattern_memory.rare_learning import RarePatternLearner
from tools.run_fpem_dppm_rare import stable_prediction


class RareLearnerTests(unittest.TestCase):
    def test_g1_stable_replaces_only_the_anchor(self):
        key = torch.zeros(2, 3, 64)
        base = torch.ones(2, 3, 7)
        weight = torch.zeros(65, 7)
        weight[-1] = 0.25
        record = {"key": key, "base": base}
        prediction, query = stable_prediction(
            record, {"kind": "g1", "weight": weight, "scale": 2.0},
            torch.randn(7, 2), 0.5)
        self.assertTrue(torch.allclose(prediction, torch.full_like(base, 1.5)))
        self.assertEqual(query, {})

    def test_all_methods_no_future_query_and_roundtrip(self):
        torch.manual_seed(7)
        x = torch.randn(24, 2, 64)
        y = torch.randn(24, 2, 17)
        context = {"reference_map": torch.zeros(65, 17),
                   "starts": torch.arange(24) * 32, "span": 32}
        for method in [f"R{k}" for k in range(1, 8)]:
            with self.subTest(method=method):
                learner = RarePatternLearner(method, patterns=4, rank=3,
                                             epochs=1).fit(x, y, context)
                output = learner.query(x[:2])
                self.assertEqual(output["candidate"].shape, (2,2,4,17))
                self.assertEqual(output["weight"].shape, (2,2,4))
                self.assertTrue(torch.isfinite(output["candidate"]).all())
                self.assertTrue(((output["weight"].sum(-1)+output["null_weight"])-1).abs().max()<1e-5)
                restored = RarePatternLearner.from_export(learner.export())
                self.assertTrue(torch.allclose(restored.predict_candidates(x[:2]),
                                               output["candidate"], atol=1e-6))

    def test_future_labels_never_enter_query(self):
        model = RarePatternLearner("R3", patterns=4, rank=3, epochs=1)
        x = torch.randn(24,1,64)
        model.fit(x,torch.randn(24,1,9),
                  {"reference_map":torch.zeros(65,9),
                   "starts":torch.arange(24)*32,"span":32})
        self.assertEqual(model.query(x[:1])["candidate"].shape[-1],9)

    def test_group_optimizer_updates_rare_mapping_without_final_reliability(self):
        torch.manual_seed(9)
        x=torch.randn(24,1,64)
        y=torch.randn(24,1,9)
        model=RarePatternLearner("R3",patterns=4,rank=3,epochs=2,
            optimizer_config={"head_lr":1e-3,"dictionary_ratio":.3,
                              "basis_ratio":.1,"route_ratio":.3})
        model.fit(x,y,{"reference_map":torch.zeros(65,9),
                       "starts":torch.arange(24)*32,"span":32})
        grad=model.stats["loss_trace"][-1]["grad_norm"]
        self.assertGreater(grad["mapping"],0.)
        self.assertGreater(grad["basis"],0.)
        self.assertTrue(torch.isfinite(model.query(x[:2])["candidate"]).all())


if __name__ == "__main__":
    unittest.main()
