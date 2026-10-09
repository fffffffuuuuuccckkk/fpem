import json
import tempfile
import unittest
from pathlib import Path

from tools.summarize_predictive_env_backbones import BACKBONES, DATASETS, VARIANTS, summarize


class BackboneSummaryTest(unittest.TestCase):
    def test_summary_compares_each_backbone_with_its_own_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for backbone in BACKBONES:
                for dataset in DATASETS:
                    for label, experiment in VARIANTS:
                        path = root / backbone / dataset / label / experiment
                        path.mkdir(parents=True)
                        mse = 1.0 if label == "A0_backbone" else 0.9
                        (path / "metrics_and_diagnostics.json").write_text(
                            json.dumps({"MSE": mse, "MAE": mse})
                        )
            text = summarize(root)
            self.assertIn("cyclenet MSE_wins=8/8", text)
            self.assertIn("itransformer MSE_wins=8/8", text)
            self.assertIn("+10.000000", text)


if __name__ == "__main__":
    unittest.main()
