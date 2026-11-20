import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import summarize_predictive_env_all_datasets as summary


class FiveWaySummaryTest(unittest.TestCase):
    def test_five_way_summary_and_relative_improvements(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for dataset_index, dataset in enumerate(summary.DATASETS):
                reference = root / dataset / "shared_reference.pt"
                reference.parent.mkdir(parents=True)
                reference.write_bytes(f"reference-{dataset}".encode())
                reference_hash = hashlib.sha256(reference.read_bytes()).hexdigest()
                for variant_index, variant in enumerate(summary.VARIANTS):
                    label, experiment, mode, scale, fusion, conditional_gain, isolated = variant
                    mse = 1.0 if experiment == "A0" else 0.9 + variant_index * 0.01
                    row = {
                        "dataset_name": dataset,
                        "experiment": experiment,
                        "predictive_env_refactor_mode": mode,
                        "fusion_scale_calibration": scale,
                        "variant_fusion_mode": fusion,
                        "lambda_var_conditional_gain": conditional_gain,
                        "z_specific_encoder_gradient_isolated": isolated,
                        "representation_constraint": "classification",
                        "decomposition_type": "complementary_gate",
                        "environment_count": 3,
                        "seed": 2021,
                        "optimization_epochs": 10,
                        "stage_epochs": 2,
                        "lambda_h_anchor": 0.0,
                        "reference_checkpoint_sha256": reference_hash,
                        "reference_source": "loaded",
                        "environment_updates": [],
                        "MSE": mse,
                        "MAE": mse / 2,
                        "inv_MSE": mse,
                        "inv_minus_full": 0.0,
                    }
                    destination = root / dataset / label / experiment
                    destination.mkdir(parents=True)
                    (destination / "metrics_and_diagnostics.json").write_text(
                        json.dumps(row), encoding="utf-8"
                    )

            with patch.object(sys, "argv", ["summary", str(root)]):
                summary.main()
            text = (root / "all_datasets_summary.txt").read_text(encoding="utf-8")
            self.assertEqual(text.count(" A0_patchtst "), 8)
            self.assertEqual(text.count(" C_gradient_isolated "), 8)
            self.assertIn("A_current MSE_wins=8/8", text)
            self.assertIn("overall_best_variant_by_mean_relative_MSE: A_current", text)
            self.assertIn("relative_MSE_improvement_vs_A0_percent", text)


if __name__ == "__main__":
    unittest.main()
