import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from models.environment_quality import EnvironmentQualityMonitor
from models.predictive_env.eiil_environment import (
    PredictiveConflictEnvironment,
    _adjusted_rand_index,
    _normalized_mutual_info,
)
from tools.summarize_environment_quality import write_outputs


class EnvironmentQualityTest(unittest.TestCase):
    def test_quality_metrics_and_payload_do_not_consume_global_rng(self):
        sample_count = 18
        prediction = torch.linspace(-1, 1, sample_count).reshape(-1, 1, 1)
        target = torch.zeros_like(prediction)
        sample_id = torch.arange(sample_count)
        manager = PredictiveConflictEnvironment(
            sample_count,
            env_num=3,
            steps=2,
            diagnostic_samples=18,
            quality_diagnostics=True,
            random_partition_repeats=17,
        )
        before = torch.random.get_rng_state().clone()
        _, record = manager.update(prediction, target, sample_id)
        after = torch.random.get_rng_state()
        self.assertTrue(torch.equal(before, after))
        self.assertEqual(record["random_partition_repeats"], 17)
        self.assertGreaterEqual(record["random_partition_p_value"], 1 / 18)
        self.assertLessEqual(record["random_partition_p_value"], 1.0)
        self.assertEqual(manager.last_quality_payload["q"].shape, (18, 3))
        self.assertAlmostEqual(
            record["gradient_separation_gain"],
            record["gradient_separation"] - 1.0,
            places=5,
        )
        _, second = manager.update(prediction * 1.01, target, sample_id)
        self.assertTrue(np.isfinite(second["stage_ARI_to_previous"]))
        self.assertTrue(np.isfinite(second["stage_NMI_to_previous"]))

    def test_assignment_agreement(self):
        left = np.array([0, 0, 1, 1, 2, 2])
        permuted = np.array([2, 2, 0, 0, 1, 1])
        self.assertAlmostEqual(_adjusted_rand_index(left, permuted), 1.0)
        self.assertAlmostEqual(_normalized_mutual_info(left, permuted), 1.0)

    def test_monitor_writes_requested_artifacts(self):
        q = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9]])
        payload = {
            "sample_id": np.arange(4),
            "g_scale": np.array([-1.0, 1.0, -0.8, 0.9]),
            "g_bias": np.array([-0.5, 0.4, -0.3, 0.6]),
            "q": q,
            "hard_env": q.argmax(1),
            "sample_mse": np.ones(4),
            "sample_mae": np.ones(4),
            "environment_centroids": np.array([[-0.7, -0.3], [0.8, 0.5]]),
            "random_partition_conflicts": np.linspace(0.0, 1.0, 20),
            "conflict_ours": np.asarray(1.2),
        }
        record = {field: 1.0 for field in (
            "D_within_grad", "D_between_grad", "gradient_separation",
            "gradient_separation_gain", "conflict_ours",
            "random_partition_conflict_mean", "random_partition_conflict_std",
            "random_partition_z_score", "random_partition_p_value",
            "min_environment_mass", "normalized_assignment_entropy", "max_q_mean",
            "max_q_p10", "max_q_p50", "max_q_p90",
            "environment_similarity_correlation", "stage_ARI_to_previous",
            "stage_NMI_to_previous", "mean_q_change",
        )}
        record.update(
            random_partition_repeats=20,
            environment_mass=[0.5, 0.5],
            per_env_MSE=[1.0, 1.0],
            per_env_MAE=[1.0, 1.0],
            per_env_mean_sample_loss=[1.0, 1.0],
            env_gradient_scale=[-0.7, 0.8],
            env_gradient_bias=[-0.3, 0.5],
        )
        with tempfile.TemporaryDirectory() as temporary:
            monitor = EnvironmentQualityMonitor(temporary, "toy")
            monitor.capture(0, record, payload)
            monitor.finalize({"var_acc": 0.8, "inv_acc": 0.4})
            for name in (
                "environment_quality_summary.txt",
                "environment_quality_stages.csv",
                "environment_gradient_final.png",
                "environment_gradient_stage_0.png",
                "environment_conflict_null.png",
                "environment_separation.png",
                "environment_stability.png",
                "environment_gradient_final.csv",
            ):
                self.assertTrue((Path(temporary) / name).is_file(), name)
            rows, csv_path = write_outputs(
                temporary, Path(temporary) / "all_datasets.txt"
            )
            self.assertEqual(len(rows), 1)
            self.assertTrue(csv_path.is_file())


if __name__ == "__main__":
    unittest.main()
