import unittest
from types import SimpleNamespace

import numpy as np

from tools.visualize_environment_representations import (
    environment_for_visualization,
    stratified_selection,
)


class EnvironmentRepresentationTsneTest(unittest.TestCase):
    def test_stratified_selection_is_deterministic_and_covers_environments(self):
        labels = np.repeat(np.arange(3), (30, 20, 10))
        first = stratified_selection(labels, 15)
        second = stratified_selection(labels, 15)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 15)
        self.assertEqual(set(labels[first]), {0, 1, 2})
        self.assertTrue(np.all(first[:-1] < first[1:]))

    def test_training_supervision_replays_deterministic_shuffle(self):
        q = np.eye(3, dtype=np.float32)[np.arange(12) % 3]
        environment = {"q": q, "hard_env": q.argmax(1)}
        run_args = SimpleNamespace(
            epochs=10,
            stage_epochs=2,
            environment_supervision="shuffled",
            seed=2021,
        )
        first, mode, stage = environment_for_visualization(
            environment, run_args, "training_supervision"
        )
        second, _, _ = environment_for_visualization(
            environment, run_args, "training_supervision"
        )
        np.testing.assert_array_equal(first["q"], second["q"])
        self.assertEqual(mode, "shuffled")
        self.assertEqual(stage, 4)
        self.assertCountEqual(first["hard_env"], environment["hard_env"])
        self.assertFalse(np.array_equal(first["q"], environment["q"]))


if __name__ == "__main__":
    unittest.main()
