import unittest

import numpy as np

from tools.visualize_environment_representations import stratified_selection


class EnvironmentRepresentationTsneTest(unittest.TestCase):
    def test_stratified_selection_is_deterministic_and_covers_environments(self):
        labels = np.repeat(np.arange(3), (30, 20, 10))
        first = stratified_selection(labels, 15)
        second = stratified_selection(labels, 15)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 15)
        self.assertEqual(set(labels[first]), {0, 1, 2})
        self.assertTrue(np.all(first[:-1] < first[1:]))


if __name__ == "__main__":
    unittest.main()
