import unittest

import numpy as np

from tools.run_fpem_eiil3_r3_then_f03 import corrected_record


class ReverseStageTest(unittest.TestCase):
    def test_fixed_r3_residual_target(self):
        y = np.array([[2., 3.]], dtype=np.float32)
        r3 = np.array([[.5, -.2]], dtype=np.float32)
        record = {"y": y, "design": np.array([[1.]], dtype=np.float32)}
        adjusted = corrected_record(record, r3)
        np.testing.assert_allclose(adjusted["y"] + r3, y)
        self.assertIs(adjusted["design"], record["design"])
        np.testing.assert_allclose(record["y"], y)


if __name__ == "__main__":
    unittest.main()
