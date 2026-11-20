import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from models.predictive_env.periodic_exog import (
    periodic_features_for_dataset,
    periodic_features_from_dates,
)


class PeriodicExogTest(unittest.TestCase):
    def test_calendar_phase_not_window_phase(self):
        dates = pd.date_range("2024-01-01", periods=24 * 8, freq="h")
        feature = periodic_features_from_dates(dates, "ETTh1")
        self.assertEqual(feature.shape, (192, 4))
        np.testing.assert_allclose(feature[0, :2], feature[24, :2], atol=1e-6)
        np.testing.assert_allclose(feature[0], feature[24 * 7], atol=1e-6)
        self.assertGreater(np.max(np.abs(feature[0] - feature[24])), 0.1)

    def test_minute_resolution_and_modes(self):
        dates = pd.date_range("2024-01-01", periods=96, freq="15min")
        all_features = periodic_features_from_dates(dates, "ETTm2")
        day = periodic_features_from_dates(dates, "ETTm2", mode="day")
        week = periodic_features_from_dates(dates, "ETTm2", mode="week")
        np.testing.assert_allclose(all_features[:, :2], day)
        np.testing.assert_allclose(all_features[:, 2:], week)
        self.assertEqual(day.shape, (96, 2))

    def test_daily_exchange_rate_uses_real_weekday_phase(self):
        dates = pd.date_range("2024-01-01", periods=14, freq="D")
        week = periodic_features_from_dates(
            dates, "ExchangeRate", mode="week"
        )
        self.assertEqual(week.shape, (14, 2))
        np.testing.assert_allclose(week[0], week[7], atol=1e-6)
        self.assertGreater(np.max(np.abs(week[0] - week[1])), 0.1)
        day = periodic_features_from_dates(dates, "ExchangeRate", mode="day")
        np.testing.assert_allclose(day, np.tile([[0.0, 1.0]], (14, 1)), atol=1e-6)

    def test_split_alignment_custom(self):
        with tempfile.TemporaryDirectory() as tmp:
            dates = pd.date_range("2024-01-01", periods=100, freq="h")
            path = Path(tmp) / "Electricity.csv"
            pd.DataFrame({"date": dates}).to_csv(path, index=False)
            # Custom TEST starts at n-n_test-seq_len = 74.
            dataset = SimpleNamespace(root_path=tmp, data_path=path.name,
                                      seq_len=6, data_x=np.zeros((26, 1)))
            stamp = periodic_features_for_dataset(dataset, "Electricity", "test")
            expected = periodic_features_from_dates(dates[74:], "Electricity")
            np.testing.assert_allclose(stamp, expected)

    def test_invalid_timestamp(self):
        with self.assertRaises(ValueError):
            periodic_features_from_dates(["2024-01-01 00:05"], "ETTh1")

    def test_native_timef_split(self):
        marks = np.arange(24, dtype=np.float32).reshape(6, 4) / 24
        dataset = SimpleNamespace(timeenc=1, data_stamp=marks,
                                  data_x=np.zeros((6, 2), dtype=np.float32))
        result = periodic_features_for_dataset(dataset, "ETTh1", "train", mode="timeF")
        np.testing.assert_array_equal(result, marks)
        result[0, 0] = -10
        self.assertNotEqual(dataset.data_stamp[0, 0], result[0, 0])


if __name__ == "__main__":
    unittest.main()
