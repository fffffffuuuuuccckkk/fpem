import os

import numpy as np

from utils.metrics import backfill_readable_metrics, save_readable_metrics


def test_save_and_backfill_readable_metrics(tmp_path):
    first = tmp_path / "experiment_a"
    first.mkdir()
    values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    np.save(first / "metrics.npy", values)
    save_readable_metrics(str(first), "experiment_a", values, dtw="Not calculated")
    text = (first / "metrics.txt").read_text(encoding="utf-8")
    assert "MAE: 0.1" in text
    assert "MSE: 0.2" in text
    assert "DTW: Not calculated" in text

    second = tmp_path / "experiment_b"
    second.mkdir()
    np.save(second / "metrics.npy", values * 2)
    written = backfill_readable_metrics(str(tmp_path))
    assert len(written) == 2
    assert os.path.isfile(second / "metrics.txt")
    summary = (tmp_path / "metrics_summary.txt").read_text(encoding="utf-8")
    assert "experiment_a" in summary
    assert "experiment_b" in summary
