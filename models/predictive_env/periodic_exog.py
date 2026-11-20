"""History-only, absolute-calendar periodic covariates for FPEM.

The phase is defined by the timestamp (weekday and time of day), never by
the start of a forecasting window.  No future target or future timestamp is
needed by the encoder.  The split boundaries mirror data_provider/data_loader.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


_PERIODS = {
    "etth1": (24, 168),
    "etth2": (24, 168),
    "ettm1": (96, 672),
    "ettm2": (96, 672),
    "electricity": (24, 168),
    "traffic": (24, 168),
    "weather": (144, 1008),
    "exchangerate": (1, 7),
    "solar": (144, 1008),
}


def periodic_features_from_dates(dates, dataset_name: str, mode: str = "both") -> np.ndarray:
    """Return [N,2/4] sine-cosine features using absolute calendar phase.

    For example, ETTm data sampled every 15 minutes uses 96 positions per day
    and 672 per week.  A Monday 00:00 has the same phase in every split.
    """
    key = dataset_name.lower().replace(".csv", "")
    if key not in _PERIODS:
        raise ValueError("unsupported periodic dataset: {}".format(dataset_name))
    if mode not in ("day", "week", "both"):
        raise ValueError("mode must be day, week or both")
    stamps = pd.DatetimeIndex(pd.to_datetime(dates))
    if stamps.hasnans:
        raise ValueError("date column contains missing timestamps")
    day_period, week_period = _PERIODS[key]
    step_minutes = 1440 // day_period
    minute_of_day = stamps.hour.to_numpy() * 60 + stamps.minute.to_numpy()
    if np.any(minute_of_day % step_minutes):
        raise ValueError("timestamps are not aligned with {} minute grid".format(step_minutes))
    day_step = minute_of_day // step_minutes
    week_step = stamps.weekday.to_numpy() * day_period + day_step
    pieces = []
    if mode in ("day", "both"):
        angle = (2.0 * np.pi / day_period) * day_step
        pieces.extend((np.sin(angle), np.cos(angle)))
    if mode in ("week", "both"):
        angle = (2.0 * np.pi / week_period) * week_step
        pieces.extend((np.sin(angle), np.cos(angle)))
    result = np.stack(pieces, axis=-1).astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise ValueError("periodic features contain nonfinite values")
    return result


def _split_bounds(dataset_name: str, split: str, count: int, seq_len: int):
    if split not in ("train", "val", "test"):
        raise ValueError("split must be train, val or test")
    key = dataset_name.lower().replace(".csv", "")
    position = {"train": 0, "val": 1, "test": 2}[split]
    if key in ("etth1", "etth2"):
        left = [0, 12 * 30 * 24 - seq_len, 16 * 30 * 24 - seq_len]
        right = [12 * 30 * 24, 16 * 30 * 24, 20 * 30 * 24]
    elif key in ("ettm1", "ettm2"):
        left = [0, 12 * 30 * 24 * 4 - seq_len, 16 * 30 * 24 * 4 - seq_len]
        right = [12 * 30 * 24 * 4, 16 * 30 * 24 * 4, 20 * 30 * 24 * 4]
    else:
        n_train = int(count * 0.7)
        n_test = int(count * 0.2)
        left = [0, n_train - seq_len, count - n_test - seq_len]
        right = [n_train, count - n_test, count]
    start, end = left[position], right[position]
    if start < 0 or end > count or start >= end:
        raise ValueError("invalid split bounds for {} {}".format(dataset_name, split))
    return start, end


def periodic_features_for_dataset(dataset, dataset_name: str, split: str,
                                  mode: str = "both") -> np.ndarray:
    """Build a split-aligned grid; sample i reads rows i:i+seq_len only.

    Sine/cosine modes read the absolute CSV dates.  ``timeF`` is the native
    iTransformer/TSL normalized calendar encoding, already built by the
    dataset from the same split's timestamps.  It remains history-only.
    """
    if mode == "timeF":
        if getattr(dataset, "timeenc", None) != 1:
            raise ValueError("timeF requires dataset timeenc=1")
        features = np.asarray(dataset.data_stamp, dtype=np.float32)
        if features.ndim != 2 or features.shape[0] != len(dataset.data_x):
            raise ValueError("timeF calendar/data split length mismatch")
        if not np.isfinite(features).all():
            raise ValueError("timeF contains nonfinite values")
        return features.copy()
    data_path = Path(dataset.root_path) / dataset.data_path
    if not data_path.is_file():
        raise FileNotFoundError("calendar CSV required for token A0: {}".format(data_path))
    dates = pd.read_csv(str(data_path), usecols=["date"])["date"]
    start, end = _split_bounds(dataset_name, split, len(dates), int(dataset.seq_len))
    if len(dataset.data_x) != end - start:
        raise ValueError("calendar/data split length mismatch: {} != {}".format(
            len(dataset.data_x), end - start))
    return periodic_features_from_dates(dates.iloc[start:end], dataset_name, mode)
