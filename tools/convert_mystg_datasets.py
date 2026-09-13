#!/usr/bin/env python3
"""Convert MySTG split NPY datasets to Time-Series-Library CSV files.

The MySTG export keeps normalized calendar components instead of the original
date strings. This converter reconstructs a regular date index, validates it
against those components, and records every upstream inconsistency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


SPLITS: Tuple[str, ...] = ("train", "val", "test")
ETT_COLUMNS = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")


@dataclass(frozen=True)
class DatasetSpec:
    source_name: str
    output_path: str
    preferred_start_year: int
    preferred_start_month: int
    known_columns: Tuple[str, ...] | None = None


DATASETS: Tuple[DatasetSpec, ...] = (
    DatasetSpec("Electricity", "dataset/electricity/electricity.csv", 2016, 7),
    DatasetSpec("ETTh2", "dataset/ETT-small/ETTh2.csv", 2016, 7, ETT_COLUMNS),
    DatasetSpec("ETTm1", "dataset/ETT-small/ETTm1.csv", 2016, 7, ETT_COLUMNS),
    DatasetSpec("ETTm2", "dataset/ETT-small/ETTm2.csv", 2016, 7, ETT_COLUMNS),
    DatasetSpec("ExchangeRate", "dataset/exchange_rate/exchange_rate.csv", 1990, 1),
    DatasetSpec("Weather", "dataset/weather/weather.csv", 2020, 1),
    DatasetSpec("Traffic", "dataset/traffic/traffic.csv", 2016, 7),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Directory containing Electricity/, ETTh2/, ... split NPY files.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
        help="Time-Series-Library-FPEM repository root.",
    )
    parser.add_argument("--chunk-rows", type=int, default=1024)
    return parser.parse_args()


def sha256_file(path: Path, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_components(features: np.ndarray, frequency_minutes: int) -> Tuple[np.ndarray, ...]:
    if features.ndim != 2 or features.shape[1] != 4:
        raise ValueError(f"timestamp features must have shape [T, 4], got {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("timestamp features contain NaN or infinity")

    minute_of_day = np.rint(features[:, 0].astype(np.float64) * 1440.0).astype(np.int64)
    weekday = np.rint(features[:, 1].astype(np.float64) * 7.0).astype(np.int64)
    day = np.rint(features[:, 2].astype(np.float64) * 31.0).astype(np.int64) + 1
    day_of_year = np.rint(features[:, 3].astype(np.float64) * 366.0).astype(np.int64) + 1

    if np.any(minute_of_day < 0) or np.any(minute_of_day >= 1440):
        raise ValueError("decoded minute-of-day is outside [0, 1440)")
    if np.any(weekday < 0) or np.any(weekday > 6):
        raise ValueError("decoded weekday is outside [0, 6]")
    if np.any(day < 1) or np.any(day > 31):
        raise ValueError("decoded day-of-month is outside [1, 31]")
    if np.any(day_of_year < 1) or np.any(day_of_year > 366):
        raise ValueError("decoded day-of-year is outside [1, 366]")
    if np.any(minute_of_day % frequency_minutes != 0):
        raise ValueError("decoded time-of-day is not aligned to the declared frequency")
    return minute_of_day, weekday, day, day_of_year


def reconstruct_timestamps(
    features: np.ndarray,
    frequency_minutes: int,
    preferred_start_year: int,
    preferred_start_month: int,
) -> Tuple[pd.DatetimeIndex, int, Dict[str, int], bool, bool]:
    minute_of_day, weekday, day, day_of_year = _decode_components(features, frequency_minutes)

    # Some upstream generators computed time-of-day from the row index even
    # when the source CSV began after midnight. Infer the real clock offset
    # from the first day boundary; day and weekday remain authoritative.
    day_changes = np.flatnonzero(day[1:] != day[:-1]) + 1
    if len(day_changes):
        start_minute = int((-int(day_changes[0]) * frequency_minutes) % 1440)
    else:
        start_minute = int(minute_of_day[0])

    doy_channel_reliable = not np.allclose(features[:, 2], features[:, 3], atol=1e-6)
    candidates: List[Tuple[int, int, pd.DatetimeIndex, int, int]] = []
    for base_year in range(1970, 2101):
        for base_month in range(1, 13):
            try:
                start = pd.Timestamp(
                    year=base_year,
                    month=base_month,
                    day=int(day[0]),
                ) + pd.Timedelta(minutes=start_minute)
            except ValueError:
                continue
            stamps = pd.date_range(
                start=start,
                periods=len(features),
                freq=pd.Timedelta(minutes=frequency_minutes),
            )
            if not np.array_equal(stamps.weekday.to_numpy(), weekday):
                continue
            if not np.array_equal(stamps.day.to_numpy(), day):
                continue
            observed_minutes = stamps.hour.to_numpy() * 60 + stamps.minute.to_numpy()
            tod_mismatches = int(np.count_nonzero(observed_minutes != minute_of_day))
            doy_mismatches = int(
                np.count_nonzero(stamps.dayofyear.to_numpy() != day_of_year)
            )
            candidates.append(
                (base_year, base_month, stamps, tod_mismatches, doy_mismatches)
            )
    used_regular_fallback = not candidates
    if candidates:
        base_year, base_month, stamps, _, _ = min(
            candidates,
            key=lambda item: (
                item[4] if doy_channel_reliable else 0,
                abs(item[0] - preferred_start_year),
                abs(item[1] - preferred_start_month),
                item[0],
                item[1],
            ),
        )
    else:
        # Weather contains a few upstream calendar discontinuities. A regular
        # synthetic index is safer for forecasting than duplicate/out-of-order
        # dates, and every disagreement is recorded in the manifest.
        base_year = preferred_start_year
        base_month = preferred_start_month
        start = pd.Timestamp(
            year=base_year, month=base_month, day=int(day[0])
        ) + pd.Timedelta(minutes=start_minute)
        stamps = pd.date_range(
            start=start,
            periods=len(features),
            freq=pd.Timedelta(minutes=frequency_minutes),
        )

    observed_minutes = stamps.hour.to_numpy() * 60 + stamps.minute.to_numpy()
    mismatches = {
        "time_of_day": int(np.count_nonzero(observed_minutes != minute_of_day)),
        "weekday": int(np.count_nonzero(stamps.weekday.to_numpy() != weekday)),
        "day_of_month": int(np.count_nonzero(stamps.day.to_numpy() != day)),
        "day_of_year": int(
            np.count_nonzero(stamps.dayofyear.to_numpy() != day_of_year)
        ),
    }
    return stamps, base_year, mismatches, doy_channel_reliable, used_regular_fallback


def make_columns(spec: DatasetSpec, width: int) -> List[str]:
    if spec.known_columns is not None:
        if len(spec.known_columns) != width:
            raise ValueError(
                f"{spec.source_name}: expected {len(spec.known_columns)} variables, got {width}"
            )
        return list(spec.known_columns)
    if width < 1:
        raise ValueError(f"{spec.source_name}: dataset has no variables")
    return [str(index) for index in range(width - 1)] + ["OT"]


def load_split_files(source_dir: Path) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, str]]:
    data_arrays: List[np.ndarray] = []
    timestamp_arrays: List[np.ndarray] = []
    hashes: Dict[str, str] = {}
    for split in SPLITS:
        data_path = source_dir / f"{split}_data.npy"
        timestamp_path = source_dir / f"{split}_timestamps.npy"
        if not data_path.is_file() or not timestamp_path.is_file():
            raise FileNotFoundError(f"missing {split} files in {source_dir}")
        data = np.load(data_path, mmap_mode="r", allow_pickle=False)
        timestamps = np.load(timestamp_path, mmap_mode="r", allow_pickle=False)
        if data.ndim != 2:
            raise ValueError(f"{data_path}: expected [T, C], got {data.shape}")
        if timestamps.shape != (len(data), 4):
            raise ValueError(
                f"{timestamp_path}: shape {timestamps.shape} does not match {len(data)} rows"
            )
        data_arrays.append(data)
        timestamp_arrays.append(timestamps)
        hashes[data_path.name] = sha256_file(data_path)
        hashes[timestamp_path.name] = sha256_file(timestamp_path)
    return data_arrays, timestamp_arrays, hashes


def write_csv(
    output_path: Path,
    arrays: Sequence[np.ndarray],
    timestamps: pd.DatetimeIndex,
    columns: Sequence[str],
    chunk_rows: int,
) -> Tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary_path.exists():
        temporary_path.unlink()

    row_offset = 0
    wrote_header = False
    try:
        for array in arrays:
            for start in range(0, len(array), chunk_rows):
                stop = min(start + chunk_rows, len(array))
                values = np.asarray(array[start:stop])
                if not np.isfinite(values).all():
                    bad_count = int(values.size - np.isfinite(values).sum())
                    raise ValueError(f"found {bad_count} NaN/inf values while writing {output_path}")
                frame = pd.DataFrame(values, columns=columns)
                date_values = timestamps[row_offset + start : row_offset + stop].strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                frame.insert(0, "date", date_values)
                frame.to_csv(
                    temporary_path,
                    mode="a",
                    header=not wrote_header,
                    index=False,
                    float_format="%.10g",
                )
                wrote_header = True
            row_offset += len(array)
        os.replace(temporary_path, output_path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise
    return row_offset, output_path.stat().st_size


def convert_dataset(
    spec: DatasetSpec, source_root: Path, project_root: Path, chunk_rows: int
) -> Dict[str, object]:
    source_dir = source_root / spec.source_name
    meta_path = source_dir / "meta.json"
    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    frequency_minutes = int(meta["frequency (minutes)"])

    arrays, timestamp_arrays, source_hashes = load_split_files(source_dir)
    widths = {array.shape[1] for array in arrays}
    if len(widths) != 1:
        raise ValueError(f"{spec.source_name}: split variable counts differ: {sorted(widths)}")
    width = widths.pop()
    split_rows = {split: int(len(array)) for split, array in zip(SPLITS, arrays)}
    expected_shape = tuple(meta.get("shape", ()))
    if expected_shape and expected_shape != (sum(split_rows.values()), width):
        raise ValueError(
            f"{spec.source_name}: meta shape {expected_shape} differs from split shape "
            f"{(sum(split_rows.values()), width)}"
        )

    all_timestamp_features = np.concatenate(timestamp_arrays, axis=0)
    timestamps, inferred_year, timestamp_mismatches, doy_reliable, calendar_fallback = reconstruct_timestamps(
        all_timestamp_features,
        frequency_minutes,
        spec.preferred_start_year,
        spec.preferred_start_month,
    )
    columns = make_columns(spec, width)
    output_path = project_root / spec.output_path
    rows_written, bytes_written = write_csv(
        output_path, arrays, timestamps, columns, chunk_rows
    )

    # Cheap post-write checks catch truncation and column-order mistakes without
    # loading very wide datasets back into memory.
    header = pd.read_csv(output_path, nrows=0).columns.tolist()
    if header != ["date", *columns]:
        raise RuntimeError(f"{output_path}: unexpected output header")
    if rows_written != len(timestamps):
        raise RuntimeError(f"{output_path}: wrote {rows_written}, expected {len(timestamps)}")

    return {
        "dataset": spec.source_name,
        "output": str(output_path),
        "rows": rows_written,
        "variables": width,
        "split_rows": split_rows,
        "frequency_minutes": frequency_minutes,
        "start": str(timestamps[0]),
        "end": str(timestamps[-1]),
        "inferred_start_year": inferred_year,
        "irregular_intervals": 0,
        "timestamp_feature_mismatches": timestamp_mismatches,
        "day_of_year_feature_reliable": doy_reliable,
        "regular_calendar_fallback": calendar_fallback,
        "non_finite_values": 0,
        "bytes": bytes_written,
        "csv_sha256": sha256_file(output_path),
        "source_sha256": source_hashes,
    }


def write_manifests(project_root: Path, reports: Iterable[Dict[str, object]]) -> None:
    reports = list(reports)
    json_path = project_root / "dataset" / "mystg_import_manifest.json"
    text_path = project_root / "dataset" / "mystg_import_manifest.txt"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(reports, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "MySTG -> Time-Series-Library dataset import",
        "All values and timestamp features came from TRAIN/VAL/TEST source files only.",
        "CSV layout: date first, numeric predictors next, OT target last.",
        "",
    ]
    for report in reports:
        lines.extend(
            [
                f"[{report['dataset']}]",
                f"output: {report['output']}",
                f"shape: {report['rows']} rows x {report['variables']} variables",
                f"split_rows: {report['split_rows']}",
                f"frequency_minutes: {report['frequency_minutes']}",
                f"date_range: {report['start']} -> {report['end']}",
                f"inferred_start_year: {report['inferred_start_year']}",
                f"irregular_intervals: {report['irregular_intervals']}",
                f"timestamp_feature_mismatches: {report['timestamp_feature_mismatches']}",
                f"day_of_year_feature_reliable: {report['day_of_year_feature_reliable']}",
                f"regular_calendar_fallback: {report['regular_calendar_fallback']}",
                f"non_finite_values: {report['non_finite_values']}",
                f"csv_sha256: {report['csv_sha256']}",
                "",
            ]
        )
    text_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.chunk_rows <= 0:
        raise ValueError("--chunk-rows must be positive")
    source_root = args.source_root.expanduser().resolve()
    project_root = args.project_root.expanduser().resolve()

    reports: List[Dict[str, object]] = []
    for spec in DATASETS:
        print(f"[convert] {spec.source_name} -> {spec.output_path}", flush=True)
        report = convert_dataset(spec, source_root, project_root, args.chunk_rows)
        reports.append(report)
        print(
            f"[ok] {spec.source_name}: {report['rows']} rows, "
            f"{report['variables']} variables, {report['bytes']} bytes",
            flush=True,
        )
    write_manifests(project_root, reports)
    print(f"[done] imported {len(reports)} datasets", flush=True)


if __name__ == "__main__":
    main()
