"""Convert the canonical Solar-AL benchmark to Dataset_Custom's date/OT CSV."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

URL = ("https://raw.githubusercontent.com/laiguokun/"
       "multivariate-time-series-data/master/solar-energy/solar_AL.txt.gz")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--archive", required=True)
    args = p.parse_args()
    output, archive = Path(args.output), Path(args.archive)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not archive.is_file():
        error = None
        for attempt in range(5):
            try:
                with urllib.request.urlopen(URL, timeout=90) as response:
                    data = response.read()
                gzip.decompress(data[:1024] + data[1024:])
                archive.write_bytes(data)
                break
            except Exception as exc:
                error = exc
                time.sleep(min(30, 2 ** attempt))
        else:
            raise RuntimeError("Solar-AL source download failed") from error
    if output.is_file():
        print("REUSE", output)
        return
    with gzip.open(archive, "rt") as source:
        values = np.loadtxt(source, delimiter=",", dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError(f"invalid Solar-AL shape/values: {values.shape}")
    names = [f"solar_{i:03d}" for i in range(values.shape[1] - 1)] + ["OT"]
    frame = pd.DataFrame(values, columns=names)
    frame.insert(0, "date", pd.date_range("2006-01-01", periods=len(frame), freq="10min"))
    frame.to_csv(output, index=False)
    meta = {"source": URL, "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "rows": len(frame), "channels": values.shape[1],
            "timestamp_note": "synthetic regular 10-minute index; original file has no timestamps"}
    output.with_suffix(".provenance.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(meta)


if __name__ == "__main__":
    main()
