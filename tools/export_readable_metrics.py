"""Export readable metric files for historical experiment results."""

import argparse
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.metrics import backfill_readable_metrics, rebuild_readable_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default=os.path.join(PROJECT_ROOT, "results"))
    args = parser.parse_args()
    written = backfill_readable_metrics(args.results_dir)
    summary_path = rebuild_readable_metrics(args.results_dir)
    print("Wrote {} metrics.txt files".format(len(written)))
    print("Summary: {}".format(summary_path))


if __name__ == "__main__":
    main()
