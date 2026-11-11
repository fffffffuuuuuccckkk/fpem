"""Print stored FPEM reference metadata without loading any training data."""

import argparse
import json

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    payload = torch.load(args.path, map_location="cpu", weights_only=False)
    print(json.dumps(payload.get("signature"), indent=2))


if __name__ == "__main__":
    main()
