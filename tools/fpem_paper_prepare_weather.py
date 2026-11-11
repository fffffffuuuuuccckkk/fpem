"""Copy Weather short-horizon run config into the independent paper root."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--source", type=Path, required=True)
p.add_argument("--destination", type=Path, required=True)
args = p.parse_args()
source = json.loads((args.source / "base_run_config.json").read_text())
args.destination.mkdir(parents=True, exist_ok=True)
reference = Path(source["reference_checkpoint"])
if not reference.is_file():
    reference = args.destination / "shared_reference.pt"
source["reference_checkpoint"] = str(reference.resolve())
target = args.destination / "base_run_config.json"
if target.exists():
    old = json.loads(target.read_text())
    if old != source:
        raise ValueError(f"Existing paper config differs: {target}")
else:
    target.write_text(json.dumps(source, indent=2) + "\n")
print(target, "reference_exists", reference.is_file(), flush=True)
