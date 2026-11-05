import sys
import torch

for name in sys.argv[1:]:
    value = torch.load(name, map_location="cpu", weights_only=False)
    print(name, type(value).__name__, list(value) if isinstance(value, dict) else "")
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, dict):
                print("  ", key, list(item)[:25])
