#!/usr/bin/env python3
"""Make a compact, immutable mirror of Stage-1 labels for another server."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

METHODS=("G3_KerHRM_lite","G4_ANM_MM_lite","G7_Ridge_Update",
         "B0_Raw_History","B1_Random_Balanced","B2_ShuffleY_G1")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",required=True)
    parser.add_argument("--destination",required=True)
    args=parser.parse_args()
    source=Path(args.source);destination=Path(args.destination)
    destination.mkdir(parents=True,exist_ok=True)
    with np.load(source/"case_cache.npz") as loaded:
        np.savez_compressed(destination/"case_indices.npz",C_ids=loaded["C_ids"],
                            C_channels=loaded["C_channels"])
    shutil.copy2(source/"case_cache.json",destination/"case_cache.json")
    for method in METHODS:
        target=destination/"methods"/method
        target.mkdir(parents=True,exist_ok=True)
        for name in ("assignments_train.npz","prototype_meta.json"):
            shutil.copy2(source/"methods"/method/name,target/name)
    print(destination)


if __name__=="__main__":main()
