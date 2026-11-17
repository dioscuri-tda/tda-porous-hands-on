r"""
Assemble per-structure descriptor vectors into one table per descriptor kind.

The descriptor step leaves one small `.npy` vector per structure and direction:

    <descroot>/<kind>/<direction tag>/<name>.npy        e.g. work/desc/ph/1_0_0/rtp_000.npy

This script turns every kind into a single wide CSV, one row per structure:

    <outdir>/<kind>.csv      columns: name, <kind>_<tag>_0, <kind>_<tag>_1, ...

so the column `ph_1_0_0_17` is element 17 of the persistence image computed
along direction [1 0 0]. A structure with a missing vector is an error (no
partial tables), and so is a NaN.

Usage
-----
    python common/collect_descriptors.py --descroot work/desc --outdir descriptors \
        --kinds ph ecp
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd


def tag_order(tag: str):
    """Axes first ([1 0 0], [0 1 0], [0 0 1]), then the diagonals."""
    v = [int(c) for c in tag.split("_")]
    return (sum(c != 0 for c in v), [-abs(c) for c in v], [-c for c in v])


def collect_kind(kind_dir: str, kind: str) -> pd.DataFrame:
    tags = sorted((t for t in os.listdir(kind_dir) if os.path.isdir(os.path.join(kind_dir, t))),
                  key=tag_order)
    if not tags:
        sys.exit(f"no direction directories in {kind_dir}")
    blocks, names_ref = [], None
    for tag in tags:
        d = os.path.join(kind_dir, tag)
        files = sorted(f for f in os.listdir(d) if f.endswith(".npy"))
        names = [f[:-4] for f in files]
        if names_ref is None:
            names_ref = names
        elif names != names_ref:
            sys.exit(f"{kind}/{tag}: structures differ from the other directions - rerun that stage")
        mat = np.stack([np.load(os.path.join(d, f)).ravel() for f in files])
        blocks.append(pd.DataFrame(mat, columns=[f"{kind}_{tag}_{i}" for i in range(mat.shape[1])]))
        print(f"  {kind}/{tag}: {mat.shape[0]} structures x {mat.shape[1]} features")
    out = pd.concat(blocks, axis=1)
    out.insert(0, "name", names_ref)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--descroot", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--kinds", nargs="+", required=True)
    a = p.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    for kind in a.kinds:
        kind_dir = os.path.join(a.descroot, kind)
        if not os.path.isdir(kind_dir):
            sys.exit(f"missing {kind_dir} - was the '{kind}' stage run?")
        df = collect_kind(kind_dir, kind)
        if df.isna().any().any():
            sys.exit(f"{kind}: NaNs present - investigate before training")
        path = os.path.join(a.outdir, f"{kind}.csv")
        df.to_csv(path, index=False, float_format="%.6g")
        print(f"  -> {path}  ({df.shape[0]} rows x {df.shape[1] - 1} features)")


if __name__ == "__main__":
    main()
