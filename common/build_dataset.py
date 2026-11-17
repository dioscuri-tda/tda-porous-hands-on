r"""
Join the topological descriptors with the labels into one training table.

Reads the label table (one row per structure, key column `name`) and any
number of descriptor tables written by `collect_descriptors.py`, inner-joins
them on `name` and writes a single CSV: labels first, descriptor columns after.

Usage
-----
    python common/build_dataset.py --labels data/stiffness.csv \
        --descriptors descriptors/ph.csv descriptors/ecp.csv \
        --output dataset.csv
"""

import argparse
import os
import sys

import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", required=True)
    p.add_argument("--descriptors", nargs="+", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    df = pd.read_csv(a.labels)
    n_labels = len(df)
    print(f"labels      {a.labels}: {n_labels} structures, {df.shape[1] - 1} columns")
    for path in a.descriptors:
        if not os.path.exists(path):
            sys.exit(f"missing descriptor table {path}\n"
                     f"  run 01_compute_descriptors.sh first, or use --reference "
                     f"to take the precomputed tables")
        d = pd.read_csv(path)
        clash = (set(d.columns) & set(df.columns)) - {"name"}
        if clash:
            sys.exit(f"{path}: columns already present: {sorted(clash)[:5]}")
        df = df.merge(d, on="name", how="inner", validate="one_to_one")
        print(f"descriptors {path}: {len(d)} structures, {d.shape[1] - 1} features")

    if len(df) != n_labels:
        sys.exit(f"only {len(df)} of {n_labels} structures have all descriptors")
    if df.isna().any().any():
        sys.exit("NaNs in the joined table")

    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    df.to_csv(a.output, index=False)
    groups = sorted({c.split("_")[0] for d in a.descriptors
                     for c in pd.read_csv(d, nrows=0).columns if c != "name"})
    print(f"-> {a.output}: {df.shape[0]} rows x {df.shape[1]} columns "
          f"(descriptor groups: {', '.join(groups)})")


if __name__ == "__main__":
    main()
